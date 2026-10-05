"""QDrive in ROUNDS: the sequential state-preparation plan cut into many small chained jobs, each scored before the next is sent.

Why (docs/qdrive-field-notes.md, docs/qdrive-solution-design.md): one QDrive job per scene stalls or times out, and a stalled job tells
you nothing. Here every independent component of the target graph (built-in bay window: patches+lamps 14 qubits, polarity 6; living room: 11 and 8) is prepared by a
chain of jobs. Round r is job r of every component; job r continues job r-1's returned circuit (`initial_circuit` = its output ASSET
uuid) and holds only `--layers-per-round` layers of targets. After each job comes back it is simulated locally and compared with the
state the plan says the component should be in at that point (total variation and fidelity); the run stops, before spending the next
credit, when a round fails or lands further than `--max-tv` from its target. Finished chains are merged locally into one circuit.

Piece-wise outputs: everything is kept per step under <out>/qdrive/rounds/<step>/ with a state file that records every job id, so a run
can be stopped and resumed (completed steps are never resent), and `assemble` builds the show's `qdrive/circuit.qasm` from whatever is
done: all components finished -> circuit.qasm; otherwise circuit_partial.qasm (unfinished components still |0...0>), never circuit.qasm.
A component can be filled by a LOCAL circuit (exact state preparation, no Moth call) with --local-fill; provenance.json then says so and
the show does not call the result a Moth result.

    python -m src.quantum.qdrive_rounds plan     --calib runs/harrietlivingroom/calib --out runs/harrietlivingroom/solve
    python -m src.quantum.qdrive_rounds run      --calib ... --out ... --max-steps 2                          # preview: sends nothing
    python -m src.quantum.qdrive_rounds run      --calib ... --out ... --max-steps 2 --approve-credits 2     # sends 2 jobs (1 credit each)
    python -m src.quantum.qdrive_rounds assemble --calib ... --out ... [--local-fill]
    python -m src.quantum.qdrive_rounds canary   --out ... --approve-credits 1                                # known-good 2-qubit job: is the service up?
"""
import argparse
import glob
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from src.capture.labels import DEFAULT_CALIB_DIR, DEFAULT_RUN_DIR
from src.quantum import qdrive_plan as qp
from src.quantum.moth_client import MothClient, MothError, NotApproved, InvalidPayload, estimate_credits, output_asset_id
from src.targets.payloads import payload_sha256

ENGINE = "qdrive-api-v1"
LAYERS_PER_ROUND = 2
MAX_TV = 0.10                      # R3 in the solution design expects TV < 0.05; a round further off than this stops the run


class StaleRounds(MothError):
    """The rounds on disk were planned for different targets than the ones loaded now."""


# ---------------------------------------------------------------------------------------------------------- plan
@dataclass
class Step:
    id: str
    comp: int                      # index into RoundsPlan.comps
    index: int                     # position in the component's chain (0 = first job, no initial_circuit)
    lo: int                        # layers lo..hi-1 of the component's plan
    hi: int


@dataclass
class CompPlan:
    qubits: list                   # global qubit ids; local index = position
    prep: object                   # qdrive_plan.StatePrepPlan (local indices)


@dataclass
class RoundsPlan:
    n_qubits: int
    comps: list
    steps: list
    settings: dict
    sha: str

    def steps_of(self, comp):
        return [s for s in self.steps if s.comp == comp]

    def order(self):
        """Round-robin: job 0 of every component, then job 1 of every component, and so on."""
        return sorted(self.steps, key=lambda s: (s.index, s.comp))

    def params(self, step, seed=None):
        c = self.settings
        return qp.state_prep_params(self.comps[step.comp].prep, step.lo, step.hi, seed=c["seed"] if seed is None else seed,
                                    update_method=c["update_method"], shots=c["shots"], chained=step.index > 0, coupling_map=c["coupling_map"])


def marginal(p, n, qubits):
    """Distribution over `qubits` (local bit i = qubits[i]) from a probability vector over n qubits (qubit q is bit q)."""
    idx = np.arange(len(p))
    key = np.zeros(len(p), int)
    for i, q in enumerate(qubits):
        key |= ((idx >> q) & 1) << i
    out = np.zeros(2 ** len(qubits))
    np.add.at(out, key, np.asarray(p, float) / np.sum(p))
    return out


def plan_rounds(moment, prob, n, layers_per_round=LAYERS_PER_ROUND, seed=7, update_method="spectral", shots=1024, coupling_map=True, tol=1e-3):
    """Cut the scene into components and each component's state-preparation layers into chained steps.
    `moment` gives the target graph (qdrive_plan.TargetMoments); `prob` is the exact distribution over the n qubits."""
    pairs = moment.pairs()
    comps, steps = [], []
    for ci, comp in enumerate(qp.components(moment.qubits(), pairs)):
        prep = qp.plan_state_prep(comp, [p for p in pairs if p[0] in comp and p[1] in comp], marginal(prob, n, comp), tol)
        comps.append(CompPlan(list(comp), prep))
        L = len(prep.layers)
        for k, lo in enumerate(range(0, L, max(1, layers_per_round))):
            steps.append(Step(f"c{ci}s{k}", ci, k, lo, min(L, lo + max(1, layers_per_round))))
    settings = dict(layers_per_round=layers_per_round, seed=seed, update_method=update_method, shots=shots, coupling_map=coupling_map)
    rp = RoundsPlan(n, comps, steps, settings, "")
    rp.sha = payload_sha256({"engine": ENGINE, "settings": settings, "comps": [c.qubits for c in comps],
                             "steps": [dict(id=s.id, params=rp.params(s)) for s in steps]})
    return rp


def step_problems(rp, step):
    """Things the lab found risky (nothing here is a hard rule, except what PreparedJob.problems already blocks)."""
    out = []
    params = rp.params(step)
    big = max((len(t["qubits"]) for t in params["targets"] if t), default=0)
    if big > qp.MAX_GROUP:
        out.append(f"{step.id}: a {big}-qubit group (the lab timed out at 8, finished at 3; 4 is unvalidated)")
    width = len(rp.comps[step.comp].qubits)
    if width > 8:
        out.append(f"{step.id}: {width}-qubit register (8 qubits ran; 11 is unvalidated; 19 never ran)")
    return out


def describe(rp):
    rows = [f"{len(rp.comps)} component(s), {len(rp.steps)} job(s) = {len(rp.steps)} credit(s) at {estimate_credits(ENGINE)} credit per job:"]
    for s in rp.order():
        prep = rp.comps[s.comp].prep
        tg = [t for layer in prep.layers[s.lo:s.hi] for t in layer]
        rows.append(f"  {s.id}: round {s.index}, qubits {rp.comps[s.comp].qubits[0]}-{rp.comps[s.comp].qubits[-1]} ({prep.n}q), layers {s.lo}-{s.hi - 1}, "
                    f"{len(tg)} target(s), groups of {sorted({len(t['qubits']) for t in tg})}, {'continues ' + rp.steps_of(s.comp)[s.index - 1].id if s.index else 'starts fresh'}")
    return rows


# ---------------------------------------------------------------------------------------------------------- state on disk
def rounds_dir(base):
    return os.path.join(base, "rounds")


def load_state(base, rp):
    path = os.path.join(rounds_dir(base), "rounds_state.json")
    if not os.path.exists(path):
        return dict(plan_sha=rp.sha, steps={})
    with open(path) as f:
        st = json.load(f)
    if st.get("plan_sha") != rp.sha:
        raise StaleRounds(f"{path} was written for a different plan ({st.get('plan_sha', '')[:10]} != {rp.sha[:10]}). The targets or the round "
                          f"settings changed: use --restart (keeps the old folder, renamed) or a different --out.")
    return st


def save_state(base, st):
    os.makedirs(rounds_dir(base), exist_ok=True)
    path = os.path.join(rounds_dir(base), "rounds_state.json")
    with open(path + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(path + ".tmp", path)


def restart(base):
    d = rounds_dir(base)
    if os.path.isdir(d):
        os.rename(d, f"{d}_old_{time.strftime('%Y%m%d_%H%M%S')}")


def _read(path):
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------------------------------------- scoring
def score_piece(qasm_text, psi_target):
    """(tv, fidelity) of a returned circuit against the state the plan wanted at that step: total variation between Z-basis
    distributions, and |<target|circuit>|^2 (the stricter one: it sees phases and X/Y structure)."""
    from src.quantum import sampler_local as sl
    psi, n = sl.statevector_from_qasm(qasm_text)
    t = np.asarray(psi_target, complex)
    if len(psi) != len(t):
        raise ValueError(f"the circuit has {n} qubits but the step's target state has {int(np.log2(len(t)))}")
    p, q = np.abs(psi) ** 2, np.abs(t) ** 2
    return 0.5 * float(np.abs(p / p.sum() - q / q.sum()).sum()), float(abs(np.vdot(t / np.linalg.norm(t), psi / np.linalg.norm(psi))) ** 2)


def _mux_ry(qc, angles, controls, target):
    """Exact multiplexed Ry: Ry(angles[c]) on target for the control value c (controls[0] least significant), by the standard recursion
    on the most significant control (2^k CNOTs)."""
    if not controls:
        qc.ry(float(angles[0]), target)
        return
    half = len(angles) // 2
    lo, hi = np.asarray(angles[:half], float), np.asarray(angles[half:], float)
    _mux_ry(qc, (lo + hi) / 2, controls[:-1], target)
    qc.cx(controls[-1], target)
    _mux_ry(qc, (lo - hi) / 2, controls[:-1], target)
    qc.cx(controls[-1], target)


def local_circuit(psi):
    """OpenQASM 3 that prepares the pure state psi exactly (no Moth call): the stand-in for a piece Moth did not deliver.
    Built as a tree of multiplexed Ry rotations, which is exact for the real non-negative amplitudes every plan state has. (Qiskit's own
    prepare_state silently drops amplitudes about 1e-4 of the largest: on the 11-qubit living-room component it was only 42% faithful.)
    A state with negative or complex amplitudes falls back on prepare_state."""
    from qiskit import QuantumCircuit, qasm3, transpile
    psi = np.asarray(psi, complex)
    psi = psi / np.linalg.norm(psi)
    n = int(np.log2(len(psi)))
    qc = QuantumCircuit(n)
    if np.abs(psi.imag).max() > 1e-12 or psi.real.min() < -1e-12:
        qc.prepare_state(psi, list(range(n)))
    else:
        axes = np.abs(psi) ** 2
        axes = axes.reshape((2,) * n)                                # axis a is qubit n-1-a
        for t in range(n - 1, -1, -1):
            controls = list(range(t + 1, n))                         # controls[i] = qubit t+1+i
            m = axes.sum(axis=tuple(range(n - t, n))) if t else axes
            angles = []
            for c in range(2 ** len(controls)):
                idx = tuple((c >> (len(controls) - 1 - a)) & 1 for a in range(len(controls)))
                row = m[idx] if controls else m
                tot = row[0] + row[1]
                angles.append(2 * np.arcsin(np.sqrt(np.clip(row[1] / tot, 0, 1))) if tot > 1e-300 else 0.0)
            _mux_ry(qc, angles, controls, t)
    return qasm3.dumps(transpile(qc, basis_gates=["u", "cx"], optimization_level=1))


# ---------------------------------------------------------------------------------------------------------- a local stand-in engine
class LocalClient:
    """An IDEAL local QDrive: every job returns a circuit that prepares exactly the state the plan wants after the job's last layer,
    and, like the real service, only accepts a chained job whose initial_circuit is the asset it issued for the previous job. It exists to
    rehearse the whole rounds pipeline (state file, resume, scoring, assembling, the show) with no network and no credits, and it
    is recorded as local everywhere: the assembled circuit is never called a Moth result. It says nothing about how the real engine converges."""
    is_local = True

    def __init__(self, rp):
        self.rp, self.assets = rp, {}

    def prepare(self, engine_id, params, input_files=None):
        from src.quantum.moth_client import PreparedJob
        return PreparedJob(engine_id, params, input_files)

    def _step(self, params):
        return next(s for s in self.rp.steps if self.rp.params(s) == params)

    def run(self, engine_id, params, *, approved=False, out_dir=None, input_files=None):
        import uuid
        step = self._step(params)
        if step.index:
            prev = self.rp.steps_of(step.comp)[step.index - 1]
            if input_files != {"initial_circuit": self.assets.get(prev.id)}:
                raise MothError(f"local engine: {step.id} must continue the asset of {prev.id}")
        job_id = f"local-{uuid.uuid4()}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"job_{job_id}.json"), "w") as f:
            json.dump({"job_id": job_id, "engine_id": "local-ideal", "estimated_credits": 0}, f)
        path = os.path.join(out_dir, "circuit")
        with open(path, "w") as f:
            f.write(self.circuit_for(step))
        self.assets[step.id] = asset = str(uuid.uuid4())
        return dict(job_id=job_id, result={"outputs": [{"slot": "circuit", "output_asset_id": asset}], "result": None}, inline=None, files={"circuit": path})

    def circuit_for(self, step):
        return local_circuit(self.rp.comps[step.comp].prep.layer_psi[step.hi - 1])


# ---------------------------------------------------------------------------------------------------------- running
def _job_id_in(step_dir):
    files = sorted(glob.glob(os.path.join(step_dir, "job_*.json")), key=os.path.getmtime)
    return os.path.basename(files[-1])[4:-5] if files else None


def pending_steps(rp, st, retry_failed=False, components=None):
    """(queue, blocked): steps that would be sent next, in send order, and {step id: why not} for chains stopped by a failed or
    unrecorded job. Completed steps are never queued."""
    queue, blocked, stopped = [], {}, set()
    for s in rp.order():
        if components is not None and s.comp not in components:
            continue
        rec = st["steps"].get(s.id, {})
        if rec.get("status") == "completed":
            continue
        if s.comp in stopped:
            blocked[s.id] = "an earlier step of this component is not completed"
            continue
        if rec.get("status") in ("failed", "submitted") and not retry_failed:
            blocked[s.id] = (f"{rec['status']} (job {rec.get('job_id') or 'unknown'}): pass --retry-failed to send it again"
                             + ("; it was submitted but no outcome was recorded, so it may have run and been billed" if rec["status"] == "submitted" else ""))
            stopped.add(s.comp)
            continue
        queue.append(s)
    return queue, blocked


def run_rounds(rp, base, client=None, *, approve_credits=None, max_steps=None, max_tv=MAX_TV, retry_failed=False, components=None, log=print):
    """Send the next `max_steps` jobs (all if None), one at a time, scoring each. Returns dict(sent, ok, halted, failed, preview).
    Nothing is sent unless approve_credits equals the credits of the jobs queued now. `components` restricts the run to those components."""
    st = load_state(base, rp)
    queue, blocked = pending_steps(rp, st, retry_failed, components)
    to_run = queue if max_steps is None else queue[:max_steps]
    local = bool(getattr(client, "is_local", False))
    per = 0 if local else estimate_credits(ENGINE)
    for sid, why in blocked.items():
        log(f"  blocked {sid}: {why}")
    for s in to_run:
        for p in step_problems(rp, s):
            log(f"  note: {p}")
    log(f"{len(queue)} job(s) pending, {len(to_run)} queued now = {len(to_run) * per} credit(s): {', '.join(s.id for s in to_run) or 'none'}")
    out = dict(sent=[], ok=True, halted=None, failed=None, preview=approve_credits is None and not local)
    if not to_run:
        return out
    if approve_credits is None and not local:
        log("(preview only: nothing was sent. Add --approve-credits "
            f"{len(to_run) * per} to send these; MOTH_API_KEY must be set in the environment or in .env.)")
        return out
    if not local and approve_credits != len(to_run) * per:
        raise NotApproved(f"--approve-credits {approve_credits} does not match the {len(to_run) * per} credit(s) of the {len(to_run)} job(s) queued; "
                          f"each QDrive job costs {per}")
    client = client or MothClient()
    for s in to_run:
        sdir = os.path.join(rounds_dir(base), s.id)
        prev = rp.steps_of(s.comp)[s.index - 1] if s.index else None
        input_files = None
        if prev is not None:
            asset = st["steps"].get(prev.id, {}).get("asset_id")
            if not asset:
                raise MothError(f"{s.id} continues {prev.id}, which has no recorded output asset id")
            input_files = {"initial_circuit": asset}
        params = rp.params(s)
        job = client.prepare(ENGINE, params, input_files)
        if job.problems():
            raise InvalidPayload(f"{s.id}: " + "; ".join(job.problems()))
        rec = st["steps"][s.id] = dict(status="submitted", submitted_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), job_id=None,
                                       payload_sha=payload_sha256({"engine": ENGINE, "params": params, "input_files": input_files}))
        save_state(base, st)                                    # a crash from here on leaves a visible "submitted" record, never a lost job
        log(f"sending {s.id} (round {s.index}, {len(params['targets']) - sum(1 for t in params['targets'] if t is None)} target(s)"
            f"{', continuing ' + prev.id if prev else ''})")
        t0 = time.time()
        try:
            res = client.run(ENGINE, params, approved=True, out_dir=sdir, input_files=input_files)
        except (MothError, TimeoutError, OSError) as e:
            rec.update(status="failed", job_id=_job_id_in(sdir), error=str(e)[:500], secs=round(time.time() - t0, 1))
            save_state(base, st)
            out.update(ok=False, failed=s.id)
            log(f"  {s.id} FAILED after {rec['secs']} s (job {rec['job_id']}): {rec['error']}")
            if "engine_timeout" in rec["error"]:
                log("  engine_timeout is the service not answering (a worker that never started shows an empty progress and fails at about 62 s, "
                    "whatever the payload). Run `canary` (1 credit, a known-good 2-qubit job) before sending anything else: if it fails too, wait.")
            return out
        if local:
            rec["local"] = True
        rec.update(status="completed", job_id=res["job_id"], secs=round(time.time() - t0, 1), asset_id=output_asset_id(res["result"]),
                   circuit=os.path.relpath(res["files"].get("circuit", ""), base) if res["files"].get("circuit") else None)
        out["sent"].append(s.id)
        if not rec["circuit"]:
            rec.update(status="failed", error="the job completed but returned no circuit file")
            save_state(base, st)
            out.update(ok=False, failed=s.id)
            log(f"  {s.id}: no circuit came back (job {res['job_id']})")
            return out
        tv, fid = score_piece(_read(res["files"]["circuit"]), rp.comps[s.comp].prep.layer_psi[s.hi - 1])
        rec.update(tv=round(tv, 4), fidelity=round(fid, 4))
        save_state(base, st)
        log(f"  {s.id} done in {rec['secs']} s (job {res['job_id'][:8]}): TV to the planned state {tv:.4f}, fidelity {fid:.4f}")
        if not rec["asset_id"] and s.hi < len(rp.comps[s.comp].prep.layers):
            out.update(ok=False, halted=s.id)
            log(f"  {s.id} returned no output asset id, so the next round of this component cannot continue it: stopping")
            return out
        if max_tv is not None and tv > max_tv:
            out.update(ok=False, halted=s.id)
            log(f"  stopping: TV {tv:.4f} is above --max-tv {max_tv}. The next round would build on this circuit; look before spending more "
                f"(rerun with a higher --max-tv to continue anyway).")
            return out
    return out


# ---------------------------------------------------------------------------------------------------------- assembling
def assemble(rp, base, local_fill=False):
    """Merge the finished chains into one circuit. Returns dict(text, complete, pieces, missing). A component counts as finished when its
    LAST step completed (a chained job's circuit is the old gates plus the new ones, so the last circuit is the whole component)."""
    st = load_state(base, rp)
    parts, pieces, missing = [], [], []
    for ci, comp in enumerate(rp.comps):
        steps = rp.steps_of(ci)
        done = []
        for s in steps:
            rec = st["steps"].get(s.id, {})
            if rec.get("status") == "completed" and rec.get("circuit") and os.path.exists(os.path.join(base, rec["circuit"])):
                done.append(s)
            else:
                break
        piece = dict(component=ci, qubits=comp.qubits, steps_done=len(done), steps_total=len(steps),
                     jobs=[st["steps"][s.id]["job_id"] for s in done])
        if len(done) == len(steps):
            text, piece["source"] = _read(os.path.join(base, st["steps"][steps[-1].id]["circuit"])), \
                "local" if any(st["steps"][s.id].get("local") for s in done) else "qdrive"
            piece.update(tv=st["steps"][steps[-1].id].get("tv"), fidelity=st["steps"][steps[-1].id].get("fidelity"))
        elif local_fill:
            text, piece["source"] = local_circuit(comp.prep.layer_psi[-1]), "local"
        else:
            piece["source"] = "missing"
            missing.append(ci)
            pieces.append(piece)
            continue
        parts.append((text, comp.qubits))
        pieces.append(piece)
    return dict(text=qp.merge_circuits(parts, rp.n_qubits) if parts else None, complete=not missing, pieces=pieces, missing=missing)


def write_assembled(asm, rp, base):
    """circuit.qasm (complete only) or circuit_partial.qasm, plus provenance.json. The show reads provenance.json to word its label."""
    os.makedirs(base, exist_ok=True)
    full, part = os.path.join(base, "circuit.qasm"), os.path.join(base, "circuit_partial.qasm")
    if asm["text"] is not None:
        with open(full if asm["complete"] else part, "w") as f:
            f.write(asm["text"])
    if asm["complete"] and os.path.exists(part):
        os.remove(part)
    prov = dict(kind="qdrive-rounds", complete=asm["complete"], plan_sha=rp.sha, pieces=asm["pieces"], written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                from_moth=asm["complete"] and all(p["source"] == "qdrive" for p in asm["pieces"]),
                local_pieces=[p["component"] for p in asm["pieces"] if p["source"] == "local"])
    with open(os.path.join(base, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    return prov


def circuit_provenance(circuit_path):
    """What a circuit written by write_assembled is, for the show's label: dict(label, from_moth, untested), or None when the folder has
    no provenance.json (the show then falls back on its older rules)."""
    path = os.path.join(os.path.dirname(os.path.abspath(circuit_path)), "provenance.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            prov = json.load(f)
    except (OSError, ValueError):
        return None
    if prov.get("kind") != "qdrive-rounds":
        return None
    srcs = [p.get("source") for p in prov.get("pieces", [])]
    if "local" in srcs and "qdrive" not in srcs and prov.get("complete"):
        return dict(label="LOCAL rehearsal of the QDrive rounds (ideal local engine, not a Moth result), simulated locally", from_moth=False, untested=True)
    if prov.get("from_moth"):
        jobs = sum(len(p.get("jobs", [])) for p in prov["pieces"])
        label = f"Moth QDrive circuit built in {jobs} chained jobs, merged and simulated locally"
        return dict(label=label, from_moth=True, untested=False)
    if "local" in srcs:
        n_q = srcs.count("qdrive")
        label = (f"circuit assembled from {n_q} Moth QDrive component(s) and {srcs.count('local')} LOCALLY prepared component(s), simulated locally "
                 f"(not a pure Moth result)")
        return dict(label=label, from_moth=False, untested=True)
    return dict(label="partial QDrive circuit (unfinished components left in |0>), simulated locally (not a Moth result)", from_moth=False, untested=True)


# ---------------------------------------------------------------------------------------------------------- the scene
def oracle_probabilities(labels, targets):
    """The scene's exact Z-statistics oracle over all qubits (what qdrive_lab/scene.py rebuilds), from the drawing (a labels dict, or the path of a
    labels.json) and the calibration."""
    from src.capture.labels import load_labels
    from src.geometry.planes import build_scene
    from src.targets.consistency import scene_problem
    from src.quantum import sampler_local as sl
    from src.quantum.sampler_mock import Sampler
    m = targets["meta"]
    s = m["sampler"]
    labels = load_labels(labels) if isinstance(labels, (str, os.PathLike)) else labels
    nx, ny = m["grid"]
    sampler = Sampler(build_scene(labels), nx, ny, kh=s["kh"], kf=s["kf"], J_in=s["J_in"], J_cross=s["J_cross"], J_pol=s["J_pol"],
                      min_cover=labels["grid"]["min_cover"])
    why = scene_problem(m, sampler.cells)
    if why:
        raise ValueError(f"{why}: pass the labels.json the calibration was made from (--labels), or run the calibration again")
    layout = dict(m["qubit_map"])
    layout.setdefault("n_qubits", m["n_qubits"])
    psi = sl.oracle_state(sampler, layout)
    p = np.abs(psi) ** 2
    return p / p.sum(), layout["n_qubits"]


def scene_labels(calib, labels=None):
    """The drawing a calibration was made from: the given labels.json, else the one next to the calib folder (a studio project), else the built-in
    bay window (six panes), which is what src.targets.calibrate_from_mock calibrates when it is given no labels."""
    from src.capture.labels import bay_window_labels, fill_defaults, load_labels, validate
    if labels:
        return load_labels(labels)
    beside = os.path.join(os.path.dirname(os.path.abspath(calib)), "labels.json")
    return load_labels(beside) if os.path.exists(beside) else validate(fill_defaults(bay_window_labels()))


def plan_for_scene(calib, labels=None, **kw):
    with open(os.path.join(calib, "targets.json")) as f:
        targets = json.load(f)
    prob, n = oracle_probabilities(scene_labels(calib, labels), targets)
    return plan_rounds(qp.TargetMoments(targets["bloch"], targets["relationships"]), prob, n, **kw), targets


# ---------------------------------------------------------------------------------------------------------- canary
CANARY_PARAMS = {"machine": "aer", "n_qubits": 2, "seed": 7, "shots": 1024, "update_method": "spectral", "tomography": 0, "sample": False,
                 "targets": [{"qubits": [0, 1], "expvals": {"ZI": 0.0, "IZ": 0.0, "ZZ": 1.0}}, None]}


def canary(out, client=None, approve_credits=None, log=print):
    """The lab's T07 (completed in 7 s on 4 Oct 2026), unchanged: tells a service problem from a payload problem. 1 credit."""
    per = estimate_credits(ENGINE)
    log(f"canary: a 2-qubit Bell target that finished in 7 s when the service was healthy; costs {per} credit(s)")
    if approve_credits is None:
        log("(preview only: nothing was sent)")
        return None
    if approve_credits != per:
        raise NotApproved(f"--approve-credits {approve_credits} does not match the canary's {per} credit(s)")
    client = client or MothClient()
    t0 = time.time()
    try:
        res = client.run(ENGINE, dict(CANARY_PARAMS), approved=True, out_dir=os.path.join(out, "canary"))
    except (MothError, TimeoutError) as e:
        log(f"canary FAILED after {time.time() - t0:.0f} s: {e}\n  The service is not healthy; sending the rounds now would only fail.")
        return False
    log(f"canary passed in {time.time() - t0:.0f} s (job {res['job_id'][:8]}): the service is answering, so a failing round points at its payload")
    return True


# ---------------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "assemble", "canary"])
    ap.add_argument("--calib", default=DEFAULT_CALIB_DIR, help="a calibration folder (targets.json); a studio project's is <project>/calib")
    ap.add_argument("--labels", default=None, help="labels.json of the scene (default: next to the calib folder, else the built-in bay window)")
    ap.add_argument("--out", default=DEFAULT_RUN_DIR, help="the solve folder; everything is kept under <out>/qdrive/ (a studio project's is <project>/solve)")
    ap.add_argument("--layers-per-round", type=int, default=LAYERS_PER_ROUND, help="growth layers in each job (fewer = more, smaller jobs)")
    ap.add_argument("--max-steps", type=int, default=None, help="send at most this many jobs in this run")
    ap.add_argument("--approve-credits", type=int, default=None, help="must equal the credits of the jobs queued; omit to preview")
    ap.add_argument("--max-tv", type=float, default=MAX_TV, help="stop before the next round when a job lands further than this from its planned state")
    ap.add_argument("--component", type=int, action="append", default=None, help="run: only this component's chain (repeatable; see `plan` for the numbers)")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--restart", action="store_true", help="move the existing rounds folder aside and plan from scratch")
    ap.add_argument("--local-fill", action="store_true", help="assemble: prepare unfinished components locally (exact, no Moth); marked as such")
    ap.add_argument("--no-coupling-map", action="store_true", help="leave coupling_map out of every job (the lab never isolated its effect)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--engine", choices=["moth", "local"], default="moth",
                    help="local: an ideal stand-in engine, no network, no credits; kept apart in <out>/qdrive_local/ and always labelled local")
    a = ap.parse_args(argv)
    base = os.path.join(a.out, "qdrive_local" if a.engine == "local" else "qdrive")
    if a.cmd == "canary":
        ok = canary(a.out, approve_credits=a.approve_credits)
        raise SystemExit(0 if ok in (True, None) else 1)
    rp, targets = plan_for_scene(a.calib, a.labels, layers_per_round=a.layers_per_round, seed=a.seed, coupling_map=not a.no_coupling_map)
    if a.restart:
        restart(base)
    print("\n".join(describe(rp)))
    if a.cmd == "plan":
        print(f"\nplan {rp.sha[:12]}; nothing was sent.")
        return
    if a.cmd == "run":
        res = run_rounds(rp, base, LocalClient(rp) if a.engine == "local" else None, approve_credits=a.approve_credits, max_steps=a.max_steps, max_tv=a.max_tv, retry_failed=a.retry_failed, components=a.component)
        if res["preview"]:
            return
    from src.quantum import solver
    solver.finish_rounds(rp, base, targets, local_fill=a.local_fill)
    if a.cmd == "run" and not res["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
