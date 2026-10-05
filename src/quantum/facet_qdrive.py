#!/usr/bin/env python3
"""Prepare the relief's facet register with Moth's QDrive, in small chained jobs, one spend decision at a time.

    python -m src.quantum.facet_qdrive plan  --labels runs/studio/labels.json --out runs/studio/solve/qdrive_facets
    python -m src.quantum.facet_qdrive send  --labels ... --out ...                          # preview of the next job: sends nothing
    python -m src.quantum.facet_qdrive send  --labels ... --out ... --approve-credits 1      # sends the NEXT job (1 credit)
    python -m src.quantum.facet_qdrive send  --labels ... --out /tmp/rehearsal --local       # ideal local stand-in, no network, never a Moth result
    python -m src.quantum.aer_relief --labels ... --out ... --facet-circuit <out>/circuit.qasm   # Aer then executes the whole relief on THAT facet state

QDrive takes target correlations, not gates (docs/qdrive-field-notes.md). The facet register of the relief is the part of the circuit that is
a state-preparation problem: F qubits whose neighbours are coupled by diagonal phases. The plan here follows the lab's findings: one target
per coupled pair carrying EVERY Pauli word of that pair's reduced density matrix (X, Y and Z words, identity-padded), targets that share a qubit
in successive layers (a `null` ends a layer), and a job holds only a couple of layers, because jobs near 20 qubits never completed. Job k continues
job k-1's returned circuit (`initial_circuit` = its output ASSET uuid). After every job the returned circuit is simulated locally and compared with
the ideal facet state (fidelity, moment error); that number is reported whatever it is, because the engine takes one gate layer per update and has
no convergence guarantee. Whatever circuit is complete is then the facet register of the Aer run, with the polarity superposition lifted on top.

Nothing is sent without --approve-credits equal to the cost of the job named in the preview, and the job record is written before the
engine is polled. `--local` returns, for job j, the ideal preparation with only the first part of the couplings applied: it rehearses the
chain (state file, resume, scoring, assembling, the Aer run) and says nothing about how the real engine converges.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from src.quantum import aer_relief as ar
from src.quantum import qdrive_plan as qp
from src.quantum import sampler_local as sl
from src.quantum.moth_client import InvalidPayload, MothClient, MothError, NotApproved, estimate_credits, output_asset_id
from src.quantum.relief_state import facet_block, facet_state
from src.targets.payloads import payload_sha256

ENGINE = "qdrive-api-v1"
LAYERS_PER_JOB = 2
TOL = 1e-3


# ------------------------------------------------------------------ the plan
def facet_groups(spec):
    """One pair group per distinct coupling, and a one-qubit group for every facet no coupling touches."""
    pairs = sorted({tuple(sorted((i, j))) for i, j, *_ in spec.edges})
    seen = {q for p in pairs for q in p}
    return pairs + [(q,) for q in range(spec.F) if q not in seen]


def plan_chain(spec, layers_per_job=LAYERS_PER_JOB, seed=7, update_method="spectral", shots=1024):
    """dict(layers (groups), target_layers (the target dicts), edges, jobs (QDrive params, job k > 0 without n_qubits), sha). Targets are the full reduced density matrix of each group in the ideal
    facet state; a single facet whose state is a pure product has all three of its words in its target."""
    from src.quantum.motte_mock import rdm
    psi = facet_state(spec)
    ordered = qp.order_groups(facet_groups(spec))
    layers = qp.pack_layers(ordered)
    edges = [list(e) for e in sorted({tuple(sorted((i, j))) for i, j, *_ in spec.edges})]
    targets = [[dict(qubits=list(g), expvals=qp.rdm_words(rdm(psi, spec.F, list(g)), len(g), TOL)) for g in layer] for layer in layers]
    jobs = []
    for k in range(0, len(targets), layers_per_job):
        body = []
        for layer in targets[k:k + layers_per_job]:
            body += [dict(t) for t in layer] + [None]
        params = dict(machine="aer", targets=body, update_method=update_method, shots=shots, tomography=0, sample=False, seed=seed)
        if k == 0:
            params["n_qubits"] = spec.F
        if edges:
            params["coupling_map"] = edges
        jobs.append(params)
    return dict(layers=layers, target_layers=targets, edges=edges, jobs=jobs, layers_per_job=layers_per_job,
                sha=payload_sha256({"engine": ENGINE, "jobs": jobs, "spec": ar.spec_fingerprint(spec)}))


def ideal_circuit(spec, upto=None):
    """OpenQASM 3 of the ideal preparation (hand-built gates, exact). upto = how many of the couplings' layers to apply (None: all)."""
    from qiskit import qasm3
    edges = spec.edges
    if upto is not None:
        keep = {tuple(sorted(g)) for layer in plan_chain(spec)["layers"][:upto] for g in layer if len(g) == 2}
        edges = [e for e in spec.edges if tuple(sorted((e[0], e[1]))) in keep]
    return qasm3.dumps(facet_block(spec, edges))


# ------------------------------------------------------------------ state on disk
def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def chain_state(folder, plan, spec):
    st = _read_json(os.path.join(folder, "chain.json"))
    if st and st.get("plan_sha") == plan["sha"]:
        return st
    return dict(kind="qdrive-facets-chain", plan_sha=plan["sha"], spec=ar.spec_fingerprint(spec), steps={})


def save_chain(folder, st):
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, "chain.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(folder, "chain.json"))


def read_state(folder):
    """state.json of the chain (kind qdrive-facets), or None."""
    st = _read_json(os.path.join(folder, "state.json"))
    return st if st and st.get("kind") == "qdrive-facets" else None


def next_index(plan, st, retry=False):
    """(index of the next job to send or None when the chain is done, blocking reason or None). A job recorded as submitted without a result is an orphan: the
    credit may be spent, so nothing more is sent until it is looked at (retry=True overrides)."""
    for k in range(len(plan["jobs"])):
        rec = st["steps"].get(str(k))
        if rec and rec.get("status") == "completed":
            continue
        if rec and rec.get("status") == "submitted" and not retry:
            return k, f"job {k} was submitted (job id {rec.get('job_id') or 'unknown'}) and no result was saved: its credit may be spent; check it in Moth, then resend with --retry"
        return k, None
    return None, None


def prepared(plan, st, k, client):
    """(PreparedJob, input_files) for job k of the chain."""
    input_files = None
    if k:
        asset = (st["steps"].get(str(k - 1)) or {}).get("asset_id")
        if not asset:
            raise MothError(f"job {k} continues job {k - 1}, which has no recorded output asset id")
        input_files = {"initial_circuit": asset}
    return client.prepare(ENGINE, plan["jobs"][k], input_files), input_files


def preview_job(plan, st, k):
    """The PreparedJob for job k as it would be sent, for display and validation; a placeholder asset id stands in for one that does not exist yet."""
    asset = (st["steps"].get(str(k - 1)) or {}).get("asset_id") or "00000000-0000-0000-0000-000000000000"
    return MothClient().prepare(ENGINE, plan["jobs"][k], {"initial_circuit": asset} if k else None)


def provenance_of(circuit_path):
    """What a circuit written by this module is, for labels: dict(label, from_moth, untested), or None for a file this module did not write."""
    prov = _read_json(os.path.join(os.path.dirname(os.path.abspath(circuit_path)), "provenance.json"))
    if not prov or prov.get("kind") != "qdrive-facets":
        return None
    if prov.get("local"):
        return dict(label="LOCAL rehearsal of the QDrive facet chain (ideal stand-in, not a Moth result)", from_moth=False, untested=True)
    if prov.get("from_moth"):
        return dict(label=f"facet register prepared by Moth QDrive in {len(prov['jobs'])} chained job(s)", from_moth=True, untested=False)
    return dict(label="partial QDrive facet circuit", from_moth=False, untested=True)


# ------------------------------------------------------------------ scoring and assembling
def score_circuit(spec, qasm_text):
    """Fidelity of a returned facet circuit to the ideal facet state, and the rms error of its single-qubit moments. Reported even when poor."""
    from src.quantum.complementary import pauli_expectation
    psi, n = sl.statevector_from_qasm(qasm_text)
    if n != spec.F:
        raise ValueError(f"the circuit has {n} qubits, the facet register has {spec.F}")
    ref = facet_state(spec)
    psi = psi / np.linalg.norm(psi)
    err = [pauli_expectation(psi, spec.F, {q: c}) - pauli_expectation(ref, spec.F, {q: c}) for q in range(spec.F) for c in "xyz"]
    return dict(fidelity=float(abs(np.vdot(ref, psi)) ** 2), rms_single_moment_error=float(np.sqrt(np.mean(np.square(err)))))


def assemble(spec, plan, st, folder, labels_path=None):
    """Write circuit.qasm (all jobs done) or circuit_partial.qasm (some), provenance.json and state.json from what the chain holds. Returns state.json's dict."""
    done = [k for k in range(len(plan["jobs"])) if (st["steps"].get(str(k)) or {}).get("status") == "completed"]
    complete = len(done) == len(plan["jobs"])
    last = st["steps"].get(str(done[-1])) if done else None
    if last is None:
        return None
    with open(os.path.join(folder, last["circuit"])) as f:
        text = f.read()
    for name in ("circuit.qasm", "circuit_partial.qasm"):
        if os.path.exists(os.path.join(folder, name)):
            os.remove(os.path.join(folder, name))
    with open(os.path.join(folder, "circuit.qasm" if complete else "circuit_partial.qasm"), "w") as f:
        f.write(text)
    jobs = [dict(index=k, job_id=st["steps"][str(k)].get("job_id"), local=bool(st["steps"][str(k)].get("local"))) for k in done]
    local = any(j["local"] for j in jobs)
    prov = dict(kind="qdrive-facets", complete=complete, local=local, from_moth=complete and not local, jobs=jobs, plan_sha=plan["sha"])
    with open(os.path.join(folder, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    sc = score_circuit(spec, text)
    state = dict(kind="qdrive-facets", complete=complete, local=local, from_moth=prov["from_moth"], jobs_done=len(done), jobs_total=len(plan["jobs"]), n_qubits=spec.F,
                 credits=sum(0 if j["local"] else 1 for j in jobs), score=sc, plan_sha=plan["sha"], spec=ar.spec_fingerprint(spec),
                 labels_sha=hashlib.sha256(open(labels_path, "rb").read()).hexdigest() if labels_path and os.path.exists(labels_path) else None,
                 circuit="circuit.qasm" if complete else "circuit_partial.qasm", circuit_sha256=hashlib.sha256(text.encode()).hexdigest(), jobs=jobs,
                 note=("LOCAL ideal stand-in, not a Moth result. " if local else "") + "the returned circuit, simulated locally, against the ideal facet state")
    with open(os.path.join(folder, "state.json"), "w") as f:
        json.dump(state, f, indent=1)
    return state


# ------------------------------------------------------------------ a local stand-in engine
class LocalFacetClient:
    """An IDEAL local QDrive for the facet register: job j returns the ideal preparation with the couplings of the first (j + 1) jobs' layers applied, and,
    like the real service, accepts a chained job only with the asset it issued for the previous one. It rehearses the chain with no network and no
    credits; it is recorded as local everywhere and says nothing about how the real engine converges."""
    is_local = True

    def __init__(self, spec, plan):
        self.spec, self.plan = spec, plan

    def asset(self, k):
        """The asset id the service issues for job k: a function of the plan, so a fresh process still recognises the previous job's, as the real service would."""
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.plan['sha']}:{k}"))

    def prepare(self, engine_id, params, input_files=None):
        return MothClient().prepare(engine_id, params, input_files)

    def run(self, engine_id, params, *, approved=False, out_dir=None, input_files=None):
        import uuid
        k = next(i for i, p in enumerate(self.plan["jobs"]) if p == params)
        if k and input_files != {"initial_circuit": self.asset(k - 1)}:
            raise MothError(f"local engine: job {k} must continue the asset of job {k - 1}")
        job_id = f"local-{uuid.uuid4()}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"job_{job_id}.json"), "w") as f:
            json.dump({"job_id": job_id, "engine_id": "local-ideal", "estimated_credits": 0}, f)
        path = os.path.join(out_dir, "circuit")
        done_layers = (k + 1) * self.plan["layers_per_job"]
        with open(path, "w") as f:
            f.write(ideal_circuit(self.spec, None if done_layers >= len(self.plan["layers"]) else done_layers))
        asset = self.asset(k)
        return dict(job_id=job_id, result={"outputs": [{"slot": "circuit", "output_asset_id": asset}], "result": None}, inline=None, files={"circuit": path})


# ------------------------------------------------------------------ sending
def send_next(spec, folder, labels_path=None, client=None, approve_credits=None, local=False, retry=False, jobs=1, log=print, show_request=False):
    """Send the next `jobs` job(s) of the chain (default one), scoring each. Nothing is sent unless approve_credits equals their cost. Returns dict(sent, state, preview)."""
    plan = plan_chain(spec)
    st = chain_state(folder, plan, spec)
    k0, blocked = next_index(plan, st, retry)
    out = dict(sent=[], state=read_state(folder), preview=False, plan=plan)
    if k0 is None:
        log(f"the chain is complete ({len(plan['jobs'])} job(s)); nothing to send.")
        return out
    if blocked:
        raise MothError(blocked)
    queue = list(range(k0, min(len(plan["jobs"]), k0 + max(1, jobs))))
    client = LocalFacetClient(spec, plan) if local else client
    per = 0 if getattr(client, "is_local", False) else estimate_credits(ENGINE)
    log(f"job(s) {', '.join(str(k) for k in queue)} of {len(plan['jobs'])} queued = {len(queue) * (per or 0)} credit(s)")
    probe = preview_job(plan, st, queue[0])
    n_targets = sum(1 for t in probe.params["targets"] if t)
    log(f"  job {queue[0]}: {n_targets} target(s) on {spec.F} qubits" + (f", continuing job {queue[0] - 1}" if queue[0] else "") + (f"\n{probe.describe(max_ops=3)}" if show_request else ""))
    if probe.problems():
        raise InvalidPayload("; ".join(probe.problems()))
    if not getattr(client, "is_local", False):
        if approve_credits is None:
            out["preview"] = True
            log(f"\n(preview only: nothing was sent. Add --approve-credits {len(queue) * per} to send; MOTH_API_KEY must be set in the environment or in .env.)")
            return out
        if approve_credits != len(queue) * per:
            raise NotApproved(f"--approve-credits {approve_credits} does not match the {len(queue) * per} credit(s) of the {len(queue)} job(s) queued")
    client = client or MothClient()
    os.makedirs(folder, exist_ok=True)
    for k in queue:
        sdir = os.path.join(folder, "jobs", str(k))
        job, input_files = prepared(plan, st, k, client)
        rec = st["steps"][str(k)] = dict(status="submitted", submitted_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), job_id=None, local=bool(getattr(client, "is_local", False)),
                                         payload_sha=payload_sha256({"engine": ENGINE, "params": plan["jobs"][k], "input_files": input_files}))
        save_chain(folder, st)                                    # a crash from here on leaves a visible "submitted" record, never a lost job
        log(f"sending job {k}" + (f", continuing job {k - 1}" if k else ""))
        t0 = time.time()
        try:
            res = client.run(ENGINE, plan["jobs"][k], approved=True, out_dir=sdir, input_files=input_files)
        except (MothError, TimeoutError, OSError) as e:
            rec.update(status="failed", error=str(e)[:500], secs=round(time.time() - t0, 1))
            save_chain(folder, st)
            log(f"  job {k} FAILED after {rec['secs']} s: {rec['error']}")
            raise
        rec.update(status="completed", job_id=res["job_id"], secs=round(time.time() - t0, 1), asset_id=output_asset_id(res["result"]),
                   circuit=os.path.relpath(res["files"]["circuit"], folder) if res["files"].get("circuit") else None)
        if not rec["circuit"]:
            rec.update(status="failed", error="the job completed but returned no circuit file")
            save_chain(folder, st)
            raise MothError(f"job {k} (id {res['job_id']}) returned no circuit")
        save_chain(folder, st)
        out["sent"].append(k)
        with open(os.path.join(folder, rec["circuit"])) as f:
            sc = score_circuit(spec, f.read())
        rec.update(fidelity=round(sc["fidelity"], 4), rms_single_moment_error=round(sc["rms_single_moment_error"], 4))
        save_chain(folder, st)
        log(f"  job {k} done in {rec['secs']} s (job {res['job_id'][:8]}): fidelity to the ideal facet state {sc['fidelity']:.4f}, moment error {sc['rms_single_moment_error']:.4f}")
        if not rec["asset_id"] and k + 1 < len(plan["jobs"]):
            raise MothError(f"job {k} returned no output asset id, so job {k + 1} cannot continue it")
    out["state"] = assemble(spec, plan, st, folder, labels_path)
    log(f"wrote {os.path.join(folder, out['state']['circuit'])} ({out['state']['jobs_done']}/{out['state']['jobs_total']} jobs done; fidelity {out['state']['score']['fidelity']:.4f})")
    return out


# ------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "send", "assemble"])
    ap.add_argument("--labels", help="labels.json from the pen tool (default: the built-in bay window)")
    ap.add_argument("--out", required=True, help="folder for the chain (the studio uses <project>/solve/qdrive_facets)")
    ap.add_argument("--approve-credits", type=int, default=None, help="must equal the credits of the job(s) queued; omit to preview only")
    ap.add_argument("--jobs", type=int, default=1, help="how many jobs to send now (each is 1 credit); default the next one")
    ap.add_argument("--local", action="store_true", help="rehearse with an ideal local stand-in: no network, no credits, never a Moth result")
    ap.add_argument("--show-request", action="store_true", help="print the exact request of the next job")
    ap.add_argument("--retry", action="store_true", help="send again a job that was recorded as submitted with no result")
    ap.add_argument("--kappa", type=float), ap.add_argument("--n-dirs", type=int), ap.add_argument("--entangle", type=float)
    ap.add_argument("--contrast", type=float), ap.add_argument("--observations")
    a = ap.parse_args(argv)
    from src.capture.labels import bay_window_labels, load_labels
    labels = load_labels(a.labels) if a.labels else bay_window_labels()
    flags = {k: v for k, v in dict(kappa=a.kappa, n_dirs=a.n_dirs, entangle=a.entangle, observations=a.observations).items() if v is not None}
    spec = ar.build_spec(labels, flags.get("kappa", 1.0), flags.get("n_dirs"), flags.get("entangle", 0.8), flags.get("observations"))[2]
    if a.cmd == "plan":
        plan = plan_chain(spec)
        print(f"{spec.F} facet qubits, {len(plan['edges'])} couplings: {len(plan['layers'])} layer(s) of targets in {len(plan['jobs'])} chained job(s) = {len(plan['jobs'])} credit(s)")
        for k, p in enumerate(plan["jobs"]):
            print(f"  job {k}: {sum(1 for t in p['targets'] if t)} target(s)" + ("" if k == 0 else f", continues job {k - 1}"))
        print(f"plan sha256 {plan['sha']}\nNothing here sends anything.")
        return
    if a.cmd == "assemble":
        plan = plan_chain(spec)
        st = assemble(spec, plan, chain_state(a.out, plan, spec), a.out, a.labels)
        print("nothing finished yet" if st is None else f"wrote {os.path.join(a.out, st['circuit'])}: fidelity {st['score']['fidelity']:.4f}")
        return
    try:
        send_next(spec, a.out, a.labels, approve_credits=a.approve_credits, local=a.local, retry=a.retry, jobs=a.jobs, show_request=a.show_request)
    except (MothError, TimeoutError) as e:
        raise SystemExit(f"stopped: {e}")


if __name__ == "__main__":
    main()
