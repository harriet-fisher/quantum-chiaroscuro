#!/usr/bin/env python3
"""Dynamic relief: the echo's taps modulate the relief, one circuit per echo depth, every one executed on Aer.

    python -m src.quantum.echo_relief --labels runs/studio/labels.json --out runs/studio/solve/echo                      # taps from the LOCAL echo (free)
    python -m src.quantum.echo_relief --labels ... --out ... --source moth                                              # preview of Moth's otoc-echo-v1 job: sends nothing
    python -m src.quantum.echo_relief --labels ... --out ... --source moth --approve-credits 1                          # sends it (1 credit), then runs on Aer
    python -m src.show --labels runs/studio/labels.json --echo-run runs/studio/solve/echo                                # perform: one frame per echo depth

The echo is a kicked-Ising lattice run forward, kicked at one site, and run back: the tap F(site, depth) = <X> + i<Y> says how much a touch on one site
is still felt `depth` steps later. Our reading (domain_moth.dynamic_specs, taps_to_relief, which is OURS, not the engine's definition): at lattice
site s and depth d, |F| / max|F| scales the tilt of every facet whose centroid falls on s, and arg F turns its slope direction, so a touch on one patch
spreads across the wall as relief from one depth frame to the next. The relief circuit at each depth is the ordinary per-panel circuit with those tilts and
azimuths (the diagonal-coupling azimuth compensation is recomputed per frame), so each depth is a quantum circuit of its own: it is executed on Aer
(aer_relief.execute, lamp read in Z) and its measured shots are saved.

Taps come from one of two places: `local` (me.local_echo, our reading of the engine's description, at most 16 sites, free) or `moth` (the otoc-echo-v1 job,
1 credit, machine aer, at most 24 sites; result fields are "not documented yet", so the tap list is found defensively and the raw result is saved first). Either
way the circuits are executed here: the Moth engine supplies the dynamics, Aer executes the relief. A payload that already has a Moth result on disk is never sent
again, and nothing is sent without --approve-credits equal to the cost.

Performing steps through the depths one look at a time (the auto-cycle animates it); the controls (independent noise, dephased depth) stay classical.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from src.quantum import aer_relief as ar
from src.quantum import domain_moth as dm
from src.quantum import moth_engines as me
from src.quantum.moth_client import MothClient
from src.quantum.relief_state import ReliefState
from src.store.cache import cached_result, save_payload
from src.targets.payloads import payload_sha256

ENGINE = "otoc-echo-v1"
DEFAULT_DEPTH = 4
DEFAULT_SHOTS = 200_000                      # per depth, lamp read in Z (one run per depth)
LOCAL_MAX_SITES, ENGINE_MAX_SITES = 16, 24
MAX_DEPTH = 8
MODES = ("z",)


class EchoError(ValueError):
    pass


# ------------------------------------------------------------------ the lattice and the taps
def lattice_sites(fs, scene, width, height):
    """The lattice site (row-major over a width x height grid on the canvas) of every quantum facet: the cell its centroid falls in."""
    out = []
    for f in fs.facets[:fs.n_quantum]:
        cx = min(width - 1, int(f.centroid[0] / scene.W * width))
        cy = min(height - 1, int(f.centroid[1] / scene.H * height))
        out.append(cy * width + cx)
    return out


def check_lattice(width, height, depth, source):
    out = []
    n = width * height
    cap = LOCAL_MAX_SITES if source == "local" else ENGINE_MAX_SITES
    if not (width >= 1 and height >= 1 and n >= 2):
        out.append("the echo lattice needs at least 2 sites")
    if n > cap:
        out.append(f"a {width}x{height} lattice has {n} sites; the {'local echo stops at' if source == 'local' else 'engine (machine aer) stops at'} {cap}")
    if not 1 <= depth <= MAX_DEPTH:
        out.append(f"depth must be between 1 and {MAX_DEPTH} (one relief circuit is executed per depth)")
    return out


def parse_taps(result):
    """[(site, depth, F_re, F_im)] from an otoc-echo-v1 result, wherever the list sits (the result fields are 'not documented yet'). Accepts a list of dicts with
    the documented names site, depth, F_re, F_im, or a list of 4-number rows. Raises ValueError with the keys seen when there are none."""
    root = result.get("result", result) if isinstance(result, dict) else result
    if isinstance(root, str):
        root = json.loads(root)

    def is_taps(o):
        if not isinstance(o, list) or not o:
            return False
        if all(isinstance(t, dict) and {"site", "depth"} <= {str(k).lower() for k in t} and {"f_re", "f_im"} <= {str(k).lower() for k in t} for t in o):
            return True
        return all(isinstance(t, (list, tuple)) and len(t) == 4 and all(isinstance(x, (int, float)) for x in t) for t in o)
    taps = me._find(root, is_taps)
    if taps is None:
        raise ValueError(f"no echo taps found; top-level keys: {sorted(root) if isinstance(root, dict) else type(root).__name__}")
    out = []
    for t in taps:
        if isinstance(t, dict):
            low = {str(k).lower(): v for k, v in t.items()}
            out.append((int(low["site"]), int(low["depth"]), float(low["f_re"]), float(low["f_im"])))
        else:
            out.append((int(t[0]), int(t[1]), float(t[2]), float(t[3])))
    return out


def depth_specs(spec, sites, taps, depth, n_sites):
    return dm.dynamic_specs(spec, sites, taps, depth, n_sites)


def plan_sha(spec, source, width, height, depth, kick_site, shots, seed, moth_sha=None):
    d = dict(engine="relief-echo", spec=ar.spec_fingerprint(spec), source=source, lattice=[width, height], depth=depth, kick_site=kick_site,
             shots=int(shots), seed=int(seed), moth=moth_sha)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def moth_job(width, height, depth, kick_site, shots=4096, seed=7):
    """(params, sha256, PreparedJob) of the otoc-echo-v1 request: no key, no network."""
    params = me.echo_payload(width=width, height=height, depth=depth, kick_site=kick_site, shots=shots, seed=seed)["params"]
    return params, payload_sha256({"engine": ENGINE, "params": params}), MothClient().prepare(ENGINE, params)


# ------------------------------------------------------------------ a saved run
def read_state(folder):
    try:
        with open(os.path.join(folder, "state.json")) as f:
            d = json.load(f)
        return d if d.get("kind") == "relief-echo" and os.path.exists(os.path.join(folder, "shots.npz")) else None
    except (OSError, ValueError):
        return None


def run(labels, labels_path, out, source="local", depth=DEFAULT_DEPTH, width=4, height=4, kick_site=None, shots=DEFAULT_SHOTS, seed=ar.DEFAULT_SEED, flags=None,
        approve_credits=None, client=None, moth_shots=4096, log=print):
    """Get the echo taps (local, or Moth's otoc-echo-v1), build the relief circuit of every depth and execute them on Aer; save everything under `out`.
    Returns state.json's dict, or None for a Moth preview. A run with the same plan hash is reused."""
    flags = dict(flags or {})
    scene, fs, spec = ar.build_spec(labels, flags.get("kappa", 1.0), flags.get("n_dirs"), flags.get("entangle", 0.8), flags.get("observations"))
    problems = ar.check_size(spec) + check_lattice(width, height, depth, source)
    if problems:
        raise EchoError("; ".join(problems))
    n_sites = width * height
    kick_site = n_sites // 2 if kick_site is None else int(kick_site)
    if not 0 <= kick_site < n_sites:
        raise EchoError(f"kick site {kick_site} is outside the {n_sites}-site lattice")
    sites = lattice_sites(fs, scene, width, height)
    moth_sha = job_id = credits = None
    have = read_state(out)
    if source == "moth":
        params, moth_sha, job = moth_job(width, height, depth, kick_site, moth_shots)
        if job.problems():
            raise EchoError("the echo request cannot be sent: " + "; ".join(job.problems()))
    sha = plan_sha(spec, source, width, height, depth, kick_site, shots, seed, moth_sha)
    if have and have.get("sha256") == sha:
        log(f"a run for exactly this echo, circuit, shot count and seed is already on disk ({out}): reusing it, nothing was executed.")
        return have
    if source == "local":
        log(f"echo taps: LOCAL reading of the engine's description ({width}x{height} lattice, depth {depth}, kick at site {kick_site})")
        taps = me.local_echo(width, height, depth, kick_site=kick_site)
        credits = 0
    else:
        moth_dir = os.path.join(out, "moth")
        cached = cached_result(moth_dir, moth_sha)
        log(job.describe(max_ops=3)[:1200])
        if cached:
            log(f"\nresult for this exact echo request is already on disk ({cached['path']}): reusing it, 0 credits.")
            raw, job_id, credits = cached["result"], os.path.basename(cached["path"])[len("raw_result_"):-5], 0
        else:
            if approve_credits is None:
                log(f"\n(preview only: nothing was sent. Add --approve-credits {job.credits} to send; MOTH_API_KEY must be set in the environment or in .env.)")
                return None
            if job.credits is None or approve_credits != job.credits:
                raise SystemExit(f"--approve-credits {approve_credits} does not match the estimated cost of {job.credits}")
            save_payload(moth_dir, {"engine": ENGINE, "params": params}, moth_sha)
            res = (client or MothClient()).run(ENGINE, params, approved=True, out_dir=moth_dir)
            raw, job_id, credits = res["result"], res["job_id"], job.credits
            log(f"\njob {job_id} completed.")
        taps = parse_taps(raw)
        log(f"echo taps: {len(taps)} from Moth's otoc-echo-v1 (job {job_id})")
    specs = depth_specs(spec, sites, taps, depth, n_sites)
    for d, sp in enumerate(specs):
        bad = ar.check_size(sp)
        if bad:
            raise EchoError(f"depth {d + 1}: " + "; ".join(bad))
    t0 = time.time()
    log(f"{depth} relief circuit(s) of {spec.n_full} qubits, {shots:,} shots each on Aer (statevector, lamp read in Z)")
    arrays, scores = {}, []
    for d, sp in enumerate(specs):
        a, pools = ar.execute(sp, shots, seed + d, prefix=f"d{d}_", modes=MODES, log=lambda m: None)
        arrays.update(a)
        sc = ar.score(sp, pools["z"])
        scores.append(sc)
        log(f"  depth {d + 1}: TV {sc['tv_wo']:.4f} (shot-noise floor {sc['tv_wo_sampling_floor']:.4f}), facet marginal error {sc['facet_marginal_max_error']:.4f}: "
            + ("consistent" if sc["consistent"] else "NOT CONSISTENT") + f"; mean tilt {float(sp.tau.mean()):.3f}")
    import qiskit
    import qiskit_aer
    os.makedirs(out, exist_ok=True)
    meta = dict(kind="relief-echo", sha256=sha, spec=ar.spec_fingerprint(spec), depth_specs=[ar.spec_fingerprint(sp) for sp in specs], depth=depth, lattice=[width, height],
                kick_site=kick_site, sites=sites, taps=[list(t) for t in taps], taps_source=source, moth_job_id=job_id, moth_sha256=moth_sha, credits=credits,
                shots=int(shots), seed=int(seed), n_qubits=spec.n_full, F=spec.F, P=spec.P, n_classical=spec.n_classical, modes=list(MODES), flags=flags,
                labels_sha=hashlib.sha256(open(labels_path, "rb").read()).hexdigest() if labels_path and os.path.exists(labels_path) else None,
                scores=scores, consistent=all(s["consistent"] for s in scores), mean_tilt=[float(sp.tau.mean()) for sp in specs], method="aer statevector",
                versions=dict(qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__), seconds=round(time.time() - t0, 2),
                executed="one relief circuit per echo depth (facet, polarity, lamp and observe registers) as gates on Aer; the taps come from "
                         + ("Moth's otoc-echo-v1" if source == "moth" else "the local echo (our reading, not Moth's engine)"))
    np.savez_compressed(os.path.join(out, "shots.npz.tmp.npz"), meta=np.array(json.dumps(meta)), **arrays)
    os.replace(os.path.join(out, "shots.npz.tmp.npz"), os.path.join(out, "shots.npz"))
    with open(os.path.join(out, "state.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(out, "taps.json"), "w") as f:
        json.dump(dict(source=source, lattice=[width, height], depth=depth, kick_site=kick_site, taps=meta["taps"]), f)
    log(f"wrote {out}/shots.npz, taps.json and state.json ({meta['seconds']}s)")
    return meta


# ------------------------------------------------------------------ the show's state
class EchoReliefState:
    """Looks that step through the echo depths: look k is a draw from the Aer run of depth (k mod depth). Every other part of the interface the show uses is the
    current depth's AerReliefState (its spec, its witnesses); `spec` is the spec of the depth most recently drawn, so a witness run describes the state on screen."""
    executed_by = "Aer"
    backend = "exact"

    def __init__(self, spec, folder, label=None):
        meta = read_state(folder)
        if meta is None:
            raise ValueError(f"no echo run in {folder!r} (expected state.json and shots.npz; make one with python -m src.quantum.echo_relief)")
        if meta["spec"] != ar.spec_fingerprint(spec):
            raise ValueError(f"the echo run in {folder!r} was made for a different drawing or different relief settings; run the solve step again")
        specs = depth_specs(spec, meta["sites"], [tuple(t) for t in meta["taps"]], meta["depth"], meta["lattice"][0] * meta["lattice"][1])
        if [ar.spec_fingerprint(sp) for sp in specs] != meta["depth_specs"]:
            raise ValueError(f"the echo run in {folder!r} does not rebuild to the circuits it executed (the tap mapping changed); run the solve step again")
        with np.load(os.path.join(folder, "shots.npz"), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
        child_meta = dict(meta, facets=None)
        self.children = [ar.AerReliefState(sp, arrays=arrays, meta=child_meta, prefix=f"d{d}_") for d, sp in enumerate(specs)]
        self.meta, self.folder, self.cursor, self.current = meta, folder, 0, 0
        src = "Moth's otoc-echo-v1" if meta["taps_source"] == "moth" else "the local echo"
        self.label = label or (f"Aer run of {meta['depth']} relief circuits ({meta['n_qubits']} qubits, {meta['shots']:,} shots each), one per echo depth; taps from {src}; "
                               f"simulated here; not a Moth result" + (" (the dynamics are Moth's, the circuits are Aer's)" if meta["taps_source"] == "moth" else ""))
        self.from_moth = meta["taps_source"] == "moth"

    @property
    def spec(self):
        return self.children[self.current].spec

    def draw(self, K, rng, g=None, control="coherent", lamp_mode="z", world=None):
        k = self.current = self.cursor
        self.cursor = (k + 1) % len(self.children)
        d = self.children[k].draw(K, rng, g=g, control=control, lamp_mode=lamp_mode, world=world)
        d.echo_depth, d.echo_total = k + 1, len(self.children)
        return d

    def __getattr__(self, name):
        if name in ("children", "current") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.children[self.current], name)


# ------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="labels.json from the pen tool (default: the built-in bay window)")
    ap.add_argument("--out", required=True, help="folder for shots.npz, taps.json and state.json (the studio uses <project>/solve/echo)")
    ap.add_argument("--source", choices=["local", "moth"], default="local", help="where the taps come from: our local echo (free) or Moth's otoc-echo-v1 (1 credit)")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--width", type=int, default=4), ap.add_argument("--height", type=int, default=4)
    ap.add_argument("--kick-site", type=int, default=None)
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="Aer shots per depth")
    ap.add_argument("--seed", type=int, default=ar.DEFAULT_SEED)
    ap.add_argument("--moth-shots", type=int, default=4096)
    ap.add_argument("--approve-credits", type=int, default=None, help="--source moth: must equal the estimated cost; omit to preview only")
    ap.add_argument("--kappa", type=float), ap.add_argument("--n-dirs", type=int), ap.add_argument("--entangle", type=float)
    ap.add_argument("--contrast", type=float), ap.add_argument("--observations")
    a = ap.parse_args(argv)
    from src.capture.labels import bay_window_labels, load_labels
    labels = load_labels(a.labels) if a.labels else bay_window_labels()
    flags = {k: v for k, v in dict(kappa=a.kappa, n_dirs=a.n_dirs, entangle=a.entangle, contrast=a.contrast, observations=a.observations).items() if v is not None}
    if not 1000 <= a.shots <= 20_000_000:
        raise SystemExit("--shots must be between 1,000 and 20,000,000")
    try:
        run(labels, a.labels, a.out, a.source, a.depth, a.width, a.height, a.kick_site, a.shots, a.seed, flags, a.approve_credits, moth_shots=a.moth_shots)
    except (EchoError, ar.AerRunError) as e:
        raise SystemExit(f"cannot run the echo: {e}")


if __name__ == "__main__":
    main()
