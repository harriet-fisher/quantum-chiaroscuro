#!/usr/bin/env python3
"""Superposed Relief executed as a circuit on Aer: the quantum Solve and Perform.

    python -m src.quantum.aer_relief --labels runs/studio/labels.json --out runs/studio/solve/aer          # 1M shots, no credits, no network
    python -m src.show --labels runs/studio/labels.json --aer-run runs/studio/solve/aer                    # perform from the saved run

The relief state is not a target to be solved for: the geometry fixes the circuit (`relief_state.full_circuit`). Solving here means COMPILING
and EXECUTING it: the whole performance circuit (facet, polarity, lamp and observation registers, measured at the end) is transpiled for
Aer's statevector simulator and run for `shots` shots, once with the lamp register read in Z and once in X. The measured bitstrings are saved
(shots.npz) together with the circuit (circuit.qasm), and the run is checked against the numpy reference state: total-variation distance of
the (observation, light, polarity) outcomes and the facet marginals, each against what the same number of shots drawn from the exact
distribution would show, plus a witness run (polarity in X, facets in Z) measured on Aer too.

Performing then needs no state at all: `AerReliefState.draw` takes a frame's observation, light and polarity outcomes from one measured shot,
and its K photons from the other shots that came out with the same three outcomes (post-selection, the way K-shot averaging has always
worked here). The classical controls (independent noise, dephased depth) are mixtures, not states; they are still drawn by the numpy reference.

Honest limits: this is a statevector simulation on this machine (at most LIMIT qubits), so it is quantum in what is executed and
measured, not in being hard to simulate, and it is not a Moth result. Only the per-panel engine fits; the domain engine's 69 qubits do not.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from src.quantum import sampler_local as sl
from src.quantum.relief_state import WORLDS, ReliefDraw, ReliefState, full_circuit, parse_observations, reference_circuit

LIMIT = 26                                   # statevector qubits: 2^26 amplitudes is 1 GiB
DEFAULT_SHOTS = 1_000_000
DEFAULT_SEED = 2026
MODES = ("z", "x")                           # lamp register read in Z (a light world per look) or in X (the light directions interfere)
RELIEF_FLAGS = ("kappa", "n_dirs", "entangle", "contrast", "observations")      # the src.show flags that shape the circuit (and the tone) of a run
WITNESS_SHOTS = 20000


class AerRunError(ValueError):
    pass


# ------------------------------------------------------------------ the spec the show would build
def build_spec(labels, kappa=1.0, n_dirs=None, entangle=0.8, observations=None):
    """(scene, facets, spec) exactly as Session.relief() builds them for the per-panel engine."""
    from src.capture.labels import fill_defaults, validate
    from src.geometry.facets import build_facets
    from src.geometry.planes import build_scene
    from src.quantum.relief_state import spec_from_facets
    scene = build_scene(validate(fill_defaults(labels)))
    fs = build_facets(scene, n_dirs=n_dirs, kappa=kappa)
    obs = parse_observations(observations) if isinstance(observations, str) or observations is None else [tuple(o) for o in observations]
    return scene, fs, spec_from_facets(fs, scene, entangle=entangle, observations=obs)


def check_size(spec):
    """A list of reasons this spec cannot be executed here (empty when it can)."""
    out = []
    if spec.n_full > LIMIT:
        out.append(f"the circuit has {spec.n_full} qubits ({spec.F} facets + {spec.P} polarity + 4 lamp/observe); the statevector run stops at {LIMIT}. Use fewer facets per panel (--n-dirs) or the rehearsal")
    if spec.P > 12:
        out.append(f"{spec.P} polarity qubits give {2 ** spec.P} outcome classes per look; the shot pool needs far more shots than it is worth")
    return out


def spec_fingerprint(spec):
    return hashlib.sha256(repr(spec.key()).encode()).hexdigest()


def plan_sha(spec, shots, seed):
    return hashlib.sha256(json.dumps(dict(engine="relief-aer", spec=spec_fingerprint(spec), shots=int(shots), seed=int(seed), modes=list(MODES)), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ executing the circuit
def _simulator():
    from qiskit_aer import AerSimulator
    return AerSimulator(method="statevector")


def run_circuit(spec, lamp_mode, shots, seed):
    """Run full_circuit(spec, lamp_mode) on Aer. Returns (g, w, o, x, count) int arrays, one row per distinct measured outcome, sorted by
    (g, w, o, x): g the observe register, w the lamp register, o the polarity bits (bit p = qubit p), x the facet bits (bit i = facet i)."""
    from qiskit import transpile
    sim = _simulator()
    counts = sim.run(transpile(full_circuit(spec, lamp_mode), sim, optimization_level=1), shots=int(shots), seed_simulator=int(seed)).result().get_counts()
    rows = np.empty((len(counts), 5), np.int64)
    for k, (key, c) in enumerate(counts.items()):
        co, cl, cb, cf = key.split(" ")                                   # registers print last-added first
        rows[k] = (int(co, 2), int(cl, 2), int(cb, 2), int(cf, 2), c)
    rows = rows[np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))]
    return tuple(rows[:, j] for j in range(5))


def witness_shots(spec, shots, seed):
    """The witness setting as a device would run it: reference circuit, polarity qubits rotated to X, everything measured. Returns (xs, zs): +/-1
    outcomes of the polarity qubits (in X) and of the facets (in Z), each (shots, P) and (shots, F)."""
    from qiskit import transpile
    qc = reference_circuit(spec)
    for p in range(spec.P):
        qc.h(spec.F + p)
    qc.measure_all()
    sim = _simulator()
    counts = sim.run(transpile(qc, sim, optimization_level=1), shots=int(shots), seed_simulator=int(seed)).result().get_counts()
    idx = np.repeat([int(k, 2) for k in counts], list(counts.values()))
    return sl.spins(idx, list(range(spec.F, spec.F + spec.P))), sl.spins(idx, list(range(spec.F)))


# ------------------------------------------------------------------ checking a run against the reference state
def _class_ids(g, w, o, P):
    return (g * 4 + w) * (1 << P) + o


def score(spec, pool, rng=None):
    """How well the Aer shots (lamp read in Z) agree with the exact distribution. Two quantities, each next to what the same number of shots
    drawn from the exact distribution would give (the sampling floor), because with a million shots over millions of outcomes a raw distance
    mostly measures shot noise:
      tv_wo     total variation of P(light w, polarity o | observation g), averaged over g, vs the floor from numpy sampling of the exact cells;
      facet     the largest |empirical - exact| of P(facet lit | g, w), vs four standard errors of that many shots.
    `consistent` is true when both are within their floors (1.5 x for the distance)."""
    rng = rng or np.random.default_rng(0)
    g, w, o, x, c = pool
    ref = ReliefState(spec)
    F, P = spec.F, spec.P
    bits = (np.arange(1 << F)[:, None] >> np.arange(F)[None, :]) & 1
    tv, floor, worst, worst_bound, shots = [], [], 0.0, 0.0, int(c.sum())
    for gi in range(len(spec.observations)):
        sel = g == gi
        n_g = int(c[sel].sum())
        if not n_g:
            continue
        cell = ref.cell(gi)                                                # (4, 2^P, 2^F), sums to 1
        p_wo = cell.sum(axis=2)
        emp = np.zeros_like(p_wo)
        np.add.at(emp, (w[sel], o[sel]), c[sel])
        tv.append(0.5 * np.abs(emp / n_g - p_wo).sum())
        floor.append(np.mean([0.5 * np.abs(rng.multinomial(n_g, p_wo.ravel() / p_wo.sum()).reshape(p_wo.shape) / n_g - p_wo).sum() for _ in range(3)]))
        for wi in range(4):
            sw = sel & (w == wi)
            n = int(c[sw].sum())
            if not n:
                continue
            lit = ((1 - bits[x[sw]]) * c[sw][:, None]).sum(axis=0) / n
            err = np.abs(lit - ref.marginals(gi, w=wi)).max()
            worst, worst_bound = max(worst, float(err)), max(worst_bound, 4 * 0.5 / np.sqrt(n))
    tv_m, floor_m = float(np.mean(tv)), float(np.mean(floor))
    return dict(shots=shots, tv_wo=tv_m, tv_wo_sampling_floor=floor_m, facet_marginal_max_error=worst, facet_marginal_bound=float(worst_bound),
                consistent=bool(tv_m <= 1.5 * floor_m + 1e-3 and worst <= worst_bound))


def witness_summary(spec, shots, seed):
    xs, zs = witness_shots(spec, shots, seed)
    rows = []
    for p in range(spec.P):
        par = zs[:, spec.facets_of(p)].prod(axis=1)
        s = xs[:, p] * par
        rows.append(dict(panel=p, visibility=float(xs[:, p].mean()), visibility_se=float(xs[:, p].std() / np.sqrt(shots)), visibility_exact=spec.visibility(p),
                         stabilizer=float(s.mean()), stabilizer_se=float(s.std() / np.sqrt(shots))))
    return dict(shots=int(shots), panels=rows,
                note="polarity qubits measured in X and facets in Z in one setting on Aer: <X_B> estimates the interference visibility, <X_B Z^F> is 1 for the ideal state "
                     "(0 for a dephased or independent-noise polarity)")


# ------------------------------------------------------------------ a saved run
def _npz_path(folder):
    return os.path.join(folder, "shots.npz")


def read_state(folder):
    try:
        with open(os.path.join(folder, "state.json")) as f:
            d = json.load(f)
        return d if d.get("kind") == "relief-aer" and os.path.exists(_npz_path(folder)) else None
    except (OSError, ValueError):
        return None


def circuit_qasm(spec):
    try:
        from qiskit import qasm3
        return qasm3.dumps(full_circuit(spec))
    except Exception as e:                                                   # noqa: BLE001 - the file is a courtesy; the run does not depend on it
        return f"// the circuit could not be exported to OpenQASM 3: {type(e).__name__}: {e}\n"


def run(labels, labels_path, out, shots=DEFAULT_SHOTS, seed=DEFAULT_SEED, flags=None, log=print):
    """Execute the scene's circuit on Aer and save everything under `out`. `flags` = the show flags of RELIEF_FLAGS (the ones that differ from the
    show's defaults). A run whose plan hash is already on disk is reused. Returns state.json's dict."""
    flags = dict(flags or {})
    scene, fs, spec = build_spec(labels, flags.get("kappa", 1.0), flags.get("n_dirs"), flags.get("entangle", 0.8), flags.get("observations"))
    problems = check_size(spec)
    if problems:
        raise AerRunError("; ".join(problems))
    sha = plan_sha(spec, shots, seed)
    have = read_state(out)
    if have and have.get("sha256") == sha:
        log(f"a run for exactly this circuit, shot count and seed is already on disk ({out}): reusing it, nothing was executed.")
        return have
    t0 = time.time()
    log(f"circuit: {spec.n_full} qubits = {spec.F} facets + {spec.P} polarity + 2 lamp + 2 observe; {shots:,} shots on Aer (statevector), seed {seed}")
    arrays, modes_s = {}, {}
    for m in MODES:
        t = time.time()
        pool = run_circuit(spec, m, shots, seed)
        for name, a in zip("gwoxc", pool):
            arrays[f"{m}_{name}"] = a
        modes_s[m] = pool
        log(f"  lamp read in {m.upper()}: {len(pool[0]):,} distinct outcomes in {time.time() - t:.1f}s")
    sc = score(spec, modes_s["z"])
    log(f"check against the reference state: TV {sc['tv_wo']:.4f} (shot-noise floor {sc['tv_wo_sampling_floor']:.4f}); worst facet marginal off by {sc['facet_marginal_max_error']:.4f} "
        f"(4 sigma {sc['facet_marginal_bound']:.4f}): " + ("consistent" if sc["consistent"] else "NOT CONSISTENT"))
    wit = witness_summary(spec, WITNESS_SHOTS, seed + 1)
    log("witness run on Aer: " + "; ".join(f"panel {r['panel']} <X_B>={r['visibility']:.3f} (ideal {r['visibility_exact']:.3f}), stabilizer {r['stabilizer']:.3f}" for r in wit["panels"]))
    import qiskit
    import qiskit_aer
    os.makedirs(out, exist_ok=True)
    meta = dict(kind="relief-aer", sha256=sha, spec=spec_fingerprint(spec), shots=int(shots), seed=int(seed), n_qubits=spec.n_full, F=spec.F, P=spec.P,
                n_classical=spec.n_classical, observations=[list(o) for o in spec.observations], modes=list(MODES), flags=flags,
                labels_sha=hashlib.sha256(open(labels_path, "rb").read()).hexdigest() if labels_path and os.path.exists(labels_path) else None,
                method="aer statevector", versions=dict(qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__), seconds=round(time.time() - t0, 2),
                score=sc, witness=wit, executed="the whole performance circuit (facet, polarity, lamp and observe registers) as gates on Aer, measured at the end; "
                "no Moth call, no credits", circuit="circuit.qasm", shots_file="shots.npz")
    np.savez_compressed(_npz_path(out) + ".tmp.npz", meta=np.array(json.dumps(meta)), **arrays)
    os.replace(_npz_path(out) + ".tmp.npz", _npz_path(out))
    with open(os.path.join(out, "circuit.qasm"), "w") as f:
        f.write(circuit_qasm(spec))
    with open(os.path.join(out, "state.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log(f"wrote {out}/shots.npz, circuit.qasm and state.json ({meta['seconds']}s)")
    return meta


# ------------------------------------------------------------------ the show's state
class AerReliefState(ReliefState):
    """A ReliefState whose coherent looks are the shots of a saved Aer run. The numpy reference is kept underneath only for the classical controls
    (they are mixtures: independent noise, dephased depth) and for the exact numbers beside the witnesses."""
    executed_by = "Aer"

    def __init__(self, spec, folder, label=None):
        super().__init__(spec, label=label or "")
        meta = read_state(folder)
        if meta is None:
            raise ValueError(f"no Aer run in {folder!r} (expected state.json and shots.npz; make one with python -m src.quantum.aer_relief)")
        if meta["spec"] != spec_fingerprint(spec):
            raise ValueError(f"the Aer run in {folder!r} was made for a different drawing or different relief settings; run the solve step again")
        self.meta, self.folder = meta, folder
        with np.load(_npz_path(folder), allow_pickle=False) as z:
            self._pools = {m: self._index(*(z[f"{m}_{n}"] for n in "gwoxc")) for m in meta["modes"]}
        self.label = label or (f"Aer run of the full circuit ({meta['n_qubits']} qubits, {meta['shots']:,} shots, statevector method, simulated here; not a Moth result)")

    @staticmethod
    def _index(g, w, o, x, c):
        return dict(g=g, w=w, o=o, x=x, cum=np.cumsum(c), classes={})

    def draw(self, K, rng, g=None, control="coherent", lamp_mode="z", world=None):
        """One frame: observation, light and polarity outcomes of ONE measured shot; the K photons from the shots with the same three outcomes."""
        if control != "coherent":
            return super().draw(K, rng, g=g, control=control, lamp_mode=lamp_mode, world=world)
        if lamp_mode not in self._pools:
            raise ValueError(f"the saved Aer run has no lamp register read in {lamp_mode.upper()}")
        sp, pool = self.spec, self._pools[lamp_mode]
        P = sp.P
        g = int(rng.integers(len(sp.observations))) if g is None else int(g)
        lo = np.searchsorted(pool["g"], g, side="left")
        hi = np.searchsorted(pool["g"], g, side="right")
        if world is not None:
            lo, hi = lo + np.searchsorted(pool["w"][lo:hi], int(world), side="left"), lo + np.searchsorted(pool["w"][lo:hi], int(world), side="right")
        cum = pool["cum"]
        base = cum[lo - 1] if lo else 0
        if hi <= lo or cum[hi - 1] == base:
            raise ValueError(f"the Aer run holds no shot with observation {g}" + ("" if world is None else f" and light world {world}"))
        e = lo + int(np.searchsorted(cum[lo:hi] - base, rng.integers(int(cum[hi - 1] - base)), side="right"))   # the measured shot that fixes the frame
        w, o = int(pool["w"][e]), int(pool["o"][e])
        cid = int(_class_ids(g, w, o, P))
        span = pool["classes"].get(cid)
        if span is None:                                                    # entries of this (g, w, o): contiguous, because the rows are sorted by (g, w, o, x)
            a = lo + np.searchsorted(pool["w"][lo:hi], w, side="left")
            b = lo + np.searchsorted(pool["w"][lo:hi], w, side="right")
            a, b = a + np.searchsorted(pool["o"][a:b], o, side="left"), a + np.searchsorted(pool["o"][a:b], o, side="right")
            before = cum[a - 1] if a else 0
            span = pool["classes"][cid] = (a, b, cum[a:b] - before)
        a, b, ccum = span
        xs = pool["x"][a + np.minimum(np.searchsorted(ccum, rng.integers(int(ccum[-1]), size=K), side="right"), b - a - 1)]
        lit = (((xs[:, None] >> np.arange(sp.F)[None, :]) & 1) == 0).mean(axis=0)
        if sp.n_classical:
            lit = np.concatenate([lit, self._classical_lit(K, rng, w, lamp_mode, "coherent")])
        pol = tuple(1 - 2 * ((o >> p) & 1) for p in range(P))
        gam, chi = sp.observations[g]
        lean = tuple(float(pv * np.cos(gam)) for pv in pol)
        return ReliefDraw(lit, WORLDS[w] if lamp_mode == "z" else (0, 0), pol, int(K), w, g, float(gam), float(chi), lean, "coherent", lamp_mode)

    def measure_witness(self, shots, rng):
        """(xs, zs) of a witness run on Aer: polarity in X, facets in Z (see witness_shots)."""
        return witness_shots(self.spec, shots, int(rng.integers(1 << 31)))


def load_state(spec, folder):
    return AerReliefState(spec, folder)


# ------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="labels.json from the pen tool (default: the built-in bay window)")
    ap.add_argument("--out", required=True, help="folder for shots.npz, circuit.qasm and state.json (the studio uses <project>/solve/aer)")
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--kappa", type=float), ap.add_argument("--n-dirs", type=int), ap.add_argument("--entangle", type=float)
    ap.add_argument("--contrast", type=float, help="not part of the circuit: recorded so that performing from this run uses the same tone")
    ap.add_argument("--observations")
    ap.add_argument("--engine", choices=["panel"], default="panel", help="only the per-panel engine fits a statevector run")
    a = ap.parse_args(argv)
    from src.capture.labels import bay_window_labels, load_labels
    labels = load_labels(a.labels) if a.labels else bay_window_labels()
    flags = {k: v for k, v in dict(kappa=a.kappa, n_dirs=a.n_dirs, entangle=a.entangle, contrast=a.contrast, observations=a.observations).items() if v is not None}
    if not 1000 <= a.shots <= 50_000_000:
        raise SystemExit("--shots must be between 1,000 and 50,000,000")
    try:
        run(labels, a.labels, a.out, a.shots, a.seed, flags)
    except AerRunError as e:
        raise SystemExit(f"cannot run on Aer: {e}")


if __name__ == "__main__":
    main()
