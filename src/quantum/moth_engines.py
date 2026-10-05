"""Moth engines for Superposed Relief: what each one is for, and payload builders. BUILDING A PAYLOAD SENDS NOTHING.

    python -m src.quantum.moth_engines plan                       # every engine, its cost and whether it fits this project
    python -m src.quantum.moth_engines build tomography --out runs/moth      # writes payload_<engine>.json, prints the exact request
    python -m src.quantum.moth_engines build qdrive-probe|qdrive-facets|echo|qpixl|shader --out runs/moth
Sending is the existing, gated path: python -m src.quantum.moth_client send PAYLOAD --engine ID --approve-credits N

Engine docs read 2 Oct 2026 (docs.mothquantum.com/docs/engines/<id>). Fit, in order of value to this project:

 tomography-api-v2 (1 credit)  VERIFIER, best first spend. Takes ANY OpenQASM 2 circuit and returns single- and two-qubit tomography plus
     quantum and classical mutual information. Send our reference relief circuit: the Bloch X of a polarity qubit IS the interference
     visibility V_p, a (polarity, facet) pair gives <Z_B X_f> (slope sign locked to depth), and I(B_p : facet) measures the entanglement that
     carries the which-depth trade-off. A Moth-computed result about OUR state, scored against the exact moments (score_tomography).
 qdrive-api-v1 (1 credit)  STATE PREPARATION for the facet register ("specify correlations, not gates"): with diagonal couplings the facet
     state has non-trivial X/Y moments and XX/YY correlators for QDrive to meet. The two jobs of 201 targets both died with engine_timeout
     (targets 56 and 61); jobs must be small and chained through initial_circuit (an input_files asset id: upload flow in the Moth docs, the
     accepted asset types and the output->input reuse are UNVERIFIED), and mixed Pauli words need the 2-qubit probe first.
 otoc-echo-v1 (1 credit)  DYNAMICS. Out-of-time-order echo on a chain or square lattice (Aer <= 24 qubits; IBM backends up to 156 sites need the
     user's own IBM token, which this client never sends). The complex tap F(site, depth) maps onto the Gauss map: |F| -> tilt, arg F ->
     slope azimuth, sign(Re F) -> raised/sunk, depth -> animation frame: a touch on one patch spreads as relief across the wall.
 qpixl-v1 (1 credit)  LOG-QUBIT ENCODING. Encodes up to 64x64 values as qubit angles and decodes them by measurement; with the Lambert map
     (lit probability per pixel) as the values, 4096 facets need ~13 qubits. A scaling statement, not a replacement for the entangled state.
 entanglement-shader-v1 (1 credit)  MATERIAL. An angle-and-phase reflectance lookup table (OSL/GLSL/EXR/HDR) from a layered, interacting
     quantum system (<= 21 qubits). Candidate non-Lambertian tone response; not wired in.
 blur-core-v1 (1 credit), coin-toss-v1 (2), labyrinth-v1 (5), tessa-image-v1 (needs an image upload), the QRC / audio / MIDI engines:
     not a fit (see PLAN), kept out of the client except blur-core / coin-toss / labyrinth prices.
"""
import argparse
import json
import os

import numpy as np

from src.quantum.relief_state import facet_state, full_state, reference_circuit
from src.store.jsonfmt import dumps_compact

PLAN = [
    ("tomography-api-v2", 1, "HIGH", "verify our actual circuit: polarity Bloch X = visibility, (polarity, facet) pair = depth-locked slope, mutual information"),
    ("qdrive-api-v1", 1, "HIGH, blocked on job size", "prepare the entangled facet register from moments; chunk small and chain through initial_circuit"),
    ("otoc-echo-v1", 1, "MEDIUM-HIGH", "dynamic relief: echo taps -> tilt (|F|), azimuth (arg F), depth sign; Aer <= 24 qubits"),
    ("qpixl-v1", 1, "MEDIUM", "scaling story: Lambert map of up to 64x64 pixels on ~13 qubits"),
    ("entanglement-shader-v1", 1, "LOW-MEDIUM", "angle-phase reflectance LUT as a quantum-made tone response"),
    ("graph-v1", 5, "MEDIUM (existing)", "exact tomography of <=20 qubits; superseded for verification by tomography-api-v2 at 1/5 the price"),
    ("blur-core-v1", 1, "LOW", "unitary blur of a grid; the frame blur must be mask-aware and live, this is neither"),
    ("coin-toss-v1", 2, "NO", "returns head/tail counts, not a bit stream: cannot supply the lamp/observe bits"),
    ("labyrinth-v1", 5, "NO", "ZZ-correlated maze of open corridors; no honest mapping to relief"),
    ("tessa-image-v1", 1, "NO (for now)", "image in, image out: encodes a normal map but returns no joint state; needs the image-asset upload flow"),
]


# ------------------------------------------------------------------ exact moments (what an engine's result is scored against)
def pauli_moments(spec, include_lifted=True):
    """{"X0": <X on qubit 0>, ...} single-qubit moments for the facet and polarity qubits, and {"X0X1": ...} for the register's edges and every
    (polarity, facet-of-its-panel) pair. Exact, from the reference statevector."""
    from src.quantum.complementary import pauli_expectation
    psi = full_state(spec)
    n = spec.n_sim
    out = {}
    for q in range(n):
        for c in "XYZ":
            out[f"{c}{q}"] = pauli_expectation(psi, n, {q: c.lower()})
    pairs = ([(i, j) for i, j, *_ in spec.edges] + [(spec.F + int(spec.panel_of[i]), i) for i in range(spec.F)]
             + [(spec.F + int(a), spec.F + int(b)) for a, b in spec.pol_graph])
    for a, b in pairs:
        for c1 in "XYZ":
            for c2 in "XYZ":
                out[f"{c1}{a}{c2}{b}"] = pauli_expectation(psi, n, {a: c1.lower(), b: c2.lower()})
    return out


# ------------------------------------------------------------------ tomography-api-v2
def tomography_payload(spec, shots=4096, provider="aer", pair_scope="relief"):
    """The reference relief circuit as OpenQASM 2, with the qubits and pairs that carry the physics: every qubit; (polarity, facet) pairs; and the
    facet couplings. pair_scope='relief' keeps the request small; 'all' asks for every pair (the engine's default)."""
    from qiskit import qasm2, transpile
    # rzz is not in OpenQASM 2's standard library (a strict parser rejects it), so it is lowered to cx rz cx for the export only
    qasm = qasm2.dumps(transpile(reference_circuit(spec), basis_gates=["h", "ry", "rz", "cz", "cx"], optimization_level=0))
    pairs = sorted({(spec.F + int(spec.panel_of[i]), i) for i in range(spec.F)} | {(i, j) for i, j, *_ in spec.edges}
                   | {(spec.F + int(a), spec.F + int(b)) for a, b in spec.pol_graph})
    params = dict(circuit_qasm=qasm, provider_name=provider, shots=shots, single_tomography=True, double_tomography=True,
                  mutual_information=True, classical_mutual_information=True, qubit_list=list(range(spec.n_sim)))
    if pair_scope == "relief":
        params["qubit_pair_list"] = [list(p) for p in pairs]
    return dict(params=params)


def _find(obj, pred, depth=0):
    """First nested value satisfying pred, searching dicts and lists (results of this engine are 'not documented yet', so no shape is assumed)."""
    if pred(obj):
        return obj
    if depth > 6:
        return None
    for v in (obj.values() if isinstance(obj, dict) else obj if isinstance(obj, list) else []):
        r = _find(v, pred, depth + 1)
        if r is not None:
            return r
    return None


def parse_tomography(result):
    """({qubit: (x, y, z)}, {(a, b): {word: value}}, mutual_info) from whatever shape a tomography result has; raises ValueError with the keys seen
    if no Bloch vectors can be found. The raw result is always saved first by the client, so a miss costs nothing but a re-parse."""
    from src.quantum.solver import UnrecognisedResult, _bloch_map, _rel_map
    root = result.get("result", result) if isinstance(result, dict) else result
    if isinstance(root, dict) and "output" in root:
        root = root["output"]
    try:
        bl = _find(root, lambda o: isinstance(o, dict) and o and all(isinstance(v, (dict, list)) for v in o.values()) and
                   all(isinstance(k, (int, str)) and str(k).isdigit() for k in o) and
                   all((isinstance(v, dict) and {"X", "Y", "Z"} <= {str(k).upper() for k in v}) or (isinstance(v, list) and len(v) == 3) for v in o.values()))
        if bl is None:
            bl = _find(root, lambda o: isinstance(o, dict) and "bloch" in o)
            bl = bl["bloch"] if bl else None
        bloch = _bloch_map(bl) if bl is not None else None
    except (UnrecognisedResult, ValueError, KeyError):
        bloch = None
    if bloch is None:
        raise ValueError(f"no per-qubit Bloch vectors found; top-level keys: {sorted(root) if isinstance(root, dict) else type(root).__name__}")
    rel = {}
    r = _find(root, lambda o: isinstance(o, dict) and ("relationships" in o or "pairs" in o or "double" in o))
    if r:
        try:
            rel = _rel_map(r.get("relationships") or r.get("pairs") or r.get("double"))
        except (UnrecognisedResult, ValueError, AttributeError):
            rel = {}
    mi = _find(root, lambda o: isinstance(o, dict) and any("mutual" in str(k).lower() for k in o))
    return {q: (v["X"], v["Y"], v["Z"]) for q, v in bloch.items()}, rel, mi


def score_tomography(parsed, spec):
    """Score a parsed tomography against the exact moments. Prints-ready dict: rms error of single-qubit Bloch components, the polarity qubits'
    visibility vs V_p = prod cos(tau), and (when pairs came back) the (polarity, facet) correlators. Reported even when poor."""
    bloch, rel, _ = parsed
    ex = pauli_moments(spec)
    errs = [bloch[q][k] - ex[f"{c}{q}"] for q in bloch if q < spec.n_sim for k, c in enumerate("XYZ")]
    out = dict(n_qubits=len(bloch), rms_single=float(np.sqrt(np.mean(np.square(errs)))) if errs else None, max_single=float(np.max(np.abs(errs))) if errs else None,
               visibility=[dict(panel=p, measured=bloch.get(spec.F + p, (None,))[0], exact=ex[f"X{spec.F + p}"], isolated=spec.visibility(p)) for p in range(spec.P)])
    pair_err = []
    for (a, b), words in rel.items():
        for w, v in words.items():
            key = f"{w[0]}{a}{w[1]}{b}" if len(w) == 2 else None
            if key in ex:
                pair_err.append(v - ex[key])
    out["rms_pairs"] = float(np.sqrt(np.mean(np.square(pair_err)))) if pair_err else None
    out["n_pair_values"] = len(pair_err)
    return out


def score_folder(folder, spec):
    """Score the newest raw_result_*.json in `folder`, print the numbers, write score.json beside it (numeric top-level fields only, which is what the
    show's scorecard prints). Raises with the result's top-level keys if no Bloch vectors can be found; the raw file is never touched."""
    import glob
    import json
    files = sorted(glob.glob(os.path.join(folder, "raw_result_*.json")), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"no raw_result_*.json in {folder}: send the job first (moth_client send ... --out {folder})")
    with open(files[-1]) as f:
        result = json.load(f)
    parsed = parse_tomography(result)
    sc = score_tomography(parsed, spec)
    vis = [v for v in sc["visibility"] if v["measured"] is not None]
    flat = {k: v for k, v in sc.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if vis:
        flat["visibility_measured_mean"] = float(np.mean([v["measured"] for v in vis]))
        flat["visibility_exact_mean"] = float(np.mean([v["exact"] for v in vis]))
    with open(os.path.join(folder, "score.json"), "w") as f:
        json.dump(flat, f, indent=1)
    print(f"scored {os.path.basename(files[-1])}")
    for v in sc["visibility"]:
        print(f"  panel {v['panel']}: Moth <X> {v['measured']}  exact {v['exact']:.3f}  isolated product law {v['isolated']:.3f}")
    print("  " + ", ".join(f"{k} {v:.3g}" for k, v in flat.items()))
    print(f"wrote {os.path.join(folder, 'score.json')}")
    return flat


# ------------------------------------------------------------------ qdrive-api-v1
def qdrive_probe_payload():
    """Smallest job that tests the one unverified syntax: a mixed Pauli word on two qubits. 2 qubits, 4 entries (1 credit). Compare the returned
    circuit's <X0 Z1> with 0.5, and its singles with 0."""
    return dict(params=dict(machine="aer", n_qubits=2, coupling_map=[[0, 1]], update_method="spectral", shots=1024, tomography=0, sample=False, seed=7,
                            targets=[{"qubits": [0], "expvals": {"Z": 0.0}}, {"qubits": [1], "expvals": {"Z": 0.0}},
                                     {"qubits": [0, 1], "expvals": {"XZ": 0.5}}, None]))


def qdrive_facet_targets(spec):
    """Facet-register targets from the exact state: every facet's X, Y, Z and, for every coupling, the XX and YY correlators (the Z-only
    correlators of a diagonal coupling are products and carry nothing). Ordered most informative first."""
    psi = facet_state(spec)
    from src.quantum.complementary import pauli_expectation
    F = spec.F
    singles = [(abs(v), {"qubits": [q], "expvals": {c: round(float(v), 4)}}) for q in range(F) for c in "XYZ" for v in [pauli_expectation(psi, F, {q: c.lower()})]]
    pairs = [(abs(v), {"qubits": [i, j], "expvals": {c * 2: round(float(v), 4)}}) for i, j, *_ in spec.edges for c in "XY"
             for v in [pauli_expectation(psi, F, {i: c.lower(), j: c.lower()})]]
    return [t for _, t in sorted(singles + pairs, key=lambda x: -x[0])]


def qdrive_facet_payloads(spec, per_job=24, rounds=1, update_method="spectral", seed=7, shots=1024):
    """The facet-register program cut into jobs of at most `per_job` targets. Job k > 0 must be given job k-1's circuit as its initial_circuit
    (input_files asset id; the flow for that is UNVERIFIED), so only job 0 is a self-contained first experiment."""
    targets = qdrive_facet_targets(spec)
    edges = sorted({tuple(sorted((i, j))) for i, j, *_ in spec.edges})
    jobs = []
    for k in range(0, len(targets), per_job):
        chunk = targets[k:k + per_job]
        body = dict(machine="aer", n_qubits=spec.F, coupling_map=[list(e) for e in edges], update_method=update_method, shots=shots, tomography=0, sample=False,
                    seed=seed, targets=(chunk + [None]) * rounds)
        jobs.append(dict(params=body, needs_initial_circuit_from=None if k == 0 else (k // per_job) - 1))
    return jobs


def lift_and_score_circuit(spec, qasm_text):
    """Whatever circuit QDrive returns for the facet register: lift the polarity superposition on top locally, report its fidelity to the ideal
    facet state and its moment errors. Printed even when poor. Returns (ReliefState, report)."""
    from src.quantum import sampler_local as sl
    from src.quantum.relief_state import ReliefState
    psi, n = sl.statevector_from_qasm(qasm_text)
    if n != spec.F:
        raise ValueError(f"circuit has {n} qubits, the facet register has {spec.F}")
    ref = facet_state(spec)
    fid = float(abs(np.vdot(ref, psi)) ** 2 / (np.vdot(psi, psi).real))
    from src.quantum.complementary import pauli_expectation
    err = [pauli_expectation(psi, spec.F, {q: c}) - pauli_expectation(ref, spec.F, {q: c}) for q in range(spec.F) for c in "xyz"]
    return ReliefState(spec, psi, label="Moth QDrive circuit for the facet register, polarity lifted locally"), \
        dict(fidelity=fid, rms_single_moment_error=float(np.sqrt(np.mean(np.square(err)))), n_qubits=n)


# ------------------------------------------------------------------ otoc-echo-v1: dynamic relief
def echo_payload(width=5, height=4, depth=6, theta_x=0.35 * np.pi, theta_z=0.15 * np.pi, theta_zz=0.25 * np.pi, kick="Z", kick_site=None, shots=4096, seed=7):
    """Square-lattice echo whose sites are the patch grid (width x height <= 24 qubits on Aer). Taps are requested with include_taps."""
    p = dict(lattice="square", width=width, height=height, depth=depth, theta_x=round(float(theta_x), 6), theta_z=round(float(theta_z), 6),
             theta_zz=round(float(theta_zz), 6), kick=kick, machine="aer", shots=shots, include_taps=True, exact=False, seed=seed)
    if kick_site is not None:
        p["kick_site"] = int(kick_site)
    return dict(params=p)


def taps_to_relief(taps, n_sites, depth, tau_max=1.0):
    """The Gauss-map reading of an echo tap list [(site, depth, F_re, F_im), ...] (field names as the docs list them: site, depth, F_re, F_im):
    |F| -> tilt tau (scaled so the largest tap is tau_max), arg F -> slope azimuth phi, sign(Re F) -> +1 raised / -1 sunk. Returns arrays of shape
    (depth, n_sites): tau, phi, sign. Sites without a tap are flat. This mapping is OURS, not the engine's definition."""
    tau, phi, sign = (np.zeros((depth, n_sites)) for _ in range(3))
    mag = max([abs(complex(t[2], t[3])) for t in taps] + [1e-12])
    for site, d, re, im in taps:
        z = complex(re, im)
        if 0 <= site < n_sites and 1 <= d <= depth:
            tau[d - 1, site] = tau_max * abs(z) / mag
            phi[d - 1, site] = np.angle(z)
            sign[d - 1, site] = 1.0 if re >= 0 else -1.0
    return tau, phi, sign


def local_echo(width=4, height=4, depth=4, theta_x=0.35 * np.pi, theta_z=0.15 * np.pi, theta_zz=0.25 * np.pi, kick_site=None):
    """A LOCAL reading of the echo engine's description (not its definition): kicked-Ising Floquet step U = exp(-i theta_zz sum ZZ/2) exp(-i theta_z
    sum Z/2) exp(-i theta_x sum X/2); start from |0..0>, evolve d steps forward, kick (Pauli Z) at kick_site, evolve d steps back, and read
    F(site, d) = <X_site> + i <Y_site>. Returns the tap list [(site, d, re, im)] for d = 1..depth. <= 16 sites."""
    n = width * height
    if n > 16:
        raise ValueError("the local echo is for <= 16 sites; use the engine for more")
    kick_site = n // 2 if kick_site is None else kick_site
    idx = np.arange(2 ** n)
    z = 1 - 2 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)
    edges = [(y * width + x, y * width + x + 1) for y in range(height) for x in range(width - 1)] + [(y * width + x, (y + 1) * width + x) for y in range(height - 1) for x in range(width)]
    diag = np.exp(-1j * (theta_zz / 2 * sum(z[:, a] * z[:, b] for a, b in edges) + theta_z / 2 * z.sum(axis=1)))
    c, s = np.cos(theta_x / 2), -1j * np.sin(theta_x / 2)
    rx = np.array([[c, s], [s, c]])

    def step(v, inverse=False):
        t = v.reshape((2,) * n)
        if inverse:
            v = v * np.conj(diag)
            t = v.reshape((2,) * n)
        for q in range(n):
            ax = n - 1 - q
            m = rx.conj().T if inverse else rx
            t = np.moveaxis(np.tensordot(m, t, axes=([1], [ax])), 0, ax)
        v = t.reshape(-1)
        return v if inverse else v * diag

    taps = []
    psi0 = np.zeros(2 ** n, complex)
    psi0[0] = 1
    for d in range(1, depth + 1):
        v = psi0.copy()
        for _ in range(d):
            v = step(v)
        v = v * z[:, kick_site]
        for _ in range(d):
            v = step(v, inverse=True)
        pr = np.abs(v) ** 2
        for site in range(n):
            t = v.reshape((2,) * n)
            ax = n - 1 - site
            a0, a1 = np.take(t, 0, axis=ax).reshape(-1), np.take(t, 1, axis=ax).reshape(-1)
            ex = 2 * np.vdot(a0, a1)                                        # <X> + i<Y> = 2 <1|rho|0>-ish transverse coherence (conjugate convention)
            taps.append((site, d, float(ex.real), float(ex.imag)))
    return taps


# ------------------------------------------------------------------ qpixl and shader
def qpixl_payload(values, shots=4096):
    """The Lambert map (one probability per pixel, 2..4096 values) as an angle array; decoding it by measurement IS Born sampling at pixel scale."""
    v = [round(float(x), 5) for x in np.asarray(values, float).ravel()]
    if not 2 <= len(v) <= 4096:
        raise ValueError("qpixl-v1 encodes 2 to 4096 values (up to 64x64)")
    return dict(params=dict(values=v, shots=shots, mode="emu", machine="aer", discretize=0, dynamic_range="none"))


def shader_payload(layers=2, incoming_rays=8, reflectance=0.2, absorption=0.95, interaction=1.0, style="peaked", resolution=60):
    return dict(params=dict(reflectance=reflectance, absorption=absorption, layers=layers, incoming_rays=incoming_rays, interaction=interaction,
                            resolution=resolution, style=style))


# ------------------------------------------------------------------ CLI
def _spec(args):
    from src.capture.labels import bay_window_labels, fill_defaults, load_labels, validate
    from src.geometry.facets import build_facets
    from src.geometry.planes import build_scene
    from src.quantum.relief_state import spec_from_facets
    labels = load_labels(args.labels) if args.labels else validate(fill_defaults(bay_window_labels()))
    scene = build_scene(labels)
    return spec_from_facets(build_facets(scene), scene, entangle=args.entangle)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    b = sub.add_parser("build")
    b.add_argument("what", choices=["tomography", "qdrive-probe", "qdrive-facets", "echo", "qpixl", "shader"])
    b.add_argument("--out", default="runs/moth")
    b.add_argument("--labels")
    b.add_argument("--entangle", type=float, default=0.8)
    sc = sub.add_parser("score", help="score a saved tomography-api-v2 result against the exact moments and write score.json (the demo scorecard reads it)")
    sc.add_argument("folder", help="directory holding raw_result_<job>.json (the `send --out` directory)")
    sc.add_argument("--labels")
    sc.add_argument("--entangle", type=float, default=0.8)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print(f"{'engine':24s}{'credits':>8s}  {'fit':26s}role")
        for eid, cr, fit, role in PLAN:
            print(f"{eid:24s}{cr:>8d}  {fit:26s}{role}")
        print("\nNothing here sends anything. Preview: python -m src.quantum.moth_client preview FILE --engine ID")
        return
    if a.cmd == "score":
        return score_folder(a.folder, _spec(a))
    from src.quantum.moth_client import MothClient
    os.makedirs(a.out, exist_ok=True)
    c = MothClient()
    if a.what == "tomography":
        items = [("tomography-api-v2", tomography_payload(_spec(a)))]
    elif a.what == "qdrive-probe":
        items = [("qdrive-api-v1", qdrive_probe_payload())]
    elif a.what == "qdrive-facets":
        items = [("qdrive-api-v1", {"params": j["params"]}) for j in qdrive_facet_payloads(_spec(a))[:1]]
    elif a.what == "echo":
        items = [("otoc-echo-v1", echo_payload())]
    elif a.what == "qpixl":
        items = [("qpixl-v1", qpixl_payload(np.linspace(0.05, 0.95, 16)))]
    else:
        items = [("entanglement-shader-v1", shader_payload())]
    for eid, payload in items:
        path = os.path.join(a.out, f"payload_{eid}_{a.what}.json")
        with open(path, "w") as f:
            f.write(dumps_compact(payload) + "\n")
        print(f"wrote {path}\n{c.prepare(eid, payload['params']).describe(max_ops=3)[:1800]}\n(nothing was sent)\n")


if __name__ == "__main__":
    main()
