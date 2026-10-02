"""Write SYNTHETIC engine results to runs/selftest/ in the real on-disk format, so solver + A/B run offline.

    python -m tests.make_selftest_fixtures
    python -m src.quantum.solver graph-v1 --calib runs/calibration_5x4 --out runs/selftest     # cache hit: 0 credits, no network
    python -m src.quantum.solver qdrive   --calib runs/calibration_5x4 --out runs/selftest
    python -m src.quantum.ab_compare --run runs/selftest

Nothing here comes from Moth. The graph-v1 response is the exact moments of the classical oracle state, shrunk and jittered to
look like an engine that misses some targets; its shape is a GUESS at the undocumented fields (the parser accepts several).
The QDrive circuit is an arbitrary hand-built 19-qubit circuit (hub-controlled rotations), not a solution to the targets.
A file named SYNTHETIC in the run directory stamps every figure.
"""
import json
import os
import sys

import numpy as np

from src.capture.labels import bay_window_labels, fill_defaults, validate
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import allocate
from src.quantum import sampler_local as sl
from src.quantum.sampler_mock import Sampler
from src.store.cache import save_payload
from src.targets.payloads import payload_sha256


def build(calib="runs/calibration_5x4", out="runs/selftest", seed=5):
    rng = np.random.default_rng(seed)
    with open(os.path.join(calib, "targets.json")) as f:
        targets = json.load(f)
    NX, NY = targets["meta"]["grid"]
    scene = build_scene(validate(fill_defaults(bay_window_labels())))
    sampler = Sampler(scene, NX, NY)
    layout = allocate(sampler.n, sampler.n_panels)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "SYNTHETIC"), "w").write("offline self-test fixtures, not Moth results\n")

    # ---- graph-v1: tomography from the oracle's exact moments, shrunk and jittered; top-20 bitstrings from its probabilities
    oracle = sl.StateSource(sl.oracle_state(sampler, layout), layout)
    n = layout["n_qubits"]
    pairs = [tuple(t["qubits"]) for t in targets["relationships"]]
    single, zz = oracle.z_moments(pairs)
    shrink = lambda v: float(np.clip(0.9 * v + rng.normal(0, 0.03), -1, 1))
    order = np.argsort(oracle.probs)[::-1][:20]
    counts = rng.multinomial(1024, oracle.probs / oracle.probs.sum())
    top = [dict(bitstring="".join("1" if (i >> q) & 1 else "0" for q in range(n)), count=int(counts[i]), probability=float(oracle.probs[i]))
           for i in order]
    inline = dict(
        tomography=dict(bloch={str(q): dict(X=0.0, Y=0.0, Z=shrink(single[q])) for q in range(n)},
                        relationships={f"{a},{b}": dict(ZZ=shrink(v), XX=0.0) for (a, b), v in zip(pairs, zz)}),
        measurements=dict(top=top, dominant_bitstring=top[0]["bitstring"], edge_agreement_score=0.5),
        coupling_map=[list(p) for p in pairs])
    with open(os.path.join(calib, "payload_graph_v1.json")) as f:
        params = json.load(f)["params"]
    d = os.path.join(out, "graph_v1")
    save_payload(d, {"engine": "graph-v1", "params": params}, payload_sha256({"engine": "graph-v1", "params": params}))
    json.dump({"outputs": None, "result": inline}, open(os.path.join(d, "raw_result_selftest.json"), "w"))

    # ---- QDrive: an arbitrary circuit with hub-controlled rotations and neighbour couplings (NOT a solution)
    from qiskit import QuantumCircuit, qasm3
    qc = QuantumCircuit(n)
    L1, L2 = layout["lamps"]
    qc.h(L1); qc.h(L2)
    for p in range(sampler.n):
        qc.cry(1.5 * sampler.hub1[p], L1, p); qc.cry(1.5 * sampler.hub2[p], L2, p)
    for a, b in zip(*np.nonzero(np.triu(sampler.J))):
        qc.cry(0.9 * np.sign(sampler.J[a, b]) * 0.8, int(a), int(b))
    pol = layout["polarity"]
    qc.h(pol[0])
    for a, b in zip(pol[:-1], pol[1:]):
        qc.cx(a, b)
    d = os.path.join(out, "qdrive")
    with open(os.path.join(calib, "payload_qdrive.json")) as f:
        qparams = json.load(f)["params"]
    save_payload(d, {"engine": "qdrive-api-v1", "params": qparams}, payload_sha256({"engine": "qdrive-api-v1", "params": qparams}))
    open(os.path.join(d, "circuit"), "w").write(qasm3.dumps(qc))
    json.dump({"outputs": [{"slot": "circuit", "content_type": "text/plain"}], "result": None}, open(os.path.join(d, "raw_result_selftest.json"), "w"))
    print(f"wrote SYNTHETIC fixtures to {out}")


if __name__ == "__main__":
    build(*sys.argv[1:3])
