"""Compare the Motte mock (H_stale vs H_fresh) with real QDrive lab results. Free; no network."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum.motte_mock import MotteMock

Z = lambda q, v=0.0: dict(qubits=[q], expvals={"Z": v})
ZZ = lambda a, b, v=1.0: dict(qubits=[a, b], expvals={"ZZ": v})
def show(tag, m, probes, lab):
    got = [round(m.moment(w, q), 3) for w, q in probes]
    print(f"  {tag:8s} mock {got}   lab {lab}")

CASES = [
 ("T02 Z,Z,ZZ one layer x1", 2, [Z(0), Z(1), ZZ(0, 1)], 1, [("Z", [0]), ("Z", [1]), ("ZZ", [0, 1]), ("XX", [0, 1])], [-0.478, -0.336, 0.801, -0.776]),
 ("T03 same x3", 2, [Z(0), Z(1), ZZ(0, 1)], 3, [("Z", [0]), ("Z", [1]), ("ZZ", [0, 1]), ("XX", [0, 1])], [-0.355, -0.343, 0.832, -0.751]),
 ("T04 {ZZ,XX} x2", 2, [dict(qubits=[0, 1], expvals={"ZZ": 1.0, "XX": 1.0})], 2, [("Z", [0]), ("Z", [1]), ("ZZ", [0, 1]), ("XX", [0, 1])], [-0.012, -0.012, 1.0, 1.0]),
 ("T07 {ZI,IZ,ZZ} x1", 2, [dict(qubits=[0, 1], expvals={"ZI": 0.0, "IZ": 0.0, "ZZ": 1.0})], 1, [("Z", [0]), ("Z", [1]), ("ZZ", [0, 1])], [0.029, 0.029, 0.999]),
 ("T08 GHZ3 3-body x2", 3, [dict(qubits=[0, 1, 2], expvals={"ZZI": 1.0, "IZZ": 1.0, "XXX": 1.0})], 2, [("Z", [0]), ("ZZ", [0, 1]), ("XXX", [0, 1, 2])], [0.002, 0.999, 0.999]),
]
for name, n, tg, rounds, probes, lab in CASES:
    print(name)
    for tag, stale in (("stale", True), ("fresh", False)):
        m = MotteMock(n, stale=stale)
        for _ in range(rounds): m.run(tg)
        show(tag, m, probes, lab)

# the real 8-qubit polarity-chain jobs (qubits 11-18 remapped to 0-7): singles Z~0 + 11 ZZ~0.9
base = json.load(open("runs/harrietlivingroom/probe_lamp/payload_qdrive_lamp.json"))["params"]["targets"]
tg = [t for t in base if t]
singles = [t for t in tg if len(t["qubits"]) == 1]; pairs = [t for t in tg if len(t["qubits"]) == 2]
def stats(m):
    s = [m.moment("Z", t["qubits"]) for t in singles]
    p = [m.moment("ZZ", t["qubits"]) for t in pairs]
    es = np.sqrt(np.mean([(a - list(t["expvals"].values())[0]) ** 2 for a, t in zip(s, singles)]))
    ep = np.sqrt(np.mean([(a - list(t["expvals"].values())[0]) ** 2 for a, t in zip(p, pairs)]))
    return round(float(es), 3), round(float(ep), 3), round(float(np.mean(np.abs(s))), 2), round(float(np.mean(p)), 2)
print("polarity chain, 8 qubits: (rms singles, rms pairs, mean|Z|, mean ZZ)")
LAB = {("sp", 1): "(0.41, 1.01)", ("sp", 2): "(0.21, 0.88)", ("sp", 4): "(0.086, 0.90)", ("p", 2): "pairs-only: rms 0.097, mean|Z| ~0.9"}
for tag, stale in (("stale", True), ("fresh", False)):
    for kind, order, rounds in (("sp", singles + pairs, 1), ("sp", singles + pairs, 2), ("sp", singles + pairs, 4), ("p", pairs, 2)):
        t0 = time.time(); m = MotteMock(8, stale=stale)
        for _ in range(rounds): m.run(order)
        print(f"  {tag} {kind} x{rounds}: mock {stats(m)}   lab {LAB[(kind, rounds)]}   [{time.time() - t0:.0f}s]")
