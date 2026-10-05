"""Sequential state-prep targets (full reduced density matrices of the prefix states) through the mock, on the real scene. Free."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene
from src.quantum import qdrive_plan as qp
from src.quantum.motte_mock import MotteMock
which = sys.argv[1] if len(sys.argv) > 1 else "pol"
t, sampler, layout, psi = scene.load(); n = layout["n_qubits"]; p = scene.probs(psi)
tm = qp.TargetMoments(t["bloch"], t["relationships"]); pairs = tm.pairs()
comp = [c for c in qp.components(tm.qubits(), pairs) if (len(c) == 8) == (which == "pol")][0]
idx = np.arange(len(p)); key = np.zeros(len(p), int)
for i, q in enumerate(comp): key |= ((idx >> q) & 1) << i
pc = np.zeros(2 ** len(comp)); np.add.at(pc, key, p)
job = qp.build_state_prep_job(comp, pairs, pc)
print(qp.describe([job])[0], "| words per target:", [len(x["expvals"]) for x in job.params["targets"] if x])
t0 = time.time(); m = MotteMock(len(comp), stale=True)
for tg in job.params["targets"]:
    if tg: m.run([tg])
local = lambda w, gq: m.moment(w, [comp.index(q) for q in gq])
words = [(q,) for q in comp] + [pq for pq in pairs if pq[0] in comp]
err = [local("Z" * len(w), list(w)) - tm(w) for w in words]
tv = 0.5 * float(np.abs(np.abs(m.psi) ** 2 - pc).sum())
print(f"mock result: rms(singles+pairs) {np.sqrt(np.mean(np.square(err))):.4f}  max {np.max(np.abs(err)):.4f}  TV vs oracle {tv:.4f}   [{time.time() - t0:.0f}s]")
