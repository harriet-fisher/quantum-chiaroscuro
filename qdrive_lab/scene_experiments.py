"""Offline strategy comparison on the REAL living-room scene, using the paper-based mock (free). Run: python qdrive_lab/scene_experiments.py [stale|fresh]"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene
from src.quantum import qdrive_plan as qp
from src.quantum.motte_mock import MotteMock

semantics = sys.argv[1] if len(sys.argv) > 1 else "stale"
stale = semantics == "stale"
t, sampler, layout, psi = scene.load()
n = layout["n_qubits"]; p_or = scene.probs(psi)
tm = qp.TargetMoments(t["bloch"], t["relationships"])
pairs = tm.pairs()
orc = qp.OracleMoments(psi, n, pairs)
comps = qp.components(tm.qubits(), pairs)
print(f"mock semantics: {semantics}; components: {[(c[0], c[-1], len(c)) for c in comps]}\n")


def marginal(comp):
    idx = np.arange(len(p_or)); key = np.zeros(len(p_or), int)
    for i, q in enumerate(comp):
        key |= ((idx >> q) & 1) << i
    out = np.zeros(2 ** len(comp)); np.add.at(out, key, p_or); return out


def report(name, comp, m, n_targets, n_updates):
    local = lambda w, gq: m.moment(w, [comp.index(q) for q in gq])
    words = [(q,) for q in comp] + [pq for pq in pairs if pq[0] in comp]
    err = [local("Z" * len(w), list(w)) - tm(w) for w in words]
    pr = np.abs(m.psi) ** 2; tv = 0.5 * float(np.abs(pr - marginal(comp)).sum())
    print(f"  {name:44s} rms(singles+pairs) {np.sqrt(np.mean(np.square(err))):.4f}  max {np.max(np.abs(err)):.3f}  TV {tv:.3f}  targets {n_targets:3d} updates {n_updates:3d}")


for comp in comps:
    print(f"component qubits {comp[0]}..{comp[-1]} ({len(comp)} qubits)")
    sp = [dict(qubits=[comp.index(q)], expvals={"Z": tm((q,))}) for q in comp]
    pp = [dict(qubits=[comp.index(a), comp.index(b)], expvals={"ZZ": tm((a, b))}) for a, b in pairs if a in comp]
    # A. naive: what we sent before (singles + overlapping pairs, ONE layer), rounds 1/2/4
    for r in (1, 2, 4):
        m = MotteMock(len(comp), stale=stale)
        for _ in range(r): m.run(sp + pp)
        report(f"A naive singles+pairs, one layer x{r}", comp, m, r * (len(sp) + len(pp)), r)
    # B/C. grouped + layer-separated
    for label, mom, k in (("B pair groups k=2 (targets only)", tm, 2), ("C triangle groups k=3 (exact words)", orc, 3)):
        for r in (1, 2, 3):
            job = qp.build_job(comp, pairs, mom, k, r)
            assert not [x for x in qp.check(job) if "width" not in x], qp.check(job)
            t0 = time.time(); m = MotteMock(len(comp), stale=stale); m.run(job.params["targets"])
            report(f"{label} x{r}", comp, m, job.n_targets, sum(1 for x in job.params["targets"] if x is None))
    # D. closed loop on top of k=3 exact groups
    def engine(params, prev):
        m = prev if prev is not None else MotteMock(len(comp), stale=stale)
        m.run(params["targets"]); return m, (lambda w, gq: m.moment(w, [comp.index(q) for q in gq]))
    print("  D closed loop, k=3 exact words, tol 0.03, gain 0.5")
    m, hist = qp.closed_loop(comp, pairs, orc, engine, tol=0.03, gain=0.5, max_jobs=5, log=lambda s: None)
    for h in hist: print(f"     job {h['job']}: rms {h['rms']:.4f} stubborn {h['stubborn']}/{h['groups']}")
    report("D after loop", comp, m, 0, 0)
    print()
