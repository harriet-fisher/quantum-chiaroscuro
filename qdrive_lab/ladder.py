"""The live validation ladder for the QDrive plan: payloads (NOT sent), local scoring, and a gated send. Run from chiaroscuro/.

    python qdrive_lab/ladder.py build                       # writes runs/harrietlivingroom/qdrive_plan/R*/payload.json, prints the plan
    python qdrive_lab/ladder.py score R1 <circuit file>     # score a returned circuit locally against the exact oracle
    python qdrive_lab/ladder.py send R1 --approve-credits 1 # the ONLY command that spends credits (1 per rung)

Every rung answers one question the planner depends on (see docs/qdrive-solution-design.md section 6).
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from src.quantum import qdrive_plan as qp

OUT = os.path.join(os.path.dirname(HERE), "runs", "harrietlivingroom", "qdrive_plan")


def synth4():
    """4 qubits: q0,q1 fair coins; q2 follows q0 xor-ish; q3 depends on q0,q1,q2 (a 4-body growth step)."""
    idx = np.arange(16); z = lambda q: 1 - 2 * ((idx >> q) & 1)
    e = 0.9 * z(0) * z(2) - 0.7 * z(1) * z(2) + 0.8 * z(2) * z(3) + 0.6 * z(0) * z(3) - 0.5 * z(1) * z(3)
    p = np.exp(e); return p / p.sum()


def ghz3():
    p = np.zeros(8); p[0] = p[7] = 0.5; return p


def scene_component(which):
    import scene
    t, sampler, layout, psi = scene.load(); n = layout["n_qubits"]; pr = scene.probs(psi)
    tm = qp.TargetMoments(t["bloch"], t["relationships"]); pairs = tm.pairs()
    comp = [c for c in qp.components(tm.qubits(), pairs) if (len(c) == 8) == (which == "pol")][0]
    idx = np.arange(len(pr)); key = np.zeros(len(pr), int)
    for i, q in enumerate(comp): key |= ((idx >> q) & 1) << i
    pc = np.zeros(2 ** len(comp)); np.add.at(pc, key, pr)
    return comp, pairs, pc, tm


def rungs():
    R = {}
    g = ghz3()
    R["R1"] = dict(why="Does growth work when the current group is ALREADY entangled with the rest (mixed reduced state)? GHZ-3 in two steps: Bell on (0,1), then (1,2).",
                   n=3, prob=g, comp=[0, 1, 2], pairs=[(0, 1), (1, 2)], builder="stateprep", expect="TV<0.03: the engine extends an entangled qubit correctly. If not, sequential growth fails and the plan needs (m+1)-body targets.")
    s4 = synth4()
    R["R2"] = dict(why="Do 4-qubit targets run (and fit the time limit), and is a 4-body growth step honoured?",
                   n=4, prob=s4, comp=[0, 1, 2, 3], pairs=[(0, 2), (1, 2), (2, 3), (0, 3), (1, 3)], builder="stateprep", expect="TV<0.05 and completes in < 60 s.")
    comp, pairs, pc, tm = scene_component("pol")
    R["R3"] = dict(why="The real polarity component (8 qubits, groups <= 3, 8 layers) with full reduced-density-matrix words.",
                   n=8, prob=pc, comp=comp, pairs=pairs, builder="stateprep", expect="TV<0.05 vs the exact oracle; singles+pairs rms<0.03.", tm=tm)
    R["R3b"] = dict(why="Same plan but Z-words only (what we have been sending): shows whether the X/Y words matter. Run only if R3 passes or to diagnose it.",
                    n=8, prob=pc, comp=comp, pairs=pairs, builder="zonly", expect="Expected worse than R3 if the full words are needed.", tm=tm)
    comp, pairs, pc, tm = scene_component("patch")
    R["R4"] = dict(why="The real patches+lamps component (11 qubits, 4-body groups, one virtual lamp-lamp edge).",
                   n=11, prob=pc, comp=comp, pairs=pairs, builder="stateprep", expect="TV<0.05; completes (11-qubit width and 4-body groups are both unvalidated).", tm=tm)
    return R


def local_job(r):
    """Jobs are always in LOCAL component indices (n_qubits = component size)."""
    comp = r["comp"]; n = r["n"]
    if r["builder"] == "stateprep":
        loc_pairs = [(comp.index(a), comp.index(b)) for a, b in r["pairs"] if a in comp and b in comp]
        return qp.build_state_prep_job(list(range(n)), loc_pairs, r["prob"])
    loc_pairs = [(comp.index(a), comp.index(b)) for a, b in r["pairs"] if a in comp and b in comp]
    return qp.build_growth_job(list(range(n)), loc_pairs, qp.OracleMoments(r["prob"], n, loc_pairs))


def score(r, circuit_text):
    from qiskit import qasm3
    from qiskit.quantum_info import Statevector
    sv = Statevector(qasm3.loads(circuit_text).remove_final_measurements(inplace=False))
    p = np.abs(sv.data) ** 2
    tv = 0.5 * float(np.abs(p - r["prob"]).sum())
    n = r["n"]; idx = np.arange(len(p)); z = lambda q: 1 - 2 * ((idx >> q) & 1)
    po = r["prob"]
    errs = [float(p @ z(a) - po @ z(a)) for a in range(n)]
    loc_pairs = [(r["comp"].index(a), r["comp"].index(b)) for a, b in r["pairs"] if a in r["comp"] and b in r["comp"]]
    errs += [float(p @ (z(a) * z(b)) - po @ (z(a) * z(b))) for a, b in loc_pairs]
    return dict(tv=round(tv, 4), rms_singles_pairs=round(float(np.sqrt(np.mean(np.square(errs)))), 4), max_err=round(float(np.max(np.abs(errs))), 4))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "score", "send"]); ap.add_argument("rung", nargs="?"); ap.add_argument("circuit", nargs="?")
    ap.add_argument("--approve-credits", type=int)
    a = ap.parse_args(); R = rungs()
    if a.cmd == "build":
        os.makedirs(OUT, exist_ok=True); total = 0
        for k, r in R.items():
            job = local_job(r); probs = qp.check(job)
            d = os.path.join(OUT, k); os.makedirs(d, exist_ok=True)
            json.dump(dict(params=job.params), open(os.path.join(d, "payload.json"), "w"), indent=0)
            nw = [len(t["expvals"]) for t in job.params["targets"] if t]
            print(f"{k}: {r['why']}\n     {job.params['n_qubits']} qubits, {len(nw)} targets (words {nw}), {sum(1 for t in job.params['targets'] if t is None)} updates, "
                  f"edges {len(job.params['coupling_map'])}; expect: {r['expect']}\n     planner flags: {[x for x in probs] or 'none'}")
            total += 1
        print(f"\npayloads in {OUT}; {total} rungs x 1 credit = {total} credits if all are sent. NOTHING WAS SENT.")
    elif a.cmd == "score":
        print(json.dumps(score(R[a.rung], open(a.circuit).read())))
    else:
        r = R[a.rung]; job = local_job(r)
        if a.approve_credits != 1:
            sys.exit("this sends ONE job (1 credit): pass --approve-credits 1")
        import lab
        res = lab.submit_job(f"ladder_{a.rung}", job.params, wait_s=330)
        print({k: res.get(k) for k in ("http", "status", "secs", "job_id", "error", "error_body")})
        if res.get("circuit_path"):
            print("score vs exact oracle:", score(r, open(res["circuit_path"]).read()))


if __name__ == "__main__":
    main()
