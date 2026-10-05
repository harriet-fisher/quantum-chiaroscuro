"""Paid trials (1 credit per accepted job; hard cap = QDRIVE_LAB_BUDGET jobs in lab.py, default 15).
Run: python qdrive_lab/trials.py STAGE      (stage1 | stage2 | ...)  or  python qdrive_lab/trials.py T03 T04
Every trial uses tiny 2-4 qubit circuits with KNOWN right answers (Bell, GHZ), scores the returned circuit locally, and writes
results/trial_<name>.json. Results are never recomputed from the API: a trial that already has a result file is skipped.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab                                              # noqa: E402

Z = lambda q, v=0.0: dict(qubits=[q], expvals={"Z": v})
ZZ = lambda a, b, v=1.0: dict(qubits=[a, b], expvals={"ZZ": v})
BELL_SCORE = [dict(qubits=[0], word="Z", value=0.0), dict(qubits=[1], word="Z", value=0.0), dict(qubits=[0, 1], word="ZZ", value=1.0),
              dict(qubits=[0, 1], word="XX", value=1.0)]          # XX=+1 is NOT requested in most trials: it shows whether a Bell STATE was built
ZONLY_SCORE = BELL_SCORE[:3]
GHZ3_SCORE = [dict(qubits=[q], word="Z", value=0.0) for q in range(3)] + \
             [dict(qubits=p, word="ZZ", value=1.0) for p in ([0, 1], [1, 2], [0, 2])] + [dict(qubits=[0, 1, 2], word="XXX", value=1.0)]


def base(**kw):
    p = dict(machine="aer", n_qubits=2, seed=7, shots=1024, update_method="spectral", tomography=0, sample=False)
    p.update(kw)
    return p


def bell_targets(rounds):
    return ([Z(0), Z(1), ZZ(0, 1), None]) * rounds


TRIALS = {}


def trial(name, stage):
    def deco(fn):
        TRIALS[name] = (stage, fn)
        return fn
    return deco


# ------------------------------------------------------------------ stage 1: basics on a Bell pair
@trial("T01_official_sample", 1)
def t01():       # the engine's own code sample, verbatim: one ZZ target, NO update() entry
    return dict(params=dict(n_qubits=2, targets=[dict(expvals={"ZZ": 1.0}, qubits=[0, 1])]), score=ZONLY_SCORE[2:] + BELL_SCORE[3:])


@trial("T02_bell_1round", 1)
def t02():
    return dict(params=base(targets=bell_targets(1)), score=BELL_SCORE)


@trial("T03_bell_3rounds", 1)
def t03():
    return dict(params=base(targets=bell_targets(3)), score=BELL_SCORE)


@trial("T04_bell_mixed_word_mapping", 1)
def t04():       # one target carrying two Pauli words: does a mapping with several keys work?
    return dict(params=base(targets=[dict(qubits=[0, 1], expvals={"ZZ": 1.0, "XX": 1.0}), None] * 2), score=BELL_SCORE)


@trial("T05_uploaded_text_asset_tomography2", 1)
def t05():       # our own QASM3 Bell circuit uploaded as a text/plain asset, used as initial_circuit; no targets, just tomography
    aid = open(os.path.join(lab.RESULTS, "bell_asset_id.txt")).read().strip()
    return dict(params=dict(tomography=2), input_files={"initial_circuit": aid}, score=BELL_SCORE)


@trial("T06_chain_by_job_ref", 1)
def t06():       # the engine's documented chaining form, off the T02 job
    jid = json.load(open(os.path.join(lab.RESULTS, "trial_T02_bell_1round.json")))["job_id"]
    return dict(params=dict(tomography=2), input_files={"initial_circuit": f"job:{jid}/circuit"}, score=BELL_SCORE)


# ------------------------------------------------------------------ stage 2: "one target = one gate on its qubit group"
BELL_GROUP = dict(qubits=[0, 1], expvals={"ZZ": 1.0, "XX": 1.0})
GHZ3_GROUP = dict(qubits=[0, 1, 2], expvals={"ZZI": 1.0, "IZZ": 1.0, "XXX": 1.0})


@trial("T07_bell_identity_padded", 2)
def t07():       # singles folded into the pair target with identity letters; no XX requested -> any (|00>+e^{i phi}|11>)/sqrt2 is right
    return dict(params=base(targets=[dict(qubits=[0, 1], expvals={"ZI": 0.0, "IZ": 0.0, "ZZ": 1.0}), None]), score=ZONLY_SCORE)


@trial("T08_ghz3_one_group_target", 2)
def t08():       # one 3-qubit target (3-body word XXX): can QDrive build GHZ-3?
    return dict(params=base(n_qubits=3, targets=[GHZ3_GROUP, None] * 2), score=GHZ3_SCORE)


@trial("T09_chain_from_output_asset", 2)
def t09():       # continue from T02's (wrong-ish) circuit by its output asset UUID; target the full Bell RDM
    aid = open(os.path.join(lab.RESULTS, "T02_circuit_asset_id.txt")).read().strip()
    return dict(params=dict(machine="aer", seed=7, targets=[BELL_GROUP, None]), input_files={"initial_circuit": aid}, score=BELL_SCORE)


@trial("T10_seed_repeat_of_T04", 2)
def t10():       # identical to T04: is the output byte-identical (determinism for a fixed seed)?
    return dict(params=base(targets=[dict(BELL_GROUP), None] * 2), score=BELL_SCORE)


@trial("T11_sample_and_tomography", 2)
def t11():       # T04 plus sample=true and tomography=2: do extra outputs / inline results appear?
    return dict(params=base(targets=[dict(BELL_GROUP), None] * 2, sample=True, tomography=2), score=BELL_SCORE)


@trial("T12_direct_update", 2)
def t12():
    return dict(params=base(targets=[dict(BELL_GROUP), None] * 2, update_method="direct"), score=BELL_SCORE)


@trial("T13_constrained_update", 2)
def t13():
    return dict(params=base(targets=[dict(BELL_GROUP), None] * 2, update_method="constrained"), score=BELL_SCORE)


@trial("T14_certainty_on_words", 2)
def t14():       # [value, certainty] per word inside the mapping
    return dict(params=base(targets=[dict(qubits=[0, 1], expvals={"ZZ": [1.0, 0.9], "XX": [1.0, 0.9]}), None] * 2), score=BELL_SCORE)


# ------------------------------------------------------------------ stage 3: the real lamp / patch data as GROUP targets
LIVING = os.path.join(os.path.dirname(lab.HERE), "runs", "harrietlivingroom", "solve", "qdrive", "payload.json")


def group_target(entries, qubits, remap=True):
    """entries: original single/pair targets (dict qubits/expvals{Z|ZZ}); returns ONE target on `qubits` whose mapping holds every
    word (identity-padded) for the group, plus the matching scoring list in the group's local indexing."""
    local = {q: i for i, q in enumerate(qubits)}
    words, score = {}, []
    for e in entries:
        if not all(q in local for q in e["qubits"]):
            continue
        w = ["I"] * len(qubits)
        for q in e["qubits"]:
            w[local[q]] = "Z"
        v = list(e["expvals"].values())[0]
        words["".join(w)] = v
        score.append(dict(qubits=[local[q] for q in e["qubits"]] if remap else e["qubits"],
                          word="Z" * len(e["qubits"]), value=v))
    return dict(qubits=list(range(len(qubits))) if remap else list(qubits), expvals=words), score


def _living():
    p = json.load(open(LIVING))["params"]
    return [t for t in p["targets"] if t], p["coupling_map"]


@trial("T15_lamp8_one_group_target", 3)
def t15():       # qubits 11-18 (the lamp) as ONE 8-qubit target holding 8 Z words and 11 ZZ words
    tg, cm = _living()
    lamp = list(range(11, 19))
    tgt, score = group_target(tg, lamp)
    cmap = [[a - 11, b - 11] for a, b in cm if a >= 11 and b >= 11]
    return dict(params=base(n_qubits=8, coupling_map=cmap, targets=[tgt, None] * 2), score=score)


@trial("T16_full19_two_disjoint_groups", 3)
def t16():       # the real 19-qubit register: patch group (0-10) and lamp group (11-18) as two disjoint group targets
    tg, cm = _living()
    ta, sa = group_target(tg, list(range(0, 11)), remap=False)
    tb, sb = group_target(tg, list(range(11, 19)), remap=False)
    score = sa + sb                                     # remap=False: scoring uses the original 19-qubit indices already
    return dict(params=base(n_qubits=19, coupling_map=cm, targets=[ta, tb, None] * 2), score=score, wait_s=330)


def run(name):
    out_path = os.path.join(lab.RESULTS, f"trial_{name}.json")
    if os.path.exists(out_path):
        print("skip (already run):", name); return json.load(open(out_path))
    spec = TRIALS[name][1]()
    res = lab.submit_job(name, spec["params"], spec.get("input_files"), wait_s=spec.get("wait_s", 200))
    if res.get("circuit_path"):
        res["score"] = lab.score(res["circuit_path"], spec["score"])
    lab.save(f"trial_{name}.json", res)
    sc = res.get("score")
    print(f"{name}: http {res['http']} status {res.get('status')} secs {res.get('secs')} job {res.get('job_id')}")
    if res.get("error_body"):
        print("   rejected:", res["error_body"][:400])
    if res.get("error"):
        print("   job error:", res["error"])
    if res.get("outputs"):
        print("   outputs:", [o.get("slot") for o in res["outputs"]], "inline result:", str(res.get("result_inline"))[:160])
    if sc:
        print("   ops:", sc["ops"], "| rms", sc["rms"])
        print("   ", [(r["word"], r["qubits"], r["want"], r["got"]) for r in sc["rows"]])
    return res


if __name__ == "__main__":
    args = sys.argv[1:] or ["stage1"]
    names = []
    for a in args:
        if a.startswith("stage"):
            names += [n for n, (s, _) in TRIALS.items() if s == int(a[5:])]
        else:
            names += [n for n in TRIALS if n.startswith(a)]
    print(f"jobs spent so far: {lab.jobs_spent()} / {lab.BUDGET}")
    for n in names:
        run(n)
    print(f"\njobs spent: {lab.jobs_spent()} / {lab.BUDGET}")
