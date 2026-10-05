#!/usr/bin/env python3
"""solve(graph, targets): run the calibrated targets on a Moth engine and score what came back (spec §8.2, handoff §6.2).

    # preview only: no key, no network, no credits (add --approve-credits N to actually send)
    python -m src.quantum.solver graph-v1                     # --calib runs/calibration_bay_4x4 --out runs/bay_run, the built-in bay window
    python -m src.quantum.solver graph-v1 --approve-credits 5
    python -m src.quantum.solver qdrive   --approve-credits 1
    # QDrive in small chained rounds (what to use when one big job stalls): python -m src.quantum.qdrive_rounds --help

graph-v1  -> <out>/graph_v1/state.json, requested_vs_achieved.{json,csv}   (tomography is exact; no state or circuit comes back)
qdrive    -> <out>/qdrive/circuit.qasm, state.json, requested_vs_achieved.{json,csv}   (circuit simulated locally with Aer)

The raw response is saved before anything is parsed, and a payload that already has a result on disk is never sent again.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from src.capture.labels import DEFAULT_CALIB_DIR, DEFAULT_RUN_DIR
from src.quantum import sampler_local as sl
from src.quantum.moth_client import MothClient
from src.store.cache import cached_result, save_payload
from src.targets.payloads import payload_sha256


class UnrecognisedResult(ValueError):
    pass


# ------------------------------------------------------------------ graph-v1 response parsing
def _bloch_map(b):
    """{qubit: {"X","Y","Z"}} from the shapes a tomography.bloch field plausibly has."""
    if isinstance(b, dict) and set(b) >= {"X", "Y", "Z"} and isinstance(b["Z"], list):
        return {q: {k: float(b[k][q]) for k in "XYZ"} for q in range(len(b["Z"]))}
    items = b.items() if isinstance(b, dict) else enumerate(b)
    out = {}
    for q, v in items:
        if isinstance(v, dict) and {"X", "Y", "Z"} <= {k.upper() for k in v}:
            out[int(q)] = {k.upper(): float(x) for k, x in v.items() if k.upper() in "XYZ"}
        elif isinstance(v, (list, tuple)) and len(v) == 3:
            out[int(q)] = dict(zip("XYZ", map(float, v)))
        else:
            raise UnrecognisedResult(f"tomography.bloch[{q!r}] has an unrecognised shape: {str(v)[:80]}")
    return out


def _rel_map(r):
    """{(a, b): {"ZZ": v, ...}} from tomography.relationships keyed "a,b". A bare number is recorded under "?" (the Pauli word is
    not stated); a 3x3 matrix is read as [first-qubit Pauli][second-qubit Pauli] in XYZ order."""
    out = {}
    for key, v in r.items():
        a, b = (int(x) for x in str(key).replace("(", "").replace(")", "").replace(" ", "").split(","))
        if isinstance(v, (int, float)):
            out[(a, b)] = {"?": float(v)}
        elif isinstance(v, dict):
            out[(a, b)] = {k.upper(): float(x) for k, x in v.items() if isinstance(x, (int, float))}
        elif isinstance(v, (list, tuple)) and len(v) == 3 and all(len(row) == 3 for row in v):
            out[(a, b)] = {p + q: float(v[i][j]) for i, p in enumerate("XYZ") for j, q in enumerate("XYZ")}
        else:
            raise UnrecognisedResult(f"tomography.relationships[{key!r}] has an unrecognised shape: {str(v)[:80]}")
    return out


def parse_graph_v1(inline):
    """Normalise a graph-v1 result. Fields are 'not documented yet' by the engine author, so accept the documented names at the
    top level or under 'output', and raise UnrecognisedResult (with the keys seen) rather than guess."""
    root = inline.get("output", inline) if isinstance(inline, dict) else None
    if not isinstance(root, dict) or "tomography" not in root:
        raise UnrecognisedResult(f"no 'tomography' in the result; top-level keys: {sorted(inline) if isinstance(inline, dict) else type(inline).__name__}")
    tomo = root["tomography"]
    meas = root.get("measurements") or []
    # Real shape [VERIFIED on the first run]: `measurements` is a list of {bitstring, count, probability}; dominant_bitstring and
    # edge_agreement_score sit beside it in `output`. A dict form ({"top": [...]}) is also accepted.
    top = meas if isinstance(meas, list) else (meas.get("top") or meas.get("counts") or meas.get("bitstrings") or [])
    if isinstance(top, dict):
        top = [{"bitstring": k, "probability": v} for k, v in top.items()]
    src = root if isinstance(meas, list) else {**root, **meas}
    return dict(bloch=_bloch_map(tomo["bloch"]), relationships=_rel_map(tomo["relationships"]), top=top,
                dominant=src.get("dominant_bitstring"), edge_agreement=src.get("edge_agreement_score"),
                coupling_map=root.get("coupling_map"), shots=root.get("shots"), backend=root.get("backend"))


def graph_v1_achieved(parsed, targets):
    """Achieved <Z> and <ZZ> aligned to the requested rows. Returns (rows, notes)."""
    rows, notes = [], []
    for t in targets["bloch"]:
        got = parsed["bloch"].get(t["qubit"], {}).get("Z")
        rows.append(dict(label=t["label"], kind="single", qubits=[t["qubit"]], requested=t["Z"], achieved=got))
    bare = False
    for t in targets["relationships"]:
        a, b = t["qubits"]
        v = parsed["relationships"].get((a, b)) or parsed["relationships"].get((b, a)) or {}
        got = v.get("ZZ")
        if got is None and "?" in v:
            got, bare = v["?"], True
        rows.append(dict(label=t["label"], kind=t["kind"], qubits=[a, b], requested=t["ZZ"], achieved=got))
    if bare:
        notes.append("relationships were bare numbers; assumed to be <ZZ> (the Pauli word is not stated in the response)")
    return rows, notes


# ------------------------------------------------------------------ scoring
def score(rows):
    have = [r for r in rows if r["achieved"] is not None]
    for r in rows:
        r["error"] = None if r["achieved"] is None else r["achieved"] - r["requested"]
    err = np.array([r["error"] for r in have]) if have else np.array([])
    by_kind = {}
    for k in sorted({r["kind"] for r in have}):
        e = np.array([r["error"] for r in have if r["kind"] == k])
        by_kind[k] = dict(n=len(e), rms=float(np.sqrt((e ** 2).mean())), max_abs=float(np.abs(e).max()))
    return dict(n_requested=len(rows), n_achieved=len(have), missing=[r["label"] for r in rows if r["achieved"] is None],
               rms=float(np.sqrt((err ** 2).mean())) if len(err) else None, max_abs=float(np.abs(err).max()) if len(err) else None,
               n_gap_over_0p1=int((np.abs(err) > 0.1).sum()), by_kind=by_kind)


def write_report(out_dir, rows, summary, state):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=1)
    with open(os.path.join(out_dir, "requested_vs_achieved.json"), "w") as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=1)
    with open(os.path.join(out_dir, "requested_vs_achieved.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "kind", "qubits", "requested", "achieved", "error"])
        for r in rows:
            w.writerow([r["label"], r["kind"], "-".join(map(str, r["qubits"])), r["requested"], r["achieved"], r["error"]])


def print_summary(name, summary, notes=()):
    print(f"\n{name}: achieved {summary['n_achieved']}/{summary['n_requested']} requested values; "
          f"rms error {summary['rms']:.4f}, max |error| {summary['max_abs']:.4f}, {summary['n_gap_over_0p1']} off by more than 0.1")
    for k, v in summary["by_kind"].items():
        print(f"   {k:18s} n={v['n']:3d}  rms {v['rms']:.4f}  max {v['max_abs']:.4f}")
    for n in notes:
        print("   note:", n)
    if summary["missing"]:
        print(f"   missing from the response: {len(summary['missing'])} (e.g. {summary['missing'][:2]})")


# ------------------------------------------------------------------ QDrive circuit, evaluated locally
def qdrive_achieved(qasm_text, targets, layout):
    """Simulate the returned circuit locally and read the requested <Z>, <ZZ> exactly. Returns (rows, StateSource)."""
    src = sl.source_from_qasm(qasm_text, layout, label="QDrive circuit (local Aer statevector)")
    pairs = [tuple(t["qubits"]) for t in targets["relationships"]]
    single, zz = src.z_moments(pairs)
    rows = [dict(label=t["label"], kind="single", qubits=[t["qubit"]], requested=t["Z"], achieved=float(single[t["qubit"]]))
            for t in targets["bloch"]]
    rows += [dict(label=t["label"], kind=t["kind"], qubits=list(t["qubits"]), requested=t["ZZ"], achieved=float(v))
             for t, v in zip(targets["relationships"], zz)]
    return rows, src


# ------------------------------------------------------------------ running an engine
def _load(calib):
    with open(os.path.join(calib, "targets.json")) as f:
        targets = json.load(f)
    return targets


def run_engine(engine, calib, out, approve_credits):
    fname, eid = dict(graph_v1=("payload_graph_v1.json", "graph-v1"), qdrive=("payload_qdrive.json", "qdrive-api-v1"))[engine]
    with open(os.path.join(calib, fname)) as f:
        params = json.load(f)["params"]
    client = MothClient()
    job = client.prepare(eid, params)
    sha, run_dir = payload_sha256({"engine": eid, "params": params}), os.path.join(out, engine)
    print(job.describe(max_ops=6))
    cached = cached_result(run_dir, sha)
    if cached:
        print(f"\nresult for this exact payload is already on disk ({cached['path']}): reusing it, 0 credits.")
        return cached["result"], None, run_dir, job
    if approve_credits is None:
        print("\n(preview only: nothing was sent. Add --approve-credits "
              f"{job.credits} to send; MOTH_API_KEY must be set in the environment or in .env.)")
        return None, None, run_dir, job
    if job.credits is None or approve_credits != job.credits:
        raise SystemExit(f"--approve-credits {approve_credits} does not match the estimated cost of {job.credits}")
    save_payload(run_dir, {"engine": eid, "params": params}, sha)
    out_run = client.run(eid, params, approved=True, out_dir=run_dir)
    print(f"\njob {out_run['job_id']} completed.")
    return out_run["result"], out_run, run_dir, job


def solve_graph_v1(calib, out, approve_credits=None):
    res, run, run_dir, job = run_engine("graph_v1", calib, out, approve_credits)
    if res is None:
        return None
    inline = res.get("result")
    if isinstance(inline, str):
        inline = json.loads(inline)
    targets = _load(calib)
    parsed = parse_graph_v1(inline)
    rows, notes = graph_v1_achieved(parsed, targets)
    summary = score(rows)
    state = dict(engine="graph-v1", mode="emu", n_qubits=job.params["num_qubits"], credits=job.credits,
                 job_id=run["job_id"] if run else None, parsed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), notes=notes,
                 qubit_map=targets["meta"]["qubit_map"], coupling_map=parsed["coupling_map"],
                 bloch={str(q): v for q, v in parsed["bloch"].items()},
                 relationships={f"{a},{b}": v for (a, b), v in parsed["relationships"].items()},
                 measurements=dict(top=parsed["top"], dominant_bitstring=parsed["dominant"], edge_agreement_score=parsed["edge_agreement"],
                                   note="graph-v1 returns only the 20 most likely bitstrings; qubit 0 is the leftmost character; bit '0' is Z = +1"),
                 summary=summary)
    write_report(run_dir, rows, summary, state)
    print_summary("graph-v1", summary, notes)
    print(f"wrote {run_dir}/state.json and requested_vs_achieved.csv")
    return state


def solve_qdrive(calib, out, approve_credits=None):
    res, run, run_dir, job = run_engine("qdrive", calib, out, approve_credits)
    if res is None:
        return None
    path = os.path.join(run_dir, "circuit")
    if not os.path.exists(path):
        raise SystemExit(f"no circuit file in {run_dir}; raw result keys: {list(res)}")
    with open(path) as f:
        qasm = f.read()
    with open(os.path.join(run_dir, "circuit.qasm"), "w") as f:
        f.write(qasm)
    targets = _load(calib)
    rows, src = qdrive_achieved(qasm, targets, targets["meta"]["qubit_map"] | dict(n_qubits=targets["meta"]["n_qubits"]))
    summary = score(rows)
    state = dict(engine="qdrive-api-v1", credits=job.credits, job_id=run["job_id"] if run else None,
                 circuit="circuit.qasm", n_qubits=src.n, qubit_map=targets["meta"]["qubit_map"], summary=summary,
                 evaluation="exact Z moments of the returned circuit, simulated locally with Aer (qiskit-qasm3-import)")
    write_report(run_dir, rows, summary, state)
    print_summary("QDrive (circuit simulated locally)", summary)
    print(f"wrote {run_dir}/circuit.qasm, state.json and requested_vs_achieved.csv")
    return state


def finish_rounds(rp, base, targets, local_fill=False):
    """Assemble whatever the QDrive rounds have delivered under `base` (= <solve>/qdrive) and, when every component is there, write the same
    files solve_qdrive writes (circuit.qasm, state.json, requested_vs_achieved.*) so the show and the studio need nothing else.
    Returns the provenance dict. An incomplete assembly goes to circuit_partial.qasm and is scored against the targets for information only."""
    from src.quantum import qdrive_rounds as qr
    asm = qr.assemble(rp, base, local_fill=local_fill)
    prov = qr.write_assembled(asm, rp, base)
    for p in asm["pieces"]:
        extra = f", TV {p['tv']} fidelity {p['fidelity']}" if p.get("tv") is not None else ""
        print(f"  component {p['component']} (qubits {p['qubits'][0]}-{p['qubits'][-1]}): {p['source']}, {p['steps_done']}/{p['steps_total']} rounds{extra}")
    if asm["text"] is None:
        print("nothing finished yet: no circuit written")
        return prov
    layout = targets["meta"]["qubit_map"] | dict(n_qubits=targets["meta"]["n_qubits"])
    rows, src = qdrive_achieved(asm["text"], targets, layout)
    summary = score(rows)
    name = "circuit.qasm" if asm["complete"] else "circuit_partial.qasm"
    state = dict(engine="qdrive-api-v1 (rounds)", circuit=name, complete=asm["complete"], n_qubits=src.n, qubit_map=targets["meta"]["qubit_map"],
                 summary=summary, pieces=asm["pieces"], plan_sha=rp.sha,
                 jobs=[j for p in asm["pieces"] for j in p.get("jobs", [])], credits=sum(len(p.get("jobs", [])) for p in asm["pieces"]),
                 evaluation="exact Z moments of the merged circuit, simulated locally with Aer (qiskit-qasm3-import)")
    if asm["complete"]:
        write_report(base, rows, summary, state)
    else:
        with open(os.path.join(base, "partial_state.json"), "w") as f:
            json.dump(dict(state, summary=summary), f, indent=1)
    print_summary("QDrive rounds (circuit simulated locally)" + ("" if asm["complete"] else " - PARTIAL, unfinished components in |0>"), summary)
    print(f"wrote {os.path.join(base, name)}" + (" and state.json, requested_vs_achieved.csv" if asm["complete"] else "") + f"; provenance: {prov['from_moth'] and 'Moth' or 'not pure Moth'}")
    return prov


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine", choices=["graph-v1", "qdrive"])
    ap.add_argument("--calib", default=DEFAULT_CALIB_DIR, help="directory with targets.json and the payload files")
    ap.add_argument("--out", default=DEFAULT_RUN_DIR)
    ap.add_argument("--approve-credits", type=int, default=None, help="must equal the estimated credits; omit to preview only")
    a = ap.parse_args(argv)
    (solve_graph_v1 if a.engine == "graph-v1" else solve_qdrive)(a.calib, a.out, a.approve_credits)


if __name__ == "__main__":
    main()
