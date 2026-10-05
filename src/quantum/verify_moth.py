#!/usr/bin/env python3
"""Verify the relief circuit on Moth: the reference circuit goes to tomography-api-v2 and what comes back is scored against the exact moments.

    python -m src.quantum.verify_moth --labels runs/studio/labels.json --out runs/studio/solve/tomography            # preview: sends nothing
    python -m src.quantum.verify_moth --labels ... --out ... --approve-credits 1                                    # sends one job (1 credit)
    python -m src.quantum.verify_moth --labels ... --out /tmp/rehearsal --local                                      # no network: an IDEAL local tomography

This is the Moth counterpart of the Aer run's own check (aer_relief.py). The Aer run executes the whole performance circuit and measures
shots; tomography-api-v2 takes the preparation part (facet and polarity qubits, no lamp or observe registers) as OpenQASM 2 and returns
single- and two-qubit moments, so it can confirm that the state is the one we think it is but cannot drive frames. Scored: the Bloch
vector of every qubit (the polarity qubit's X is the interference visibility V_p) and every (polarity, facet) and facet-facet correlator, against the
exact reference state, and, when an Aer run of the same circuit exists, the Moth visibilities beside the Aer ones.

The raw result is saved before anything is parsed, and a payload that already has a result on disk is never sent again. `--local` returns
the exact moments in the shape the parser expects: it rehearses the pipeline and proves nothing about Moth (result fields of this engine are "not
documented yet", so the parser is defensive and the real shape is UNVERIFIED); it never writes score.json, which is what the show prints as a
Moth result.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from src.quantum import aer_relief as ar
from src.quantum import moth_engines as me
from src.quantum.moth_client import MothClient
from src.store.cache import cached_result, save_payload
from src.targets.payloads import payload_sha256

ENGINE = "tomography-api-v2"


class LocalTomographyClient:
    """An IDEAL local stand-in: answers a tomography request with the exact moments of the reference state, in the shape parse_tomography reads.
    Rehearses the pipeline with no network and no credits; says nothing about the real engine."""
    is_local = True

    def __init__(self, spec):
        self.spec = spec

    def prepare(self, engine_id, params, input_files=None):
        return MothClient().prepare(engine_id, params, input_files)

    def run(self, engine_id, params, *, approved=False, out_dir=None, input_files=None):
        sp, ex = self.spec, me.pauli_moments(self.spec)
        bloch = {str(q): {c: ex[f"{c}{q}"] for c in "XYZ"} for q in params.get("qubit_list", range(sp.n_sim))}
        rel = {f"{a},{b}": {c1 + c2: ex[f"{c1}{a}{c2}{b}"] for c1 in "XYZ" for c2 in "XYZ"} for a, b in params.get("qubit_pair_list", []) if f"X{a}X{b}" in ex}
        res = {"outputs": None, "result": {"output": {"tomography": {"bloch": bloch, "relationships": rel}}}}
        job_id = f"local-{hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]}"
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"job_{job_id}.json"), "w") as f:
                json.dump({"job_id": job_id, "engine_id": "local-ideal", "estimated_credits": 0}, f)
            with open(os.path.join(out_dir, f"raw_result_{job_id}.json"), "w") as f:
                json.dump(res, f)
        return dict(job_id=job_id, result=res, inline=res["result"], files={})


def plan(labels, flags=None, shots=4096):
    """(spec, payload params, sha256, PreparedJob) for the tomography request of this scene: no key, no network."""
    flags = dict(flags or {})
    _, _, spec = ar.build_spec(labels, flags.get("kappa", 1.0), flags.get("n_dirs"), flags.get("entangle", 0.8), flags.get("observations"))
    params = me.tomography_payload(spec, shots=shots)["params"]
    sha = payload_sha256({"engine": ENGINE, "params": params})
    return spec, params, sha, MothClient().prepare(ENGINE, params)


def read_state(folder):
    try:
        with open(os.path.join(folder, "state.json")) as f:
            d = json.load(f)
        return d if d.get("kind") == "relief-tomography" else None
    except (OSError, ValueError):
        return None


def compare_with_aer(parsed_bloch, spec, aer_dir):
    """Per panel: the polarity <X> Moth measured next to the Aer witness run's and the exact value. None when there is no Aer run of this circuit."""
    meta = ar.read_state(aer_dir) if aer_dir else None
    if not meta or meta.get("spec") != ar.spec_fingerprint(spec):
        return None
    rows = []
    for r in meta["witness"]["panels"]:
        got = parsed_bloch.get(spec.F + r["panel"])
        rows.append(dict(panel=r["panel"], moth=None if got is None else float(got[0]), aer=r["visibility"], aer_se=r["visibility_se"], exact=r["visibility_exact"]))
    have = [abs(r["moth"] - r["aer"]) for r in rows if r["moth"] is not None]
    return dict(rows=rows, max_abs_difference=max(have) if have else None)


def run(labels, labels_path, out, flags=None, approve_credits=None, client=None, local=False, aer_dir=None, shots=4096, log=print):
    """Send (or reuse, or rehearse locally) the tomography of the scene's reference circuit, score it and write state.json under `out`.
    Returns state.json's dict, or None for a preview. Nothing is sent unless approve_credits equals the cost."""
    spec, params, sha, job = plan(labels, flags, shots)
    log(job.describe(max_ops=3)[:1500])
    if job.problems():
        raise SystemExit("this request cannot be sent: " + "; ".join(job.problems()))
    cached = cached_result(out, sha)
    res = None
    if cached:
        log(f"\nresult for this exact payload is already on disk ({cached['path']}): reusing it, 0 credits.")
        raw, job_id, credits, is_local = cached["result"], os.path.basename(cached["path"])[len("raw_result_"):-5], 0, False
    elif local:
        client = LocalTomographyClient(spec)
        save_payload(out, {"engine": ENGINE, "params": params}, sha)
        res = client.run(ENGINE, params, out_dir=out)
        raw, job_id, credits, is_local = res["result"], res["job_id"], 0, True
        log("\nLOCAL rehearsal: the exact moments of the reference state in the engine's shape. Nothing was sent, and this says nothing about Moth.")
    else:
        if approve_credits is None:
            log(f"\n(preview only: nothing was sent. Add --approve-credits {job.credits} to send; MOTH_API_KEY must be set in the environment or in .env.)")
            return None
        if job.credits is None or approve_credits != job.credits:
            raise SystemExit(f"--approve-credits {approve_credits} does not match the estimated cost of {job.credits}")
        save_payload(out, {"engine": ENGINE, "params": params}, sha)
        res = (client or MothClient()).run(ENGINE, params, approved=True, out_dir=out)
        raw, job_id, credits, is_local = res["result"], res["job_id"], job.credits, False
        log(f"\njob {job_id} completed.")
    parsed = me.parse_tomography(raw)
    sc = me.score_tomography(parsed, spec)
    cmp_aer = compare_with_aer({q: v for q, v in parsed[0].items()}, spec, aer_dir)
    flat = {k: v for k, v in sc.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    vis = [v for v in sc["visibility"] if v["measured"] is not None]
    if vis:
        flat["visibility_measured_mean"] = float(np.mean([v["measured"] for v in vis]))
        flat["visibility_exact_mean"] = float(np.mean([v["exact"] for v in vis]))
    state = dict(kind="relief-tomography", engine=ENGINE, sha256=sha, job_id=job_id, credits=credits, local=is_local, n_qubits=spec.n_sim, shots=shots,
                 flags=dict(flags or {}), labels_sha=hashlib.sha256(open(labels_path, "rb").read()).hexdigest() if labels_path and os.path.exists(labels_path) else None,
                 spec=ar.spec_fingerprint(spec), score=sc, aer_comparison=cmp_aer, parsed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 note="reference circuit (facet and polarity qubits only; no lamp or observe registers), scored against the exact reference state"
                      + ("; LOCAL ideal stand-in, not a Moth result" if is_local else ""))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "state.json"), "w") as f:
        json.dump(state, f, indent=1, default=float)
    if not is_local:                                                         # score.json is what the show prints as a Moth result
        with open(os.path.join(out, "score.json"), "w") as f:
            json.dump(flat, f, indent=1)
    for v in sc["visibility"]:
        log(f"  panel {v['panel']}: <X> measured {v['measured']}  exact {v['exact']:.3f}  isolated product law {v['isolated']:.3f}")
    log("  " + ", ".join(f"{k} {v:.3g}" for k, v in flat.items()))
    if cmp_aer and cmp_aer["max_abs_difference"] is not None:
        log(f"  against the Aer witness run: largest visibility difference {cmp_aer['max_abs_difference']:.3f}")
    log(f"wrote {os.path.join(out, 'state.json')}")
    return state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="labels.json from the pen tool (default: the built-in bay window)")
    ap.add_argument("--out", required=True, help="folder for the job record, raw result and state.json (the studio uses <project>/solve/tomography)")
    ap.add_argument("--approve-credits", type=int, default=None, help="must equal the estimated cost; omit to preview only")
    ap.add_argument("--local", action="store_true", help="rehearse with an ideal local stand-in: no network, no credits, never a Moth result")
    ap.add_argument("--aer-run", help="folder of an Aer run (aer_relief) of the same circuit to compare visibilities with")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--kappa", type=float), ap.add_argument("--n-dirs", type=int), ap.add_argument("--entangle", type=float)
    ap.add_argument("--contrast", type=float), ap.add_argument("--observations")
    a = ap.parse_args(argv)
    from src.capture.labels import bay_window_labels, load_labels
    labels = load_labels(a.labels) if a.labels else bay_window_labels()
    flags = {k: v for k, v in dict(kappa=a.kappa, n_dirs=a.n_dirs, entangle=a.entangle, contrast=a.contrast, observations=a.observations).items() if v is not None}
    run(labels, a.labels, a.out, flags, a.approve_credits, local=a.local, aer_dir=a.aer_run, shots=a.shots)


if __name__ == "__main__":
    main()
