"""Shared helpers for the temporary QDrive lab. Safe to delete the whole qdrive_lab/ folder; nothing else imports it.

- raw_post / raw_get: talk to the API WITHOUT the client's own pre-checks, so we see what the server really accepts.
- Ledger: every POST that created a job is appended to results/ledger.jsonl; submit_job refuses past BUDGET credits.
- Scoring: read Pauli expectations of a returned QASM3 circuit with qiskit (little-endian: qubit 0 = rightmost Pauli letter).
"""
import json
import os
import sys
import time

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from src.quantum.moth_client import API, MothClient          # noqa: E402

RESULTS = os.path.join(HERE, "results")
LEDGER = os.path.join(RESULTS, "ledger.jsonl")
BUDGET = int(os.environ.get("QDRIVE_LAB_BUDGET", "15"))
ENGINE = "qdrive-api-v1"
os.makedirs(RESULTS, exist_ok=True)
_client = MothClient()


def headers():
    return _client._headers()


def raw_post(path, body, timeout=60):
    return requests.post(f"{API}{path}", headers=headers(), json=body, timeout=(10, timeout))


def raw_get(path, timeout=60):
    return requests.get(f"{API}{path}", headers=headers(), timeout=(10, timeout))


def short(r, n=600):
    try:
        d = r.json()
    except ValueError:
        return r.text[:n]
    d.pop("$schema", None)
    return json.dumps(d)[:n]


# ---------------------------------------------------------------- ledger
def ledger():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]


def jobs_spent():
    return sum(1 for e in ledger() if e.get("job_id"))


def _log(entry):
    entry["t"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def submit_job(label, params, input_files=None, wait_s=200):
    """POST a QDrive job (raw), wait, download the circuit. Returns dict(label, http, job_id, status, error, circuit_path, secs).
    Refuses once BUDGET jobs have been created (a rejected 4xx POST creates no job and is logged but not counted)."""
    if jobs_spent() >= BUDGET:
        raise RuntimeError(f"lab budget of {BUDGET} jobs reached; not submitting '{label}'")
    body = {"params": params, **({"input_files": input_files} if input_files else {})}
    t0 = time.time()
    r = raw_post(f"/engines/{ENGINE}/process", body)
    out = dict(label=label, http=r.status_code, body_sent=body if len(json.dumps(body)) < 4000 else "(large)")
    if not r.ok:
        out["error_body"] = short(r)
        _log(dict(out, job_id=None))
        return out
    jid = r.json().get("job_id")
    out["job_id"] = jid
    _log(dict(label=label, http=r.status_code, job_id=jid))
    while True:
        st = raw_get(f"/jobs/{jid}/status").json()
        if st.get("status") in ("completed", "failed", "cancelled", "canceled"):
            break
        if time.time() - t0 > wait_s:
            out["status"] = "still running at wait limit"
            return out
        time.sleep(1.5)
    out["status"], out["secs"] = st["status"], round(time.time() - t0, 1)
    out["progress"] = st.get("progress")
    out["status_warnings"] = st.get("warnings")
    if st["status"] != "completed":
        out["error"] = st.get("error")
        return out
    res = raw_get(f"/jobs/{jid}/result").json()
    out["outputs"] = [{k: v for k, v in o.items() if k != "url"} for o in (res.get("outputs") or [])]
    out["result_inline"] = res.get("result")
    d = os.path.join(RESULTS, f"job_{jid}")
    os.makedirs(d, exist_ok=True)
    for o in res.get("outputs") or []:
        data = requests.get(o["url"], timeout=(10, 120)).content
        p = os.path.join(d, os.path.basename(str(o["slot"])) or "output")
        open(p, "wb").write(data)
        out.setdefault("files", {})[o["slot"]] = p
    out["circuit_path"] = out.get("files", {}).get("circuit")
    return out


# ---------------------------------------------------------------- scoring
def load_state(qasm_text):
    from qiskit import qasm3
    from qiskit.quantum_info import Statevector
    qc = qasm3.loads(qasm_text).remove_final_measurements(inplace=False)
    return Statevector(qc), qc


def pauli_expval(sv, n, word, qubits):
    """<P> where letter word[i] acts on qubits[i] (qubit 0 is the rightmost character of qiskit's Pauli string)."""
    from qiskit.quantum_info import Pauli
    s = ["I"] * n
    for letter, q in zip(word, qubits):
        s[n - 1 - q] = letter
    return float(sv.expectation_value(Pauli("".join(s))).real)


def score(circuit_path, targets):
    """targets: list of dicts {qubits, word, value}. Returns dict(rows, rms, n_qubits, ops)."""
    sv, qc = load_state(open(circuit_path).read())
    rows = [dict(qubits=t["qubits"], word=t["word"], want=t["value"],
                 got=round(pauli_expval(sv, qc.num_qubits, t["word"], t["qubits"]), 4)) for t in targets]
    err = [r["got"] - r["want"] for r in rows]
    return dict(rows=rows, rms=round(float(np.sqrt(np.mean(np.square(err)))) if err else 0.0, 4),
                n_qubits=qc.num_qubits, ops=dict(qc.count_ops()))


def save(name, obj):
    p = os.path.join(RESULTS, name)
    json.dump(obj, open(p, "w"), indent=1, default=str)
    return p
