"""Free probes: every request here is made to FAIL schema validation (it always carries shots=0), so no job is created and no credit
can be spent. 422 bodies list every violation, which shows which of our extra fields the schema itself accepts.
If any response is 2xx a job WAS created: it is logged to the ledger and the script stops.
Run: python qdrive_lab/probe_free.py
"""
import io
import json
import os
import struct
import sys
import zlib

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab                                              # noqa: E402

P = f"/engines/{lab.ENGINE}/process"
OUT = {}


def post(label, body, path=P, **kw):
    r = lab.raw_post(path, body, **kw)
    OUT[label] = dict(http=r.status_code, body=lab.short(r, 900))
    print(f"[{r.status_code}] {label}\n      {OUT[label]['body'][:420]}")
    if r.ok:
        lab._log(dict(label="UNEXPECTED JOB " + label, http=r.status_code, job_id=r.json().get("job_id")))
        raise SystemExit("a probe created a job; stopping")
    return r


def png_bytes(w=2, h=2):
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


T = [dict(qubits=[0, 1], expvals={"ZZ": 1.0})]
BAD = dict(shots=0)                                     # the deliberate violation

# ---- 1. which fields does the schema itself police?
post("schema: many violations at once", dict(params=dict(n_qubits=0, shots=0, tomography=3, seed=-1, sample="yes", targets=T)))
post("schema: unknown param key", dict(params=dict(n_qubits=2, targets=T, bogus_key=1, **BAD)))
post("schema: update_method bogus (schema or runtime?)", dict(params=dict(n_qubits=2, targets=T, update_method="nonsense", **BAD)))
post("schema: machine bogus", dict(params=dict(n_qubits=2, targets=T, machine="nonsense", **BAD)))
post("schema: targets as bare number list", dict(params=dict(n_qubits=2, targets=[1, 2], **BAD)))
post("schema: targets as string", dict(params=dict(n_qubits=2, targets="Z0=1", **BAD)))
post("schema: targets list of strings", dict(params=dict(n_qubits=2, targets=["a"], **BAD)))
post("schema: coupling_map string", dict(params=dict(n_qubits=2, targets=T, coupling_map="0-1", **BAD)))
post("schema: coupling_map bad edge", dict(params=dict(n_qubits=2, targets=T, coupling_map=[[0, 5]], **BAD)))
post("schema: ansatz string", dict(params=dict(n_qubits=2, targets=T, ansatz="OPENQASM 3;", **BAD)))
post("schema: n_qubits as float", dict(params=dict(n_qubits=2.5, targets=T, **BAD)))
post("schema: n_qubits huge (is there an upper cap?)", dict(params=dict(n_qubits=500, targets=T, **BAD)))
post("schema: no n_qubits and no initial_circuit", dict(params=dict(targets=T, **BAD)))

# ---- 2. input_files plumbing
post("input_files: unknown asset id", dict(params=dict(targets=T, **BAD), input_files={"initial_circuit": "00000000-0000-0000-0000-000000000000"}))
post("input_files: job reference to a missing job", dict(params=dict(targets=T, **BAD), input_files={"initial_circuit": "job:00000000-0000-0000-0000-000000000000/circuit"}))
post("input_files: unknown slot name", dict(params=dict(targets=T, n_qubits=2, **BAD), input_files={"circuit_in": "x"}))

# ---- 3. the route the official code samples use (multipart/form-data, /v1/generation/...)
FORM = "https://api.mothquantum.com/v1/generation/qdrive-api-v1/process"
r = requests.post(FORM, headers=lab.headers(), data={"_params": json.dumps(dict(n_qubits=2, targets=T, **BAD))}, timeout=60)
OUT["route: /v1/generation form _params"] = dict(http=r.status_code, body=lab.short(r, 600))
print(f"[{r.status_code}] route: /v1/generation form _params\n      {lab.short(r, 420)}")
if r.ok:
    lab._log(dict(label="UNEXPECTED JOB generation route", http=r.status_code, job_id=r.json().get("job_id"))); raise SystemExit("job created")
r = requests.post(FORM, headers=lab.headers(), data={"initial_circuit": "job:00000000-0000-0000-0000-000000000000/circuit", "_params": json.dumps(dict(tomography=2, **BAD))}, timeout=60)
OUT["route: /v1/generation form chain ref"] = dict(http=r.status_code, body=lab.short(r, 600))
print(f"[{r.status_code}] route: /v1/generation form chain ref\n      {lab.short(r, 420)}")
if r.ok:
    lab._log(dict(label="UNEXPECTED JOB generation chain", http=r.status_code, job_id=r.json().get("job_id"))); raise SystemExit("job created")

# ---- 4. assets: register / upload / complete / list / download / delete, plus type rules
print("\n--- assets ---")
data = png_bytes()
reg = lab.raw_post("/assets", dict(filename="lab_probe.png", content_type="image/png", size_bytes=len(data)))
OUT["asset: register png"] = dict(http=reg.status_code, body=lab.short(reg, 700)); print(f"[{reg.status_code}] register png: {lab.short(reg, 500)}")
aid = None
if reg.ok:
    j = reg.json(); aid = j["asset_id"]
    up = requests.put(j["upload"]["url"], data=data, headers=j["upload"].get("headers") or {}, timeout=60)
    OUT["asset: PUT"] = dict(http=up.status_code, body=up.text[:300]); print(f"[{up.status_code}] PUT bytes")
    done = lab.raw_post(f"/assets/{aid}/complete", {})
    OUT["asset: complete"] = dict(http=done.status_code, body=lab.short(done, 500)); print(f"[{done.status_code}] complete: {lab.short(done, 300)}")
    # try the uploaded PNG as QDrive's initial_circuit (text/plain slot): deliberately invalid via shots=0
    post("asset: PNG asset id as initial_circuit (type mismatch?)", dict(params=dict(targets=T, **BAD), input_files={"initial_circuit": aid}))
    dl = lab.raw_get(f"/assets/{aid}/download")
    OUT["asset: download"] = dict(http=dl.status_code, body=lab.short(dl, 300)); print(f"[{dl.status_code}] download link: {lab.short(dl, 200)}")
    de = requests.delete(f"{lab.API}/assets/{aid}", headers=lab.headers(), timeout=30)
    OUT["asset: delete"] = dict(http=de.status_code, body=de.text[:200]); print(f"[{de.status_code}] delete")

qasm = b'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n'
for ct in ("text/plain", "application/octet-stream", "application/json"):
    r = lab.raw_post("/assets", dict(filename="bell.qasm", content_type=ct, size_bytes=len(qasm)))
    OUT[f"asset: register {ct}"] = dict(http=r.status_code, body=lab.short(r, 500)); print(f"[{r.status_code}] register {ct}: {lab.short(r, 260)}")
    if r.ok:
        lab.raw_get("/assets")                              # leave a trace; clean up the pending asset
        requests.delete(f"{lab.API}/assets/{r.json()['asset_id']}", headers=lab.headers(), timeout=30)

r = lab.raw_get("/me/storage"); OUT["storage"] = dict(http=r.status_code, body=lab.short(r, 400)); print("storage:", lab.short(r, 300))
lab.save("probe_free.json", OUT)
print("\nsaved results/probe_free.json; jobs created by this script: 0")
