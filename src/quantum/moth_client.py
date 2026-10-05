"""Client for the Moth Atlas API (handoff §4.2). Importing this module never touches the network.

Flow [VERIFIED from the docs pages]:
    POST {API}/engines/<engine-id>/process   {"params": {...}}  -> {"job_id": ...}
    GET  {API}/jobs/<job_id>/status          poll until completed | failed | cancelled   (error text under "error")
    GET  {API}/jobs/<job_id>/result          -> {"outputs": [{"slot": ..., "url": ...}, ...]}

Safety, because every submit spends credits:
  * the key comes from the MOTH_API_KEY environment variable, else from MOTH_API_KEY in <project>/.env; it is read lazily
    (never at import), only that one variable is extracted, and it is never printed or logged;
  * prepare() builds and shows the exact request without a key or a network call;
  * submit() raises NotApproved unless approved=True is passed explicitly;
  * POSTs are never retried automatically (a retry could be billed twice); GETs may retry on connection errors;
  * the job id is written to disk straight after submit so a crash while polling cannot lose a paid job;
  * output URLs are presigned, so they are downloaded WITHOUT the Authorization header.

CLI (preview never needs a key; send needs MOTH_API_KEY and a typed credit approval):
    python -m src.quantum.moth_client preview PAYLOAD.json --engine graph-v1
    python -m src.quantum.moth_client send    PAYLOAD.json --engine graph-v1 --approve-credits 5 --out runs/solve
"""
import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import requests

from src.store.jsonfmt import dumps_compact

API = "https://api.mothquantum.com/api/v1"
ENV_KEY = "MOTH_API_KEY"
DOTENV = Path(__file__).resolve().parents[2] / ".env"
TERMINAL = ("completed", "failed", "cancelled")

# Credits per run as shown on the engine pages [VERIFIED, handoff §5; the rest read from docs.mothquantum.com/docs/engines/<id> on 2 Oct 2026].
# Whether cost scales with qubits or shots is not documented [UNKNOWN]. Engines not listed have no known price and cannot be sent from the CLI.
CREDITS = {"graph-v1": 5, "qdrive-api-v1": 1, "tomography-api-v2": 1, "otoc-echo-v1": 1, "qpixl-v1": 1, "entanglement-shader-v1": 1,
           "blur-core-v1": 1, "coin-toss-v1": 2, "labyrinth-v1": 5}

# Engine limits and accepted params [VERIFIED from docs.mothquantum.com engine pages, 2 Oct 2026]. The API validates params
# with additionalProperties: false (HTTP 422 on an unknown field), so unknown keys are caught here first.
ENGINES = {
    "graph-v1": dict(qubits_key="num_qubits", min_qubits=2, max_qubits=20, timeout_s=300,
                     params={"mode", "num_qubits", "shots", "coupling_map", "operations", "seed", "backend_name", "qpu_instance", "qpu_token"}),
    "qdrive-api-v1": dict(qubits_key="n_qubits", min_qubits=None, max_qubits=None, timeout_s=120,
                          params={"machine", "shots", "sample", "tomography", "update_method", "n_qubits", "seed", "ansatz", "coupling_map", "targets"}),
    # The engines below were added from their docs pages; param names are exactly the documented ones. Their result fields are mostly
    # "not documented yet" by the engine authors, so results are saved raw and parsed defensively (src/quantum/moth_engines.py).
    "tomography-api-v2": dict(qubits_key=None, min_qubits=None, max_qubits=None, timeout_s=10800,
                              params={"circuit_qasm", "backend_name", "provider_name", "shots", "qubit_list", "qubit_pair_list", "single_tomography",
                                      "double_tomography", "mutual_information", "classical_mutual_information"}),
    "otoc-echo-v1": dict(qubits_key="n_sites", min_qubits=2, max_qubits=24, timeout_s=7200,           # 24 is the Aer cap; IBM backends need the user's token
                         params={"lattice", "n_sites", "width", "height", "theta_x", "theta_z", "theta_zz", "depth", "kick", "kick_site", "machine", "shots",
                                 "allow_high_shots", "exact", "via", "fractional_gates", "disorder", "twirls", "include_taps", "include_z", "min_tap_level",
                                 "ref_floor", "seed"}),
    "qpixl-v1": dict(qubits_key=None, min_qubits=None, max_qubits=None, timeout_s=18000,
                     params={"values", "shots", "mode", "machine", "backend_name", "discretize", "dynamic_range", "allow_high_shots"}),
    "entanglement-shader-v1": dict(qubits_key=None, min_qubits=None, max_qubits=None, timeout_s=18000,
                                   params={"reflectance", "absorption", "layers", "incoming_rays", "interaction", "resolution", "style"}),
    "blur-core-v1": dict(qubits_key=None, min_qubits=None, max_qubits=None, timeout_s=300,
                         params={"values", "strength", "style", "reach", "axes", "shots", "max_qubits"}),
}
# More than this many QDrive targets in one job has never run to completion: two jobs of 201 targets on 19 qubits both died with engine_timeout
# at target 56 and 61 (2 Oct 2026 status reads), the longer after 4 minutes against a documented 120 s limit.
QDRIVE_TARGETS_WARN, QDRIVE_TARGETS_BLOCK = 30, 100
RETRY_STATUS = (429, 502, 503)      # GET only: 429 rate limit (300/min/key), 502 upstream, 503 runtime unavailable


def _dotenv_key(path):
    """MOTH_API_KEY from a .env file, or None. Understands KEY=value, export KEY=value, quotes and # comments."""
    path = Path(path)
    if not path.is_file():
        return None
    if path.stat().st_mode & 0o077:
        warnings.warn(f"{path.name} is readable by other users; run: chmod 600 {path}", stacklevel=3)
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() != ENV_KEY:
            continue
        value = value.strip()
        if value[:1] in "\"'" and value[-1:] == value[:1] and len(value) >= 2:
            return value[1:-1] or None
        return value.split(" #")[0].strip() or None
    return None


def load_key():
    """(key, source) with source 'environment' or '.env'; (None, None) if neither has it. Callers must never print the key."""
    key = os.environ.get(ENV_KEY, "").strip()
    if key:
        return key, "environment"
    key = _dotenv_key(DOTENV)
    return (key, ".env") if key else (None, None)


class MothError(RuntimeError):
    pass


class MissingKey(MothError):
    pass


class NotApproved(MothError):
    pass


class InvalidPayload(MothError):
    pass


def _is_uuid(v):
    import uuid
    try:
        uuid.UUID(str(v))
        return True
    except ValueError:
        return False


def output_asset_id(result, slot="circuit"):
    """The asset UUID of a job's output slot, from the raw result envelope. This (not 'job:<id>/<slot>') is what a chained QDrive job's
    initial_circuit must be [VERIFIED, lab trials T06/T09]. None when the envelope has no such output."""
    for o in (result or {}).get("outputs") or []:
        if o.get("slot") == slot and o.get("output_asset_id"):
            return o["output_asset_id"]
    return None


def estimate_credits(engine_id):
    """Credits for one run of engine_id, or None when the price is not known."""
    return CREDITS.get(engine_id)


@dataclass
class PreparedJob:
    engine_id: str
    params: dict
    input_files: dict = None          # {slot: asset_id}; assets are uploaded first (https://docs.mothquantum.com/docs/assets)

    @property
    def url(self):
        return f"{API}/engines/{self.engine_id}/process"

    @property
    def body(self):
        return {"params": self.params, **({"input_files": self.input_files} if self.input_files else {})}

    @property
    def credits(self):
        return estimate_credits(self.engine_id)

    def problems(self):
        """Reasons this payload would be rejected, found locally. Empty list = nothing known to be wrong."""
        spec, out = ENGINES.get(self.engine_id), []
        if spec is None:
            return out
        unknown = sorted(set(self.params) - spec["params"])
        if unknown:
            out.append(f"unknown params {unknown} (the API rejects them with 422)")
        n = self.params.get(spec["qubits_key"])
        if n is not None and spec["max_qubits"] is not None and not spec["min_qubits"] <= n <= spec["max_qubits"]:
            out.append(f"{spec['qubits_key']}={n} is outside {self.engine_id}'s documented range "
                       f"{spec['min_qubits']}..{spec['max_qubits']}")
        if self.params.get("mode") == "qpu":
            out.append("mode 'qpu' needs IBM credentials and is not supported by this client")
        if self.params.get("qpu_token") or self.params.get("qpu_instance"):
            out.append("this client never sends IBM credentials; run hardware jobs from the Moth dashboard with your own token")
        if self.engine_id in ("otoc-echo-v1",) and self.params.get("machine", "aer") != "aer":
            out.append("otoc-echo-v1: only machine 'aer' is supported by this client (IBM backends need your own token)")
        if self.engine_id == "qdrive-api-v1" and self.input_files:
            bad = [k for k, v in self.input_files.items() if k != "initial_circuit" or not _is_uuid(v)]
            if bad:
                out.append("QDrive input_files may only hold initial_circuit, as an ASSET UUID (a 'job:<id>/circuit' reference is rejected with 422)")
            if self.params.get("n_qubits") is not None:
                out.append("n_qubits must be left out when initial_circuit is given (the circuit carries its width)")
        if self.engine_id == "qdrive-api-v1" and len(self.params.get("targets") or []) > QDRIVE_TARGETS_BLOCK:
            out.append(f"{len(self.params['targets'])} QDrive targets in one job: the two jobs of 201 both died with engine_timeout; keep a job under "
                       f"{QDRIVE_TARGETS_BLOCK} (ideally {QDRIVE_TARGETS_WARN}) and chain with initial_circuit")
        return out

    def warnings(self):
        out = []
        if self.engine_id == "qdrive-api-v1" and QDRIVE_TARGETS_WARN < len(self.params.get("targets") or []) <= QDRIVE_TARGETS_BLOCK:
            out.append(f"{len(self.params['targets'])} QDrive targets: the only jobs that ran at this size timed out; expect about 4 s per target at 19 qubits")
        return out

    def describe(self, max_ops=None):
        """The request exactly as it would be sent, with the key redacted; optionally elide long lists for display."""
        shown = dict(self.params)
        if max_ops is not None:
            for name, noun in (("operations", "operations"), ("coupling_map", "edges")):
                items = shown.get(name)
                if isinstance(items, list) and len(items) > max_ops:
                    shown[name] = items[:max_ops] + [f"... {len(items) - max_ops} more {noun}"]
        body = {"params": shown, **({"input_files": self.input_files} if self.input_files else {})}
        source = load_key()[1]
        key = f"found in {source}" if source else "NOT set"
        cost = "unknown (not listed in CREDITS)" if self.credits is None else f"{self.credits} credits per run"
        issues = "".join(f"\nBLOCKED: {p}" for p in self.problems()) + "".join(f"\nWARNING: {w}" for w in self.warnings())
        return (f"POST {self.url}\n"
                f"Authorization: Bearer <{ENV_KEY}, currently {key}>\n"
                f"Content-Type: application/json\n\n"
                f"{dumps_compact(body)}\n\n"
                f"estimated cost: {cost}{issues}")


class MothClient:
    def __init__(self, api_key=None, session=None, poll_interval=2.0, timeout=900.0, retry_sleep=1.0):
        self.retry_sleep = retry_sleep
        self._api_key = api_key
        self._http = session or requests.Session()
        self.poll_interval = poll_interval
        self.timeout = timeout

    # -- request building (no key, no network)
    def prepare(self, engine_id, params, input_files=None):
        return PreparedJob(engine_id, params, input_files)

    def _headers(self):
        key = (self._api_key or load_key()[0] or "").strip()
        if not key:
            raise MissingKey(f"set {ENV_KEY} in the environment or in {DOTENV} (Atlas > API keys)")
        return {"Authorization": f"Bearer {key}"}

    def _check(self, r, what):
        if not r.ok:
            detail = r.text[:500]
            try:                                                   # RFC 7807 problem+json; 422 lists every violation
                d = r.json()
                errs = "; ".join(f"{'.'.join(map(str, e.get('location', [])))}: {e.get('message')}" for e in d.get("errors", []) if isinstance(e, dict))
                detail = " | ".join(x for x in (d.get("title"), d.get("detail"), errs) if x) or detail
            except (ValueError, AttributeError):
                pass
            raise MothError(f"{what}: HTTP {r.status_code}: {detail}")
        try:
            return r.json()
        except ValueError:
            raise MothError(f"{what}: response was not JSON: {r.text[:200]}")

    # -- network calls
    def submit(self, job, *, approved=False):
        """POST the job. Never retried. Returns the job id."""
        if not approved:
            raise NotApproved(f"refusing to submit {job.engine_id}: this costs {job.credits} credits. "
                              f"Show the payload to the user and pass approved=True only after they agree.")
        if job.problems():
            raise InvalidPayload(f"refusing to submit {job.engine_id}: " + "; ".join(job.problems()))
        r = self._http.post(job.url, headers=self._headers(), json=job.body, timeout=(10, 60))
        if r.status_code == 503:
            raise MothError(f"submit {job.engine_id}: HTTP 503, runtime unavailable. Per the docs the job was NOT recorded, so "
                            f"nothing was charged and it is safe to submit again by hand.")
        data = self._check(r, f"submit {job.engine_id}")
        if "job_id" not in data:
            raise MothError(f"submit {job.engine_id}: no job_id in response (keys: {sorted(data)})")
        return data["job_id"]

    def _get(self, path, what, tries=4):
        """GET with backoff on connection errors and 429/502/503. Only GETs are ever retried (a POST retry could bill twice)."""
        for attempt in range(tries):
            last = attempt == tries - 1
            try:
                r = self._http.get(f"{API}{path}", headers=self._headers(), timeout=(10, 60))
            except requests.ConnectionError:
                if last:
                    raise
            else:
                if r.status_code not in RETRY_STATUS or last:
                    return self._check(r, what)
            time.sleep(self.retry_sleep * 2 ** attempt)

    def status(self, job_id):
        return self._get(f"/jobs/{job_id}/status", f"status {job_id}")

    def result(self, job_id):
        return self._get(f"/jobs/{job_id}/result", f"result {job_id}")

    def wait(self, job_id):
        t0 = time.time()
        while True:
            st = self.status(job_id)
            if st.get("status") in TERMINAL:
                if st["status"] != "completed":
                    err = st.get("error")
                    if isinstance(err, dict):
                        err = f"{err.get('type')}: {err.get('message')} (retryable: {err.get('retryable')})"
                    raise MothError(f"job {job_id} {st['status']}: {err}. Whether a failed job is charged is not documented; "
                                    f"check the credit balance before resubmitting.")
                return st
            if time.time() - t0 > self.timeout:
                raise TimeoutError(f"job {job_id} still '{st.get('status')}' after {self.timeout:.0f}s; it may still run "
                                   f"(and bill) server-side, poll it later with status('{job_id}')")
            time.sleep(self.poll_interval)

    def download_outputs(self, result, out_dir):
        """Save every file output under out_dir/<slot>; returns {slot: path}. `outputs` is null for inline-result engines.
        Presigned URLs expire (expires_at), so this is called straight after the result is fetched. No Authorization header."""
        os.makedirs(out_dir, exist_ok=True)
        saved = {}
        for o in result.get("outputs") or []:
            r = requests.get(o["url"], timeout=(10, 120))
            if not r.ok:
                raise MothError(f"download {o.get('slot')}: HTTP {r.status_code} (url expires at {o.get('expires_at')})")
            path = os.path.join(out_dir, os.path.basename(str(o["slot"])) or "output")
            with open(path, "wb") as f:
                f.write(r.content)
            saved[o["slot"]] = path
        return saved

    def run(self, engine_id, params, *, approved=False, out_dir=None, input_files=None):
        """prepare -> submit -> wait -> result (-> download).

        Result envelope [VERIFIED]: file engines return {"outputs": [{"slot", "url", ...}], "result": null}; inline engines
        (graph-v1) return {"outputs": null, "result": {...}}. Returns {"job_id", "result" (raw envelope), "inline", "files"}.
        With out_dir, the job id is written before polling and the raw envelope before anything is parsed, so nothing
        that was paid for can be lost to a later bug.
        """
        job = self.prepare(engine_id, params, input_files)
        job_id = self.submit(job, approved=approved)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"job_{job_id}.json"), "w") as f:
                json.dump({"job_id": job_id, "engine_id": engine_id, "estimated_credits": job.credits,
                           "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=2)
        self.wait(job_id)
        res = self.result(job_id)
        if out_dir:
            with open(os.path.join(out_dir, f"raw_result_{job_id}.json"), "w") as f:
                json.dump(res, f, indent=1)
        files = self.download_outputs(res, out_dir) if out_dir else {}
        inline = res.get("result")
        if isinstance(inline, str):                                 # "<engine-specific JSON>" may arrive as a JSON string
            try:
                inline = json.loads(inline)
            except ValueError:
                pass
        return {"job_id": job_id, "result": res, "inline": inline, "files": files}


def _load_params(path):
    with open(path) as f:
        body = json.load(f)
    return body["params"] if set(body) == {"params"} else body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preview", "send"):
        p = sub.add_parser(name)
        p.add_argument("payload", help="JSON file: {'params': {...}} or the bare params dict")
        p.add_argument("--engine", required=True)
        p.add_argument("--max-ops", type=int, default=12, help="preview: elide operation lists longer than this")
    sub.choices["send"].add_argument("--approve-credits", type=int, required=True,
                                     help="must equal the estimated credit cost; this is the confirmation")
    sub.choices["send"].add_argument("--out", default=None, help="directory for the job record and downloaded outputs")
    args = ap.parse_args(argv)

    client = MothClient()
    job = client.prepare(args.engine, _load_params(args.payload))
    print(job.describe(max_ops=args.max_ops))
    if args.cmd == "preview":
        print("\n(nothing was sent)")
        return
    if job.credits is None:
        raise SystemExit(f"price of {args.engine} is unknown; add it to CREDITS before sending")
    if args.approve_credits != job.credits:
        raise SystemExit(f"--approve-credits {args.approve_credits} does not match the estimated cost of {job.credits}")
    out = client.run(args.engine, job.params, approved=True, out_dir=args.out)
    print(f"\njob {out['job_id']} completed; files: {list(out['files'])}; inline result: {out['inline'] is not None}")


if __name__ == "__main__":
    main()
