"""Result cache keyed by payload hash (handoff §6.2-6.3: one engine call per scene state, never pay twice for the same payload).

A run directory holds payload.json, payload.sha256 and raw_result_<job>.json. If the hash on disk matches the payload about
to be sent and a raw result exists, that result is reused and no credits are spent.
"""
import glob
import json
import os
import shutil

from src.store.jsonfmt import dumps_compact

# files a run leaves behind for ONE payload; a different payload must never inherit them
RUN_FILES = ("raw_result_*.json", "job_*.json", "state.json", "requested_vs_achieved.*", "circuit", "circuit.qasm")


def archive_stale(run_dir, sha):
    """Move what an earlier, different payload left in run_dir into run_dir/superseded/<its hash>/ (nothing paid for is ever deleted).
    Without this a payload whose job then fails would find the OLD payload's raw result next to its own hash and be served it as a cache hit.
    Returns the archive folder, or None when there was nothing to move."""
    try:
        with open(os.path.join(run_dir, "payload.sha256")) as f:
            old = f.read().strip()
    except FileNotFoundError:
        old = ""
    if old == sha:
        return None
    files = sorted({p for pat in RUN_FILES for p in glob.glob(os.path.join(run_dir, pat))})
    if not files:
        return None
    dest = os.path.join(run_dir, "superseded", (old or "unknown")[:12])
    os.makedirs(dest, exist_ok=True)
    for p in files:
        shutil.move(p, os.path.join(dest, os.path.basename(p)))
    return dest


def save_payload(run_dir, payload, sha):
    os.makedirs(run_dir, exist_ok=True)
    archive_stale(run_dir, sha)
    with open(os.path.join(run_dir, "payload.json"), "w") as f:
        f.write(dumps_compact(payload) + "\n")
    with open(os.path.join(run_dir, "payload.sha256"), "w") as f:
        f.write(sha + "\n")


def cached_result(run_dir, sha):
    """The raw result envelope previously fetched for exactly this payload, or None."""
    try:
        with open(os.path.join(run_dir, "payload.sha256")) as f:
            if f.read().strip() != sha:
                return None
    except FileNotFoundError:
        return None
    files = sorted(glob.glob(os.path.join(run_dir, "raw_result_*.json")), key=os.path.getmtime)
    if not files:
        return None
    with open(files[-1]) as f:
        return dict(path=files[-1], result=json.load(f))
