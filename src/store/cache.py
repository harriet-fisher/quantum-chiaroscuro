"""Result cache keyed by payload hash (handoff §6.2-6.3: one engine call per scene state, never pay twice for the same payload).

A run directory holds payload.json, payload.sha256 and raw_result_<job>.json. If the hash on disk matches the payload about
to be sent and a raw result exists, that result is reused and no credits are spent.
"""
import glob
import json
import os

from src.store.jsonfmt import dumps_compact


def save_payload(run_dir, payload, sha):
    os.makedirs(run_dir, exist_ok=True)
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
