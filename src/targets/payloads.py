"""Build engine payloads from a target table. Building a payload sends nothing.

graph-v1 request shape follows the documented example (handoff §5.1):
    {"num_qubits", "coupling_map", "operations": [{"type": "bloch", ...}, {"type": "relationship", ...}], "shots", "mode"}
wrapped as {"params": {...}} per the generic submit envelope (§4.2). [UNKNOWN] whether graph-v1 expects the wrapper or
the flat body shown in its docs: confirm with the dashboard's 'Try the API call' before the first send.

QDrive (qdrive-api-v1): see qdrive_payload() and QDRIVE_ASSUMPTIONS. The entry shapes are documented; their meaning is only
partly documented, so the payload is a stated bet, and its result is checked locally against the targets.
"""
import hashlib
import json
import os

import numpy as np

from src.capture.labels import DEFAULT_CALIB_DIR


def _round(v, nd=4):
    return round(max(-1.0, min(1.0, float(v))), nd) + 0.0   # + 0.0 turns -0.0 into 0.0


def graph_v1_payload(bloch, relationships, n_qubits, shots=1024, mode="emu", order="strongest_first", drop_below=0.0):
    """bloch: [{"qubit", "Z"}]; relationships: [{"qubits": [a, b], "ZZ"}] with a < b.

    Operations are applied in sequence by the engine, so order matters, and the first real run showed how much: sent as
    "all <Z> first, then <ZZ> strongest to weakest" (order="strongest_first", what the first run, runs/first_run, used), the engine returned
    ~zero correlation on every patch/lamp edge; later operations on shared qubits very likely overwrote earlier ones.
    order="weakest_first" puts the strongest targets last. drop_below omits targets with |value| below it (a ~0 target is
    already the default for a mixed qubit/pair and may only disturb earlier ones) and prunes coupling_map to the edges still
    used by a relationship, so dropped hub edges do not entangle anything.
    """
    sign = -1 if order == "strongest_first" else 1
    bl = [b for b in bloch if abs(b["Z"]) >= drop_below]
    rl = sorted((r for r in relationships if abs(r["ZZ"]) >= drop_below), key=lambda r: (sign * abs(r["ZZ"]), r["qubits"]))
    edges = sorted({tuple(r["qubits"]) for r in rl})
    ops = [{"type": "bloch", "qubit": b["qubit"], "paulis": {"Z": _round(b["Z"])}} for b in sorted(bl, key=lambda b: b["qubit"])]
    ops += [{"type": "relationship", "qubits": list(r["qubits"]), "paulis": {"ZZ": _round(r["ZZ"])}} for r in rl]
    return {"params": {"num_qubits": n_qubits, "coupling_map": [list(e) for e in edges],
                       "operations": ops, "shots": shots, "mode": mode}}


def payload_sha256(payload):
    """Cache key for a payload (store/cache.py: one engine call per scene state)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# What the QDrive docs say [VERIFIED, docs.mothquantum.com/docs/engines/qdrive-api-v1]: `targets` is an ordered list of
# {"qubits": [...], "expvals": number | [value, certainty] | expression}; a null entry calls update(). The Motte-model
# paper (arXiv:2605.22744) says the circuit is built layer by layer: targets name where Pauli expectations should go, and
# update() derives and applies the gate that moves the state toward them. These are the parts that are NOT documented:
QDRIVE_ASSUMPTIONS = [
    "VERIFIED by a failed first attempt (job f54945ab-9000-4457-97c9-c226bae97e4b, 2 Oct 2026): expvals must be a MAPPING, not a "
    "bare number ('invalid_target: target 0 is invalid: expvals must be a mapping')",
    "NOT verified: the mapping is Pauli word -> value, as in graph-v1's `paulis`, e.g. {\"Z\": 0.1} on one qubit and {\"ZZ\": 0.7} on a "
    "pair; the value may also be [value, certainty]",
    "NOT verified: targets are consumed by update(), so the full target set is repeated `rounds` times, each followed by a null entry",
    "NOT verified: tomography=0 means exact local state rather than shot-based tomography; sample=false because the circuit is sampled locally",
    "NOT verified: the 120 s job timeout is enough for the chosen qubit count and rounds",
]


def qdrive_payload(bloch, relationships, n_qubits, coupling_map, rounds=1, seed=7, update_method="spectral",
                   shots=1024, tomography=0, sample=False, machine="aer"):
    """Same targets as graph_v1_payload, written as a QDrive program: [singles..., pairs..., null] x rounds.
    rounds defaults to 1: three rounds made 201 targets, and both jobs of that size died with engine_timeout (at target 56 and 61)."""
    singles = [{"qubits": [b["qubit"]], "expvals": {"Z": _round(b["Z"])}} for b in sorted(bloch, key=lambda b: b["qubit"])]
    pairs = [{"qubits": list(r["qubits"]), "expvals": {"ZZ": _round(r["ZZ"])}}
             for r in sorted(relationships, key=lambda r: (-abs(r["ZZ"]), r["qubits"]))]
    targets = []
    for _ in range(rounds):
        targets += singles + pairs + [None]
    return {"params": {"machine": machine, "n_qubits": n_qubits, "coupling_map": [list(e) for e in coupling_map],
                       "targets": targets, "update_method": update_method, "shots": shots,
                       "tomography": tomography, "sample": sample, "seed": seed}}


QDRIVE_COMPLEMENTARY_ASSUMPTIONS = QDRIVE_ASSUMPTIONS + [
    "NOT verified: mixed Pauli words on a pair or group, e.g. {\"XZ\": 1.0} on [polarity, lamp] (word letter i acts on qubits[i]), are "
    "accepted and honoured; the docs describe weighted Pauli sums but show no example",
    "NOT verified: the engine returns a circuit whose X-type correlations actually reach these values (the Z-only run was not honoured by "
    "graph-v1 either); check locally with src.quantum.complementary --circuit before drawing any conclusion",
]


def qdrive_complementary_payload(bloch, relationships, n_qubits, coupling_map, polarity_qubits, partners, phi=3.141592653589793,
                                 rounds=1, seed=7, update_method="spectral", shots=1024, tomography=0, sample=False, machine="aer"):
    """The complementary-polarity variant of qdrive_payload (src.quantum.complementary). Differences from the Z-only program:
    the polarity qubits' Z targets and the polarity-polarity <ZZ> chain are dropped (polarity is a fair coin in Z), and each
    polarity qubit gets <Z> = 0 plus the mixed-basis correlator <X_polarity Z_lamp> = (1 - cos phi) / 2 with its partner lamp
    (1 for phi = pi); a lamp shared by three or more polarity qubits also gets the GHZ-frame stabiliser <X_lamp Z Z Z...> = 1
    (phi = pi only). coupling_map loses polarity-polarity edges and gains the polarity-lamp ones. partners: one list of lamp qubits
    per polarity qubit, as complementary.partners() returns. Building this sends nothing."""
    pol = set(polarity_qubits)
    bl = [b for b in bloch if b["qubit"] not in pol]
    rl = [r for r in relationships if not set(r["qubits"]) <= pol]
    singles = [{"qubits": [b["qubit"]], "expvals": {"Z": _round(b["Z"])}} for b in sorted(bl, key=lambda b: b["qubit"])]
    singles += [{"qubits": [q], "expvals": {"Z": 0.0}} for q in sorted(pol)]
    pairs = [{"qubits": list(r["qubits"]), "expvals": {"ZZ": _round(r["ZZ"])}} for r in sorted(rl, key=lambda r: (-abs(r["ZZ"]), r["qubits"]))]
    mixed, edges = [], {tuple(e) for e in coupling_map if not set(e) <= pol}
    by_lamp = {}
    for q, lamps in zip(polarity_qubits, partners):
        for u in lamps:
            mixed.append({"qubits": [q, u], "expvals": {"XZ": _round((1 - np.cos(phi)) / 2)}})
            edges.add(tuple(sorted((q, u))))
            by_lamp.setdefault(u, []).append(q)
    if abs(phi - np.pi) < 1e-9:
        for u, qs in sorted(by_lamp.items()):
            if len(qs) >= 3:
                mixed.append({"qubits": [u] + qs, "expvals": {"X" + "Z" * len(qs): 1.0}})
    targets = []
    for _ in range(rounds):
        targets += singles + pairs + mixed + [None]
    return {"params": {"machine": machine, "n_qubits": n_qubits, "coupling_map": [list(e) for e in sorted(edges)],
                       "targets": targets, "update_method": update_method, "shots": shots, "tomography": tomography,
                       "sample": sample, "seed": seed}}


def main(argv=None):
    """Build a graph-v1 payload variant from a calibration's targets.json (writes a file; sends nothing)."""
    import argparse
    from src.store.jsonfmt import dumps_compact
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--targets", default=os.path.join(DEFAULT_CALIB_DIR, "targets.json"))
    ap.add_argument("--order", choices=["strongest_first", "weakest_first"], default="weakest_first")
    ap.add_argument("--drop-below", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t = json.load(open(a.targets))
    p = graph_v1_payload(t["bloch"], t["relationships"], t["meta"]["n_qubits"], order=a.order, drop_below=a.drop_below)
    with open(a.out, "w") as f:
        f.write(dumps_compact(p) + "\n")
    ops = p["params"]["operations"]
    print(f"wrote {a.out}: {sum(o['type'] == 'bloch' for o in ops)} bloch + {sum(o['type'] == 'relationship' for o in ops)} relationship ops, "
          f"{len(p['params']['coupling_map'])} edges, order {a.order}, drop_below {a.drop_below}. NOT SENT.")


if __name__ == "__main__":
    main()
