#!/usr/bin/env python3
"""Estimate <Z> and <ZZ> targets from the mock's classical sampler and build the graph-v1 payload (handoff §6.2).

    python -m src.targets.calibrate_from_mock [--labels labels.json] [--grid NX NY] [--sweeps 200000] [--seed 7]
                                              [--use mc|exact] [--out runs/calibration]

NO Moth call is made. It writes targets.json and payload_graph_v1.json, prints the payload head, the estimated credit
cost and the estimator accuracy, then stops. Send with `python -m src.quantum.moth_client send ...` only after review.

What is estimated. The mock draws the two lamp spins uniformly, then Gibbs-samples the patch spins given the lamps
(sampler_mock.Sampler.sample_shots), and draws the polarity spins from an exact chain distribution. So
  <Z_L> = 0 by construction (uniform lamps), <Z_L Z_p> = E_L[ L * <s_p | L> ], <Z_p Z_q> = E_L[ <s_p s_q | L> ],
  <Z_p> = E_L[ <s_p | L> ]. Each lamp world (4 of them) is a separate chain; Monte-Carlo errors are batch-means
standard errors. For up to 20 patches the same quantities are also computed EXACTLY by enumerating all 2^n states, as a
check on the estimator (and usable as the payload source with --use exact).

Qubit order and signs: see src/graph/allocate_qubits.py. Z = +1 means lit / light from the right / frontal / raised, so
in a returned bitstring (qubit 0 leftmost) bit '0' is Z = +1.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

from src.capture.labels import bay_window_labels, fill_defaults, load_labels, validate
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import allocate, format_budget, budget
from src.graph.edges import patch_edges
from src.quantum.moth_client import MothClient
from src.quantum.sampler_mock import Sampler
from src.store.jsonfmt import dumps_compact
from src.targets import consistency
from src.targets.payloads import QDRIVE_ASSUMPTIONS, graph_v1_payload, payload_sha256, qdrive_payload

LAMP_WORLDS = list(itertools.product([-1, 1], repeat=2))   # (L1, L2), each world has probability 1/4
EXACT_MAX_PATCHES = 20
N_BATCHES = 50


def batch_se(series, n_batches=N_BATCHES):
    """Batch-means standard error of the column means of series (N x T)."""
    n = (series.shape[0] // n_batches) * n_batches
    means = series[:n].reshape(n_batches, -1, series.shape[1]).mean(axis=1)
    return means.std(axis=0, ddof=1) / np.sqrt(n_batches)


def world_series(shots, lamp, edges):
    """Per-shot series for every target in one lamp world: [<s_p> (n), <s_a s_b> (edges), L1 s_p (n), L2 s_p (n)]."""
    a, b = (np.array([e[0] for e in edges], int), np.array([e[1] for e in edges], int)) if edges else (np.array([], int),) * 2
    return np.concatenate([shots, shots[:, a] * shots[:, b], lamp[0] * shots, lamp[1] * shots], axis=1).astype(np.float64)


def estimate_patches_mc(sampler, edges, sweeps, burn, rng):
    """Monte-Carlo targets for the patch + lamp-hub block. Returns (values, standard errors), same layout as world_series."""
    vals, var = 0.0, 0.0
    for lamp in LAMP_WORLDS:
        shots = sampler.sample_shots(lamp, sweeps, rng, burn=burn, thin=1)
        series = world_series(shots, lamp, edges)
        vals = vals + series.mean(axis=0) / 4
        var = var + (batch_se(series) / 4) ** 2
    return vals, np.sqrt(var)


def all_states(n):
    """All 2^n spin configurations as an int8 matrix, row r = binary digits of r mapped to -1/+1."""
    r = np.arange(2 ** n)[:, None]
    return (((r >> np.arange(n)) & 1) * 2 - 1).astype(np.int8)


def estimate_patches_exact(sampler, edges):
    """The same quantities by enumerating the Boltzmann distribution of every lamp world exactly."""
    n = sampler.n
    S = all_states(n).astype(np.float64)
    a, b = np.array([e[0] for e in edges], int), np.array([e[1] for e in edges], int)
    feat = np.concatenate([S, S[:, a] * S[:, b]], axis=1)          # per-state values of <s_p> and <s_a s_b>
    total = 0.0
    for lamp in LAMP_WORLDS:
        h = sampler.hub1 * lamp[0] + sampler.hub2 * lamp[1]
        energy = S @ h + 0.5 * np.einsum("ij,ij->i", S @ sampler.J, S)   # same model as Sampler.sample_shots
        w = np.exp(sampler.beta * (energy - energy.max()))
        w /= w.sum()
        m = w @ feat                                                     # [<s_p> | L, <s_a s_b> | L]
        sp = m[:n]
        total = total + np.concatenate([m, lamp[0] * sp, lamp[1] * sp]) / 4
    return total


def estimate_polarity(sampler, draws, rng):
    """Monte-Carlo and exact <Z>, <ZZ over polarity-chain edges> from Sampler.sample_polarity."""
    D = np.array([sampler.sample_polarity(rng) for _ in range(draws)], float)
    edges = sampler.pol_edges
    series = np.concatenate([D, np.stack([D[:, a] * D[:, b] for a, b in edges], 1)], 1) if edges else D
    mc, se = series.mean(axis=0), batch_se(series)
    states = np.array(sampler.polarity_states(), float)
    p = sampler.polarity_weights(); p = p / p.sum()
    exact_feat = np.concatenate([states, np.stack([states[:, a] * states[:, b] for a, b in edges], 1)], 1) if edges else states
    return mc, se, p @ exact_feat


def build_targets(sampler, sweeps, burn, pol_draws, seed, use):
    rng = np.random.default_rng(seed)
    n = sampler.n
    alloc = allocate(n, sampler.n_panels)
    edges = patch_edges(sampler.J)
    ne = len(edges)

    mc, se = estimate_patches_mc(sampler, edges, sweeps, burn, rng)
    exact = estimate_patches_exact(sampler, edges) if n <= EXACT_MAX_PATCHES else None
    pmc, pse, pexact = estimate_polarity(sampler, pol_draws, rng)

    def target(key, mc_v, se_v, exact_v, **extra):
        """One target row: the payload value is the exact one when --use exact (and available), else the MC estimate."""
        use_exact = use == "exact" and exact_v is not None
        return {**extra, key: float(exact_v if use_exact else mc_v), "mc": float(mc_v), "exact": None if exact_v is None else float(exact_v),
                "se": float(se_v)}

    ex = lambda i: None if exact is None else exact[i]
    cell = lambda p: f"patch {p} (cell {sampler.cells[p]['i']},{sampler.cells[p]['j']}, panel {sampler.cells[p]['k']})"

    bloch, rel = [], []
    for p in range(n):
        bloch.append(target("Z", mc[p], se[p], ex(p), qubit=alloc["patches"][p], label=f"<Z> {cell(p)}"))
    for i, q in enumerate(alloc["lamps"]):
        bloch.append(target("Z", 0.0, 0.0, 0.0, qubit=q, label=f"<Z> lamp L{i + 1}", note="0 by construction: lamps are drawn uniformly"))
    for k, q in enumerate(alloc["polarity"]):
        bloch.append(target("Z", pmc[k], pse[k], pexact[k], qubit=q, label=f"<Z> polarity panel {k}"))

    for e, (a, b) in enumerate(edges):
        i = n + e
        rel.append(target("ZZ", mc[i], se[i], ex(i), qubits=[a, b], kind="patch-patch", label=f"<ZZ> patch {a} - patch {b}"))
    for L in (0, 1):
        for p in range(n):
            i = n + ne + L * n + p
            rel.append(target("ZZ", mc[i], se[i], ex(i), qubits=sorted([alloc["patches"][p], alloc["lamps"][L]]),
                              kind="lamp-patch", label=f"<ZZ> lamp L{L + 1} - patch {p}"))
    for e, (a, b) in enumerate(sampler.pol_edges):
        i = sampler.n_panels + e
        rel.append(target("ZZ", pmc[i], pse[i], pexact[i], qubits=sorted([alloc["polarity"][a], alloc["polarity"][b]]),
                          kind="polarity-polarity", label=f"<ZZ> polarity {a} - polarity {b}"))
    return alloc, bloch, rel


def accuracy(bloch, rel):
    """How far the Monte-Carlo estimates are from the exact values, in standard errors."""
    rows = [(t["mc"], t["exact"], t["se"]) for t in bloch + rel if t["exact"] is not None and t["se"] > 0]
    if not rows:
        return None
    d = np.array([abs(v - e) for v, e, _ in rows]); z = np.array([abs(v - e) / s for v, e, s in rows])
    return dict(n=len(rows), max_abs_err=float(d.max()), max_z=float(z.max()), rms_z=float(np.sqrt((z ** 2).mean())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="labels.json from the pen tool (default: built-in bay window)")
    ap.add_argument("--grid", nargs=2, type=int, metavar=("NX", "NY"))
    ap.add_argument("--sweeps", type=int, default=200000, help="Gibbs sweeps kept per lamp world (default 200000)")
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--pol-draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use", choices=["mc", "exact"], default="mc", help="which estimate goes into the payload")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--qd-rounds", type=int, default=1, help="QDrive: times the target set is repeated, each followed by update()")
    ap.add_argument("--kh", type=float, default=1.6, help="hub coupling to lamp L1 (mock default 1.6)")
    ap.add_argument("--kf", type=float, default=1.3, help="hub coupling to lamp L2 (mock default 1.3)")
    ap.add_argument("--j-in", type=float, default=0.9, help="coplanar patch coupling (mock default 0.9); lower it to unpin targets")
    ap.add_argument("--j-cross", type=float, default=0.6, help="crease coupling magnitude (mock default 0.6)")
    ap.add_argument("--j-pol", type=float, default=0.9, help="polarity chain coupling (mock default 0.9)")
    ap.add_argument("--out", default="runs/calibration")
    ap.add_argument("--head", type=int, default=12, help="operations to show when printing the payload")
    args = ap.parse_args(argv)

    labels = load_labels(args.labels) if args.labels else validate(fill_defaults(bay_window_labels()))
    NX, NY = args.grid or (labels["grid"]["nx"], labels["grid"]["ny"])
    scene = build_scene(labels)
    sampler = Sampler(scene, NX, NY, kh=args.kh, kf=args.kf, J_in=args.j_in, J_cross=args.j_cross, J_pol=args.j_pol,
                      min_cover=labels["grid"]["min_cover"])
    b = budget(sampler.n, sampler.n_panels)
    print(f"scene: {scene.n_panels} panels, grid {NX}x{NY}\nqubit budget: {format_budget(b)}")

    t0 = time.time()
    alloc, bloch, rel = build_targets(sampler, args.sweeps, args.burn, args.pol_draws, args.seed, args.use)
    print(f"estimated {len(bloch)} <Z> and {len(rel)} <ZZ> targets in {time.time() - t0:.0f}s "
          f"({4 * args.sweeps} Gibbs sweeps over 4 lamp worlds, seed {args.seed}, payload source: {args.use})")

    acc = accuracy(bloch, rel)
    if acc:
        print(f"estimator vs exact enumeration over {acc['n']} targets: max |err| {acc['max_abs_err']:.4f}, "
              f"max {acc['max_z']:.1f} standard errors, rms {acc['rms_z']:.2f} (about 1 expected)")
    else:
        print(f"exact enumeration skipped ({sampler.n} patches > {EXACT_MAX_PATCHES}); no accuracy check")
    cons = consistency.run_all(bloch, rel)
    print(f"consistency: {cons['n_triangles']} triangle checks, min slack {cons['min_slack']:.3f}, "
          f"{len(cons['violations']) + len(cons['bound_violations'])} violations")
    for v in (cons["bound_violations"] + cons["violations"])[:5]:
        print("  !", v)
    npn = cons["near_pinned"]
    if npn["count"]:
        print(f"WARNING near-pinned: {npn['count']} of {npn['of']} <ZZ> targets exceed |{npn['threshold']}|. Spec §3.5 wants magnitudes "
              f"below 1 so the draw stays random inside the bounds, and an engine may relax such targets silently. Lower --j-in / --kh / --kf to unpin.")
    print("variant: Z-only baseline (classically reproducible; not the complementary-polarity quantum variant)")

    payload = graph_v1_payload(bloch, rel, b["total"], shots=args.shots, mode="emu")
    job = MothClient().prepare("graph-v1", payload["params"])
    os.makedirs(args.out, exist_ok=True)
    meta = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seed=args.seed, sweeps=args.sweeps, burn=args.burn, pol_draws=args.pol_draws,
        payload_source=args.use, grid=[NX, NY], n_qubits=b["total"], qubit_map=alloc, payload_sha256=payload_sha256(payload),
        estimated_credits=job.credits, accuracy_vs_exact=acc, consistency=cons,
        sampler=dict(kh=args.kh, kf=args.kf, J_in=args.j_in, J_cross=args.j_cross, J_pol=args.j_pol, beta=sampler.beta),
        variant="z-only baseline: every target is a Z-basis correlator, so a classical correlated sampler reproduces it exactly "
                "(spec §2.3). It is NOT the quantum-specific variant: polarity is not complementary to lighting here and the "
                "polarity qubits are not entangled with the lamp/patch structure (spec §3.6, handoff §9.4).",
        quantum_specific=False,
        conventions="Z=+1: lit patch / light from right (L1) / frontal (L2) / raised relief; bitstring bit '0' is Z=+1, qubit 0 leftmost",
        patch_cells=[dict(qubit=i, i=c["i"], j=c["j"], panel=c["k"], plane_id=c["plane_id"]) for i, c in enumerate(sampler.cells)])
    with open(os.path.join(args.out, "targets.json"), "w") as f:
        f.write(dumps_compact(dict(meta=meta, bloch=bloch, relationships=rel)) + "\n")
    with open(os.path.join(args.out, "payload_graph_v1.json"), "w") as f:
        f.write(dumps_compact(payload) + "\n")
    qd = qdrive_payload(bloch, rel, b["total"], payload["params"]["coupling_map"], rounds=args.qd_rounds, seed=args.seed)
    qjob = MothClient().prepare("qdrive-api-v1", qd["params"])
    with open(os.path.join(args.out, "payload_qdrive.json"), "w") as f:
        f.write(dumps_compact(qd) + "\n")

    print("\n" + "=" * 78 + "\nPAYLOAD (graph-v1, NOT SENT)\n" + "=" * 78)
    print(job.describe(max_ops=args.head))
    print("=" * 78)
    p = payload["params"]
    print(f"{p['num_qubits']} qubits, {len(p['coupling_map'])} edges, {sum(o['type'] == 'bloch' for o in p['operations'])} bloch + "
          f"{sum(o['type'] == 'relationship' for o in p['operations'])} relationship operations, {p['shots']} shots, mode {p['mode']}")
    print(f"sha256 {meta['payload_sha256'][:16]}...   files: {args.out}/targets.json, {args.out}/payload_graph_v1.json")
    for j in (job, qjob):
        for prob in j.problems():
            print(f"NOT SENDABLE ({j.engine_id}): {prob}" + ("  -> re-run with --grid 5 4" if j.engine_id == "graph-v1" else ""))
    print("\n" + "=" * 78 + f"\nQDRIVE PAYLOAD (qdrive-api-v1, NOT SENT), {qjob.credits} credit\n" + "=" * 78)
    t = qd["params"]["targets"]
    print(f"{len(t)} target entries = {args.qd_rounds} rounds x ({len(bloch)} single + {len(rel)} pair + 1 update), "
          f"seed {args.seed}, update_method {qd['params']['update_method']}, tomography {qd['params']['tomography']}")
    print("first entries:", ", ".join(json.dumps(e) for e in t[:3]), "...")
    print("assumptions not covered by the docs:")
    for a in QDRIVE_ASSUMPTIONS:
        print("  -", a)
    print(f"files: {args.out}/payload_qdrive.json\nNothing has been sent to Moth. Total for both runs: "
          f"{(job.credits or 0) + (qjob.credits or 0)} credits.")


if __name__ == "__main__":
    main()
