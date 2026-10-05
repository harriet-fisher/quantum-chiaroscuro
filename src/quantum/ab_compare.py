#!/usr/bin/env python3
"""A/B figures: graph-v1 vs QDrive vs the classical mock vs independent noise (handoff §6.2, spec §11).

    python -m src.quantum.ab_compare --run runs/bay_run [--labels labels.json] [--out runs/bay_run/ab]

Reads what the solver wrote (graph_v1/state.json, qdrive/circuit.qasm, both requested_vs_achieved.json) and writes
  ab_requested_vs_achieved.png   achieved vs requested <Z>, <ZZ> per engine, with rms error
  ab_frames.png                  the same four lighting worlds from each source, through the identical compose code
  ab_coherence.png               spec §2.2 coherence metrics per source, independent noise as the floor
  ab_summary.json
Sources and what each can honestly be:
  mock         classical Gibbs sampler (the look test). Not quantum.
  QDrive       the returned circuit, simulated locally: a real pure state, unlimited shots, any basis.
  graph-v1     returns only the 20 most likely bitstrings, so frames from it are a TRUNCATED empirical distribution, shown for
               contrast and not as a fair frame source. Its real output is the exact tomography scored in the first figure.
  independent  fair coin per patch, lamp and polarity: the floor.
If <run>/SYNTHETIC exists, every figure is stamped SYNTHETIC (offline self-test fixtures, not Moth results).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline.coherence import coherence_metrics
from src.baseline.independent import IndependentNoise
from src.capture.labels import DEFAULT_CALIB_DIR, DEFAULT_RUN_DIR, bay_window_labels, fill_defaults, load_labels, validate
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import allocate
from src.quantum import sampler_local as sl
from src.quantum.sampler_mock import Sampler
from src.texture.frames import FrameGenerator

WORLDS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
KIND_COLOUR = {"single": "#9a9ca3", "patch-patch": "#4c9be8", "lamp-patch": "#f2a33a", "polarity-polarity": "#5fd38d"}
METRICS = ["coplanar_agreement", "crease_agreement", "light_left_right", "light_frontal_grazing", "polarity_agreement"]


def world_name(w):
    return f"light from {'right' if w[0] > 0 else 'left'}, {'frontal' if w[1] > 0 else 'grazing'}"


def mock_pool(sampler, per_world, rng):
    lamps, shots, pol = [], [], []
    for w in WORLDS:
        s = sampler.sample_shots(w, per_world, rng, burn=200, thin=3)
        lamps += [w] * per_world
        shots.append(s)
        pol += [sampler.sample_polarity(rng) for _ in range(per_world)]
    return sl.Pool(np.array(lamps), np.concatenate(shots), np.array(pol))


def independent_pool(sampler, M, rng):
    noise = IndependentNoise(sampler)
    return sl.Pool(rng.choice([-1, 1], (M, 2)), noise.sample_shots(None, M, rng), np.array([noise.sample_polarity(rng) for _ in range(M)]))


def top20_pool(top, layout, M, rng):
    """Resample graph-v1's reported top bitstrings by their probabilities (truncated: everything else is missing)."""
    bs = [t["bitstring"] for t in top]
    p = np.array([float(t.get("probability", t.get("count", 0))) for t in top], float)
    p /= p.sum()
    idx = np.array([sum((1 << q) for q, ch in enumerate(b) if ch == "1") for b in bs])      # qubit 0 is the leftmost character
    pick = idx[rng.choice(len(idx), size=M, p=p)]
    return sl.Pool(sl.spins(pick, layout["lamps"]), sl.spins(pick, layout["patches"]), sl.spins(pick, layout["polarity"]))


def load_rows(path):
    with open(path) as f:
        return json.load(f)


def fig_requested_vs_achieved(engines, out, stamp):
    fig, axs = plt.subplots(1, len(engines), figsize=(5.6 * len(engines), 5.4), facecolor="#111", squeeze=False)
    for ax, (name, d) in zip(axs[0], engines.items()):
        ax.set_facecolor("#16171a")
        for kind, c in KIND_COLOUR.items():
            r = [x for x in d["rows"] if x["kind"] == kind and x["achieved"] is not None]
            if r:
                ax.scatter([x["requested"] for x in r], [x["achieved"] for x in r], s=26, c=c, label=f"{kind} ({len(r)})", alpha=.9)
        ax.plot([-1, 1], [-1, 1], color="#666", lw=1); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        s = d["summary"]
        ax.set_title(f"{name}: rms error {s['rms']:.3f}, max {s['max_abs']:.3f}\n{s['n_gap_over_0p1']} of {s['n_achieved']} off by > 0.1", color="w", fontsize=10)
        ax.set_xlabel("requested", color="w"); ax.set_ylabel("achieved", color="w"); ax.tick_params(colors="w")
        ax.legend(facecolor="#222", labelcolor="w", fontsize=8, loc="upper left")
    fig.suptitle(stamp + "Requested vs achieved correlations (Z-type targets)", color="w")
    fig.tight_layout(); fig.savefig(out, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


def fig_frames(scene, sampler, pools, out, stamp, K, seed):
    fig, axs = plt.subplots(len(pools), len(WORLDS), figsize=(4.1 * len(WORLDS), 2.6 * len(pools)), facecolor="#111", squeeze=False)
    for r, (name, pool) in enumerate(pools.items()):
        gen = FrameGenerator(scene, sampler, pool)
        for c, w in enumerate(WORLDS):
            ax = axs[r, c]; ax.axis("off")
            try:
                img, d = gen.frame(K, np.random.default_rng(seed + c), lamp=w)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{name}: {world_name(w)}  pol {''.join('R' if p > 0 else 'S' for p in d.pol)}", color="w", fontsize=8)
            except ValueError:
                ax.set_facecolor("#000"); ax.text(.5, .5, f"{name}\nthis world is not in the data", color="#888", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(stamp + f"Same lighting worlds, K={K} shots per frame, identical compose code", color="w")
    fig.tight_layout(); fig.savefig(out, dpi=95, facecolor=fig.get_facecolor()); plt.close(fig)


def fig_coherence(metrics, out, stamp):
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor="#111"); ax.set_facecolor("#16171a")
    names = list(metrics); w = 0.8 / len(names); cols = ["#777", "#f2a33a", "#4c9be8", "#5fd38d", "#c77dff"]
    for i, n in enumerate(names):
        vals = [metrics[n][m] if metrics[n][m] is not None else 0 for m in METRICS]
        ax.bar(np.arange(len(METRICS)) + i * w, vals, w, label=n, color=cols[i % len(cols)])
    ax.set_xticks(np.arange(len(METRICS)) + 0.4 - w / 2); ax.set_xticklabels([m.replace("_", "\n") for m in METRICS], color="w", fontsize=9)
    ax.axhline(0, color="#666", lw=1); ax.tick_params(colors="w"); ax.legend(facecolor="#222", labelcolor="w")
    ax.set_title(stamp + "Coherence of the draws: do they read as one light on one set of planes? (independent noise = the floor)", color="w", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN_DIR)
    ap.add_argument("--calib", default=DEFAULT_CALIB_DIR)
    ap.add_argument("--labels")
    ap.add_argument("--out")
    ap.add_argument("--pool", type=int, default=60000)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.run, "ab")
    os.makedirs(out, exist_ok=True)
    stamp = "SYNTHETIC (offline self-test, not a Moth result)  " if os.path.exists(os.path.join(a.run, "SYNTHETIC")) else ""

    with open(os.path.join(a.calib, "targets.json")) as f:
        targets = json.load(f)
    NX, NY = targets["meta"]["grid"]
    labels = load_labels(a.labels) if a.labels else validate(fill_defaults(bay_window_labels()))
    scene = build_scene(labels)
    sampler = Sampler(scene, NX, NY, min_cover=labels["grid"]["min_cover"])
    layout = allocate(sampler.n, sampler.n_panels)
    rng = np.random.default_rng(a.seed)

    engines, pools, notes = {}, {}, []
    pools["mock (classical)"] = mock_pool(sampler, a.pool // 4, rng)
    gv = os.path.join(a.run, "graph_v1", "requested_vs_achieved.json")
    if os.path.exists(gv):
        engines["graph-v1"] = load_rows(gv)
        with open(os.path.join(a.run, "graph_v1", "state.json")) as f:
            st = json.load(f)
        top = st["measurements"]["top"]
        if top:
            pools["graph-v1 (top-20 only)"] = top20_pool(top, layout, a.pool, rng)
    else:
        notes.append("graph-v1 result not found")
    qd = os.path.join(a.run, "qdrive", "requested_vs_achieved.json")
    qasm = os.path.join(a.run, "qdrive", "circuit.qasm")
    if os.path.exists(qd) and os.path.exists(qasm):
        engines["QDrive (circuit, local sim)"] = load_rows(qd)
        with open(qasm) as f:
            src = sl.source_from_qasm(f.read(), layout, label="QDrive circuit")
        pools["QDrive circuit (local sim)"] = src.pool(a.pool, rng)
    else:
        notes.append("QDrive result not found")
    pools["independent noise"] = independent_pool(sampler, a.pool, rng)

    metrics = {n: coherence_metrics(sampler, p.lamps, p.shots, p.pol) for n, p in pools.items()}
    if engines:
        fig_requested_vs_achieved(engines, os.path.join(out, "ab_requested_vs_achieved.png"), stamp)
    fig_frames(scene, sampler, pools, os.path.join(out, "ab_frames.png"), stamp, a.K, a.seed)
    fig_coherence(metrics, os.path.join(out, "ab_coherence.png"), stamp)

    summary = dict(synthetic=bool(stamp), engines={n: d["summary"] for n, d in engines.items()}, coherence=metrics, notes=notes,
                   credits=dict(graph_v1=5 if "graph-v1" in engines else 0, qdrive=1 if any(k.startswith("QDrive") for k in engines) else 0))
    with open(os.path.join(out, "ab_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(stamp + f"A/B written to {out}")
    print(f"{'source':30s}" + "".join(f"{m[:14]:>16s}" for m in METRICS))
    for n, m in metrics.items():
        print(f"{n:30s}" + "".join(f"{(m[k] if m[k] is not None else float('nan')):>16.3f}" for k in METRICS))
    for n in notes:
        print("note:", n)


if __name__ == "__main__":
    main()
