#!/usr/bin/env python3
"""Spec §2.2 test, run: correlated draws versus plain independent noise, through the identical compose path.

    python -m src.baseline.compare [--labels labels.json] [--grid NX NY] [--draws 400] [--out runs/baseline]

Prints coherence metrics for both and writes baseline_vs_correlated.png. Every outcome in the bottom row (lamps,
polarity, patches) is an independent coin flip; the top row uses the mock's correlated sampler. Same geometry, same
windows, same relief and bevel code. Add the quantum samples as a third column when they exist.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline.coherence import coherence_metrics, draw_configs
from src.baseline.independent import IndependentNoise
from src.capture.labels import bay_window_labels, fill_defaults, load_labels, validate
from src.geometry.planes import build_scene
from src.quantum.sampler_mock import Sampler
from src.texture.compose import compose_frame


def frames(scene, sampler, seeds, K):
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        lamp = tuple(int(v) for v in rng.choice([-1, 1], 2))
        pol = sampler.sample_polarity(rng)
        lit = sampler.sample_lit(lamp, K, rng)
        out.append((compose_frame(scene, sampler.cells, sampler.NX, lit, lamp, pol), lamp, pol))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels")
    ap.add_argument("--grid", nargs=2, type=int)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--K", type=int, default=1, help="shots averaged per frame in the figure (1 = hardest)")
    ap.add_argument("--out", default="runs/baseline")
    args = ap.parse_args(argv)

    labels = load_labels(args.labels) if args.labels else validate(fill_defaults(bay_window_labels()))
    NX, NY = args.grid or (labels["grid"]["nx"], labels["grid"]["ny"])
    scene = build_scene(labels)
    corr = Sampler(scene, NX, NY, min_cover=labels["grid"]["min_cover"])
    noise = IndependentNoise(corr)

    rows = [("correlated sampler", corr), ("independent noise", noise)]
    metrics = {}
    for name, s in rows:
        metrics[name] = coherence_metrics(s, *draw_configs(s, args.draws, np.random.default_rng(1)))
    lines = ["coplanar_agreement", "crease_agreement", "light_left_right", "light_frontal_grazing", "polarity_agreement"]
    print(f"{args.draws} draws, one patch shot each, grid {NX}x{NY}\n{'metric':24s}" + "".join(f"{n:>22s}" for n, _ in rows))
    for m in lines:
        print(f"{m:24s}" + "".join(f"{metrics[n][m]:>22.3f}" if metrics[n][m] is not None else f"{'-':>22s}" for n, _ in rows))

    seeds = [3, 8, 21, 34]
    os.makedirs(args.out, exist_ok=True)
    fig, axs = plt.subplots(2, 4, figsize=(16, 5.6), facecolor="#111")
    for r, (name, s) in enumerate(rows):
        for c, (img, lamp, pol) in enumerate(frames(scene, s, seeds, args.K)):
            ax = axs[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{name}  {'R' if lamp[0] > 0 else 'L'}{'F' if lamp[1] > 0 else 'G'}  pol {''.join('R' if p > 0 else 'S' for p in pol)}",
                         color="w", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"Spec 2.2 test: same geometry and compose code, K={args.K}. Top: correlated draw. Bottom: plain independent noise.", color="w")
    fig.tight_layout()
    path = os.path.join(args.out, "baseline_vs_correlated.png")
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    print("wrote", path)


if __name__ == "__main__":
    main()
