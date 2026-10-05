#!/usr/bin/env python3
"""Standing Light - classical look test (NO quantum calls). Port of mock/standing_light_mock.py onto src/ modules.

    python -m src.run_mock [NX NY] [--labels labels.json] [--out runs/mock]

Without --labels it renders the ORIGINAL three-panel mock window (the layout of mock/standing_light_mock.py) and should reproduce the five mock figures
exactly. --stacked renders the show's default instead: the same window with two stacked panes in every wing (six panels).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.capture.labels import bay_window_labels, fill_defaults, load_labels, validate
from src.geometry.planes import build_scene
from src.quantum.sampler_mock import Sampler
from src.texture.compose import render


def show(ax, img, title, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, color="w", fontsize=9)
    ax.axis("off")


def fig_dark(nr, nc, w=4.2, h=2.6):
    fig, axs = plt.subplots(nr, nc, figsize=(nc * w, nr * h), facecolor="#111")
    return fig, np.atleast_2d(axs)


def polarity_combos(scene):
    """The four polarity patterns of figure 2. When the scene has a flat centre and tilted sides (the bay window, three panels or six panes) the
    sides and the centre are set against each other; any other scene alternates panel by panel."""
    n = scene.n_panels
    side = [abs(p.angle_deg) > 0.5 for p in scene.panels]
    if any(side) and not all(side):
        sides = lambda v: tuple(v if s else -v for s in side)
        return [((-1,) * n, "all sunk"), ((1,) * n, "all raised"),
                (sides(-1), "sides sunk, centre raised"), (sides(1), "sides raised, centre sunk")]
    alt = tuple(1 if k % 2 else -1 for k in range(n))
    return [((-1,) * n, "all sunk"), ((1,) * n, "all raised"),
            (alt, "alternating, first sunk"), (tuple(-v for v in alt), "alternating, first raised")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", nargs="*", type=int, help="NX NY (default: the labels' grid: 6 4 for the mock window, 4 4 for --stacked)")
    ap.add_argument("--labels", help="labels.json from the pen tool (default: the original three-panel mock window)")
    ap.add_argument("--stacked", action="store_true", help="without --labels: the six-pane bay window the show uses by default (two stacked windows per wing)")
    ap.add_argument("--out", default="runs/mock", help="output directory for the figures")
    args = ap.parse_args(argv)

    labels = load_labels(args.labels) if args.labels else validate(fill_defaults(bay_window_labels(stacked=args.stacked)))
    NX, NY = args.grid if len(args.grid) == 2 else (labels["grid"]["nx"], labels["grid"]["ny"])
    os.makedirs(args.out, exist_ok=True)

    scene = build_scene(labels)
    sampler = Sampler(scene, NX, NY, min_cover=labels["grid"]["min_cover"])
    print(f"patches (qubits for panels): {sampler.n}; + 2 lamps + {scene.n_panels} polarity = {sampler.n_qubits} qubits")
    glass_ok = True
    checks = 0

    def run(lamp, pol, K, seed):
        nonlocal glass_ok, checks
        img = render(scene, sampler, lamp, pol, K, np.random.default_rng(seed))
        glass_ok &= bool(np.all(img[scene.impenetrable] == 0))
        checks += 1
        return img

    lamp_name = lambda l: f"light from {'right' if l[0] > 0 else 'left'}, {'frontal' if l[1] > 0 else 'grazing'}"
    ones = lambda v: (v,) * scene.n_panels

    # geometry diagnostic
    fig, axs = fig_dark(1, 4)
    reg_img = np.where(scene.frame, 0.35 + 0.25 * (scene.region + 1) / scene.n_panels, 0.0)
    show(axs[0, 0], reg_img, "frame (projectable) by plane; black = impenetrable")
    pm = np.zeros((scene.H, scene.W))
    for c in sampler.cells:
        x0, x1, y0, y1 = c["box"]
        pm[y0:y1, x0:x1] = 0.25 + 0.2 * c["k"]
    pm[~(scene.region >= 0)] = 0
    axs[0, 1].imshow(pm * (scene.frame | True), cmap="viridis"); axs[0, 1].set_title(f"{sampler.n} patch qubits (glass cells get none)", color="w", fontsize=9); axs[0, 1].axis("off")
    show(axs[0, 2], scene.relief * scene.frame, "distance to border (relief field)", "magma")
    show(axs[0, 3], (scene.lo + scene.hi) / 2 * scene.frame, "mid of brightness window (plane + depth)")
    fig.tight_layout(); fig.savefig(f"{args.out}/fig0_geometry.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

    # knob 1: lighting world
    fig, axs = fig_dark(2, 2)
    for ax, lamp in zip(axs.ravel(), [(1, 1), (-1, 1), (1, -1), (-1, -1)]):
        show(ax, run(lamp, ones(-1), 40, 3), lamp_name(lamp))
    fig.suptitle("Knob 1: lighting world (lamp qubits), polarity sunk, K=40", color="w")
    fig.tight_layout(); fig.savefig(f"{args.out}/fig1_light_direction.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

    # knob 2: polarity
    fig, axs = fig_dark(2, 2)
    for ax, (pol, name) in zip(axs.ravel(), polarity_combos(scene)):
        show(ax, run((1, 1), pol, 40, 5), name)
    fig.suptitle("Knob 2: relief polarity (one qubit per panel), light from right, K=40", color="w")
    fig.tight_layout(); fig.savefig(f"{args.out}/fig2_polarity.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

    # knob 3: hardness
    fig, axs = fig_dark(1, 5, w=3.4, h=2.2)
    for ax, K in zip(axs.ravel(), [1, 3, 8, 32, 128]):
        show(ax, run((1, 1), ones(-1), K, 11), f"K = {K} shots averaged")
    fig.suptitle("Knob 3: hardness (shots averaged per frame)", color="w")
    fig.tight_layout(); fig.savefig(f"{args.out}/fig3_hardness.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

    # random contact sheet: all three knobs drawn at random
    fig, axs = fig_dark(3, 4, w=3.2, h=2.0)
    rng = np.random.default_rng(2026)
    for ax in axs.ravel():
        lamp = tuple(rng.choice([-1, 1], 2)); pol = sampler.sample_polarity(rng)
        K = int(rng.choice([1, 4, 16, 64]))
        img = run(lamp, pol, K, int(rng.integers(1e9)))
        pn = "".join("R" if p > 0 else "S" for p in pol)
        show(ax, img, f"{'R' if lamp[0] > 0 else 'L'}{'F' if lamp[1] > 0 else 'G'}  pol {pn}  K={K}")
    fig.suptitle("Random frames: lamp (R/L light, F/G frontal/grazing), polarity (S sunk, R raised per panel), K", color="w")
    fig.tight_layout(); fig.savefig(f"{args.out}/fig4_random_sheet.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

    print(f"glass/outside pixels exactly black in all {checks} frames: {glass_ok}")


if __name__ == "__main__":
    main()
