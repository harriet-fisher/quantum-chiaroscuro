#!/usr/bin/env python3
"""
Standing Light - classical mock (look test only, NO quantum calls).

A synthetic bay window (left / centre / right panels, glass inset in each).
Geometry decides:   - impenetrable regions (glass + outside): pure black, never calculated
                    - brightness window [lo, hi] per plane and per depth (near/far)
                    - relief profile (distance to border: trim looks sunk or raised)
A *classical stand-in sampler* decides the random draw:
                    - an Ising-style model on the patch graph + 2 "lamp" spins + 3 "polarity" spins
Three knobs (all drawn by the sampler):
  1. lighting world  : lamp spins L1 (light from left/right), L2 (frontal vs grazing)
  2. relief polarity : one spin per panel, sunk (centre dark / edges bright) vs raised
  3. hardness        : number of shots K averaged per frame (K=1 hard, large K soft)

Replace `sample_frame_config` with a QDrive / graph-v1 backed sampler later.
"""
import os
import itertools
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
W, H = 1200, 700
THETA = np.deg2rad(35)

# ------------------------------------------------------------------ geometry
PANELS = {
    0: dict(name="left",   poly=[(70, 70), (420, 150), (420, 560), (70, 640)],
            n=(np.sin(THETA), 0.0, np.cos(THETA))),
    1: dict(name="centre", poly=[(420, 150), (780, 150), (780, 560), (420, 560)],
            n=(0.0, 0.0, 1.0)),
    2: dict(name="right",  poly=[(780, 150), (1130, 70), (1130, 640), (780, 560)],
            n=(-np.sin(THETA), 0.0, np.cos(THETA))),
}
GLASS_SCALE = 0.45   # glass inset size relative to each panel (about its centroid)

# brightness windows [lo, hi]
WIN_CENTRE = (0.40, 0.95)
WIN_SIDE_NEAR = (0.25, 0.80)
WIN_SIDE_FAR = (0.10, 0.65)


def poly_mask(poly):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in poly], fill=255)
    return np.array(im) > 0


def scale_poly(poly, f):
    c = np.mean(np.array(poly, float), axis=0)
    return [tuple(c + f * (np.array(p, float) - c)) for p in poly]


REGION = -np.ones((H, W), int)
GLASS = np.zeros((H, W), bool)
for k, p in PANELS.items():
    REGION[poly_mask(p["poly"])] = k
    GLASS |= poly_mask(scale_poly(p["poly"], GLASS_SCALE))
FRAME = (REGION >= 0) & ~GLASS          # projectable pixels
IMPENETRABLE = ~FRAME                   # glass + outside: black, never calculated

# depth coordinate t (0 = near/outer end, 1 = far/inner end) and windows
yy, xx = np.mgrid[0:H, 0:W]
T = np.full((H, W), 0.5)
T[REGION == 0] = ((xx - 70) / (420 - 70))[REGION == 0]
T[REGION == 2] = ((1130 - xx) / (1130 - 780))[REGION == 2]
T = np.clip(T, 0, 1)
LO = np.zeros((H, W)); HI = np.zeros((H, W))
for k in (0, 2):
    m = REGION == k
    LO[m] = WIN_SIDE_NEAR[0] + T[m] * (WIN_SIDE_FAR[0] - WIN_SIDE_NEAR[0])
    HI[m] = WIN_SIDE_NEAR[1] + T[m] * (WIN_SIDE_FAR[1] - WIN_SIDE_NEAR[1])
m = REGION == 1
LO[m], HI[m] = WIN_CENTRE

# distance-to-border field per panel (border = crease, outer edge or glass edge)
R = np.zeros((H, W))
for k in PANELS:
    fm = FRAME & (REGION == k)
    d = ndi.distance_transform_edt(fm)
    R[fm] = (d / d.max())[fm]


# ------------------------------------------------------------------ patch graph
def build_patches(NX, NY, min_cover=0.30):
    cw, ch = W / NX, H / NY
    cells, idx = [], {}
    for j in range(NY):
        for i in range(NX):
            x0, x1, y0, y1 = int(i * cw), int((i + 1) * cw), int(j * ch), int((j + 1) * ch)
            f = FRAME[y0:y1, x0:x1]
            if f.mean() < min_cover:
                continue                      # mostly glass / outside: no qubit
            reg = REGION[y0:y1, x0:x1][f]
            k = int(np.bincount(reg, minlength=3).argmax())
            idx[(i, j)] = len(cells)
            cells.append(dict(i=i, j=j, box=(x0, x1, y0, y1), k=k))
    return cells, idx


class Sampler:
    """Classical stand-in for the quantum draw: Boltzmann sampling of an Ising-type model."""

    def __init__(self, NX=6, NY=4, kh=1.6, kf=1.3, J_in=0.9, J_cross=0.6, J_pol=0.9, beta=1.0):
        self.cells, self.idx = build_patches(NX, NY)
        self.n = len(self.cells)
        self.NX, self.NY, self.beta, self.J_pol = NX, NY, beta, J_pol
        nx = np.array([PANELS[c["k"]]["n"][0] for c in self.cells])
        nz = np.array([PANELS[c["k"]]["n"][2] for c in self.cells])
        self.hub1 = kh * nx                        # coupling to lamp L1 (left/right light)
        self.hub2 = kf * (nz - nz.mean())          # coupling to lamp L2 (frontal vs grazing)
        J = np.zeros((self.n, self.n))
        for (i, j), a in self.idx.items():
            for di, dj in ((1, 0), (0, 1)):
                b = self.idx.get((i + di, j + dj))
                if b is None:
                    continue
                same = self.cells[a]["k"] == self.cells[b]["k"]
                J[a, b] = J[b, a] = J_in if same else -J_cross   # coplanar agree, crease disagree
        self.J = J
        self.n_qubits = self.n + 2 + 3             # patches + lamps + polarity

    def sample_lit(self, lamp, K, rng, burn=60, thin=3):
        """K shots of the patch bits with the lamp spins clamped (post-selected)."""
        h = self.hub1 * lamp[0] + self.hub2 * lamp[1]
        s = rng.choice([-1, 1], self.n)
        out = []
        total = burn + K * thin
        for sweep in range(total):
            for p in rng.permutation(self.n):
                local = h[p] + self.J[p] @ s
                s[p] = 1 if rng.random() < 1 / (1 + np.exp(-2 * self.beta * local)) else -1
            if sweep >= burn and (sweep - burn) % thin == 0:
                out.append(s.copy())
        return (np.array(out) > 0).mean(axis=0)    # lit fraction per patch

    def sample_polarity(self, rng):
        states = list(itertools.product([-1, 1], repeat=3))   # +1 raised, -1 sunk
        w = np.array([np.exp(self.beta * self.J_pol * (s[0] * s[1] + s[1] * s[2])) for s in states])
        return states[rng.choice(len(states), p=w / w.sum())]


# ------------------------------------------------------------------ rendering
def gauss(x, s):
    return ndi.gaussian_filter(x, s, mode="nearest")


def render(sampler, lamp, pol, K, rng, interp_sigma=None, blend_sigma=5, w_light=0.75, w_relief=0.20, w_dir=0.70):
    lit = sampler.sample_lit(lamp, K, rng)
    cw = W / sampler.NX
    interp_sigma = interp_sigma or 0.18 * cw
    S = np.zeros((H, W))
    for k in PANELS:
        fm = FRAME & (REGION == k)
        Pk = np.zeros((H, W)); valid = np.zeros((H, W), bool)
        for c, v in zip(sampler.cells, lit):
            if c["k"] != k:
                continue
            x0, x1, y0, y1 = c["box"]
            sel = fm[y0:y1, x0:x1]
            Pk[y0:y1, x0:x1][sel] = v
            valid[y0:y1, x0:x1] |= sel
        if not valid.any():
            continue
        near = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = Pk[near[0], near[1]]
        S[fm] = (gauss(filled * fm, interp_sigma) / np.maximum(gauss(fm.astype(float), interp_sigma), 1e-6))[fm]
    # relief profile: sunk = centre dark / edges bright; raised = opposite
    pol_map = np.zeros((H, W))
    for k in PANELS:
        pol_map[REGION == k] = pol[k]
    prof = np.where(pol_map < 0, 1 - R, R)
    # directional bevel: height field h = polarity * smoothed relief; shade = n.l with n = (-hx, -hy, 1).
    # Flipping polarity flips the sign of h, so the SAME light reads as hollow or raised (crater illusion).
    hs = pol_map * gauss(R, 4)
    hy, hx = np.gradient(hs)
    lx = lamp[0]                              # +1 light from right, -1 from left
    ly = -0.5                                 # slightly from above (image y points down)
    strength = 1.0 if lamp[1] < 0 else 0.5    # grazing light = stronger bevel
    bevel = np.clip(0.5 + 40.0 * strength * (-hx * lx - hy * ly), 0, 1)
    u = np.clip((w_light * S + w_relief * prof + w_dir * bevel) / (w_light + w_relief + w_dir), 0, 1)
    val = LO + (HI - LO) * u
    fr = FRAME.astype(float)
    out = gauss(val * fr, blend_sigma) / np.maximum(gauss(fr, blend_sigma), 1e-6)   # mask-aware blur
    out = out * FRAME                                                              # impenetrable = black
    return np.clip(out, 0, 1)


# ------------------------------------------------------------------ figures
def show(ax, img, title, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, color="w", fontsize=9)
    ax.axis("off")


def fig_dark(nr, nc, w=4.2, h=2.6):
    fig, axs = plt.subplots(nr, nc, figsize=(nc * w, nr * h), facecolor="#111")
    return fig, np.atleast_2d(axs)


def main():
    import sys
    NX, NY = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (6, 4)
    sampler = Sampler(NX, NY)
    print(f"patches (qubits for panels): {sampler.n}; + 2 lamps + 3 polarity = {sampler.n_qubits} qubits")
    glass_ok = True
    checks = 0

    def run(lamp, pol, K, seed):
        nonlocal glass_ok, checks
        img = render(sampler, lamp, pol, K, np.random.default_rng(seed))
        glass_ok &= bool(np.all(img[IMPENETRABLE] == 0))
        checks += 1
        return img

    lamp_name = lambda l: f"light from {'right' if l[0] > 0 else 'left'}, {'frontal' if l[1] > 0 else 'grazing'}"

    # geometry diagnostic
    fig, axs = fig_dark(1, 4)
    reg_img = np.where(FRAME, 0.35 + 0.25 * (REGION + 1) / 3, 0.0)
    show(axs[0, 0], reg_img, "frame (projectable) by plane; black = impenetrable")
    pm = np.zeros((H, W))
    for c in sampler.cells:
        x0, x1, y0, y1 = c["box"]
        pm[y0:y1, x0:x1] = 0.25 + 0.2 * c["k"]
    pm[~(REGION >= 0)] = 0
    axs[0, 1].imshow(pm * (FRAME | True), cmap="viridis"); axs[0, 1].set_title(f"{sampler.n} patch qubits (glass cells get none)", color="w", fontsize=9); axs[0, 1].axis("off")
    show(axs[0, 2], R * FRAME, "distance to border (relief field)", "magma")
    show(axs[0, 3], (LO + HI) / 2 * FRAME, "mid of brightness window (plane + depth)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig0_geometry.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

    # knob 1: lighting world
    fig, axs = fig_dark(2, 2)
    for ax, lamp in zip(axs.ravel(), [(1, 1), (-1, 1), (1, -1), (-1, -1)]):
        show(ax, run(lamp, (-1, -1, -1), 40, 3), lamp_name(lamp))
    fig.suptitle("Knob 1: lighting world (lamp qubits), polarity sunk, K=40", color="w")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_light_direction.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

    # knob 2: polarity
    fig, axs = fig_dark(2, 2)
    combos = [((-1, -1, -1), "all sunk"), ((1, 1, 1), "all raised"),
              ((-1, 1, -1), "sides sunk, centre raised"), ((1, -1, 1), "sides raised, centre sunk")]
    for ax, (pol, name) in zip(axs.ravel(), combos):
        show(ax, run((1, 1), pol, 40, 5), name)
    fig.suptitle("Knob 2: relief polarity (one qubit per panel), light from right, K=40", color="w")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_polarity.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

    # knob 3: hardness
    fig, axs = fig_dark(1, 5, w=3.4, h=2.2)
    for ax, K in zip(axs.ravel(), [1, 3, 8, 32, 128]):
        show(ax, run((1, 1), (-1, -1, -1), K, 11), f"K = {K} shots averaged")
    fig.suptitle("Knob 3: hardness (shots averaged per frame)", color="w")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_hardness.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

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
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_random_sheet.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)

    print(f"glass/outside pixels exactly black in all {checks} frames: {glass_ok}")


if __name__ == "__main__":
    main()
