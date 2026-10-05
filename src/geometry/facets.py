"""Facets: the qubits of Superposed Relief (v2 handoff sections 2.1 and 4.1).

A facet is a set of projectable pixels that share one surface normal, and that normal IS the qubit's state. Under a distant light
every pixel with the same normal is lit identically, so one qubit can stand for all of them, wherever they are: the outer-left
bevel and the glass's right-hand bevel are one facet when both slope the same way. Per panel there are
  band       `n_dirs` orientation classes of the sloping surface (bevel / swell), found by clustering the slope direction
  interior   one facet for the flat plateau (its tilt is ~0; it carries the panel's base shading)
Glass pixels never get a facet and render exactly 0; the glass edge is a border like any other, so it carries a bevel too.

Geometry only fences the values and sets the tilt budget. A classical height field h of the distance d to the nearest border
(a chamfer of width `band_frac` of the panel) gives every pixel a slope s = -grad h; a facet's slope is the mean over its pixels, so
its Bloch vector in the panel-local frame (z = panel normal, x right, y down in the image) is
        b = (sin tau cos phi, sin tau sin phi, cos tau).
Which of the two depth maps (h or -h) is "raised" is NOT decided here: that is the polarity qubit's job.

The tilt budget (handoff section 5). The interference between a panel's raised and sunk depth maps has visibility
V_panel = prod cos(tau_i) ~ exp(-sum tau_i^2 / 2), so tilts are rescaled per panel to sum(tau_i^2) = kappa^2, a number that does not
depend on how many facets the panel is cut into.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from src.mask.masked_blur import gauss

TAU_MAX = 1.1            # radians: no facet is tilted further than this (about 63 degrees)
EXACT_QUBITS = 20        # the per-panel engine simulates facets + one polarity qubit per panel as an exact statevector; this many is the working size
EXACT_QUBITS_MAX = 22    # and this many is the most it will attempt (a few hundred MB per frame table); beyond it, use the domain engine
N_DIRS_MAX = 4           # slope-orientation facets per panel: the four sides of a window pane


def auto_n_dirs(n_panels, max_qubits=EXACT_QUBITS, cap=N_DIRS_MAX):
    """The largest number of orientation facets per panel (at most `cap`) for which the per-panel engine still fits `max_qubits`: every panel
    brings n_dirs facet qubits and one polarity qubit, so n_dirs <= max_qubits / n_panels - 1. Three panels get 4 (15 qubits); the six panes of the
    built-in bay window get 2 (18 qubits, a 22-qubit circuit with the lamp and observe registers); eight panels get 1."""
    return int(max(1, min(cap, max_qubits // max(1, n_panels) - 1)))


@dataclass
class Facet:
    idx: int
    panel: int
    kind: str                       # "band" | "interior"
    centroid: tuple                 # (x, y) canvas pixels
    box: tuple                      # (x0, x1, y0, y1)
    area: int
    slope: tuple                    # raw mean slope (sx, sy) = mean of -grad h over the facet's pixels
    tau: float = 0.0                # tilt after the per-panel budget
    phi: float = 0.0                # slope azimuth (radians, image coordinates)
    anchor: tuple = None            # (x, y): the deepest interior pixel of the facet's largest piece, where overlays mark it
    classical: bool = False         # a plateau facet whose tilt is ~0 is a constant |0>: no qubit, lit by the classical Lambert value
    domain: int = -1                # domain relief: the polarity qubit (depth domain) this facet belongs to
    edge: int = -1                  # domain relief: the polygon edge whose bevel strip this facet is a piece of


@dataclass
class FacetSet:
    facets: list
    label: np.ndarray               # (H, W) facet index per pixel, -1 where there is no facet (glass, outside)
    edges: list                     # [(i, j, kind, length_px)] kind: "coplanar" | "crease"
    height: np.ndarray = field(repr=False, default=None)
    kappa: float = 1.0
    params: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.facets)

    @property
    def n_quantum(self):
        """Facets that carry a qubit. They are always the first n_quantum facets; classical plateau facets come last."""
        return sum(1 for f in self.facets if not f.classical)

    @property
    def qtau(self):
        return self.tau[:self.n_quantum]

    @property
    def qphi(self):
        return self.phi[:self.n_quantum]

    @property
    def qpanel_of(self):
        return self.panel_of[:self.n_quantum]

    @property
    def panel_of(self):
        return np.array([f.panel for f in self.facets], int)

    @property
    def tau(self):
        return np.array([f.tau for f in self.facets])

    @property
    def phi(self):
        return np.array([f.phi for f in self.facets])

    def of_panel(self, k):
        return [f.idx for f in self.facets if f.panel == k]

    def with_tilt(self, tau, phi):
        """A copy whose facets carry the given tilts (e.g. from the echo engine); geometry and edges are shared."""
        import copy
        out = copy.copy(self)
        out.facets = [copy.copy(f) for f in self.facets]
        for f, t, p in zip(out.facets, tau, phi):
            f.tau, f.phi = float(t), float(p)
        return out


# ------------------------------------------------------------------ height field
def height_field(scene, band_frac=0.12, bevel_slope=0.7, dome_slope=0.0):
    """(h, d, widths): classical raised-relief height per pixel in pixel units, the distance to the nearest border, and the bevel
    width in pixels per panel. h = a ramp of slope `bevel_slope` across the band (plus an optional shallow dome). The sunk relief is
    -h; it is never built, it is the parity operator's image of this one."""
    h = np.zeros((scene.H, scene.W))
    dist = np.zeros((scene.H, scene.W))
    widths = {}
    for k in range(scene.n_panels):
        fm = scene.frame & (scene.region == k)
        if not fm.any():
            continue
        ys, xs = np.nonzero(scene.region == k)
        w = max(3.0, band_frac * float(xs.max() - xs.min() + 1))
        d = ndi.distance_transform_edt(fm)
        h[fm] = (bevel_slope * np.minimum(d, w) + dome_slope * d)[fm]
        dist[fm] = d[fm]
        widths[k] = w
    return h, dist, widths


def slope_field(scene, h, sigma=1.5):
    """(sx, sy) = -grad of the lightly smoothed height field; smoothing never mixes panels with glass or each other.

    The gradient is taken per panel, with the height field zero outside it (a border is where the ramp starts). Differencing across the
    summed field of all panels, as an earlier version did, halved the slope on the shared edge of two panels, so the seam pixels were
    classed flat and joined the interior facets: band facets of neighbouring panels then never touched, and the only cross-panel edges
    in the facet graph connected two untilted plateaus."""
    sx, sy = np.zeros_like(h), np.zeros_like(h)
    for k in range(scene.n_panels):
        fm = (scene.frame & (scene.region == k))
        if not fm.any():
            continue
        fmf = fm.astype(float)
        hk = np.where(fm, gauss(h * fmf, sigma) / np.maximum(gauss(fmf, sigma), 1e-6), 0.0)
        hy, hx = np.gradient(hk)
        sx[fm], sy[fm] = -hx[fm], -hy[fm]
    return sx, sy


# ------------------------------------------------------------------ clustering
def _kmeans_dirs(vecs, m, iters=16):
    """k-means on unit vectors (rows) with farthest-point seeding; returns labels 0..m-1. Deterministic."""
    n = len(vecs)
    if m <= 1 or n == 0:
        return np.zeros(n, int)
    sub = vecs if n <= 8000 else vecs[np.linspace(0, n - 1, 8000).astype(int)]
    seeds = [sub[0]]
    for _ in range(m - 1):
        d = np.min([((sub - s) ** 2).sum(1) for s in seeds], axis=0)
        if d.max() < 1e-9:
            break
        seeds.append(sub[np.argmax(d)])
    cent = np.array(seeds, float)
    for _ in range(iters):
        lab = np.argmin(((sub[:, None, :] - cent[None]) ** 2).sum(2), axis=1)
        for j in range(len(cent)):
            sel = sub[lab == j]
            if len(sel):
                c = sel.mean(axis=0)
                nrm = np.linalg.norm(c)
                cent[j] = c / nrm if nrm > 1e-9 else cent[j]
    out = np.empty(n, int)
    for a in range(0, n, 200_000):
        out[a:a + 200_000] = np.argmin(((vecs[a:a + 200_000, None, :] - cent[None]) ** 2).sum(2), axis=1)
    return out


def build_facets(scene, n_dirs=None, band_frac=0.12, bevel_slope=0.7, dome_slope=0.0, flat_frac=0.5, kappa=1.0, min_pixels=40, interior="drop"):
    """Cut every panel into n_dirs orientation facets (the sloping surface) and one flat facet, and give each a tilt.

    interior="drop" (default): the flat plateau of a panel has tilt ~0, so its qubit would be a constant |0> that nothing couples to
    (a dome does not help: the mean slope of a symmetric plateau cancels). It is kept as a *classical* facet, ordered after the quantum
    ones, lit by the deterministic Lambert value of the flat panel. interior="keep" gives it a qubit as before.

    A pixel is 'sloping' when |slope| >= flat_frac * the steepest slope in its panel; sloping pixels are clustered by slope
    direction into n_dirs classes. Every projectable pixel ends up in exactly one facet; pieces smaller than min_pixels are merged
    into the nearest facet of the same panel. n_dirs=None (default) picks auto_n_dirs(scene.n_panels): 4 for three panels, 2 for the six panes of the
    built-in bay window, 1 for eight. With n_dirs = 2 the bay window has 12 band facets (qubits) and 6 classical plateau facets.
    """
    if n_dirs is None:
        n_dirs = auto_n_dirs(scene.n_panels)
    h, dist, widths = height_field(scene, band_frac, bevel_slope, dome_slope)
    sx, sy = slope_field(scene, h)
    mag = np.hypot(sx, sy)
    label = -np.ones((scene.H, scene.W), int)
    kinds, panels = [], []
    for k in range(scene.n_panels):
        fm = scene.frame & (scene.region == k)
        if not fm.any():
            continue
        steep = fm & (mag >= flat_frac * mag[fm].max())
        flat = fm & ~steep
        if steep.any():
            yx = np.argwhere(steep)
            v = np.stack([sx[steep], sy[steep]], axis=1)
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            cls = _kmeans_dirs(v, n_dirs)
            for c in range(cls.max() + 1):
                sel = yx[cls == c]
                if len(sel) >= min_pixels:
                    label[sel[:, 0], sel[:, 1]] = len(kinds)
                    kinds.append("band"), panels.append(k)
        if flat.sum() >= min_pixels:
            label[flat] = len(kinds)
            kinds.append("interior"), panels.append(k)
    for k in range(scene.n_panels):                                   # stragglers join the nearest facet of their panel
        fm = scene.frame & (scene.region == k)
        todo, have = fm & (label < 0), fm & (label >= 0)
        if todo.any() and have.any():
            near = ndi.distance_transform_edt(~have, return_distances=False, return_indices=True)
            label[todo] = label[near[0][todo], near[1][todo]]
    if interior == "drop":                                            # classical plateau facets go last, so qubit i is facet i
        order = sorted(range(len(kinds)), key=lambda i: (kinds[i] == "interior", i))
        new = np.empty(len(kinds), int)
        new[order] = np.arange(len(kinds))
        label = np.where(label >= 0, new[np.maximum(label, 0)], -1)
        kinds, panels = [kinds[i] for i in order], [panels[i] for i in order]
    elif interior != "keep":
        raise ValueError("interior must be 'drop' or 'keep'")
    facets = []
    for i, (kind, k) in enumerate(zip(kinds, panels)):
        m = label == i
        ys, xs = np.nonzero(m)
        e = ndi.distance_transform_edt(m)
        ay, ax = np.unravel_index(int(np.argmax(e)), e.shape)
        facets.append(Facet(i, k, kind, (float(xs.mean()), float(ys.mean())), (int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1),
                            int(m.sum()), (float(sx[m].mean()), float(sy[m].mean())), anchor=(float(ax), float(ay)),
                            classical=bool(interior == "drop" and kind == "interior")))
    fs = FacetSet(facets, label, [], h, kappa, dict(n_dirs=n_dirs, band_frac=band_frac, bevel_slope=bevel_slope, dome_slope=dome_slope,
                                                    flat_frac=flat_frac, interior=interior))
    fs.edges = facet_edges(scene, fs)
    set_tilt_budget(fs, kappa)
    return fs


def facet_edges(scene, fs, min_len=3):
    """Pairs of facets whose pixels touch, with the length of the shared boundary. Facets of one plane are 'coplanar' (their
    shading should agree), facets of different planes meet at a 'crease'."""
    lab = fs.label
    pairs = {}
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        sel = (a >= 0) & (b >= 0) & (a != b)
        lo, hi = np.minimum(a[sel], b[sel]), np.maximum(a[sel], b[sel])
        keys, counts = np.unique(lo * 100_000 + hi, return_counts=True)
        for kk, c in zip(keys, counts):
            pairs[(int(kk // 100_000), int(kk % 100_000))] = pairs.get((int(kk // 100_000), int(kk % 100_000)), 0) + int(c)
    plane = [scene.panels[f.panel].plane_id for f in fs.facets]
    return [(i, j, "coplanar" if plane[i] == plane[j] else "crease", n) for (i, j), n in sorted(pairs.items()) if n >= min_len]


def set_tilt_budget(fs, kappa=1.0):
    """tau_i = arctan |mean slope|, rescaled per panel so that sum(tau_i^2) = kappa^2 (each capped at TAU_MAX). The azimuth phi_i is the
    direction of the mean slope. A panel whose facets are all flat stays flat (nothing to superpose)."""
    fs.kappa = float(kappa)
    for k in sorted({f.panel for f in fs.facets}):
        mine = [f for f in fs.facets if f.panel == k and not f.classical]
        for f in fs.facets:
            if f.panel == k and f.classical:
                f.tau, f.phi = 0.0, 0.0
        raw = np.array([np.arctan(np.hypot(*f.slope)) for f in mine])
        norm = np.sqrt((raw ** 2).sum())
        scale = kappa / norm if norm > 1e-9 else 0.0
        for f, r in zip(mine, raw):
            f.tau = float(min(r * scale, TAU_MAX))
            f.phi = float(np.arctan2(f.slope[1], f.slope[0])) if np.hypot(*f.slope) > 1e-12 else 0.0
    return fs


def panel_budget(fs):
    """{panel: dict(n, sum_tau2, visibility)}: the tilt budget each panel uses and the interference visibility it buys."""
    out = {}
    for k in sorted({f.panel for f in fs.facets}):
        t = np.array([f.tau for f in fs.facets if f.panel == k])
        out[k] = dict(n=len(t), sum_tau2=float((t ** 2).sum()), visibility=float(np.prod(np.cos(t))))
    return out
