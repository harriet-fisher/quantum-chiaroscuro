"""Depth domains: the spatial, graph-shaped version of the facet set (the "domain relief").

The per-panel relief (geometry/facets.py) has one polarity qubit per panel and orientation-class facets, so each panel is its own tiny
5-qubit system and the facet graph's only long-range structure is a handful of couplings. Here the wall is cut into *local* pieces:

  facet   a piece of the bevel strip that runs along one polygon edge (a panel's outer edge, or the edge of a glass pane): a few hundred
          pixels with one slope. Its qubit's Bloch vector is that slope, exactly as before.
  domain  `group_size` consecutive facets around one loop (an outer boundary or a glass boundary), sharing ONE polarity qubit, so one depth
          decision covers a small stretch of bevel. The tilt budget is per domain: sum tau^2 = group_size * tau^2, so the interference
          visibility V_d = prod cos(tau_i) stays put (about 0.6 for four facets at tau = 0.5) however many domains the wall has.
  graph   domains are nodes; two domains are joined when their facets touch. Along a boundary and within a plane the edge is coplanar,
          across a shared edge between panels it is a crease. The polarity qubits are coupled along this graph (Ising), so entanglement runs
          across the geometric edges, including across the creases between panels.

Plateaus (tilt ~0) stay classical: one facet per panel with no qubit.

The crater gauge. A raised bevel facing right lit from the right looks like a sunk bevel facing left lit from the left, so what the eye sees
is (depth sign) x (which way the bevel faces) x (which side the light is on). `lock_sign` is the sign of a domain's mean facing (the x component
of its tilt direction): coupling the lamp to a domain's polarity with that sign makes a pane's ring of bevels, which face both ways, inherently
frustrated under one shared light (see frustration()).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from src.geometry.facets import TAU_MAX, Facet, FacetSet, facet_edges, height_field, slope_field


@dataclass
class DomainSet:
    facets: FacetSet
    n_domains: int
    domain_of: np.ndarray                       # (n_quantum,) polarity qubit of each quantum facet
    domain_panel: list                          # panel index of each domain
    domain_loop: list                           # ("outer" | "glass", panel) of each domain
    domain_centroid: list                       # (x, y) canvas pixels
    domain_facing: np.ndarray                   # mean tilt direction (radians) of each domain
    edges: list                                 # [(a, b, kind, touching_px)] kind "coplanar" | "crease"
    lock_sign: np.ndarray                       # (n_domains,) -1, 0 or +1: the crater gauge of the domain
    params: dict = field(default_factory=dict)

    @property
    def n_quantum(self):
        return self.facets.n_quantum

    @property
    def n_sim(self):
        return self.n_quantum + self.n_domains

    def facets_of(self, d):
        return [int(i) for i in np.nonzero(self.domain_of == d)[0]]

    def visibility(self, d=None):
        if d is None:
            return np.array([self.visibility(k) for k in range(self.n_domains)])
        return float(np.prod(np.cos(self.facets.qtau[self.domain_of == d])))

    def sweep_order(self):
        """Domains sorted along the longer canvas axis: a good MPS site order, since neighbours stay close."""
        c = np.array(self.domain_centroid)
        axis = 0 if np.ptp(c[:, 0]) >= np.ptp(c[:, 1]) else 1
        return [int(i) for i in np.lexsort((c[:, 1 - axis], c[:, axis]))]


def _polygon_edges(scene):
    """{panel: [(p0, p1, loop_id)]} for the outer polygon of every panel and the polygon of every glass/off-limits shape inside it."""
    from src.geometry.planes import _canvas_polys
    _, _, blocked = _canvas_polys(scene.labels, (scene.W, scene.H))
    out = {}
    for k, p in enumerate(scene.panels):
        pts = [tuple(map(float, q)) for q in p.polygon]
        out[k] = [(pts[i], pts[(i + 1) % len(pts)], ("outer", k, 0)) for i in range(len(pts))]
    for gi, poly in enumerate(blocked):
        c = np.mean(np.asarray(poly, float), axis=0)
        x, y = int(np.clip(round(c[0]), 0, scene.W - 1)), int(np.clip(round(c[1]), 0, scene.H - 1))
        k = int(scene.region[y, x])
        if k < 0:
            continue
        pts = [tuple(map(float, q)) for q in poly]
        out[k] += [(pts[i], pts[(i + 1) % len(pts)], ("glass", k, gi + 1)) for i in range(len(pts))]
    return out


def _segment_distance(px, py, p0, p1):
    """Distance from points to the segment p0-p1 and the arc length of the closest point from p0."""
    d = np.array(p1) - np.array(p0)
    L = float(np.hypot(*d))
    t = np.clip(((px - p0[0]) * d[0] + (py - p0[1]) * d[1]) / max(L * L, 1e-12), 0, 1)
    cx, cy = p0[0] + t * d[0], p0[1] + t * d[1]
    return np.hypot(px - cx, py - cy), t * L, L


def build_domains(scene, seg_len=220.0, group_size=4, tau=0.5, band_frac=0.12, bevel_slope=0.7, flat_frac=0.5, min_pixels=60,
                  touch_px=3, crease_sign=-1.0):
    """Cut every panel into bevel-strip facets of about `seg_len` pixels along each polygon edge, group `group_size` consecutive facets of a
    loop into a depth domain, and give every facet a tilt (every domain's budget sum tau^2 = group_size * tau^2).

    touch_px: a domain pair is an edge of the domain graph if their facets share at least this many boundary pixels; neighbours around a loop are
    always joined.
    crease_sign: sign of the polarity coupling across a crease (-1: the depths want to differ across a shared edge, so domain walls sit on the
    creases; +1: the surface wants to read as one object)."""
    h, _, _ = height_field(scene, band_frac, bevel_slope, 0.0)
    sx, sy = slope_field(scene, h)
    mag = np.hypot(sx, sy)
    poly = _polygon_edges(scene)
    label = -np.ones((scene.H, scene.W), int)
    pieces = []                                                       # dicts: panel, loop, edge order, piece, pixel coordinates
    for k in range(scene.n_panels):
        fm = scene.frame & (scene.region == k)
        if not fm.any():
            continue
        steep = fm & (mag >= flat_frac * mag[fm].max())
        yx = np.argwhere(steep)
        if not len(yx):
            continue
        px, py = yx[:, 1] + 0.0, yx[:, 0] + 0.0
        edges = poly[k]
        D = np.empty((len(edges), len(yx)))
        ARC = np.empty_like(D)
        LEN = np.empty(len(edges))
        for e, (p0, p1, _) in enumerate(edges):
            D[e], ARC[e], LEN[e] = _segment_distance(px, py, p0, p1)
        best = np.argmin(D, axis=0)
        arc = ARC[best, np.arange(len(yx))]
        m_e = np.maximum(1, np.round(LEN / seg_len)).astype(int)
        piece = np.minimum((arc / (LEN[best] / m_e[best])).astype(int), m_e[best] - 1)
        for e in range(len(edges)):
            for q in range(m_e[e]):
                sel = (best == e) & (piece == q)
                if sel.sum():
                    pieces.append(dict(panel=k, loop=edges[e][2], edge=e, piece=q, yx=yx[sel]))
    # tiny pieces (a sliver at a corner) are absorbed by the nearest big piece of the same panel
    big = [p for p in pieces if len(p["yx"]) >= min_pixels]
    for p in pieces:
        if len(p["yx"]) < min_pixels:
            cands = [b for b in big if b["panel"] == p["panel"]]
            if cands:
                c = p["yx"].mean(axis=0)
                tgt = min(cands, key=lambda b: float(((b["yx"].mean(axis=0) - c) ** 2).sum()))
                tgt["yx"] = np.vstack([tgt["yx"], p["yx"]])
    pieces = big
    # order pieces around each loop, then chunk each loop into domains
    pieces.sort(key=lambda p: (p["panel"], p["loop"][0] != "outer", p["loop"][2], p["edge"], p["piece"]))
    loops = {}
    for i, p in enumerate(pieces):
        loops.setdefault((p["panel"], p["loop"]), []).append(i)
    domain_of_piece, dom_panel, dom_loop = {}, [], []
    for (k, loop), idxs in loops.items():
        nd = max(1, int(round(len(idxs) / group_size)))
        for chunk in np.array_split(np.array(idxs), nd):
            for i in chunk:
                domain_of_piece[int(i)] = len(dom_panel)
            dom_panel.append(k)
            dom_loop.append(loop[:1] + (k,))
    for i, p in enumerate(pieces):
        label[p["yx"][:, 0], p["yx"][:, 1]] = i
    # plateaus: one classical facet per panel, then any projectable pixel without a facet joins the nearest facet of its panel
    kinds = ["band"] * len(pieces)
    panels = [p["panel"] for p in pieces]
    for k in range(scene.n_panels):
        fm = scene.frame & (scene.region == k)
        flat = fm & (label < 0)
        if flat.sum() >= min_pixels:
            label[flat] = len(kinds)
            kinds.append("interior")
            panels.append(k)
    for k in range(scene.n_panels):
        fm = scene.frame & (scene.region == k)
        todo, have = fm & (label < 0), fm & (label >= 0)
        if todo.any() and have.any():
            near = ndi.distance_transform_edt(~have, return_distances=False, return_indices=True)
            label[todo] = label[near[0][todo], near[1][todo]]
    facets = []
    for i, (kind, k) in enumerate(zip(kinds, panels)):
        m = label == i
        ys, xs = np.nonzero(m)
        e = ndi.distance_transform_edt(m)
        ay, ax = np.unravel_index(int(np.argmax(e)), e.shape)
        f = Facet(i, k, kind, (float(xs.mean()), float(ys.mean())), (int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1), int(m.sum()),
                  (float(sx[m].mean()), float(sy[m].mean())), anchor=(float(ax), float(ay)), classical=(kind == "interior"))
        if kind == "band":
            f.domain, f.edge = domain_of_piece[i], pieces[i]["edge"]
        facets.append(f)
    fs = FacetSet(facets, label, [], h, 1.0, dict(seg_len=seg_len, group_size=group_size, tau=tau, band_frac=band_frac, bevel_slope=bevel_slope,
                                                  flat_frac=flat_frac, interior="drop", domains=True))
    fs.edges = facet_edges(scene, fs)
    nq = fs.n_quantum
    assert all(not f.classical for f in facets[:nq])                  # band pieces were created first, plateaus last
    nd = len(dom_panel)
    domain_of = np.array([f.domain for f in facets[:nq]], int)
    # tilts: per domain, the raw tilts are rescaled to sum tau_i^2 = n_d tau^2
    for d in range(nd):
        mine = [f for f in facets[:nq] if f.domain == d]
        raw = np.array([np.arctan(np.hypot(*f.slope)) for f in mine])
        norm = float(np.sqrt((raw ** 2).sum()))
        scale = tau * np.sqrt(len(mine)) / norm if norm > 1e-9 else 0.0
        for f, r in zip(mine, raw):
            f.tau = float(min(r * scale, TAU_MAX))
            f.phi = float(np.arctan2(f.slope[1], f.slope[0])) if np.hypot(*f.slope) > 1e-12 else 0.0
    for f in facets[nq:]:
        f.tau = f.phi = 0.0
    plane = [p.plane_id for p in scene.panels]
    acc = {}
    for i, j, kind, n in fs.edges:
        if i >= nq or j >= nq or domain_of[i] == domain_of[j]:
            continue
        a, b = sorted((int(domain_of[i]), int(domain_of[j])))
        acc[(a, b)] = acc.get((a, b), 0) + int(n)
    dedges = {(a, b): ("coplanar" if plane[dom_panel[a]] == plane[dom_panel[b]] else "crease", n) for (a, b), n in sorted(acc.items()) if n >= touch_px}
    by_loop = {}
    for d in range(nd):
        by_loop.setdefault(dom_loop[d], []).append(d)
    for ds_ in by_loop.values():                                      # neighbours around one loop are always joined, even if the strips only meet at a corner
        for a, b in zip(ds_, ds_[1:] + ds_[:1]):
            if a != b:
                dedges.setdefault((min(a, b), max(a, b)), ("coplanar", 0))
    dedges = [(a, b, kind, n) for (a, b), (kind, n) in sorted(dedges.items())]
    cen, facing, lock = [], np.zeros(nd), np.zeros(nd)
    for d in range(nd):
        mine = [f for f in facets[:nq] if f.domain == d]
        w = np.array([f.area for f in mine], float)
        cen.append((float(np.average([f.centroid[0] for f in mine], weights=w)), float(np.average([f.centroid[1] for f in mine], weights=w))))
        v = np.array([[f.tau * np.cos(f.phi), f.tau * np.sin(f.phi)] for f in mine]).sum(axis=0)
        facing[d] = float(np.arctan2(v[1], v[0]))
        lock[d] = 0.0 if abs(v[0]) < 0.35 * max(sum(f.tau for f in mine), 1e-9) * 0.5 else float(np.sign(v[0]))
    return DomainSet(fs, nd, domain_of, dom_panel, dom_loop, cen, facing, dedges, lock,
                     dict(seg_len=seg_len, group_size=group_size, tau=tau, crease_sign=crease_sign, touch_px=touch_px))


def frustration(ds, lock=False, locked=None):
    """Frustration of the signed domain graph (coplanar +, crease crease_sign), optionally with the lamp as one more node joined to every
    domain by its lock sign. Returns dict(nodes, edges, frustrated_cycles, cycle_basis): the number of cycles of a fundamental cycle basis whose
    edge signs multiply to -1 (no assignment of +/-1 to the nodes can satisfy every edge of such a cycle: an Ising state cannot be fully
    ferro/antiferro-consistent, so the ground state is degenerate or entangled). Not the frustration index, which is NP-hard.
    locked: only these domains are joined to the lamp (default: every domain with a nonzero lock sign)."""
    cs = ds.params["crease_sign"]
    edges = [(a, b, 1.0 if kind == "coplanar" else cs) for a, b, kind, _ in ds.edges]
    n = ds.n_domains
    if lock:
        for d in range(n):
            if ds.lock_sign[d] != 0 and (locked is None or d in locked):
                edges.append((d, n, float(ds.lock_sign[d])))
        n += 1
    adj = {i: [] for i in range(n)}
    for a, b, s in edges:
        adj[a].append((b, s))
        adj[b].append((a, s))
    gauge, seen = {}, set()
    tree = set()
    for root in range(n):
        if root in gauge:
            continue
        gauge[root] = 1.0
        stack = [root]
        while stack:
            u = stack.pop()
            for v, s in adj[u]:
                if v not in gauge:
                    gauge[v] = gauge[u] * s
                    tree.add((min(u, v), max(u, v)))
                    stack.append(v)
    nontree = [(a, b, s) for a, b, s in edges if (min(a, b), max(a, b)) not in tree]
    frustrated = sum(1 for a, b, s in nontree if gauge[a] * gauge[b] * s < 0)
    return dict(nodes=n, edges=len(edges), cycle_basis=len(nontree), frustrated_cycles=int(frustrated))
