"""Sanity checks on a target table before it is sent anywhere (handoff §8 'Targets' stage: Bloch length <= 1).

Z-diagonal targets must also satisfy the triangle (cut-polytope) inequalities for every triangle a, b, c of the
coupling map, because they come from +/-1 variables:
    x + y + z >= -1,   x - y - z >= -1,   -x + y - z >= -1,   -x - y + z >= -1      (x = <ab>, y = <bc>, z = <ac>)
Estimates from a sampler satisfy them up to Monte-Carlo error; hand-designed targets may not.
"""


def scene_problem(meta, cells, grid=None, n_panels=None):
    """None when a calibration (targets.json 'meta') was made for exactly this patch grid, else a sentence saying how it differs. `cells` are a
    Sampler's patch cells (i, j, k = panel); `grid` (nx, ny) and `n_panels` only make the sentence more helpful. The payloads are built from the
    targets, so sending them for another scene would pay for the wrong state; files without `patch_cells` (older calibrations) cannot be checked and pass."""
    cells_meta = meta.get("patch_cells")
    if cells_meta is None:
        return None
    theirs = [(c["i"], c["j"], c["panel"]) for c in cells_meta]
    mine = [(c["i"], c["j"], c["k"]) for c in cells]
    if theirs == mine:
        return None
    g = meta.get("grid") or ["?", "?"]
    there = f"{len(theirs)} patches on a {g[0]}x{g[1]} grid, {1 + max((t[2] for t in theirs), default=-1)} panels"
    here = f"{len(mine)} patches" + (f" on a {grid[0]}x{grid[1]} grid" if grid else "") + (f", {n_panels} panels" if n_panels else "")
    return f"the targets were made for a different scene ({there}) than this one ({here})"


def check_bounds(bloch, relationships, n_sigma=3.0):
    """Each |<Z>| and |<ZZ>| must be <= 1 (allowing n_sigma standard errors)."""
    bad = []
    for b in bloch:
        if abs(b["Z"]) > 1 + n_sigma * b.get("se", 0):
            bad.append(f"|<Z>| > 1 on qubit {b['qubit']}: {b['Z']:.4f}")
    for r in relationships:
        if abs(r["ZZ"]) > 1 + n_sigma * r.get("se", 0):
            bad.append(f"|<ZZ>| > 1 on {r['qubits']}: {r['ZZ']:.4f}")
    return bad


def triangles(edges):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    out = []
    for a, b in edges:
        for c in adj[a] & adj[b]:
            if c > b:
                out.append((a, b, c))
    return sorted(set(out))


def check_triangles(relationships, n_sigma=3.0):
    """Returns dict(n_triangles, violations=[...], min_slack) over triangles whose three ZZ targets are all present."""
    val = {tuple(r["qubits"]): (r["ZZ"], r.get("se", 0.0)) for r in relationships}
    edges = sorted(val)
    violations, min_slack, n = [], float("inf"), 0
    for a, b, c in triangles(edges):
        (x, sx), (y, sy), (z, sz) = val[(a, b)], val[(b, c)], val[(a, c)]
        tol = n_sigma * (sx ** 2 + sy ** 2 + sz ** 2) ** 0.5
        n += 1
        for sgn, name in [((1, 1, 1), "x+y+z"), ((1, -1, -1), "x-y-z"), ((-1, 1, -1), "-x+y-z"), ((-1, -1, 1), "-x-y+z")]:
            slack = sgn[0] * x + sgn[1] * y + sgn[2] * z + 1
            min_slack = min(min_slack, slack)
            if slack < -tol:
                violations.append(f"triangle {(a, b, c)}: {name} >= -1 violated by {-slack:.4f} (tolerance {tol:.4f})")
    return dict(n_triangles=n, violations=violations, min_slack=None if n == 0 else min_slack)


def near_pinned(relationships, threshold=0.9):
    """Targets so close to +/-1 that they leave little 'random within bounds' (spec §3.5), and that an engine may relax
    silently (spec §12.3). Returns the offending rows' labels, strongest first."""
    rows = sorted((r for r in relationships if abs(r["ZZ"]) > threshold), key=lambda r: -abs(r["ZZ"]))
    return dict(threshold=threshold, count=len(rows), of=len(relationships), labels=[r["label"] for r in rows])


def run_all(bloch, relationships):
    bounds = check_bounds(bloch, relationships)
    tri = check_triangles(relationships)
    return dict(ok=not bounds and not tri["violations"], bound_violations=bounds, near_pinned=near_pinned(relationships), **tri)
