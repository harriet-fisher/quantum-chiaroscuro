"""Edges of the patch graph and of the polarity chain (handoff §7.3 steps 4-5, §9.1)."""
import numpy as np


def patch_couplings(cells, idx, J_in=0.9, J_cross=0.6):
    """Symmetric coupling matrix over 4-neighbour cells: coplanar agree (+J_in), crease disagree (-J_cross)."""
    n = len(cells)
    J = np.zeros((n, n))
    for (i, j), a in idx.items():
        for di, dj in ((1, 0), (0, 1)):
            b = idx.get((i + di, j + dj))
            if b is None:
                continue
            same = cells[a]["plane_id"] == cells[b]["plane_id"]
            J[a, b] = J[b, a] = J_in if same else -J_cross
    return J


def patch_edges(J):
    """Undirected patch-patch edges (a < b) where J is non-zero."""
    a, b = np.nonzero(np.triu(J))
    return [(int(x), int(y)) for x, y in zip(a, b)]


def polarity_edges(scene, touch_px=6):
    """Chain of polarity qubits: one edge between each pair of panels that touch."""
    return scene.panel_adjacency(touch_px)
