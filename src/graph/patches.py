"""Patch grid: one qubit per grid cell that is projectable enough (handoff §7.3 step 3)."""
import numpy as np


def build_patches(scene, NX, NY, min_cover=0.30):
    """Cells with at least min_cover projectable pixels become patches; cells that are mostly glass or
    outside get no qubit. A patch's panel k is the majority panel among its projectable pixels."""
    W, H = scene.W, scene.H
    cw, ch = W / NX, H / NY
    cells, idx = [], {}
    for j in range(NY):
        for i in range(NX):
            x0, x1, y0, y1 = int(i * cw), int((i + 1) * cw), int(j * ch), int((j + 1) * ch)
            f = scene.frame[y0:y1, x0:x1]
            if f.mean() < min_cover:
                continue                      # mostly glass / outside: no qubit
            reg = scene.region[y0:y1, x0:x1][f]
            k = int(np.bincount(reg, minlength=scene.n_panels).argmax())
            idx[(i, j)] = len(cells)
            cells.append(dict(i=i, j=j, box=(x0, x1, y0, y1), k=k, plane_id=scene.panels[k].plane_id))
    return cells, idx
