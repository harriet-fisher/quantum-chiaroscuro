"""Per-patch lit fractions -> smooth per-panel field, normalised so nothing leaks across creases or into glass."""
import numpy as np
from scipy import ndimage as ndi

from src.mask.masked_blur import gauss


def interpolate_lit(scene, cells, lit, interp_sigma):
    S = np.zeros((scene.H, scene.W))
    for k in range(scene.n_panels):
        fm = scene.frame & (scene.region == k)
        Pk = np.zeros((scene.H, scene.W))
        valid = np.zeros((scene.H, scene.W), bool)
        for c, v in zip(cells, lit):
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
    return S
