"""Relief field R: normalised distance to the nearest border (crease, outer edge or glass edge), per panel."""
import numpy as np
from scipy import ndimage as ndi


def relief_field(region, frame, n_panels):
    R = np.zeros(region.shape)
    for k in range(n_panels):
        fm = frame & (region == k)
        if not fm.any():
            continue
        d = ndi.distance_transform_edt(fm)
        R[fm] = (d / d.max())[fm]
    return R
