"""Depth coordinate t (0 = near/outer end, 1 = far/inner end) per panel pixel."""
import numpy as np


def depth_coordinate(region, panels):
    H, W = region.shape
    _, xx = np.mgrid[0:H, 0:W]
    t = np.full((H, W), 0.5)
    for p in panels:
        if p.depth_x is None:
            continue
        near, far = p.depth_x
        m = region == p.idx
        t[m] = ((xx - near) / (far - near))[m]
    return np.clip(t, 0, 1)
