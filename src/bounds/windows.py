"""Brightness windows [lo, hi] per plane and per depth (handoff §2.3, mock parameters in §2.6)."""
import numpy as np

WIN_CENTRE = (0.40, 0.95)
WIN_SIDE_NEAR = (0.25, 0.80)
WIN_SIDE_FAR = (0.10, 0.65)

FLAT_ANGLE_DEG = 0.5   # panels with |angle| below this count as facing the viewer


def default_window(angle_deg, polygon):
    """Default window fields for a panel, used by the pen tool when the user has not set any.

    Flat panel   -> one constant window (the mock's centre panel).
    Angled panel -> near/far windows; depth runs along x from the outer edge (near) to the inner edge (far).
                    A normal tilted toward +x (angle > 0) has its outer end on the left.
    """
    if abs(angle_deg) < FLAT_ANGLE_DEG:
        return dict(window_near=list(WIN_CENTRE), window_far=None, depth_x=None)
    xs = [float(p[0]) for p in polygon]
    x_min, x_max = min(xs), max(xs)
    depth_x = [x_min, x_max] if angle_deg > 0 else [x_max, x_min]
    return dict(window_near=list(WIN_SIDE_NEAR), window_far=list(WIN_SIDE_FAR), depth_x=depth_x)


def build_windows(region, t, panels):
    """Per-pixel lo / hi arrays from each panel's window_near / window_far and the depth coordinate t."""
    lo = np.zeros(region.shape)
    hi = np.zeros(region.shape)
    for p in panels:
        m = region == p.idx
        if p.window_far is None:
            lo[m], hi[m] = p.window_near
        else:
            lo[m] = p.window_near[0] + t[m] * (p.window_far[0] - p.window_near[0])
            hi[m] = p.window_near[1] + t[m] * (p.window_far[1] - p.window_near[1])
    return lo, hi
