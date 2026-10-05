"""Relief profile and directional bevel (handoff §9.3).

Height field h = polarity * smoothed relief; shade = n.l with n = (-hx, -hy, 1). Flipping polarity flips the sign
of h, so the same light reads as hollow or raised (the crater illusion).
"""
import numpy as np

from src.mask.masked_blur import gauss


def polarity_map(scene, pol):
    if len(pol) != scene.n_panels:
        raise ValueError(f"polarity has {len(pol)} entries but the scene has {scene.n_panels} panels")
    pol_map = np.zeros((scene.H, scene.W))
    for k in range(scene.n_panels):
        pol_map[scene.region == k] = pol[k]
    return pol_map


def relief_profile(pol_map, R):
    """sunk = centre dark / edges bright; raised = opposite."""
    return np.where(pol_map < 0, 1 - R, R)


def directional_bevel(pol_map, R, lamp, gain=40.0, smooth_R=None):
    """smooth_R = gauss(R, 4), which depends only on the scene; texture.fast passes it in so it is computed once."""
    hs = pol_map * (gauss(R, 4) if smooth_R is None else smooth_R)
    hy, hx = np.gradient(hs)
    lx = lamp[0]                              # +1 light from right, -1 from left
    ly = -0.5                                 # slightly from above (image y points down)
    strength = 1.0 if lamp[1] < 0 else 0.5    # grazing light = stronger bevel
    return np.clip(0.5 + gain * strength * (-hx * lx - hy * ly), 0, 1)
