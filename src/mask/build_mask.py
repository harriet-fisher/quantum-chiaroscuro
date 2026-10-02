"""Rasterise labelled polygons into region / impenetrable masks (handoff §2.4, §7.3).

Convention used everywhere downstream:
  region[y, x]   = index of the panel covering the pixel, -1 outside every panel
  blocked        = glass / off-limits pixels (never projected, never given qubits)
  frame          = projectable pixels = inside a panel and not blocked
  impenetrable   = ~frame  (glass + off-limits + outside): pure black, no qubits
"""
import numpy as np
from PIL import Image, ImageDraw


def poly_mask(poly, size):
    W, H = size
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in poly], fill=255)
    return np.array(im) > 0


def scale_poly(poly, f):
    """Scale a polygon about the mean of its vertices."""
    c = np.mean(np.array(poly, float), axis=0)
    return [tuple(c + f * (np.array(p, float) - c)) for p in poly]


def build_masks(size, panel_polys, blocked_polys):
    """Return (region, blocked, frame, impenetrable). Later panels overwrite earlier ones where they overlap."""
    W, H = size
    region = -np.ones((H, W), int)
    blocked = np.zeros((H, W), bool)
    for k, poly in enumerate(panel_polys):
        region[poly_mask(poly, size)] = k
    for poly in blocked_polys:
        blocked |= poly_mask(poly, size)
    frame = (region >= 0) & ~blocked
    return region, blocked, frame, ~frame
