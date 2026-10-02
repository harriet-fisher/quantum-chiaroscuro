"""Test patterns for calibration and the photographed black-glass check (handoff §10.1 step 5, §11.10-11.11, §13).

Every pattern is a uint8 (h, w) array in PROJECTOR pixels, so it goes through exactly the same path to the wall as a scene
frame. Two kinds, because only one of them is bound by the glass rule:

  scene patterns        "black", "white" (full light on every projectable pixel, black on glass and outside): obey the
                        hard rule and are what the black-glass photograph is taken of.
  calibration patterns  "outline", "align", "dots", "grid", "edge": draw lines and dots wherever they need to be (the
                        dots deliberately fall anywhere), so they are NOT scene frames and are never glass-checked.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.projector.calibrate import apply_homography, dot_positions, BIG_DOT_RATIO

SCENE_PATTERNS = ("black", "white")
CALIBRATION_PATTERNS = ("outline", "align", "dots", "grid", "edge")
ALL = SCENE_PATTERNS + CALIBRATION_PATTERNS
DOC = {
    "black": "projector black: the ambient + black-level reference photo",
    "white": "full light on every projectable pixel, glass black: the photographed black-glass test",
    "outline": "panel outlines (solid) and glass outlines (dashed) through the calibration: do they land on the real edges?",
    "align": "outline + the four numbered landmark markers to drag onto their physical features (corner calibration)",
    "dots": "nine dots, top-left one larger: photograph it (and 'black') to calibrate a camera or check the photographed test",
    "grid": "uniform grid in projector pixels: shows keystone and lens distortion on the surface",
    "edge": "the edge of the projector frame: shows where the frame lands on the object",
}


def _canvas(size):
    return Image.new("L", (int(size[0]), int(size[1])), 0)


def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:                                              # old Pillow
        return ImageFont.load_default()


def black(size):
    return np.zeros((int(size[1]), int(size[0])), np.uint8)


def white(warper):
    return warper.white()


def dots(size, level=255):
    w, h = size
    im = _canvas(size)
    d = ImageDraw.Draw(im)
    r = max(4.0, 0.012 * min(w, h))
    for i, (x, y) in enumerate(dot_positions(size)):
        rr = r * (BIG_DOT_RATIO if i == 0 else 1.0)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=level)
    return np.array(im)


def grid(size, step_frac=0.1, level=200):
    w, h = size
    im = _canvas(size)
    d = ImageDraw.Draw(im)
    lw = max(2, round(min(w, h) / 360))
    for f in np.arange(0, 1.0001, step_frac):
        x, y = f * (w - 1), f * (h - 1)
        d.line([(x, 0), (x, h - 1)], fill=level, width=lw)
        d.line([(0, y), (w - 1, y)], fill=level, width=lw)
    return np.array(im)


def edge(size, level=255):
    w, h = size
    im = _canvas(size)
    lw = max(3, round(min(w, h) / 200))
    ImageDraw.Draw(im).rectangle([0, 0, w - 1, h - 1], outline=level, width=lw)
    return np.array(im)


def _dashed(draw, pts, level, width, dash=14, gap=9):
    pts = list(pts) + [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length == 0:
            continue
        ux, uy, t = (x1 - x0) / length, (y1 - y0) / length, 0.0
        while t < length:
            e = min(t + dash, length)
            draw.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * e, y0 + uy * e)], fill=level, width=width)
            t += dash + gap


def outline(size, H, panel_polys, glass_polys, level=255):
    """Panel outlines solid, glass/off-limits dashed, from canvas polygons mapped through H."""
    im = _canvas(size)
    d = ImageDraw.Draw(im)
    lw = max(2, round(min(size) / 360))
    for poly in panel_polys:
        q = [tuple(p) for p in apply_homography(H, poly)]
        d.line(q + [q[0]], fill=level, width=lw, joint="curve")
    for poly in glass_polys:
        _dashed(d, [tuple(p) for p in apply_homography(H, poly)], int(level * 0.8), lw)
    return np.array(im)


def align(size, H, panel_polys, glass_polys, landmarks_canvas, selected=None, level=255):
    """outline() plus a crosshair, circle and number at each landmark's projector position; `selected` is drawn larger."""
    im = Image.fromarray(outline(size, H, panel_polys, glass_polys, int(level * 0.55)))
    d = ImageDraw.Draw(im)
    r0 = max(10, round(min(size) / 45))
    fs = max(14, round(min(size) / 38))
    font = _font(fs)
    lw = max(2, round(min(size) / 300))
    for i, p in enumerate(apply_homography(H, landmarks_canvas)):
        r = r0 * (1.5 if i == selected else 1.0)
        d.line([(p[0] - 2 * r, p[1]), (p[0] + 2 * r, p[1])], fill=level, width=lw)
        d.line([(p[0], p[1] - 2 * r), (p[0], p[1] + 2 * r)], fill=level, width=lw)
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=level, width=lw)
        d.text((p[0] + r + 4, p[1] - r - fs), str(i + 1), fill=level, font=font)
    return np.array(im)


def render(name, *, size, warper=None, H=None, panel_polys=(), glass_polys=(), landmarks_canvas=(), selected=None):
    """Pattern by name. warper is needed for 'white'; H (canvas -> projector) and the polygons for 'outline' and 'align'."""
    if name == "black":
        return black(size)
    if name == "white":
        return white(warper)
    if name == "dots":
        return dots(size)
    if name == "grid":
        return grid(size)
    if name == "edge":
        return edge(size)
    if name == "outline":
        return outline(size, H, panel_polys, glass_polys)
    if name == "align":
        return align(size, H, panel_polys, glass_polys, landmarks_canvas, selected)
    raise KeyError(f"unknown pattern {name!r}; choose from {ALL}")
