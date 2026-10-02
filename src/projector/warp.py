"""Warp composed frames into projector pixels and keep the impenetrable regions exactly black there (handoff §2.4, §8, §10.1).

The hard rule (glass, holes and anything the projector must not light are pure black in every frame) was enforced on the
canvas by `apply_black`. Warping resamples the image, so the rule has to be enforced again in the space the projector
actually shows: Warper builds, once per (calibration, mask, output size), the set of projector pixels that are allowed to
carry light, and every frame it outputs is multiplied by that set after quantisation, so "exactly 0" is a property of the
final bytes, not of an interpolation. A pixel is allowed only if

  1. it maps (through H^-1) to a place where ALL the bilinear neighbours are projectable on the canvas, and
  2. it is at least `margin_px` projector pixels away from anything that is not (a safety margin: a projector that has
     crept a pixel or two must not light the glass).

Both make the lit region slightly smaller than the drawn panels, never larger.
"""
import numpy as np
from scipy import ndimage as ndi


class GlassLeak(AssertionError):
    """A pixel that must be black is not."""


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + 0.5


def source_coordinates(H, size):
    """For every projector pixel (x, y) in a grid of `size`, the canvas position H^-1 (x, y). Returns (sx, sy) float64."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    Hi = np.linalg.inv(np.asarray(H, float))
    d = Hi[2, 0] * xx + Hi[2, 1] * yy + Hi[2, 2]
    return (Hi[0, 0] * xx + Hi[0, 1] * yy + Hi[0, 2]) / d, (Hi[1, 0] * xx + Hi[1, 1] * yy + Hi[1, 2]) / d


def is_affine(H, tol=1e-12):
    return abs(H[2, 0]) < tol and abs(H[2, 1]) < tol


def warp_image(img, H, size, cval=0.0):
    """Resample a canvas image (2-D float) into a projector grid of `size` = (w, h) through H (canvas -> projector), bilinear.
    Anything that maps outside the canvas is `cval`."""
    img = np.asarray(img, float)
    w, h = size
    H = np.asarray(H, float) / H[2, 2]
    if is_affine(H):
        Hi = np.linalg.inv(H)
        return ndi.affine_transform(img, [[Hi[1, 1], Hi[1, 0]], [Hi[0, 1], Hi[0, 0]]], offset=[Hi[1, 2], Hi[0, 2]],
                                    output_shape=(h, w), order=1, mode="constant", cval=cval, prefilter=False)
    sx, sy = source_coordinates(H, size)
    return ndi.map_coordinates(img, [sy, sx], order=1, mode="constant", cval=cval, prefilter=False)


class Warper:
    def __init__(self, H, frame_mask, size, margin_px=2):
        """H: canvas -> projector; frame_mask: canvas bool, True where light is allowed; size: projector (w, h)."""
        self.H, self.size, self.margin_px = np.asarray(H, float) / H[2, 2], (int(size[0]), int(size[1])), int(margin_px)
        coverage = warp_image(frame_mask.astype(float), self.H, self.size)
        lit = coverage >= 1 - 1e-9
        if self.margin_px > 0 and (~lit).any():
            lit = ndi.binary_erosion(lit, _disk(self.margin_px), border_value=1)
        self.lit = lit
        self.n_lit = int(lit.sum())

    @property
    def must_be_black(self):
        return ~self.lit

    def __call__(self, img, gain=1.0):
        """A composed canvas frame (float 0..1) -> uint8 projector frame with every forbidden pixel exactly 0."""
        out = warp_image(img, self.H, self.size)
        g = float(np.clip(gain, 0.0, 1.0))
        q = np.rint(np.clip(out, 0.0, 1.0) * 255.0 * g).astype(np.uint8)
        q[~self.lit] = 0
        return q

    def white(self, level=1.0):
        """The 'white in frame' test pattern: full light everywhere light is allowed, black everywhere else."""
        return (self.lit * np.uint8(round(255 * level))).astype(np.uint8)

    def check(self, frame_u8):
        """Raise GlassLeak if any forbidden pixel of a finished projector frame is not exactly 0. Returns the lit fraction."""
        if frame_u8.shape != (self.size[1], self.size[0]):
            raise GlassLeak(f"frame is {frame_u8.shape[::-1]}, expected {self.size}")
        bad = int((frame_u8[~self.lit] != 0).sum())
        if bad:
            raise GlassLeak(f"{bad} projector pixels that must be black are lit")
        return float((frame_u8 > 0).mean())

    def project_points(self, pts):
        from src.projector.calibrate import apply_homography
        return apply_homography(self.H, pts)
