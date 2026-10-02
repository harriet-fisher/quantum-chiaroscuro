"""Cached composer for live performance (handoff §10.2): the same frame as texture.compose.compose_frame, ~20x faster.

compose_frame() spends ~1.2 s per frame, almost all of it in interpolate_lit(): two full-size Gaussians per panel, one of
which (the normaliser) never changes and the other of which is LINEAR in the per-patch lit fractions. So both can be
prepared once per scene:

    S(x) = sum_c lit_c * B_c(x),    B_c = gauss(indicator of the pixels filled from patch c) / gauss(panel mask)

FastComposer.compose() then costs one tensor dot, one gradient and one blur. The result equals compose_frame() to rounding
error (tests/test_fast.py pins it); compose_frame stays the reference and is what the mock port is checked against. As
everywhere downstream, nothing here decides how the surface looks: lit, lamp and pol are measurement outcomes passed in.
"""
import numpy as np
from scipy import ndimage as ndi

from src.mask.apply_black import apply_black
from src.mask.masked_blur import gauss
from src.texture.bevel import directional_bevel, relief_profile


class FastComposer:
    def __init__(self, scene, cells, NX, interp_sigma=None, w_light=0.75, w_relief=0.20, w_dir=0.70):
        self.scene, self.cells, self.NX = scene, cells, NX
        cw = scene.W / NX
        self.interp_sigma = interp_sigma or 0.18 * cw
        self.w = (w_light, w_relief, w_dir)
        self.basis = self._basis()
        self.smooth_R = gauss(scene.relief, 4)
        self.panel_masks = [scene.region == k for k in range(scene.n_panels)]
        self.frame_f = scene.frame.astype(float)
        self._denominators = {}

    def _basis(self):
        """(n_cells, H, W): the response of the interpolated lit field to a unit lit fraction in each patch."""
        sc = self.scene
        B = np.zeros((len(self.cells), sc.H, sc.W))
        for k in range(sc.n_panels):
            fm = sc.frame & (sc.region == k)
            ids = np.full((sc.H, sc.W), -1)
            for ci, c in enumerate(self.cells):
                if c["k"] != k:
                    continue
                x0, x1, y0, y1 = c["box"]
                ids[y0:y1, x0:x1][fm[y0:y1, x0:x1]] = ci
            valid = ids >= 0
            if not valid.any():
                continue
            near = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
            filled = ids[near[0], near[1]]
            denom = np.maximum(gauss(fm.astype(float), self.interp_sigma), 1e-6)
            for ci in np.unique(filled[fm]):
                B[ci][fm] = (gauss(((filled == ci) & fm).astype(float), self.interp_sigma) / denom)[fm]
        return B

    def _blend_denominator(self, sigma):
        d = self._denominators.get(sigma)
        if d is None:
            d = self._denominators[sigma] = np.maximum(gauss(self.frame_f, sigma), 1e-6)
        return d

    def polarity_map(self, pol):
        pm = np.zeros((self.scene.H, self.scene.W))
        for m, v in zip(self.panel_masks, pol):
            pm[m] = v
        return pm

    def compose(self, lit, lamp, pol, blend_sigma=5):
        sc = self.scene
        S = np.tensordot(np.asarray(lit, float), self.basis, axes=1)
        pol_map = self.polarity_map(pol)
        prof = relief_profile(pol_map, sc.relief)
        bevel = directional_bevel(pol_map, sc.relief, lamp, smooth_R=self.smooth_R)
        w_light, w_relief, w_dir = self.w
        u = np.clip((w_light * S + w_relief * prof + w_dir * bevel) / sum(self.w), 0, 1)
        val = sc.lo + (sc.hi - sc.lo) * u
        out = gauss(val * self.frame_f, blend_sigma) / self._blend_denominator(blend_sigma)
        return np.clip(apply_black(out, sc.frame), 0, 1)
