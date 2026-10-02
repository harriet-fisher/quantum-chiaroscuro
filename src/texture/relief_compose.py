"""Compose a Superposed Relief frame (v2 handoff section 4.4): measured facet outcomes -> pixels. No look is computed here.

    lit_i  = fraction of the K photons that landed on facet i (bit 0)
    S      = sum_i lit_i B_i(x),   B_i = gauss(indicator of facet i) / gauss(its panel's projectable pixels)   (sums to 1 per panel)
    value  = lo + (hi - lo) S                                       the geometry's brightness window for that plane and depth
    out    = masked_blur(value) * frame_mask                        glass is multiplied by 0: exactly black, as before

There is no bevel function, no relief profile and no polarity weighting in here: the shading at a bevel is the measurement outcome of
the facet qubits, whose Bloch vectors are the surface normals, under the light axis the circuit drew. The classical bevel code
(texture.bevel) stays importable for the mock and the classical baseline only, and nothing in this module imports it.
"""
import numpy as np

from src.mask.apply_black import apply_black
from src.mask.masked_blur import gauss


class ReliefComposer:
    def __init__(self, scene, facets, interp_sigma=2.0, contrast=1.0, pivot=None):
        self.scene, self.facets = scene, facets
        self.interp_sigma = float(interp_sigma)
        self.contrast, self.pivot = float(contrast), pivot
        self.frame_f = scene.frame.astype(float)
        self.basis = self._basis()
        self._denominators = {}

    def _basis(self):
        sc, fs = self.scene, self.facets
        B = np.zeros((fs.n, sc.H, sc.W))
        for k in range(sc.n_panels):
            fm = (sc.frame & (sc.region == k)).astype(float)
            if not fm.any():
                continue
            denom = np.maximum(gauss(fm, self.interp_sigma), 1e-6)
            for i in fs.of_panel(k):
                B[i] = np.where(fm > 0, gauss(((fs.label == i) & (fm > 0)).astype(float), self.interp_sigma) / denom, 0.0)
        return B

    def _blend_denominator(self, sigma):
        d = self._denominators.get(sigma)
        if d is None:
            d = self._denominators[sigma] = np.maximum(gauss(self.frame_f, sigma), 1e-6)
        return d

    def field(self, lit):
        """The interpolated lit field S in [0, 1] over the projectable pixels."""
        return np.tensordot(np.asarray(lit, float), self.basis, axes=1)

    def compose(self, lit, blend_sigma=5, contrast=None):
        sc = self.scene
        S = self.field(lit)
        c = self.contrast if contrast is None else float(contrast)
        if c != 1.0:                                     # exposure of the projection, not a scene property: stretch about the pivot
            p = 0.5 if self.pivot is None else self.pivot
            S = np.clip(p + c * (S - p), 0, 1)
        val = sc.lo + (sc.hi - sc.lo) * S
        out = gauss(val * self.frame_f, blend_sigma) / self._blend_denominator(blend_sigma)
        return np.clip(apply_black(out, sc.frame), 0, 1)


class SparseReliefComposer(ReliefComposer):
    """The same composition with each facet's interpolation weight stored only inside its (margin-padded) bounding box, in float32. A dense
    basis costs facets x H x W doubles, which is 100 MB for the 15-facet bay window and 4 GB at 600 facets; this one is a few MB. Identical to the
    dense composer except for the Gaussian tail beyond four sigma, which is below 4e-4."""

    def _basis(self):
        sc, fs = self.scene, self.facets
        margin = int(4 * self.interp_sigma) + 2
        denom = {}
        out = []
        for i, f in enumerate(fs.facets):
            k = f.panel
            if k not in denom:
                fm = (sc.frame & (sc.region == k)).astype(float)
                denom[k] = (fm, np.maximum(gauss(fm, self.interp_sigma), 1e-6))
            fm, dn = denom[k]
            x0, x1, y0, y1 = f.box
            y0, y1, x0, x1 = max(y0 - margin, 0), min(y1 + margin, sc.H), max(x0 - margin, 0), min(x1 + margin, sc.W)
            ind = ((fs.label[y0:y1, x0:x1] == i) & (fm[y0:y1, x0:x1] > 0)).astype(float)
            w = np.where(fm[y0:y1, x0:x1] > 0, gauss(ind, self.interp_sigma) / dn[y0:y1, x0:x1], 0.0)
            out.append((y0, y1, x0, x1, w.astype(np.float32)))
        return out

    def field(self, lit):
        S = np.zeros((self.scene.H, self.scene.W))
        for v, (y0, y1, x0, x1, w) in zip(np.asarray(lit, float), self.basis):
            if v:
                S[y0:y1, x0:x1] += v * w
        return S


def make_composer(scene, facets, **kw):
    """Dense composer for a handful of facets, sparse beyond that."""
    return (ReliefComposer if facets.n <= 24 else SparseReliefComposer)(scene, facets, **kw)
