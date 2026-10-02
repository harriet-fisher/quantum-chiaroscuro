"""Compose one frame (spec §3.7): measured lit fractions + relief + bevel -> brightness window -> mask-aware blur
-> multiply by the impenetrable mask.

compose_frame() is the classical half and takes the MEASUREMENT OUTCOMES as data:
    lit   lit fraction per patch (average over K shots of the patch qubits; bit 0 = Z +1 = lit)
    lamp  (L1, L2) lamp-qubit outcomes: light from right (+1) / left (-1); frontal (+1) / grazing (-1)
    pol   one polarity-qubit outcome per panel: raised (+1) / sunk (-1)
Nothing in it decides how the surface looks: geometry only fences the values (windows, relief field) and the
measurement chooses where inside the fences each value lands. Feed it outcomes from the quantum system, from the
classical mock sampler, or from independent noise (src/baseline) and compare.

render() is the mock-era convenience wrapper that also draws the patch shots from a Sampler.
"""
import numpy as np

from src.mask.apply_black import apply_black
from src.mask.masked_blur import masked_blur
from src.texture.bevel import directional_bevel, polarity_map, relief_profile
from src.texture.interpolate import interpolate_lit


def compose_frame(scene, cells, NX, lit, lamp, pol, interp_sigma=None, blend_sigma=5, w_light=0.75, w_relief=0.20, w_dir=0.70):
    cw = scene.W / NX
    interp_sigma = interp_sigma or 0.18 * cw
    S = interpolate_lit(scene, cells, lit, interp_sigma)
    pol_map = polarity_map(scene, pol)
    prof = relief_profile(pol_map, scene.relief)
    bevel = directional_bevel(pol_map, scene.relief, lamp)
    u = np.clip((w_light * S + w_relief * prof + w_dir * bevel) / (w_light + w_relief + w_dir), 0, 1)
    val = scene.lo + (scene.hi - scene.lo) * u
    out = masked_blur(val, scene.frame, blend_sigma)
    out = apply_black(out, scene.frame)
    return np.clip(out, 0, 1)


def render(scene, sampler, lamp, pol, K, rng, **kw):
    """Draw K patch shots from `sampler` with the lamps clamped, then compose. Same arguments as the original mock."""
    lit = sampler.sample_lit(lamp, K, rng)
    return compose_frame(scene, sampler.cells, sampler.NX, lit, lamp, pol, **kw)
