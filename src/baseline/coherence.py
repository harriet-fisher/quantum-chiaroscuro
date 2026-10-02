"""Coherence metrics on raw outcomes (spec §13): does a batch of draws read as ONE light on ONE set of planes?

Inputs are measurement outcomes, so the same function scores the mock sampler, independent noise, and later the quantum
samples: lamps (N, 2) in +/-1, shots (N, n_patches) in +/-1 (one shot per draw), pol (N, n_panels) in +/-1.
"""
import numpy as np

from src.graph.edges import patch_edges


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def coherence_metrics(sampler, lamps, shots, pol):
    """All values lie in [-1, 1]. Independent noise scores ~0 on every line."""
    edges = patch_edges(sampler.J)
    prod = {e: (shots[:, e[0]] * shots[:, e[1]]).mean() for e in edges}
    coplanar = [v for e, v in prod.items() if sampler.J[e] > 0]
    crease = [v for e, v in prod.items() if sampler.J[e] < 0]
    nx = np.sign(sampler.hub1)                                    # side the patch faces (0 for the flat centre)
    nz = sampler.hub2
    light_lr = (shots * nx).sum(axis=1) / max(np.abs(nx).sum(), 1)
    light_fg = (shots * nz).sum(axis=1) / max(np.abs(nz).sum(), 1e-9)
    pol_agree = [float((pol[:, a] * pol[:, b]).mean()) for a, b in sampler.pol_edges]
    return dict(
        coplanar_agreement=float(np.mean(coplanar)) if coplanar else None,      # neighbours on one plane share a state
        crease_agreement=float(np.mean(crease)) if crease else None,            # across a crease they differ (negative)
        light_left_right=_corr(lamps[:, 0].astype(float), light_lr),            # does lamp L1 decide which wing is lit?
        light_frontal_grazing=_corr(lamps[:, 1].astype(float), light_fg),       # does lamp L2 decide centre vs wings?
        polarity_agreement=float(np.mean(pol_agree)) if pol_agree else None)    # neighbouring panels agree on sunk/raised


def draw_configs(sampler, N, rng):
    """N independent frame configurations (one patch shot each) from any sampler with the Sampler interface."""
    lamps = np.array([rng.choice([-1, 1], 2) for _ in range(N)])
    shots = np.array([sampler.sample_shots(tuple(l), 1, rng)[0] for l in lamps])
    pol = np.array([sampler.sample_polarity(rng) for _ in range(N)])
    return lamps, shots, pol
