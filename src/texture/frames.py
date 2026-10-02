"""Frame generator: measurement outcomes of one register -> many lighting frames (handoff §9.2).

A frame is one lighting world with K patch shots, all taken from the SAME pool of register measurements:
  1. the lighting world (L1, L2) is the lamp outcome of a randomly chosen shot (so its frequency is the state's own,
     or you can hold it);
  2. K shots whose lamps equal that world are drawn without replacement (post-selection), and the lit fraction per patch
     is their mean;
  3. polarity is read from the first of those K shots (once per frame), so it stays jointly consistent with them;
  4. compose_frame() turns (lit, lamp, pol) into pixels. Geometry only fences the values.
Nothing here samples from a classical model: pass any Pool (quantum state, mock, or independent noise).
"""
from dataclasses import dataclass

import numpy as np

from src.texture.compose import compose_frame


@dataclass
class FrameDraw:
    lamp: tuple
    pol: tuple
    lit: np.ndarray
    K: int
    shot_ids: np.ndarray


class PoolIndex:
    """Shots bucketed by lamp outcome and (optionally) polarity, for fast conditioned draws."""

    def __init__(self, pool):
        self.pool = pool
        code = ((pool.lamps[:, 0] > 0) * 2 + (pool.lamps[:, 1] > 0)).astype(int)
        self.by_lamp = {(l1, l2): np.nonzero(code == (a * 2 + b))[0]
                        for a, l1 in enumerate((-1, 1)) for b, l2 in enumerate((-1, 1))}
        self.polcode = ((pool.pol > 0) * (1 << np.arange(pool.pol.shape[1]))).sum(axis=1)

    def available(self):
        return {k: len(v) for k, v in self.by_lamp.items()}


def _held(values):
    """(mask, want) bit codes for a tuple of +1 / -1 / None entries; None = not held."""
    mask = sum(1 << i for i, v in enumerate(values) if v is not None)
    return mask, sum(1 << i for i, v in enumerate(values) if v is not None and v > 0)


def draw_frame(index, K, rng, lamp=None, pol=None):
    """One FrameDraw. lamp and pol hold the lighting world / polarity fixed by post-selection. Each may be None (nothing held)
    or a tuple whose entries are +1, -1 or None, so a single lamp bit or a single panel can be held while the rest stay free."""
    pool = index.pool
    lamp_held = lamp is not None and all(v is not None for v in lamp)
    pmask, pwant = _held(pol) if pol is not None else (0, 0)
    if lamp_held:
        world = (int(lamp[0]), int(lamp[1]))
    elif lamp is None and not pmask:
        j = rng.integers(len(pool.lamps))
        world = (int(pool.lamps[j, 0]), int(pool.lamps[j, 1]))
    else:
        ok = np.ones(len(pool.lamps), bool)
        for i, v in enumerate(lamp or (None, None)):
            if v is not None:
                ok &= pool.lamps[:, i] == v
        if pmask:
            ok &= (index.polcode & pmask) == pwant
        cand = np.nonzero(ok)[0]
        if len(cand) == 0:
            raise ValueError(f"no shots in the pool with lamp {tuple(lamp or (None, None))} and polarity {tuple(pol) if pol is not None else None}; "
                             "enlarge the pool or choose a look the state actually produces")
        j = rng.choice(cand)
        world = (int(pool.lamps[j, 0]), int(pool.lamps[j, 1]))
    ids = index.by_lamp[world]
    if pmask:
        ids = ids[(index.polcode[ids] & pmask) == pwant]
    if len(ids) == 0:
        raise ValueError(f"no shots in the pool with lamp {world}" + (f" and polarity {tuple(pol)}" if pmask else "")
                         + "; enlarge the pool or choose a world the state actually produces")
    take = rng.choice(ids, size=K, replace=len(ids) < K)
    lit = (pool.shots[take] > 0).mean(axis=0)
    return FrameDraw(world, tuple(int(v) for v in pool.pol[take[0]]), lit, K, take)


class FrameGenerator:
    """Draws frames from a Pool and composes them on a Scene. `structure` is any object with .cells and .NX
    (the mock Sampler or a patch-graph holder); it supplies geometry only, never randomness."""

    def __init__(self, scene, structure, pool, **compose_kw):
        self.scene, self.structure, self.index, self.kw = scene, structure, PoolIndex(pool), compose_kw

    def frame(self, K, rng, lamp=None, pol=None):
        d = draw_frame(self.index, K, rng, lamp, pol)
        img = compose_frame(self.scene, self.structure.cells, self.structure.NX, d.lit, d.lamp, d.pol, **self.kw)
        return img, d

    def frames(self, n, K, rng, lamp=None, pol=None):
        for _ in range(n):
            yield self.frame(K, rng, lamp, pol)
