"""FastComposer must reproduce compose_frame (the reference the mock port is checked against) to rounding error.

    python -m unittest tests.test_fast -v
"""
import time
import unittest

import numpy as np

from src.capture.labels import bay_window_labels, fill_defaults, validate
from src.geometry.planes import build_scene
from src.quantum.sampler_mock import Sampler
from src.texture.compose import compose_frame
from src.texture.fast import FastComposer


class FastMatchesReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = build_scene(validate(fill_defaults(bay_window_labels())))
        cls.sampler = Sampler(cls.scene, 5, 4)
        cls.fast = FastComposer(cls.scene, cls.sampler.cells, cls.sampler.NX)

    def test_equal_to_reference_for_random_outcomes_and_blends(self):
        rng = np.random.default_rng(0)
        for blend in (5, 2, 9):
            for _ in range(3):
                lit = rng.random(self.sampler.n)
                lamp = tuple(int(v) for v in rng.choice([-1, 1], 2))
                pol = tuple(int(v) for v in rng.choice([-1, 1], self.scene.n_panels))
                ref = compose_frame(self.scene, self.sampler.cells, self.sampler.NX, lit, lamp, pol, blend_sigma=blend)
                got = self.fast.compose(lit, lamp, pol, blend_sigma=blend)
                self.assertLess(np.abs(ref - got).max(), 1e-9)

    def test_impenetrable_pixels_are_exactly_zero(self):
        got = self.fast.compose(np.ones(self.sampler.n), (1, 1), (1, 1, 1))
        self.assertTrue(np.all(got[self.scene.impenetrable] == 0))
        self.assertGreater(got.max(), 0.5)

    def test_is_much_faster_than_the_reference(self):
        lit = np.full(self.sampler.n, 0.5)
        t = time.perf_counter(); self.fast.compose(lit, (1, 1), (1, 1, 1)); fast_s = time.perf_counter() - t
        t = time.perf_counter(); compose_frame(self.scene, self.sampler.cells, self.sampler.NX, lit, (1, 1), (1, 1, 1)); ref_s = time.perf_counter() - t
        self.assertLess(fast_s, ref_s / 3)


if __name__ == "__main__":
    unittest.main()
