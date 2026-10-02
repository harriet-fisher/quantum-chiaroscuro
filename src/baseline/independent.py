"""Independent-noise baseline (spec §2.2 test, §8.2 baseline/): the same patches and lamps, but every outcome is a fair
coin flip with no correlations. Same interface as quantum.sampler_mock.Sampler, so it drops into render()/compose_frame().

If swapping the real sampler for this leaves the illusion intact, the correlations were not doing the work.
"""
import numpy as np


class IndependentNoise:
    def __init__(self, sampler):
        self.cells, self.idx, self.n = sampler.cells, sampler.idx, sampler.n
        self.NX, self.NY, self.n_panels, self.n_qubits = sampler.NX, sampler.NY, sampler.n_panels, sampler.n_qubits
        self.J, self.pol_edges, self.hub1, self.hub2 = sampler.J, sampler.pol_edges, sampler.hub1, sampler.hub2

    def sample_shots(self, lamp, K, rng, burn=0, thin=1):
        return rng.choice([-1, 1], (K, self.n))              # lamps are ignored: that is the point

    def sample_lit(self, lamp, K, rng, burn=0, thin=1):
        return (self.sample_shots(lamp, K, rng) > 0).mean(axis=0)

    def sample_polarity(self, rng):
        return tuple(int(v) for v in rng.choice([-1, 1], self.n_panels))
