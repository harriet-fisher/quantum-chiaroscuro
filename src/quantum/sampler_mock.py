"""Classical stand-in for the quantum draw: Boltzmann sampling of an Ising-type model on the patch graph.

Lamp spins are clamped per frame (they are sampled uniformly by the caller); patch spins are drawn by Gibbs
sampling given the lamps; the polarity spins are drawn from an exact chain distribution. Not quantum: this is
the look-test sampler and the source of the targets that calibrate_from_mock.py hands to the Moth engines.
"""
import itertools

import numpy as np

from src.graph.allocate_qubits import budget
from src.graph.edges import patch_couplings, polarity_edges
from src.graph.lamp_hub import hub_fields
from src.graph.patches import build_patches


class Sampler:
    def __init__(self, scene, NX=6, NY=4, kh=1.6, kf=1.3, J_in=0.9, J_cross=0.6, J_pol=0.9, beta=1.0, min_cover=0.30):
        self.scene = scene
        self.cells, self.idx = build_patches(scene, NX, NY, min_cover)
        self.n = len(self.cells)
        self.NX, self.NY, self.beta, self.J_pol = NX, NY, beta, J_pol
        self.hub1, self.hub2 = hub_fields(self.cells, scene.panels, kh, kf)   # couplings to lamps L1, L2
        self.J = patch_couplings(self.cells, self.idx, J_in, J_cross)
        self.pol_edges = polarity_edges(scene)
        self.n_panels = scene.n_panels
        self.n_qubits = budget(self.n, self.n_panels)["total"]            # patches + 2 lamps + polarity

    def sample_shots(self, lamp, K, rng, burn=60, thin=3):
        """K shots (rows of +/-1 patch spins) with the lamp spins clamped (post-selected)."""
        h = self.hub1 * lamp[0] + self.hub2 * lamp[1]
        s = rng.choice([-1, 1], self.n)
        out = []
        total = burn + K * thin
        for sweep in range(total):
            for p in rng.permutation(self.n):
                local = h[p] + self.J[p] @ s
                s[p] = 1 if rng.random() < 1 / (1 + np.exp(-2 * self.beta * local)) else -1
            if sweep >= burn and (sweep - burn) % thin == 0:
                out.append(s.copy())
        return np.array(out)

    def sample_lit(self, lamp, K, rng, burn=60, thin=3):
        """Lit fraction per patch over K shots."""
        return (self.sample_shots(lamp, K, rng, burn, thin) > 0).mean(axis=0)

    def polarity_states(self):
        return list(itertools.product([-1, 1], repeat=self.n_panels))   # +1 raised, -1 sunk

    def polarity_weights(self):
        return np.array([np.exp(self.beta * self.J_pol * sum(s[a] * s[b] for a, b in self.pol_edges))
                         for s in self.polarity_states()])

    def sample_polarity(self, rng):
        states = self.polarity_states()
        w = self.polarity_weights()
        return states[rng.choice(len(states), p=w / w.sum())]
