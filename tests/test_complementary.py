"""The complementary-polarity experiment: witness maths against textbook states, sampler conventions against exact values, and
the structure of the stand-in state. (These check the TOOLS; whether a Moth/QDrive circuit shows any of it is untested.)

    python -m unittest tests.test_complementary -v
"""
import unittest

import numpy as np

from src.capture.labels import bay_window_labels, fill_defaults, validate
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import allocate
from src.quantum import complementary as c
from src.quantum import sampler_local as sl
from src.quantum.sampler_mock import Sampler


def haar(n, rng):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def chsh_of(psi, n, a, b):
    return c.max_chsh(c.correlation_matrix(c.reduced_pair(psi, n, a, b))[0])


class Witness(unittest.TestCase):
    def test_textbook_states(self):
        bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
        cz_plus = np.array([1, 1, 1, -1], complex) / 2
        product = np.kron([1, 0], [0.6, 0.8]).astype(complex)
        ghz = np.zeros(8, complex); ghz[0] = ghz[7] = 1 / np.sqrt(2)
        self.assertAlmostEqual(chsh_of(bell, 2, 0, 1), 2 * np.sqrt(2), places=9)
        self.assertAlmostEqual(chsh_of(cz_plus, 2, 0, 1), 2 * np.sqrt(2), places=9)
        self.assertLessEqual(chsh_of(product, 2, 0, 1), 2 + 1e-9)
        self.assertAlmostEqual(chsh_of(ghz, 3, 0, 1), 2.0, places=9)           # a pair of a GHZ state is a classical mixture

    def test_z_only_correlations_never_exceed_the_classical_bound(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            p = rng.dirichlet(np.ones(8))                    # a diagonal (Z-only) state: random probabilities, no coherence
            T, _, _ = c.correlation_matrix(np.diag(np.repeat(p, 1))[:4, :4] / p[:4].sum())
            self.assertLessEqual(c.max_chsh(T), 2 + 1e-9)

    def test_sampled_correlation_matches_exact_for_arbitrary_axes(self):
        """Pins bloch_unitary and the spin convention: a.T.b from the density matrix equals what sampling the rotated qubits gives."""
        rng = np.random.default_rng(1)
        psi, n = haar(4, rng), 4
        T, _, _ = c.correlation_matrix(c.reduced_pair(psi, n, 1, 3))
        for _ in range(4):
            va, vb = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 3)))
            E, se = c.sample_correlation(psi, n, 1, 3, va, vb, 80000, rng)
            self.assertAlmostEqual(E, float(va @ T @ vb), delta=5 * se)

    def test_bloch_unitary_sends_the_axis_to_z(self):
        rng = np.random.default_rng(2)
        for _ in range(5):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            U = sl.bloch_unitary(v)
            n_sigma = sum(v[i] * c.PAULI["xyz"[i]] for i in range(3))
            np.testing.assert_allclose(U @ n_sigma @ U.conj().T, c.PAULI["z"], atol=1e-12)

    def test_arbitrary_basis_matches_the_named_one(self):
        psi = haar(3, np.random.default_rng(3))
        lay = dict(patches=[0], lamps=[1, 2], polarity=[], n_qubits=3)
        a = sl.StateSource(psi, lay, {0: "x"}).probs
        b = sl.StateSource(psi, lay, {0: (1, 0, 0)}).probs
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_optimal_settings_reach_the_exact_value(self):
        rng = np.random.default_rng(4)
        bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
        est = c.chsh_estimate(bell, 2, 0, 1, 40000, rng)
        self.assertGreater(est["sigmas_above_2"], 10)
        self.assertAlmostEqual(est["S"], est["exact"], delta=5 * est["se"])


class Mermin(unittest.TestCase):
    AXES = {0: ("x", "y", 1), 1: ("z", "y", -1), 2: ("z", "y", -1), 3: ("z", "y", -1)}

    def cp_state(self, m):
        n = m + 1
        idx = np.arange(2 ** n)
        bit = lambda q: (idx >> q) & 1
        return np.exp(1j * np.pi * bit(0) * sum(bit(q) for q in range(1, n))) / np.sqrt(2 ** n), n

    def test_ghz_in_the_cz_frame_reaches_the_quantum_maximum(self):
        for m, want in ((2, 4.0), (3, 8.0)):
            psi, n = self.cp_state(m)
            self.assertAlmostEqual(c.mermin_exact(psi, n, list(range(n)), self.AXES), want, places=9)
            self.assertEqual(c.mermin_bounds(n), (2 ** (n // 2), 2 ** (n - 1)))

    def test_a_product_state_stays_within_the_local_bound(self):
        psi, n = haar(1, np.random.default_rng(5)), 1
        full = np.kron(np.kron(np.kron(psi, haar(1, np.random.default_rng(6))), haar(1, np.random.default_rng(7))), haar(1, np.random.default_rng(8)))
        self.assertLessEqual(abs(c.mermin_exact(full, 4, [0, 1, 2, 3], self.AXES)), c.mermin_bounds(4)[0] + 1e-9)

    def test_sampling_agrees_with_exact(self):
        psi, n = self.cp_state(2)
        est = c.mermin_estimate(psi, n, [0, 1, 2], self.AXES, 20000, np.random.default_rng(9))
        self.assertAlmostEqual(est["value"], 4.0, delta=5 * est["se"] + 1e-9)
        self.assertGreater(est["sigmas_above_bound"], 10)


class StandIn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = build_scene(validate(fill_defaults(bay_window_labels())))
        cls.sampler = Sampler(cls.scene, 4, 4)                                 # the default scene: 12 patches + 2 lamps + 6 polarity = 20 qubits
        cls.layout = allocate(cls.sampler.n, cls.sampler.n_panels)
        cls.psi = c.complementary_state(cls.sampler, cls.layout, "hub")
        cls.n = cls.layout["n_qubits"]

    def test_normalised_and_geometry_picks_the_partner_lamp(self):
        self.assertAlmostEqual(float(np.linalg.norm(self.psi)), 1.0, places=9)
        L1, L2 = self.layout["lamps"]
        want = [[L2] if abs(p.angle_deg) < 0.5 else [L1] for p in self.scene.panels]
        self.assertEqual(c.partners(self.sampler, self.layout, "hub"), want)                   # the wings follow L1, the flat centre follows L2
        self.assertEqual(want, [[L1], [L1], [L2], [L2], [L1], [L1]])                           # six panes: two per wing

    def test_z_statistics_of_lighting_are_exactly_the_baseline_and_polarity_is_a_fair_coin(self):
        base = sl.oracle_state(self.sampler, self.layout)
        L = self.layout
        z_roles = L["patches"] + L["lamps"]
        a = sl.StateSource(base, L)
        b = sl.StateSource(self.psi, L)
        pairs = [(0, 2), (3, 8), (L["lamps"][0], 5), (L["lamps"][1], 11)]
        np.testing.assert_allclose(a.z_moments(pairs)[0][z_roles], b.z_moments(pairs)[0][z_roles], atol=1e-9)
        np.testing.assert_allclose(a.z_moments(pairs)[1], b.z_moments(pairs)[1], atol=1e-9)
        np.testing.assert_allclose(b.z_moments([])[0][L["polarity"]], 0, atol=1e-12)

    def test_x_reading_locks_polarity_to_its_lamp_and_z_reading_does_not(self):
        rng = np.random.default_rng(10)
        L = self.layout
        for basis, want in (("x", 1.0), ("z", 0.0)):
            pool = c.make_pool(self.psi, L, basis, 40000, rng)
            corr = c.polarity_lamp_correlation(pool)
            part = c.partners(self.sampler, L, "hub")
            for k in range(self.scene.n_panels):
                j = part[k][0] - L["lamps"][0]
                self.assertAlmostEqual(corr[k, j], want, delta=0.03, msg=f"panel {k} read in {basis}")

    def test_control_with_no_coupling_is_a_product_state_and_never_violates(self):
        psi = c.complementary_state(self.sampler, self.layout, "none")
        pool = c.make_pool(psi, self.layout, "x", 20000, np.random.default_rng(11))
        np.testing.assert_allclose(c.polarity_lamp_correlation(pool), 0, atol=0.03)
        for q in self.layout["polarity"]:
            self.assertAlmostEqual(c.bloch_length(psi, self.n, q), 1.0, places=9)           # pure |+>: definite in X

    def test_witnesses_on_the_stand_in(self):
        """Six panes: the two centre panes both follow L2 and the four wing panes both follow L1, so each lamp is recorded by two polarity qubits
        (and L1 also by the patches): no single pane-lamp pair violates CHSH any more (monogamy), but the many-party Mermin test over the lamp and its
        partners does."""
        L = self.layout
        for k in range(self.scene.n_panels):
            lamp = L["lamps"][0 if abs(self.scene.panels[k].angle_deg) > 0.5 else 1]
            self.assertAlmostEqual(chsh_of(self.psi, self.n, L["polarity"][k], lamp), 2.0, places=3, msg=f"panel {k}")
        psi2 = c.complementary_state(self.sampler, self.layout, "lamp2")
        g = c.lamp_groups(L, c.partners(self.sampler, L, "lamp2"))[0]
        self.assertEqual(len(g["group"]), 1 + self.scene.n_panels)                                  # L2 and one polarity qubit per pane
        m = c.mermin_exact(psi2, self.n, g["group"], g["axes"])
        bound, top = c.mermin_bounds(len(g["group"]))
        self.assertGreater(m, bound + 1)                                 # the Mermin test beats the local bound
        self.assertLessEqual(m, top + 1e-9)

    def test_one_pane_per_wing_violates_chsh_with_its_lamp_as_the_first_build_found(self):
        """The three-panel window the first build used: the flat centre panel is alone on L2, so its pair with L2 is not diluted by a second record."""
        scene = build_scene(validate(fill_defaults(bay_window_labels(stacked=False))))
        sampler = Sampler(scene, 5, 4)
        layout = allocate(sampler.n, sampler.n_panels)
        psi = c.complementary_state(sampler, layout, "hub")
        n = layout["n_qubits"]
        self.assertGreater(chsh_of(psi, n, layout["polarity"][1], layout["lamps"][1]), 2.5)           # centre polarity with L2: violates
        self.assertAlmostEqual(chsh_of(psi, n, layout["polarity"][0], layout["lamps"][0]), 2.0, places=3)   # L1 is recorded by the patches
        psi2 = c.complementary_state(sampler, layout, "lamp2")
        g = c.lamp_groups(layout, c.partners(sampler, layout, "lamp2"))[0]
        m = c.mermin_exact(psi2, n, g["group"], g["axes"])
        self.assertGreater(m, c.mermin_bounds(4)[0] + 1)                 # four-party Mermin beats the local bound of 4
        self.assertLessEqual(m, c.mermin_bounds(4)[1] + 1e-9)


if __name__ == "__main__":
    unittest.main()
