"""Conventions and plumbing of the quantum frame source, checked against qiskit and against exact enumeration.

    python -m unittest tests.test_frames -v          (needs qiskit, qiskit-aer, qiskit-qasm3-import)
"""
import unittest

import numpy as np

from src.baseline.coherence import coherence_metrics
from src.capture.labels import bay_window_labels, fill_defaults, validate
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import allocate
from src.graph.edges import patch_edges
from src.quantum import sampler_local as sl
from src.quantum.sampler_mock import Sampler
from src.targets.calibrate_from_mock import estimate_patches_exact
from src.texture.frames import FrameGenerator, PoolIndex, draw_frame

SCENE = SAMPLER = LAYOUT = ORACLE = None


def setUpModule():
    global SCENE, SAMPLER, LAYOUT, ORACLE
    SCENE = build_scene(validate(fill_defaults(bay_window_labels())))
    SAMPLER = Sampler(SCENE, 5, 4)                                  # 14 patches + 2 lamps + 3 polarity = 19 qubits
    LAYOUT = allocate(SAMPLER.n, SAMPLER.n_panels)
    ORACLE = sl.StateSource(sl.oracle_state(SAMPLER, LAYOUT), LAYOUT, label="oracle")


class Conventions(unittest.TestCase):
    def test_qasm3_x_on_qubit_3_sets_bit_3(self):
        psi, n = sl.statevector_from_qasm('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[4] q;\nx q[3];\n')
        self.assertEqual((n, int(np.argmax(abs(psi)))), (4, 8))
        self.assertEqual(sl.spins([8], [0, 1, 2, 3]).tolist(), [[1, 1, 1, -1]])      # bit 0 -> +1 (Z=+1), bit 1 -> -1

    def test_index_to_bitstring_matches_qiskit_probabilities(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        qc = QuantumCircuit(5)
        qc.h(0); qc.ry(0.7, 2); qc.cx(0, 3); qc.rx(1.1, 4); qc.cz(2, 4); qc.x(1)
        ref = Statevector(qc).probabilities_dict()                   # keys print qubit 4 ... qubit 0
        psi = Statevector(qc).data
        for idx in range(32):
            self.assertAlmostEqual(abs(psi[idx]) ** 2, ref.get(format(idx, "05b"), 0.0), places=12)
        bits = sl.spins([13], [0, 1, 2, 3, 4])[0]                    # 13 = 0b01101: qubits 0,2,3 are 1
        self.assertEqual(bits.tolist(), [-1, 1, -1, -1, 1])

    def test_basis_rotation_matches_qiskit(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        qc = QuantumCircuit(4)
        qc.ry(0.4, 0); qc.rx(1.3, 1); qc.cx(1, 2); qc.ry(2.0, 3)
        base = Statevector(qc).data
        for q in range(4):
            qc2 = qc.copy(); qc2.h(q)
            np.testing.assert_allclose(sl.apply_1q(base, sl.H, q, 4), Statevector(qc2).data, atol=1e-12)
        qc3 = qc.copy(); qc3.sdg(2); qc3.h(2)
        np.testing.assert_allclose(sl.apply_1q(base, sl.SDG_H, 2, 4), Statevector(qc3).data, atol=1e-12)

    def test_x_measurement_of_a_zero_qubit_is_a_fair_coin(self):
        psi = np.zeros(8, complex); psi[0] = 1
        src = sl.StateSource(psi, dict(patches=[0], lamps=[1, 2], polarity=[], n_qubits=3), measure_basis={0: "x"})
        shots = src.pool(20000, np.random.default_rng(0)).shots[:, 0]
        self.assertAlmostEqual(shots.mean(), 0.0, delta=0.03)


class QasmLoading(unittest.TestCase):
    def test_graph_state_with_local_rotations_round_trips_through_qasm3(self):
        from qiskit import QuantumCircuit, qasm3
        from qiskit.quantum_info import Statevector
        n = 8
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7), (1, 6)]:
            qc.cz(a, b)
        for q in range(n):
            qc.ry(0.3 * (q + 1), q); qc.rz(0.2 * q, q)
        text = qasm3.dumps(qc)
        psi, k = sl.statevector_from_qasm(text)
        self.assertEqual(k, n)
        ref = Statevector(qc).data
        self.assertAlmostEqual(abs(np.vdot(ref, psi)), 1.0, places=9)           # equal up to global phase

    def test_handwritten_qasm3_and_final_measurements_are_dropped(self):
        text = '''OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
cx q[0], q[1];
ry(0.5) q[2];
c = measure q;
'''
        psi, n = sl.statevector_from_qasm(text)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(float(np.linalg.norm(psi)), 1.0, places=9)
        p = np.abs(psi) ** 2
        self.assertAlmostEqual(p[0b000] + p[0b011], np.cos(0.25) ** 2, places=9)   # Bell pair on q0,q1 (|00>,|11>), q2 = ry(0.5)|0> is 0 w.p. cos^2(0.25)

    def test_wrong_qubit_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "layout needs"):
            sl.source_from_qasm('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\nh q[0];\n', LAYOUT)


class OracleMatchesTheMock(unittest.TestCase):
    def test_oracle_z_moments_equal_exact_enumeration_of_the_mock(self):
        """Independent code paths and opposite bit conventions must agree to rounding error."""
        edges = patch_edges(SAMPLER.J)
        n, ne = SAMPLER.n, len(edges)
        exact = estimate_patches_exact(SAMPLER, edges)               # [<s_p>, <s_a s_b>, L1*s_p, L2*s_p] with <.> over the mock model
        pairs = list(edges) + [(p, LAYOUT["lamps"][0]) for p in range(n)] + [(p, LAYOUT["lamps"][1]) for p in range(n)]
        single, zz = ORACLE.z_moments(pairs)
        np.testing.assert_allclose(single[:n], exact[:n], atol=1e-9)
        np.testing.assert_allclose(zz, exact[n:], atol=1e-9)
        np.testing.assert_allclose(single[LAYOUT["lamps"]], 0, atol=1e-12)       # lamps are uniform
        a, b = LAYOUT["polarity"][:2]
        self.assertAlmostEqual(float(ORACLE.z_moments([(a, b)])[1][0]), float(np.tanh(0.9)), places=9)

    def test_sampled_pool_reproduces_coherence_metrics_of_the_mock(self):
        pool = ORACLE.pool(60000, np.random.default_rng(1))
        m = coherence_metrics(SAMPLER, pool.lamps, pool.shots, pool.pol)
        self.assertGreater(m["coplanar_agreement"], 0.88)
        self.assertLess(m["crease_agreement"], 0.0)
        self.assertGreater(m["light_left_right"], 0.95)
        self.assertAlmostEqual(m["polarity_agreement"], np.tanh(0.9), delta=0.03)


class Frames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = ORACLE.pool(40000, np.random.default_rng(2))
        cls.gen = FrameGenerator(SCENE, SAMPLER, cls.pool)

    def test_frames_are_conditioned_on_their_lighting_world_and_polarity_is_joint(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            d = draw_frame(self.gen.index, 8, rng)
            self.assertTrue(np.all(self.pool.lamps[d.shot_ids] == np.array(d.lamp)))
            self.assertEqual(d.pol, tuple(int(v) for v in self.pool.pol[d.shot_ids[0]]))
            self.assertTrue(np.all((d.lit >= 0) & (d.lit <= 1)))

    def test_holding_lamp_and_polarity(self):
        d = draw_frame(self.gen.index, 4, np.random.default_rng(4), lamp=(1, -1), pol=(1, 1, 1))
        self.assertEqual((d.lamp, d.pol), ((1, -1), (1, 1, 1)))

    def test_partial_holds_fix_only_what_is_named(self):
        rng = np.random.default_rng(6)
        seen_l2, seen_pol = set(), set()
        for _ in range(60):
            d = draw_frame(self.gen.index, 3, rng, lamp=(-1, None), pol=(None, 1, None))
            self.assertEqual(d.lamp[0], -1)
            self.assertEqual(d.pol[1], 1)
            seen_l2.add(d.lamp[1]); seen_pol.add((d.pol[0], d.pol[2]))
        self.assertEqual(seen_l2, {-1, 1})            # the free lamp bit and the free panels still vary
        self.assertGreater(len(seen_pol), 1)

    def test_impossible_condition_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "no shots"):
            draw_frame(PoolIndex(sl.Pool(np.array([[1, 1]]), np.ones((1, SAMPLER.n)), np.ones((1, 3)))), 1, np.random.default_rng(0), lamp=(-1, -1))

    def test_composed_frames_keep_glass_exactly_black_and_vary(self):
        imgs = [img for img, _ in self.gen.frames(6, 4, np.random.default_rng(5))]
        for img in imgs:
            self.assertTrue(np.all(img[SCENE.impenetrable] == 0))
            self.assertGreater(img.max(), 0.5)
        self.assertGreater(np.abs(imgs[0] - imgs[3]).max(), 0.05)


if __name__ == "__main__":
    unittest.main()
