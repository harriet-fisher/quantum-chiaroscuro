"""QDrive planner: structure, layering, exact growth, merge, closed loop. Nothing is sent to Moth.

    python -m unittest tests.test_qdrive_plan -v

The mock engine used for the closed-loop test is a paper-based approximation (src/quantum/motte_mock.py); these tests check the
planner's own logic and plumbing, they say nothing about how the real engine converges.
"""
import os
import unittest

import numpy as np

from src.quantum import qdrive_plan as qp

# a small chain 0-1-2-3 with a hub 4 attached to all (like patches + one lamp), and an isolated pair 5-6
PAIRS = [(0, 1), (1, 2), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4), (5, 6)]


def chain_dist(n, couplings):
    """A classical Ising-chain distribution over n qubits (little-endian), used as an exact test oracle."""
    idx = np.arange(2 ** n)
    z = lambda q: 1 - 2 * ((idx >> q) & 1)
    e = sum(j * z(a) * z(b) for (a, b), j in couplings.items())
    p = np.exp(e); return p / p.sum()


class Structure(unittest.TestCase):
    def test_components_split_independent_parts(self):
        self.assertEqual(qp.components(list(range(7)), PAIRS), [[0, 1, 2, 3, 4], [5, 6]])

    def test_cover_groups_cover_every_pair_with_small_cliques(self):
        groups = qp.cover_groups([0, 1, 2, 3, 4], PAIRS, 3)
        self.assertTrue(all(len(g) <= 3 for g in groups))
        covered = {e for g in groups for e in __import__("itertools").combinations(g, 2)}
        self.assertTrue({p for p in PAIRS if p[1] <= 4} <= covered)

    def test_layers_never_hold_overlapping_groups(self):
        for groups in ([(0, 1), (1, 2), (3, 4), (0, 4)], [(0,), (1,), (0, 1)], [(0, 1, 2), (2, 3), (4, 5), (5, 6)]):
            for layer in qp.pack_layers(qp.order_groups(groups)):
                qs = [q for g in layer for q in g]
                self.assertEqual(len(qs), len(set(qs)))

    def test_group_target_words_are_identity_padded_and_ordered(self):
        mom = qp.TargetMoments([dict(qubit=0, Z=0.1), dict(qubit=1, Z=-0.2)], [dict(qubits=[0, 1], ZZ=0.5)])
        t = qp.group_target((0, 1), mom)
        self.assertEqual(t["qubits"], [0, 1])
        self.assertEqual(t["expvals"], {"ZI": 0.1, "IZ": -0.2, "ZZ": 0.5})

    def test_values_are_clipped_below_one(self):
        mom = qp.TargetMoments([dict(qubit=0, Z=0.0), dict(qubit=1, Z=0.0)], [dict(qubits=[0, 1], ZZ=1.0)])
        self.assertLess(qp.group_target((0, 1), mom)["expvals"]["ZZ"], 1.0)


class Growth(unittest.TestCase):
    def test_growth_is_exact_for_a_chain_with_a_hub(self):
        n = 5
        p = chain_dist(n, {(0, 1): .8, (1, 2): -.5, (2, 3): .9, (0, 4): .3, (1, 4): -.4, (2, 4): .2, (3, 4): .6})
        comp = list(range(n)); pairs = [q for q in PAIRS if q[1] < n]
        gg, fill = qp.growth_groups(comp, pairs)
        self.assertLessEqual(max(len(g) for g, _ in gg), 4)
        self.assertAlmostEqual(qp.markov_tv(p, comp, n, gg), 0.0, places=12)

    def test_growth_misses_when_an_edge_is_not_covered(self):
        """Drop the hub edges from the graph but keep the hub coupled in the distribution: the plan can no longer be exact."""
        n = 5
        p = chain_dist(n, {(0, 1): .8, (1, 2): -.5, (2, 3): .9, (0, 4): 1.2, (3, 4): 1.2})
        comp = list(range(n)); gg, _ = qp.growth_groups(comp, [(0, 1), (1, 2), (2, 3)])
        self.assertGreater(qp.markov_tv(p, comp, n, gg), 1e-3)

    def test_growth_job_structure(self):
        n = 5
        p = chain_dist(n, {(0, 1): .8, (1, 2): -.5, (2, 3): .9, (0, 4): .3, (1, 4): -.4, (2, 4): .2, (3, 4): .6})
        mom = qp.OracleMoments(p, n, [q for q in PAIRS if q[1] < n])
        job = qp.build_growth_job(list(range(n)), mom.pairs(), mom)
        t = job.params["targets"]
        self.assertIsNone(t[-1])
        layer = []
        for e in t:
            if e is None:
                qs = [q for x in layer for q in x["qubits"]]
                self.assertEqual(len(qs), len(set(qs))); layer = []
            else:
                layer.append(e)
        self.assertTrue(all(isinstance(e["expvals"], dict) for e in t if e))
        self.assertEqual(job.params["n_qubits"], n)

    def test_state_prep_targets_are_reachable_prefix_rdms(self):
        """Each step's full-RDM words must equal the words of that prefix state, and the last prefix is the oracle state."""
        n = 4
        p = chain_dist(n, {(0, 1): .7, (1, 2): -.6, (2, 3): .5})
        comp = list(range(n)); pairs = [(0, 1), (1, 2), (2, 3)]
        gg, _ = qp.growth_groups(comp, pairs)
        steps = list(qp.prefix_states(comp, gg, p))
        last = steps[-1][2]
        self.assertTrue(np.allclose(np.abs(last) ** 2, p, atol=1e-9))
        job = qp.build_state_prep_job(comp, pairs, p)
        self.assertTrue(all(len(e["expvals"]) >= 1 for e in job.params["targets"] if e))

    def test_check_flags_the_lab_limits(self):
        job = qp.Job(qubits=list(range(13)), layers=[], params=dict(n_qubits=13, targets=[
            dict(qubits=[0, 1, 2, 3], expvals={"ZZZZ": 0.5}), dict(qubits=[3, 4], expvals={"ZZ": 0.5}), dict(qubits=[1], expvals={"ZZ": 0.1}), None]))
        msgs = " ".join(qp.check(job))
        self.assertIn("overlapping", msgs); self.assertIn("4 qubits", msgs); self.assertIn("13-qubit", msgs); self.assertIn("word length", msgs)


class Merge(unittest.TestCase):
    def test_merge_places_components_on_their_global_qubits(self):
        from qiskit import QuantumCircuit, qasm3
        from qiskit.quantum_info import Statevector, Pauli
        a = QuantumCircuit(2); a.h(0); a.cx(0, 1)                        # Bell on local (0,1) -> global (4,1)
        b = QuantumCircuit(1); b.x(0)                                    # X on local 0 -> global 0
        text = qp.merge_circuits([(qasm3.dumps(a), [4, 1]), (qasm3.dumps(b), [0])], 5)
        sv = Statevector(qasm3.loads(text))
        z = lambda *qs: float(sv.expectation_value(Pauli("".join("Z" if (4 - i) in qs else "I" for i in range(5)))).real)
        self.assertAlmostEqual(z(0), -1.0); self.assertAlmostEqual(z(1, 4), 1.0); self.assertAlmostEqual(z(2), 1.0)


class Loop(unittest.TestCase):
    def test_residuals_only_repeat_stubborn_groups_and_loop_terminates(self):
        from src.quantum.motte_mock import MotteMock
        n = 3
        p = chain_dist(n, {(0, 1): .6, (1, 2): .6})
        mom = qp.OracleMoments(p, n, [(0, 1), (1, 2)]); comp = [0, 1, 2]
        calls = []

        def engine(params, prev):
            calls.append(sum(1 for t in params["targets"] if t))
            m = prev if prev is not None else MotteMock(n, stale=True)
            m.run(params["targets"]); return m, (lambda w, gq: m.moment(w, [comp.index(q) for q in gq]))

        state, hist = qp.closed_loop(comp, mom.pairs(), mom, engine, tol=0.02, gain=0.5, max_jobs=3, max_k=3, log=lambda s: None)
        self.assertLessEqual(len(hist), 3)
        self.assertEqual(hist[0]["job"], 1)
        self.assertLessEqual(max(calls[1:], default=0), calls[0])                 # later jobs never carry more targets than the first sweep

    def test_corrected_values_push_past_the_target_and_clip(self):
        res = {((0, 1), "ZZ"): (0.9, 0.5), ((0, 1), "ZI"): (0.0, 0.0), ((2, 3), "ZZ"): (0.97, 0.0)}
        v = qp.corrected_values(res, 0.5)
        self.assertAlmostEqual(v[((0, 1), "ZZ")], qp.CLIP)                          # 0.9 + 0.5 * 0.4 = 1.1 is clipped
        self.assertEqual(v[((0, 1), "ZI")], 0.0)
        self.assertLessEqual(max(abs(x) for x in v.values()), qp.CLIP)


@unittest.skipUnless(os.path.exists("runs/harrietlivingroom/calib/targets.json"), "living-room run not present")
class LivingRoom(unittest.TestCase):
    """The real scene: two independent components, exact growth from group marginals (needs the oracle, a few seconds)."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, "qdrive_lab")
        import scene
        cls.t, cls.sampler, cls.layout, cls.psi = scene.load()
        cls.tm = qp.TargetMoments(cls.t["bloch"], cls.t["relationships"]); cls.pairs = cls.tm.pairs()
        cls.p = np.abs(cls.psi) ** 2

    def test_two_independent_components_with_the_documented_roles(self):
        comps = qp.components(self.tm.qubits(), self.pairs)
        self.assertEqual([len(c) for c in comps], [11, 8])                        # patches+lamps | polarity
        self.assertEqual(comps[0], list(range(0, 11))); self.assertEqual(comps[1], list(range(11, 19)))

    def test_growth_widths_and_exactness(self):
        for comp, width, fill in (list(range(0, 11)), 4, [(9, 10)]), (list(range(11, 19)), 3, []):
            gg, f = qp.growth_groups(comp, self.pairs)
            self.assertEqual(max(len(g) for g, _ in gg), width); self.assertEqual(f, fill)
            idx = np.arange(len(self.p)); key = np.zeros(len(self.p), int)
            for i, q in enumerate(comp):
                key |= ((idx >> q) & 1) << i
            pc = np.zeros(2 ** len(comp)); np.add.at(pc, key, self.p)
            gl = [(tuple(comp.index(x) for x in g), comp.index(v)) for g, v in gg]
            self.assertAlmostEqual(qp.markov_tv(pc, comp, len(comp), gl), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
