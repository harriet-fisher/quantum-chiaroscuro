"""Superposed Relief executed on Aer: the run, its check against the reference state, the shot-pool sampler the show draws from, and a session that uses it.

    python -m unittest tests.test_aer_relief -v

One 22-qubit run (200k shots, about three seconds) is made once for the whole module.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.capture.labels import bay_window_labels
from src.quantum import aer_relief as ar
from src.quantum import relief_witness as rw
from src.quantum.relief_state import ReliefSpec, ReliefState

TMP = tempfile.TemporaryDirectory()
RUN = os.path.join(TMP.name, "aer")
SHOTS = 200_000
META = None
SPEC = None


def setUpModule():
    global META, SPEC
    META = ar.run(bay_window_labels(), None, RUN, shots=SHOTS, seed=5, log=lambda m: None)
    SPEC = ar.build_spec(bay_window_labels())[2]


def tearDownModule():
    TMP.cleanup()


class TheRun(unittest.TestCase):
    def test_it_executes_the_whole_circuit_and_saves_what_was_measured(self):
        self.assertEqual((META["n_qubits"], META["F"], META["P"], META["shots"]), (22, 12, 6, SHOTS))
        for name in ("shots.npz", "circuit.qasm", "state.json"):
            self.assertTrue(os.path.exists(os.path.join(RUN, name)), name)
        with np.load(os.path.join(RUN, "shots.npz"), allow_pickle=False) as z:
            for m in "zx":
                self.assertEqual(int(z[f"{m}_c"].sum()), SHOTS)
                self.assertLess(int(z[f"{m}_x"].max()), 2 ** 12)
                self.assertLess(int(z[f"{m}_o"].max()), 2 ** 6)
                self.assertLessEqual(int(z[f"{m}_g"].max()), 3)
        with open(os.path.join(RUN, "circuit.qasm")) as f:
            self.assertIn("OPENQASM 3", f.read())

    def test_the_shots_agree_with_the_reference_state_within_shot_noise(self):
        sc = META["score"]
        self.assertTrue(sc["consistent"], sc)
        self.assertLess(sc["tv_wo"], 1.5 * sc["tv_wo_sampling_floor"] + 1e-3)
        self.assertLess(sc["facet_marginal_max_error"], sc["facet_marginal_bound"])

    def test_the_check_would_catch_a_run_of_the_wrong_circuit(self):
        other = ar.build_spec(bay_window_labels(), kappa=2.0)[2]               # a different relief: the same shots must not pass as its sampling
        with np.load(os.path.join(RUN, "shots.npz"), allow_pickle=False) as z:
            pool = tuple(z[f"z_{n}"] for n in "gwoxc")
        self.assertFalse(ar.score(other, pool)["consistent"])

    def test_the_witness_run_on_aer_sees_the_stabilizer_and_the_visibility(self):
        for r in META["witness"]["panels"]:
            self.assertGreater(r["stabilizer"], 0.99)                          # <X_B Z^F> = 1 for the ideal state
            self.assertLess(abs(r["visibility"] - r["visibility_exact"]), 5 * r["visibility_se"] + 0.01)

    def test_a_run_for_the_same_circuit_shots_and_seed_is_reused_not_executed(self):
        with mock.patch.object(ar, "run_circuit", side_effect=AssertionError("executed again")):
            again = ar.run(bay_window_labels(), None, RUN, shots=SHOTS, seed=5, log=lambda m: None)
        self.assertEqual(again["sha256"], META["sha256"])

    def test_the_plan_hash_follows_the_circuit_the_shots_and_the_seed(self):
        base = ar.plan_sha(SPEC, SHOTS, 5)
        self.assertNotEqual(base, ar.plan_sha(SPEC, SHOTS + 1, 5))
        self.assertNotEqual(base, ar.plan_sha(SPEC, SHOTS, 6))
        self.assertNotEqual(base, ar.plan_sha(ar.build_spec(bay_window_labels(), entangle=0.3)[2], SHOTS, 5))
        self.assertEqual(base, ar.plan_sha(ar.build_spec(bay_window_labels())[2], SHOTS, 5))

    def test_a_circuit_too_big_for_a_statevector_is_refused_with_the_reason(self):
        n = ar.LIMIT                                                           # n facets + 2 polarity + 4 > LIMIT
        big = ReliefSpec(np.full(n, 0.3), np.zeros(n), np.arange(n) % 2, [0.0, 0.0])
        self.assertTrue(any("statevector" in p for p in ar.check_size(big)))
        self.assertEqual(ar.check_size(SPEC), [])


class TheSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aer = ar.AerReliefState(SPEC, RUN)
        cls.ref = ReliefState(SPEC)

    def frames(self, state, n, seed, **kw):
        rng = np.random.default_rng(seed)
        return [state.draw(32, rng, **kw) for _ in range(n)]

    def test_looks_have_the_distribution_of_the_reference_state(self):
        for mode in ("z", "x"):
            a, r = self.frames(self.aer, 3000, 1, lamp_mode=mode), self.frames(self.ref, 3000, 2, lamp_mode=mode)
            np.testing.assert_allclose(np.mean([d.lit for d in a], 0), np.mean([d.lit for d in r], 0), atol=0.03)
            np.testing.assert_allclose(np.mean([d.pol for d in a], 0), np.mean([d.pol for d in r], 0), atol=0.09)
            self.assertEqual(len(a[0].lit), SPEC.F + SPEC.n_classical)
        obs = np.bincount([d.obs for d in self.frames(self.aer, 4000, 3)], minlength=4) / 4000
        np.testing.assert_allclose(obs, 0.25, atol=0.03)

    def test_a_look_carries_the_outcomes_of_one_measured_shot(self):
        rng = np.random.default_rng(4)
        with np.load(os.path.join(RUN, "shots.npz"), allow_pickle=False) as z:
            seen = {(int(g), int(w), int(o)) for g, w, o in zip(z["z_g"], z["z_w"], z["z_o"])}
        for _ in range(300):
            d = self.aer.draw(8, rng)
            o = sum(1 << p for p, v in enumerate(d.pol) if v < 0)
            self.assertIn((d.obs, d.world, o), seen)

    def test_the_light_world_can_be_pinned_for_experiments_and_pins_it(self):
        rng = np.random.default_rng(5)
        self.assertEqual({self.aer.draw(8, rng, g=3, world=2).world for _ in range(50)}, {2})

    def test_the_controls_are_mixtures_and_stay_classical(self):
        rng = np.random.default_rng(6)
        for control in ("noise", "dephased"):
            d = self.aer.draw(16, rng, control=control)
            self.assertEqual(d.control, control)
            self.assertEqual(len(d.lit), SPEC.F + SPEC.n_classical)

    def test_a_run_made_for_another_drawing_or_other_settings_is_refused(self):
        other = ar.build_spec(bay_window_labels(), kappa=0.5)[2]
        with self.assertRaises(ValueError) as cm:
            ar.AerReliefState(other, RUN)
        self.assertIn("different", str(cm.exception))
        with self.assertRaises(ValueError):
            ar.AerReliefState(SPEC, os.path.join(TMP.name, "nowhere"))

    def test_witnesses_are_measured_on_aer_and_say_so(self):
        cert = rw.certificate(self.aer, np.random.default_rng(7), 20000, chsh=False)
        self.assertIn("Aer", cert["sampled_by"])
        self.assertGreater(min(cert["sampled"]["coherent"]["stabilizer"]), 0.97)
        self.assertLess(max(np.abs(cert["sampled"]["dephased"]["stabilizer"])), 0.1)          # the controls stay what they were
        self.assertTrue(any("measured by" in line for line in rw.describe_certificate(cert)))
        self.assertNotIn("Aer", rw.certificate(self.ref, np.random.default_rng(7), 2000, chsh=False)["sampled_by"])


class InTheShow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.ui.session import Session
        cls.session = Session(bay_window_labels(), out_dir=os.path.join(TMP.name, "show"), pool_size=2000, run_dir=os.path.join(TMP.name, "run"), calib_dir=os.path.join(TMP.name, "c"),
                              projector_size=(640, 360), aer_run=RUN, complementary_report=os.path.join(TMP.name, "none.json"))

    def test_the_relief_source_draws_from_the_aer_run_and_says_where_it_came_from(self):
        s = self.session
        self.assertEqual(s.prov["executed"], "aer")
        self.assertTrue(s.prov["quantum_backed"] and not s.prov["from_moth"])
        self.assertIn("Aer run of the full circuit", s.prov["label"])
        self.assertIsInstance(s.rstate, ar.AerReliefState)
        from src.ui import caption
        self.assertIn("Aer simulator", caption.provenance_line(s.prov))
        s.next_new()
        self.assertEqual(s.current.source, "relief")
        self.assertEqual(len(s.current.draw.lit), SPEC.F + SPEC.n_classical)

    def test_the_operator_witness_run_is_measured_on_aer(self):
        out = self.session.witness_run(shots=4000)
        self.assertIn("Aer", out["certificate"]["sampled_by"])

    def test_the_domain_engine_and_a_circuit_file_cannot_be_combined_with_an_aer_run(self):
        from src.ui.session import Session
        with self.assertRaises(ValueError):
            Session(bay_window_labels(), out_dir=os.path.join(TMP.name, "d"), engine="domain", aer_run=RUN, complementary_report=os.path.join(TMP.name, "none.json"))
        with self.assertRaises(ValueError):
            Session(bay_window_labels(), out_dir=os.path.join(TMP.name, "e"), circuit=os.path.join(TMP.name, "x.qasm"), aer_run=RUN, complementary_report=os.path.join(TMP.name, "none.json"))


if __name__ == "__main__":
    unittest.main()
