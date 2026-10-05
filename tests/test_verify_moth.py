"""The Moth tomography check of the relief circuit: the request, scoring, the local rehearsal, caching and the spend gate. A recording fake stands in for
Moth; nothing here can reach the network.

    python -m unittest tests.test_verify_moth -v
"""
import json
import os
import tempfile
import unittest

import numpy as np

from src.capture.labels import bay_window_labels
from src.quantum import aer_relief as ar
from src.quantum import moth_engines as me
from src.quantum import verify_moth as vm

TMP = tempfile.TemporaryDirectory()
AER = os.path.join(TMP.name, "aer")
LABELS = bay_window_labels()
SPEC = None


def setUpModule():
    global SPEC
    ar.run(LABELS, None, AER, shots=100_000, seed=3, log=lambda m: None)
    SPEC = ar.build_spec(LABELS)[2]


def tearDownModule():
    TMP.cleanup()


class FakeMoth:
    """What the Moth service would return, as far as the parser can tell: exact moments (optionally nudged), saved like the real client saves them."""
    is_local = False

    def __init__(self, nudge=0.0, result=None):
        self.calls, self.nudge, self.result = [], nudge, result

    def run(self, engine_id, params, *, approved=False, out_dir=None, input_files=None):
        self.calls.append((engine_id, approved))
        assert approved is True
        if self.result is None:
            res = vm.LocalTomographyClient(SPEC).run(engine_id, params)["result"]
            if self.nudge:
                for v in res["result"]["output"]["tomography"]["bloch"].values():
                    v["X"] += self.nudge
        else:
            res = self.result
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "job_fake-1.json"), "w") as f:
            json.dump({"job_id": "fake-1"}, f)
        with open(os.path.join(out_dir, "raw_result_fake-1.json"), "w") as f:
            json.dump(res, f)
        return dict(job_id="fake-1", result=res, inline=res.get("result"), files={})


def out(name):
    return os.path.join(TMP.name, name)


class TheRequest(unittest.TestCase):
    def test_it_is_the_reference_circuit_with_the_pairs_that_carry_the_physics(self):
        spec, params, sha, job = vm.plan(LABELS)
        self.assertEqual((job.credits, job.problems()), (1, []))
        self.assertEqual(params["qubit_list"], list(range(spec.n_sim)))
        self.assertIn("qreg q[18]", params["circuit_qasm"])
        from src.quantum.moth_client import ENGINES
        self.assertTrue(set(params) <= ENGINES[vm.ENGINE]["params"])
        self.assertEqual(sha, vm.plan(LABELS)[2])
        self.assertNotEqual(sha, vm.plan(LABELS, dict(entangle=0.3))[2])
        self.assertNotEqual(sha, vm.plan(LABELS, shots=1000)[2])

    def test_a_preview_sends_nothing_and_needs_no_client(self):
        folder = out("preview")
        self.assertIsNone(vm.run(LABELS, None, folder, client=FakeMoth(), log=lambda m: None))
        self.assertFalse(os.path.exists(os.path.join(folder, "state.json")))

    def test_the_approval_must_equal_the_cost(self):
        for credits in (0, 2, 5):
            with self.assertRaises(SystemExit):
                vm.run(LABELS, None, out("wrong"), approve_credits=credits, client=FakeMoth(), log=lambda m: None)


class ScoringAResult(unittest.TestCase):
    def test_a_faithful_result_scores_zero_and_agrees_with_the_aer_run(self):
        folder, fake = out("faithful"), FakeMoth()
        st = vm.run(LABELS, None, folder, approve_credits=1, client=fake, aer_dir=AER, log=lambda m: None)
        self.assertEqual((fake.calls, st["credits"], st["local"], st["job_id"]), ([(vm.ENGINE, True)], 1, False, "fake-1"))
        self.assertLess(st["score"]["rms_single"], 1e-9)
        self.assertLess(st["score"]["rms_pairs"], 1e-9)
        self.assertEqual(st["score"]["n_pair_values"], 225)
        self.assertLess(st["aer_comparison"]["max_abs_difference"], 0.05)           # Moth's exact visibility vs the Aer witness run's sampled one
        for name in ("state.json", "score.json", "payload.sha256", "raw_result_fake-1.json"):
            self.assertTrue(os.path.exists(os.path.join(folder, name)), name)
        self.assertLess(json.load(open(os.path.join(folder, "score.json")))["rms_single"], 1e-9)

    def test_a_poor_result_is_reported_as_poor_not_hidden(self):
        st = vm.run(LABELS, None, out("poor"), approve_credits=1, client=FakeMoth(nudge=0.2), aer_dir=AER, log=lambda m: None)
        self.assertGreater(st["score"]["rms_single"], 0.05)
        self.assertGreater(st["aer_comparison"]["max_abs_difference"], 0.1)

    def test_a_result_for_the_same_payload_is_reused_and_costs_nothing(self):
        folder = out("cache")
        vm.run(LABELS, None, folder, approve_credits=1, client=FakeMoth(), log=lambda m: None)
        again = FakeMoth()
        st = vm.run(LABELS, None, folder, client=again, log=lambda m: None)
        self.assertEqual((again.calls, st["credits"], st["job_id"]), ([], 0, "fake-1"))

    def test_an_unrecognised_result_keeps_the_raw_file_and_says_what_it_saw(self):
        folder = out("odd")
        with self.assertRaises(ValueError) as cm:
            vm.run(LABELS, None, folder, approve_credits=1, client=FakeMoth(result={"outputs": None, "result": {"mystery": 1}}), log=lambda m: None)
        self.assertIn("mystery", str(cm.exception))
        self.assertTrue(os.path.exists(os.path.join(folder, "raw_result_fake-1.json")))            # what was paid for is never lost to a parser
        self.assertFalse(os.path.exists(os.path.join(folder, "state.json")))

    def test_a_circuit_from_another_drawing_is_not_compared_with_this_aer_run(self):
        other = ar.build_spec(LABELS, entangle=0.3)[2]
        self.assertIsNone(vm.compare_with_aer({}, other, AER))


class TheLocalRehearsal(unittest.TestCase):
    def test_it_is_ideal_never_called_a_moth_result_and_writes_no_score_json(self):
        folder = out("local")
        st = vm.run(LABELS, None, folder, local=True, aer_dir=AER, log=lambda m: None)
        self.assertTrue(st["local"])
        self.assertEqual(st["credits"], 0)
        self.assertIn("not a Moth result", st["note"])
        self.assertLess(st["score"]["rms_single"], 1e-9)
        self.assertFalse(os.path.exists(os.path.join(folder, "score.json")))           # the show prints score.json as a Moth result
        self.assertIsNotNone(vm.read_state(folder))

    def test_the_stand_in_answers_in_the_shape_the_parser_reads(self):
        res = vm.LocalTomographyClient(SPEC).run(vm.ENGINE, vm.plan(LABELS)[1])["result"]
        bloch, rel, _ = me.parse_tomography(res)
        self.assertEqual(len(bloch), SPEC.n_sim)
        self.assertAlmostEqual(bloch[SPEC.F][0], SPEC.visibility(0), places=6)           # the polarity qubit's X is the interference visibility
        self.assertTrue(rel)


if __name__ == "__main__":
    unittest.main()
