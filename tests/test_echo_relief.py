"""Dynamic relief from the echo: the taps (local, or Moth's engine through a recording fake), one circuit per echo depth executed on Aer, and the sampler that steps
through the depths. Nothing here can reach the network.

    python -m unittest tests.test_echo_relief -v
"""
import json
import os
import tempfile
import unittest

import numpy as np

from src.capture.labels import bay_window_labels
from src.quantum import aer_relief as ar
from src.quantum import domain_moth as dm
from src.quantum import echo_relief as er
from src.quantum import moth_engines as me
from src.quantum.relief_state import ReliefState

LABELS = bay_window_labels()
TMP = tempfile.TemporaryDirectory()
LOG = lambda m: None
SCENE, FS, SPEC = ar.build_spec(LABELS)
RUN = os.path.join(TMP.name, "echo")
META = None


def setUpModule():
    global META
    META = er.run(LABELS, None, RUN, "local", depth=3, shots=60_000, seed=9, log=LOG)


def tearDownModule():
    TMP.cleanup()


class FakeEcho:
    """Moth's otoc-echo-v1 as far as the tap parser can tell: the local echo's taps, in the documented field names, saved like the real client saves them."""
    is_local = False

    def __init__(self, result=None):
        self.calls, self.result = [], result

    def run(self, engine_id, params, *, approved=False, out_dir=None, input_files=None):
        assert approved is True
        self.calls.append(engine_id)
        res = self.result or {"outputs": None, "result": {"output": {"taps": [dict(site=s, depth=d, F_re=re, F_im=im) for s, d, re, im in
                                                                                  me.local_echo(params["width"], params["height"], params["depth"], kick_site=params["kick_site"])]}}}
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "job_echo-1.json"), "w") as f:
            json.dump({"job_id": "echo-1"}, f)
        with open(os.path.join(out_dir, "raw_result_echo-1.json"), "w") as f:
            json.dump(res, f)
        return dict(job_id="echo-1", result=res, inline=res.get("result"), files={})


def out(name):
    return os.path.join(TMP.name, name)


class TheTaps(unittest.TestCase):
    def test_facets_map_to_lattice_sites_over_the_canvas(self):
        sites = er.lattice_sites(FS, SCENE, 4, 4)
        self.assertEqual(len(sites), SPEC.F)
        self.assertTrue(all(0 <= s < 16 for s in sites))
        self.assertGreater(len(set(sites)), 3)
        self.assertEqual(er.lattice_sites(FS, SCENE, 1, 1), [0] * SPEC.F)

    def test_the_tap_parser_finds_the_list_wherever_it_sits_and_says_what_it_saw_when_it_cannot(self):
        taps = [(1, 1, 0.5, -0.25), (3, 2, -0.1, 0.2)]
        rows = [dict(site=s, depth=d, F_re=a, F_im=b) for s, d, a, b in taps]
        for wrapped in ({"result": {"output": {"taps": rows}}}, {"taps": rows}, {"result": json.dumps({"a": {"b": rows}})}, {"result": {"taps": [list(t) for t in taps]}}):
            self.assertEqual(er.parse_taps(wrapped), taps)
        with self.assertRaises(ValueError) as cm:
            er.parse_taps({"result": {"mystery": 1}})
        self.assertIn("mystery", str(cm.exception))

    def test_lattice_and_depth_limits_follow_the_source(self):
        self.assertEqual(er.check_lattice(4, 4, 4, "local"), [])
        self.assertTrue(er.check_lattice(5, 4, 4, "local"))
        self.assertEqual(er.check_lattice(6, 4, 4, "moth"), [])
        self.assertTrue(er.check_lattice(5, 5, 4, "moth"))
        self.assertTrue(er.check_lattice(4, 4, 0, "local") and er.check_lattice(4, 4, er.MAX_DEPTH + 1, "local"))

    def test_the_depth_circuits_modulate_tilt_and_azimuth_and_keep_the_geometry(self):
        taps = me.local_echo(4, 4, 3, kick_site=8)
        specs = er.depth_specs(SPEC, er.lattice_sites(FS, SCENE, 4, 4), taps, 3, 16)
        self.assertEqual(len(specs), 3)
        self.assertGreater(max(np.abs(a.tau - b.tau).max() for a in specs for b in specs), 0.05)           # the depths really differ
        for sp in specs:
            self.assertEqual((sp.F, sp.P, sp.n_full), (SPEC.F, SPEC.P, SPEC.n_full))
            self.assertEqual(ar.check_size(sp), [])
            self.assertNotEqual(ar.spec_fingerprint(sp), ar.spec_fingerprint(SPEC))


class TheRun(unittest.TestCase):
    def test_one_circuit_per_depth_is_executed_on_aer_and_each_agrees_with_its_own_reference(self):
        self.assertEqual((META["depth"], META["taps_source"], META["credits"], META["n_qubits"]), (3, "local", 0, 22))
        self.assertEqual(len(META["scores"]), 3)
        self.assertTrue(META["consistent"], META["scores"])
        with np.load(os.path.join(RUN, "shots.npz"), allow_pickle=False) as z:
            for d in range(3):
                self.assertEqual(int(z[f"d{d}_z_c"].sum()), 60_000)
        self.assertEqual(META["modes"], ["z"])
        for name in ("state.json", "shots.npz", "taps.json"):
            self.assertTrue(os.path.exists(os.path.join(RUN, name)), name)

    def test_the_same_request_is_reused_and_a_different_one_is_not(self):
        import unittest.mock as mock
        with mock.patch.object(ar, "execute", side_effect=AssertionError("executed again")):
            self.assertEqual(er.run(LABELS, None, RUN, "local", depth=3, shots=60_000, seed=9, log=LOG)["sha256"], META["sha256"])
        base = er.plan_sha(SPEC, "local", 4, 4, 3, 8, 60_000, 9)
        self.assertEqual(base, META["sha256"])
        for other in (er.plan_sha(SPEC, "local", 4, 4, 4, 8, 60_000, 9), er.plan_sha(SPEC, "local", 3, 4, 3, 6, 60_000, 9), er.plan_sha(SPEC, "moth", 4, 4, 3, 8, 60_000, 9, "x")):
            self.assertNotEqual(base, other)

    def test_what_cannot_run_is_refused_with_the_reason_before_anything_executes(self):
        for kw in (dict(width=5, height=4), dict(depth=er.MAX_DEPTH + 1), dict(kick_site=99)):
            with self.assertRaises(er.EchoError):
                er.run(LABELS, None, out("refused"), "local", log=LOG, **kw)
        self.assertFalse(os.path.exists(out("refused")))


class TheMothTaps(unittest.TestCase):
    def test_a_preview_sends_nothing(self):
        fake = FakeEcho()
        self.assertIsNone(er.run(LABELS, None, out("m_preview"), "moth", depth=2, shots=20_000, client=fake, log=LOG))
        self.assertEqual(fake.calls, [])

    def test_the_approval_must_equal_the_cost(self):
        for credits in (0, 2):
            with self.assertRaises(SystemExit):
                er.run(LABELS, None, out("m_wrong"), "moth", depth=2, shots=20_000, approve_credits=credits, client=FakeEcho(), log=LOG)

    def test_the_moth_taps_drive_the_aer_circuits_and_the_result_is_cached_for_the_next_run(self):
        folder, fake = out("moth"), FakeEcho()
        st = er.run(LABELS, None, folder, "moth", depth=2, shots=20_000, seed=9, approve_credits=1, client=fake, log=LOG)
        self.assertEqual((fake.calls, st["credits"], st["taps_source"], st["moth_job_id"]), (["otoc-echo-v1"], 1, "moth", "echo-1"))
        self.assertTrue(st["consistent"])
        self.assertTrue(os.path.exists(os.path.join(folder, "moth", "raw_result_echo-1.json")))
        self.assertEqual(er.EchoReliefState(SPEC, folder).from_moth, True)
        again = FakeEcho()
        st2 = er.run(LABELS, None, folder, "moth", depth=2, shots=30_000, seed=9, client=again, log=LOG)       # new Aer shots, same Moth request: no credit
        self.assertEqual((again.calls, st2["credits"]), ([], 0))

    def test_an_unrecognised_echo_result_keeps_the_raw_file_and_says_what_it_saw(self):
        folder = out("m_odd")
        with self.assertRaises(ValueError) as cm:
            er.run(LABELS, None, folder, "moth", depth=2, shots=20_000, approve_credits=1, client=FakeEcho(result={"outputs": None, "result": {"mystery": 1}}), log=LOG)
        self.assertIn("mystery", str(cm.exception))
        self.assertTrue(os.path.exists(os.path.join(folder, "moth", "raw_result_echo-1.json")))
        self.assertFalse(os.path.exists(os.path.join(folder, "state.json")))


class InTheShow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = er.EchoReliefState(SPEC, RUN)

    def test_looks_step_through_the_depths_and_say_which(self):
        rng = np.random.default_rng(0)
        seen = [self.state.draw(8, rng) for _ in range(7)]
        self.assertEqual([d.echo_depth for d in seen], [1, 2, 3, 1, 2, 3, 1])
        self.assertTrue(all(d.echo_total == 3 for d in seen))
        self.assertEqual(self.state.current, 0)

    def test_each_depth_draws_from_its_own_circuit(self):
        a = self.state.children
        for k, child in enumerate(a):
            self.assertAlmostEqual(float(child.spec.tau.mean()), META["mean_tilt"][k], places=6)
        ref = [ReliefState(c.spec) for c in a]
        rng1, rng2 = np.random.default_rng(1), np.random.default_rng(2)
        for k in range(3):
            got = np.mean([a[k].draw(32, rng1, g=0, world=0).lit for _ in range(400)], 0)
            want = np.mean([ref[k].draw(32, rng2, g=0, world=0).lit for _ in range(400)], 0)
            np.testing.assert_allclose(got, want, atol=0.06)

    def test_the_witness_describes_the_depth_on_screen(self):
        from src.quantum import relief_witness as rw
        self.state.draw(8, np.random.default_rng(3))
        cert = rw.certificate(self.state, np.random.default_rng(3), 4000, chsh=False)
        self.assertIn("Aer", cert["sampled_by"])
        self.assertEqual(self.state.spec, self.state.children[self.state.current].spec)

    def test_a_run_for_another_drawing_or_settings_or_a_changed_mapping_is_refused(self):
        other = ar.build_spec(LABELS, kappa=0.5)[2]
        with self.assertRaises(ValueError) as cm:
            er.EchoReliefState(other, RUN)
        self.assertIn("different", str(cm.exception))
        with self.assertRaises(ValueError):
            er.EchoReliefState(SPEC, out("nowhere"))
        from unittest import mock
        with mock.patch.object(er, "depth_specs", side_effect=lambda *a, **k: dm.dynamic_specs(a[0], a[1], a[2], a[3], a[4], tau_gain=0.1)):
            with self.assertRaises(ValueError) as cm:
                er.EchoReliefState(SPEC, RUN)
        self.assertIn("does not rebuild", str(cm.exception))

    def test_a_session_plays_it_and_captions_the_depth(self):
        from src.ui import caption
        from src.ui.session import Session
        s = Session(bay_window_labels(), out_dir=out("show"), pool_size=1000, projector_size=(640, 360), echo_run=RUN, complementary_report=out("none.json"))
        self.assertTrue(s.prov["echo"] and s.prov["executed"] == "aer" and not s.prov["from_moth"])
        self.assertIsInstance(s.rstate, er.EchoReliefState)
        depths = []
        for _ in range(4):
            s.next_new()
            depths.append(s.current.draw.echo_depth)
        self.assertEqual(depths, [2, 3, 1, 2])                                     # the first look was drawn when the session started
        cap = caption.describe_relief(s.current.draw, s.panel_names, 8, s.prov)
        self.assertIn("echo depth 2 of 3", cap["headline"])
        self.assertIn("one per echo depth", cap["provenance"])
        self.assertIn("local echo", cap["provenance"])
        self.assertIn("Aer", s.witness_run(2000)["certificate"]["sampled_by"])
        for kw in (dict(aer_run=out("x")), dict(engine="domain"), dict(circuit=out("x.qasm"))):
            with self.assertRaises(ValueError):
                Session(bay_window_labels(), out_dir=out("bad"), echo_run=RUN, complementary_report=out("none.json"), **kw)


if __name__ == "__main__":
    unittest.main()
