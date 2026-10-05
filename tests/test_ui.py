"""Performance mode: key table, captions, session actions, demo flow, calibration, re-solve gate, and the HTTP server.

    python -m unittest tests.test_ui -v
"""
import http.client
import io
import json
import os
import tempfile
import threading
import unittest

import numpy as np
from PIL import Image

from src.capture.labels import DEFAULT_CALIB_DIR, bay_window_labels
from src.projector import calibrate as cal
from src.projector import patterns
from src.texture.frames import FrameDraw
from src.ui import caption, keys
from src.ui.operator_panel import make_server
from src.ui.session import LADDER, Knobs, Session

TMP = tempfile.TemporaryDirectory()
SESSION = None


def setUpModule():
    global SESSION
    labels = bay_window_labels()                          # the default scene: six panes on a 4x4 grid, 12 + 2 + 6 = 20 qubits, exactly as the app runs it
    SESSION = Session(labels, out_dir=os.path.join(TMP.name, "show"), pool_size=8000, run_dir=os.path.join(TMP.name, "run"), calib_dir=DEFAULT_CALIB_DIR,
                      projector_size=(1280, 720), complementary_report=os.path.join(TMP.name, "none.json"))


def tearDownModule():
    TMP.cleanup()


def reset(s=SESSION):
    s = s or SESSION
    s.knobs = Knobs(pol_hold=[None] * s.scene.n_panels)
    s.pattern, s.align = None, None
    s.demo["active"] = False
    s.set_source("oracle", "z")
    s.set_output(1280, 720)
    return s


def png_array(data):
    return np.array(Image.open(io.BytesIO(data)))


class Keys(unittest.TestCase):
    def test_table_is_consistent(self):
        seen = set()
        for b in keys.BINDINGS:
            self.assertIn(b["action"], Session.ACTIONS, b)
            for k in b["keys"]:
                self.assertNotIn((k.lower(), b.get("demo", False)), seen, f"duplicate binding for {k!r}")
                seen.add((k.lower(), b.get("demo", False)))
        self.assertTrue(keys.help_rows())
        json.dumps(keys.bindings_json())

    def test_demo_keys_shadow_the_plain_ones(self):
        space = [b for b in keys.BINDINGS if " " in b["keys"]]
        self.assertEqual({(b["action"], b.get("demo", False)) for b in space}, {("next", False), ("demo_next", True)})


class Captions(unittest.TestCase):
    d = FrameDraw((-1, 1), (1, -1, 1), np.zeros(3), 8, np.zeros(1, int))
    classical = dict(quantum_backed=False, from_moth=False, untested=False, pol_basis="z")
    simulated = dict(quantum_backed=True, from_moth=False, untested=True, pol_basis="x")
    moth = dict(quantum_backed=True, from_moth=True, untested=False, pol_basis="z")

    def test_words_follow_the_measurement_outcomes(self):
        c = caption.describe(self.d, ["left", "centre", "right"], 8, self.classical)
        self.assertEqual(c["headline"], "Light from the left, frontal")
        self.assertEqual([p["state"] for p in c["panels"]], ["raised", "sunk", "raised"])
        self.assertIn("crisp", c["hardness"])

    def test_never_claims_more_than_the_provenance_supports(self):
        cl = caption.describe(self.d, ["a", "b", "c"], 8, self.classical)
        self.assertIn("no quantum calls", cl["provenance"])
        self.assertNotIn("measurement chose", cl["explain"])
        sim = caption.describe(self.d, ["a", "b", "c"], 8, self.simulated)["provenance"]
        self.assertIn("simulated on this laptop", sim)
        self.assertIn("not a Moth result", sim)
        self.assertIn("untested", sim)
        self.assertIn("X basis", sim)
        self.assertIn("QDrive", caption.describe(self.d, ["a", "b", "c"], 8, self.moth)["provenance"])

    def test_noise_caption_says_it_is_noise(self):
        c = caption.describe(self.d, ["a", "b", "c"], 8, self.simulated, noise=True)
        self.assertIn("noise", c["headline"].lower())
        self.assertIn("no quantum state", c["provenance"])


class SessionLooks(unittest.TestCase):
    def setUp(self):
        self.s = reset()

    def test_every_frame_sent_has_the_glass_exactly_black(self):
        s = self.s
        before = s.glass["frames_checked"]
        for _ in range(25):
            s.next_new()
            u8 = png_array(s.hub.latest()[1])
            self.assertTrue(np.all(u8[s.warper.must_be_black] == 0))
        self.assertEqual(s.glass["violations"], 0)
        self.assertGreaterEqual(s.glass["frames_checked"] - before, 25)

    def test_holds_are_respected_and_released(self):
        s = self.s
        s.hold_world(1, -1)
        s.hold_pol_all(1)
        for _ in range(10):
            s.next_new()
            self.assertEqual((s.current.draw.lamp, s.current.draw.pol), ((1, -1), (1,) * 6))
        s.release_lamp()
        s.hold_pol(1, -1)
        seen = set()
        for _ in range(40):
            s.next_new()
            self.assertEqual(s.current.draw.pol[1], -1)
            seen.add(s.current.draw.lamp)
        self.assertGreater(len(seen), 1)

    def test_impossible_hold_is_an_error_message_not_a_crash(self):
        s = self.s
        s.set_source("mock", "z")
        s.pool.pol[:] = 1                                           # no sunk shots anywhere: holding "sunk" cannot be satisfied
        s.index = type(s.index)(s.pool)
        s._pools[("mock", "z", None, None)] = (s.index, s.pool, s.prov)
        s.hold_pol_all(-1)
        self.assertIsNotNone(s.error)
        self.assertEqual(s.state()["error"], s.error)
        s.hold_pol_all(None)
        s.next_new()
        self.assertIsNone(s.error)
        s._pools.pop(("mock", "z", None, None))

    def test_hardness_ladder_and_sweep(self):
        s = self.s
        s.K(8)
        s.K_step(1)
        self.assertEqual(s.knobs.K, 16)
        s.K_step(-2)
        self.assertEqual(s.knobs.K, 4)
        s.hardness_sweep_set(True)
        ks = [s.current.draw.K]
        for _ in range(8):
            s.next_new()
            ks.append(s.current.draw.K)
        self.assertEqual(ks[:8], LADDER)
        s.K(32)
        self.assertFalse(s.knobs.hardness_sweep)

    def test_blend_changes_the_pixels_not_the_draw(self):
        s = self.s
        s.next_new()
        look, a = s.current, png_array(s.hub.latest()[1]).astype(int)
        s.blend(12)
        b = png_array(s.hub.latest()[1]).astype(int)
        self.assertIs(s.current, look)
        self.assertGreater(np.abs(a - b).max(), 3)
        self.assertTrue(np.all(b[s.warper.must_be_black] == 0))
        s.gain(0.5)
        self.assertLessEqual(png_array(s.hub.latest()[1]).max(), 128)

    def test_history_back_and_forward_replays_the_same_looks(self):
        s = self.s
        s.next_new(); s.next_new(); s.next_new()
        ids = [s.current.id]
        s.prev(); s.prev()
        self.assertEqual(s.current.id, ids[0] - 2)
        s.next()
        self.assertEqual(s.current.id, ids[0] - 1)
        s.next()
        self.assertEqual(s.current.id, ids[0])                                          # back at the newest look
        s.next()
        self.assertEqual(s.current.id, ids[0] + 1)                                      # one more press draws a new look

    def test_blackout_noise_and_patterns(self):
        s = self.s
        s.blackout_set(True)
        self.assertEqual(png_array(s.hub.latest()[1]).max(), 0)
        s.blackout_set(False)
        self.assertGreater(png_array(s.hub.latest()[1]).max(), 0)
        s.noise_set(True)
        self.assertTrue(s.current.noise)
        self.assertFalse(s.live_caption()["provenance"].startswith("Each look"))
        s.noise_set(False)
        s.set_pattern("white")
        w = png_array(s.hub.latest()[1])
        self.assertTrue(np.all(w[s.warper.must_be_black] == 0))
        self.assertGreater((w > 0).mean(), 0.2)
        s.pattern_cycle()
        self.assertEqual(s.pattern, "outline")
        s.set_pattern(None)
        with self.assertRaises(ValueError):
            s.set_pattern("nonsense")
        with self.assertRaises(ValueError):
            s.dispatch("rm_rf")

    def test_output_size_follows_the_window_and_pngs_match_it(self):
        s = self.s
        s.set_output(1600, 900, fullscreen=True, dpr=2.0)
        self.assertEqual(s.size, (1600, 900))
        self.assertEqual(png_array(s.hub.latest()[1]).shape, (900, 1600))
        self.assertTrue(s.state()["output"]["fullscreen"])
        s.set_output(1280, 720)

    def test_complementary_source_locks_polarity_to_its_lamp_and_is_labelled_untested(self):
        s = self.s
        s.set_source("complementary")
        self.assertEqual(s.pol_basis, "x")
        self.assertTrue(s.prov["untested"] and not s.prov["from_moth"])
        for _ in range(15):
            s.next_new()
            self.assertEqual(set(s.current.draw.pol), {s.current.draw.lamp[1]})       # lamp2 coupling: every panel follows L2
        s.pol_basis_toggle()
        self.assertEqual(s.pol_basis, "z")
        pols = {s.next_new() or s.current.draw.pol for _ in range(30)}
        self.assertGreater(len(pols), 3)                                                # read in Z the polarity is free
        with self.assertRaises(ValueError):
            s.set_source("mock", "x")


class Calibration(unittest.TestCase):
    def setUp(self):
        self.s = reset()

    def test_corner_alignment_changes_the_mapping_and_saves_it(self):
        s = self.s
        base_lit = s.warper.n_lit
        s.align_start()
        self.assertEqual(s.pattern, "align")
        q0 = [list(p) for p in s.align["quad"]]
        self.assertTrue(s.align_move(0, q0[0][0] + 30, q0[0][1] + 20)["ok"])
        self.assertFalse(s.align_move(1, q0[3][0], q0[3][1])["ok"])               # crossing two markers over is refused
        r = s.align_apply()
        self.assertTrue(os.path.exists(r["path"]))
        self.assertEqual(s.cal.method, "corners")
        self.assertNotEqual(s.warper.n_lit, base_lit)
        s.next_new()
        self.assertTrue(np.all(png_array(s.hub.latest()[1])[s.warper.must_be_black] == 0))
        self.assertEqual(cal.load_calibration(r["path"]).method, "corners")
        s.align_reset()
        self.assertEqual(s.cal.method, "scale")

    def test_markers_stay_on_their_spots_when_the_window_changes_size(self):
        s = self.s
        s.align_start()
        before = np.array(s.align["quad"])
        s.set_output(1920, 1080)
        after = np.array(s.align["quad"])
        np.testing.assert_allclose((after + 0.5) / np.array([1920 / 1280, 1080 / 720]), before + 0.5, atol=1e-9)
        s.align_cancel()
        s.set_output(1280, 720)

    def test_margin_changes_the_lit_area_and_snapshot_writes_the_photo_test_frame(self):
        s = self.s
        a = s.warper.n_lit
        s.set_margin(8)
        self.assertLess(s.warper.n_lit, a)
        s.set_margin(2)
        r = s.snapshot()
        self.assertTrue(os.path.exists(os.path.join(r["dir"], "frame_projector.png")))
        for f in ("frame_projector.png",):
            os.remove(os.path.join(r["dir"], f))
        if os.path.exists(s.calib_path):
            os.remove(s.calib_path)


class Demo(unittest.TestCase):
    def setUp(self):
        self.s = reset()

    def test_demo_walks_every_beat_and_the_audience_follows(self):
        s = self.s
        s.demo_toggle()
        v = s.demo_view()
        self.assertEqual((v["index"], v["step"]), (0, "1. The problem"))
        self.assertTrue(s.knobs.noise and s.knobs.auto)
        kinds = [s.audience()["kind"]]
        for _ in range(v["total"] + 2):                                              # past the end must clamp, not crash
            s.demo_next()
            kinds.append(s.audience()["kind"])
        self.assertEqual(s.demo_view()["index"], v["total"] - 1)
        self.assertEqual({"slide", "live", "overlay"}, set(kinds))
        s.demo_prev()
        self.assertEqual(s.demo_view()["index"], v["total"] - 2)
        s.demo_toggle()
        self.assertIsNone(s.demo_view())
        self.assertEqual(s.audience()["kind"], "live")
        s.auto_set(False)

    def test_slides_are_written_from_the_provenance_at_that_moment(self):
        s = self.s
        s.demo_toggle()
        for _ in range(30):
            s.demo_next()
        last = s.audience()
        self.assertEqual(last["title"], "Honest scorecard")
        self.assertTrue(last["bullets"][0].startswith("Each look"))                   # the demo's last live beat switched to the complementary stand-in
        self.assertIn("untested", last["bullets"][0])
        self.assertTrue(any("not quantum-specific" in b.lower() for b in last["bullets"]))
        s.demo_toggle()
        s.auto_set(False)

    def test_missing_engine_results_are_said_plainly(self):
        s = self.s
        s.demo_toggle()
        for _ in range(5):
            s.demo_next()
        slide = s.audience()
        self.assertEqual(s.demo_view()["step"], "4. Run: requested vs achieved")
        self.assertTrue(any("No engine results" in b for b in slide["bullets"]))
        s.demo_toggle()
        s.auto_set(False)


def forbid_solver(*a, **k):
    raise AssertionError("a test reached the real Moth solver")


def clean_run_dir(s):
    for e in ("qdrive", "graph_v1"):
        d = os.path.join(s.run_dir, e)
        for f in (os.listdir(d) if os.path.isdir(d) else []):
            os.remove(os.path.join(d, f))


class Resolve(unittest.TestCase):
    def setUp(self):
        self.s = reset()
        self.s.allow_spend = False
        clean_run_dir(self.s)
        self.s._solve_fn = forbid_solver               # any test that does not install its own stub must not reach Moth

    def tearDown(self):
        clean_run_dir(self.s)
        self.s.allow_spend, self.s._solve_fn = False, None

    def test_plan_shows_the_cost_and_a_spend_needs_allow_spend_and_a_matching_hash(self):
        s = self.s
        plan = s.resolve_plan("qdrive")
        self.assertEqual((plan["credits"], plan["cached"], plan["allow_spend"]), (1, False, False))
        self.assertEqual(s.resolve_plan("graph_v1")["credits"], 5)
        with self.assertRaises(PermissionError):
            s.resolve_confirm("qdrive", plan["sha256"], 1)
        s.allow_spend = True
        with self.assertRaisesRegex(ValueError, "changed"):
            s.resolve_confirm("qdrive", "0" * 64, 1)
        with self.assertRaisesRegex(ValueError, "changed"):
            s.resolve_confirm("qdrive", plan["sha256"], 99)
        called = threading.Event()
        calls = []

        def fake(calib, out, approve):
            calls.append((calib, out, approve))
            called.set()

        s._solve_fn = fake
        s.resolve_confirm("qdrive", plan["sha256"], 1)
        self.assertTrue(called.wait(5))
        self.assertEqual(calls[0][2], 1)                                               # exactly the credits that were shown
        s.allow_spend = False
        s._solve_fn = None

    def test_a_payload_that_already_has_a_result_costs_nothing_and_needs_no_permission(self):
        s = self.s
        plan = s.resolve_plan("qdrive")
        d = os.path.join(s.run_dir, "qdrive")
        os.makedirs(d, exist_ok=True)
        for name, text in (("payload.sha256", plan["sha256"] + "\n"), ("raw_result_x.json", "{}")):
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        again = s.resolve_plan("qdrive")
        self.assertEqual((again["credits"], again["cached"]), (0, True))
        got = []
        done = threading.Event()
        s._solve_fn = lambda c, o, a: (got.append(a), done.set())
        s.resolve_confirm("qdrive", again["sha256"], 0)
        self.assertTrue(done.wait(5))
        self.assertIsNone(got[0])                                                      # None = the solver sends nothing
        s._solve_fn = None


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = reset()
        clean_run_dir(cls.s)
        cls.s._solve_fn = forbid_solver
        cls.server, cls.token, cls.base = make_server(cls.s)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.s._solve_fn = None
        cls.server.shutdown()
        cls.server.server_close()

    def req(self, method, path, body=None, headers=None, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = dict(headers or {})
        if host:
            h["Host"] = host
        c.request(method, path, json.dumps(body) if body is not None else None, h)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def test_pages_state_and_frames(self):
        for path in ("/", "/output", "/audience", "/web/common.css"):
            code, body = self.req("GET", path)
            self.assertEqual(code, 200, path)
        code, page = self.req("GET", "/")
        self.assertIn(self.token.encode(), page)
        code, data = self.req("GET", "/api/state")
        st = json.loads(data)
        self.assertEqual(st["scene"]["panels"], ["left_top", "left_bottom", "centre_top", "centre_bottom", "right_top", "right_bottom"])
        code, png = self.req("GET", "/api/latest.png")
        self.assertEqual(code, 200)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        code, ov = self.req("GET", "/api/overlay.png")
        self.assertEqual(Image.open(io.BytesIO(ov)).size, (self.s.scene.W, self.s.scene.H))
        self.assertEqual(self.req("GET", "/api/frame/99999999.png")[0], 404)

    def test_actions_need_the_token_and_a_loopback_host(self):
        ok = {"X-Token": self.token}
        self.assertEqual(self.req("POST", "/api/action", dict(name="next"), {})[0], 403)
        self.assertEqual(self.req("POST", "/api/action", dict(name="next"), {"X-Token": "wrong"})[0], 403)
        self.assertEqual(self.req("POST", "/api/action", dict(name="next"), ok, host="evil.example")[0], 403)
        self.assertEqual(self.req("GET", "/", host="evil.example")[0], 403)
        before = self.s.hub.seq
        code, data = self.req("POST", "/api/action", dict(name="next"), ok)
        self.assertEqual(code, 200)
        self.assertGreater(self.s.hub.seq, before)
        code, data = self.req("POST", "/api/action", dict(name="rm_rf"), ok)
        self.assertEqual(code, 400)
        self.assertIn("unknown action", json.loads(data)["error"])
        code, data = self.req("POST", "/api/action", dict(name="pattern", args=dict(name="white")), ok)
        self.assertEqual(code, 200)                                                   # an action argument called "name" works over HTTP too
        self.assertEqual(self.s.pattern, "white")
        self.s.set_pattern(None)

    def test_output_report_resizes_without_a_token(self):
        code, _ = self.req("POST", "/api/output/report", dict(w=1500, h=844, dpr=1.0, fullscreen=True))
        self.assertEqual(code, 200)
        self.assertEqual(self.s.size, (1500, 844))
        self.assertEqual(self.req("POST", "/api/output/report", dict(w="x"))[0], 400)
        self.s.set_output(1280, 720)

    def test_resolve_over_http_never_spends_by_default(self):
        ok = {"X-Token": self.token}
        code, data = self.req("POST", "/api/resolve/plan", dict(engine="qdrive"), ok)
        plan = json.loads(data)
        self.assertEqual(code, 200)
        code, data = self.req("POST", "/api/resolve/confirm", dict(engine="qdrive", sha256=plan["sha256"], credits=plan["credits"]), ok)
        self.assertEqual(code, 403)
        self.assertIn("allow-spend", json.loads(data)["error"])

    def test_server_sent_events_deliver_state_then_frames(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/api/events?role=audience")
        r = c.getresponse()
        self.assertEqual(r.getheader("Content-Type"), "text/event-stream")
        events = []
        buf = b""
        while len(events) < 2:
            buf += r.fp.readline()
            if buf.endswith(b"\n\n"):
                events.append(buf.decode().split("\n")[0])
                buf = b""
        self.assertEqual(events[0], "event: state")
        self.assertIn(events[1], ("event: frame", "event: state"))
        c.close()


if __name__ == "__main__":
    unittest.main()
