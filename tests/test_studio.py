"""The studio: step order and staleness, the commands it builds, and the Moth spend gate. No real process is started and nothing can reach
the network: children are a recording fake.

    python -m unittest tests.test_studio -v
"""
import http.client
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from src import studio
from src.capture.labels import DEFAULT_CALIB_DIR
from src.studio import Studio, StageError, make_server
from src.targets.payloads import payload_sha256

CALIB = DEFAULT_CALIB_DIR


class FakeProc:
    def __init__(self, code=0):
        self.stdout, self.code = io.StringIO("line one\n"), code

    def wait(self, timeout=None):
        return self.code

    def terminate(self):
        pass

    kill = terminate


class Recorder:
    def __init__(self, code=0):
        self.calls, self.code = [], code

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        return FakeProc(self.code)


def wait_for(cond, secs=3):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rec = Recorder()
        self.s = Studio(os.path.join(self.tmp.name, "proj"), spawn=self.rec, open_browser=False, log=lambda m: None)

    def tearDown(self):
        for j in self.s.jobs.values():                               # let the after-exit bookkeeping thread finish before the folder goes
            wait_for(lambda: not j.running)
        time.sleep(0.15)
        self.tmp.cleanup()

    def status(self):
        return {st["id"]: st["status"] for st in self.s.state()["stages"]}

    def with_targets(self, remember=True):
        """The demo scene (six panes, 4x4 grid) plus its calibrated payloads, as if step 2 had succeeded."""
        self.s.use_demo_scene()
        shutil.copytree(CALIB, self.s.calib)
        path = os.path.join(self.s.calib, "payload_qdrive.json")           # keep one round of targets (older calibrations stored three, 201 targets, which the client refuses)
        with open(path) as f:
            payload = json.load(f)
        t = payload["params"]["targets"]
        payload["params"]["targets"] = t[:t.index(None) + 1]
        with open(path, "w") as f:
            json.dump(payload, f)
        if remember:
            self.s._remember(targets_for=studio.sha_file(self.s.labels_path))


class Order(Base):
    def test_fresh_project_only_offers_drawing(self):
        self.assertEqual(self.status(), dict(draw="ready", targets="blocked", preview="blocked", solve="blocked", final="blocked"))
        self.assertEqual(self.s.state()["next"], "draw")

    def test_blocked_steps_refuse_to_start(self):
        with self.assertRaises(StageError):
            self.s.start("targets")
        self.assertEqual(self.rec.calls, [])

    def test_demo_scene_unlocks_targets_and_is_never_overwritten(self):
        self.s.use_demo_scene()
        self.assertEqual(self.status()["draw"], "done")
        self.assertEqual(self.status()["targets"], "ready")
        with self.assertRaises(StageError):
            self.s.use_demo_scene()

    def test_targets_command_uses_project_paths_and_options(self):
        self.s.use_demo_scene()
        self.s.start("targets", dict(grid=[5, 4], sweeps=5000))
        argv = self.rec.calls[0]
        self.assertEqual(argv[1:3], ["-m", "src.targets.calibrate_from_mock"])
        self.assertEqual(argv[argv.index("--labels") + 1], self.s.labels_path)
        self.assertEqual(argv[argv.index("--out") + 1], self.s.calib)
        self.assertEqual(argv[argv.index("--grid") + 1:argv.index("--grid") + 3], ["5", "4"])
        self.assertEqual(argv[argv.index("--sweeps") + 1], "5000")

    def test_bad_options_are_refused_before_anything_runs(self):
        self.s.use_demo_scene()
        for opts in (dict(grid=[0, 4]), dict(grid=[5, 999]), dict(sweeps=3)):
            with self.assertRaises(StageError):
                self.s.start("targets", opts)
        self.assertEqual(self.rec.calls, [])

    def test_success_records_which_drawing_the_targets_came_from(self):
        self.s.use_demo_scene()
        self.s.start("targets")
        sha = studio.sha_file(self.s.labels_path)
        self.assertTrue(wait_for(lambda: self.s._record().get("targets_for") == sha))

    def test_failure_records_nothing(self):
        self.s.spawn = Recorder(code=2)
        self.s.use_demo_scene()
        self.s.start("targets")
        self.assertTrue(wait_for(lambda: self.s.jobs["targets"].returncode == 2))
        time.sleep(0.2)
        self.assertNotIn("targets_for", self.s._record())
        self.assertEqual(self.status()["targets"], "failed")

    def test_changing_the_drawing_makes_everything_downstream_stale(self):
        self.with_targets()
        self.assertEqual(self.status()["targets"], "done")
        self.assertEqual(self.status()["preview"], "ready")
        with open(self.s.labels_path) as f:
            labels = json.load(f)
        labels["grid"]["nx"] = 4
        with open(self.s.labels_path, "w") as f:
            json.dump(labels, f)
        st = self.status()
        self.assertEqual(st["targets"], "stale")
        self.assertEqual((st["preview"], st["solve"], st["final"]), ("blocked",) * 3)
        self.assertEqual(self.s.state()["next"], "targets")

    def test_targets_the_studio_did_not_make_are_not_trusted(self):
        self.with_targets(remember=False)
        self.assertEqual(self.status()["targets"], "stale")

    def test_show_gets_project_paths_and_never_spend_permission(self):
        self.with_targets()
        self.s.allow_spend = True
        self.s.start("preview")
        argv = self.rec.calls[0]
        self.assertEqual(argv[1:3], ["-m", "src.show"])
        for flag, want in (("--labels", self.s.labels_path), ("--calib", self.s.calib), ("--run", self.s.solve), ("--source", "relief")):
            self.assertEqual(argv[argv.index(flag) + 1], want)
        self.assertNotIn("--allow-spend", argv)
        self.assertIn("--no-browser", argv)
        self.assertEqual(self.status()["preview"], "done")

    def test_final_needs_a_circuit_for_the_current_payload(self):
        self.with_targets()
        self.assertEqual(self.status()["final"], "blocked")
        run = os.path.join(self.s.solve, "qdrive")
        os.makedirs(run)
        with open(os.path.join(run, "circuit.qasm"), "w") as f:
            f.write("OPENQASM 3.0;\n")
        self.assertEqual(self.status()["final"], "blocked")           # a circuit alone, with no result for this payload, is not enough
        plan = self.s.plan("qdrive")
        with open(os.path.join(run, "payload.sha256"), "w") as f:
            f.write(plan["sha256"] + "\n")
        with open(os.path.join(run, "raw_result_x.json"), "w") as f:
            f.write("{}")
        self.assertEqual(self.status()["solve"], "done")
        self.assertEqual(self.status()["final"], "ready")
        self.s.start("final")
        argv = self.rec.calls[-1]
        self.assertEqual(argv[argv.index("--source") + 1], "circuit")
        self.assertEqual(argv[argv.index("--circuit") + 1], os.path.join(run, "circuit.qasm"))

    def test_pen_tool_will_not_resume_a_photo_drawing_without_the_photo(self):
        self.s.use_demo_scene()
        with open(self.s.labels_path) as f:
            labels = json.load(f)
        labels["source"] = "photo"
        with open(self.s.labels_path, "w") as f:
            json.dump(labels, f)
        with self.assertRaises(StageError):
            self.s.start("draw")
        self.assertEqual(self.rec.calls, [])


class ShowKnobs(Base):
    def setUp(self):
        super().setUp()
        self.with_targets()

    def argv(self, show, stage="preview", **extra):
        self.s.start(stage, dict(show=show, **extra))
        return self.rec.calls[-1]

    def value(self, argv, flag):
        return argv[argv.index(flag) + 1]

    def test_every_show_flag_has_a_knob_and_defaults_match_the_show(self):
        import argparse
        import re
        with open(os.path.join(studio.ROOT, "src", "show.py")) as f:
            src = f.read()
        flags = set(re.findall(r'add_argument\("--([a-z0-9-]+)"', src))
        owned = {"labels", "out", "run", "calib", "calibration", "circuit"} | {k["flag"][2:] for k in studio.knob_schema()}
        self.assertEqual(flags - owned, set())
        self.assertEqual({k["key"] for k in studio.knob_schema()} - {f.replace("-", "_") for f in flags}, set())

    def test_defaults_send_no_flags_beyond_the_old_command(self):
        argv = self.argv({})
        self.assertEqual(argv[argv.index("--source"):], ["--source", "relief", "--port", argv[-2], "--no-browser"])

    def test_values_become_flags_and_blank_or_default_ones_do_not(self):
        argv = self.argv(dict(kappa="1.5", n_dirs=6, entangle="", contrast=1.15, lamp_mode="x", projector_size="1280x720", lan=True, game=False))
        self.assertEqual((self.value(argv, "--kappa"), self.value(argv, "--n-dirs"), self.value(argv, "--lamp-mode"), self.value(argv, "--projector-size")),
                         ("1.5", "6", "x", "1280x720"))
        for absent in ("--entangle", "--contrast", "--game"):
            self.assertNotIn(absent, argv)
        self.assertIn("--lan", argv)

    def test_domain_knobs_and_the_studio_engine_default(self):
        argv = self.argv(dict(seg_len=100, game=True, evolve_steps=3, leaf_prefer="seam", backend="mps"), engine="domain")
        self.assertEqual(self.value(argv, "--engine"), "domain")
        self.assertIn("--game", argv)
        self.assertEqual((self.value(argv, "--seg-len"), self.value(argv, "--evolve-steps"), self.value(argv, "--leaf-prefer")), ("100.0", "3", "seam"))
        self.s.engine = "domain"
        self.assertEqual(self.value(self.argv({}), "--engine"), "domain")
        self.assertNotIn("--engine", self.argv(dict(engine="panel")))

    def test_classical_source_replaces_the_relief_source_and_refuses_relief_knobs(self):
        self.assertEqual(self.value(self.argv(dict(source="oracle", pool=5000)), "--source"), "oracle")
        n = len(self.rec.calls)
        for show in (dict(source="oracle", kappa=2), dict(tau=0.2), dict(source="oracle", engine="domain")):
            with self.assertRaises(StageError):
                self.s.start("preview", dict(show=show))
        self.assertEqual(len(self.rec.calls), n)

    def test_bad_values_are_refused_before_anything_runs(self):
        n = len(self.rec.calls)
        bad = (dict(kappa="nan"), dict(kappa=99), dict(n_dirs=2.5), dict(projector_size="big"), dict(lamp_mode="y"), dict(nope=1), dict(port=0, lan="yes"),
               dict(calib="/no/such/folder"), dict(lamp_mode="x", engine="domain", game=True))
        for show in bad:
            with self.assertRaises(StageError, msg=show):
                self.s.start("preview", dict(show=show, engine="domain") if show.get("game") else dict(show=show))
        self.assertEqual(len(self.rec.calls), n)

    def test_the_show_gets_spend_permission_only_if_the_studio_has_it(self):
        with self.assertRaises(StageError):
            self.s.start("preview", dict(show=dict(allow_spend=True)))
        self.s.allow_spend = True
        self.assertIn("--allow-spend", self.argv(dict(allow_spend=True)))

    def test_path_overrides_replace_the_project_paths(self):
        argv = self.argv(dict(calib=CALIB, run=self.tmp.name))
        self.assertEqual(self.value(argv, "--calib"), os.path.abspath(os.path.join(studio.ROOT, CALIB)))
        self.assertEqual(self.value(argv, "--run"), os.path.abspath(self.tmp.name))
        self.assertEqual(argv.count("--calib"), 1)

    def test_own_port_and_no_browser(self):
        opened = []
        self.s.open_browser = True
        self.s._open_when_up = lambda job, port: opened.append(port)
        argv = self.argv(dict(port=51999, no_browser=True))
        self.assertEqual(self.value(argv, "--port"), "51999")
        time.sleep(0.1)
        self.assertEqual(opened, [])
        self.argv(dict())
        self.assertTrue(wait_for(lambda: opened))

    def test_performance_step_takes_only_the_common_knobs_and_keeps_its_circuit(self):
        run = os.path.join(self.s.solve, "qdrive")
        os.makedirs(run)
        with open(os.path.join(run, "circuit.qasm"), "w") as f:
            f.write("OPENQASM 3.0;\n")
        plan = self.s.plan("qdrive")
        with open(os.path.join(run, "payload.sha256"), "w") as f:
            f.write(plan["sha256"] + "\n")
        with open(os.path.join(run, "raw_result_x.json"), "w") as f:
            f.write("{}")
        argv = self.argv(dict(seed=7, projector_size="800x600"), "final")
        self.assertEqual((self.value(argv, "--source"), self.value(argv, "--seed"), self.value(argv, "--projector-size")), ("circuit", "7", "800x600"))
        for show in (dict(kappa=2), dict(lamp_mode="x")):
            with self.assertRaises(StageError):
                self.s.start("final", dict(show=show))

    def test_state_carries_the_schema(self):
        st = self.s.state()
        self.assertIn("lamp_mode", {k["key"] for k in st["knobs"]})
        self.assertEqual(st["engine"], "panel")


class SpendGate(Base):
    def setUp(self):
        super().setUp()
        self.with_targets()

    def test_plan_shows_hash_and_cost_and_sends_nothing(self):
        p = self.s.plan("qdrive")
        self.assertEqual((p["credits"], p["cached"]), (1, False))
        self.assertEqual(len(p["sha256"]), 64)
        self.assertEqual(self.rec.calls, [])

    def test_preview_only_studio_refuses_to_spend(self):
        p = self.s.plan("qdrive")
        with self.assertRaises(PermissionError):
            self.s.confirm("qdrive", p["sha256"], p["credits"])
        self.assertEqual(self.rec.calls, [])

    def test_a_confirm_for_a_different_payload_or_cost_is_refused(self):
        self.s.allow_spend = True
        p = self.s.plan("qdrive")
        for sha, credits in (("0" * 64, p["credits"]), (p["sha256"], 0), (p["sha256"], 5)):
            with self.assertRaises(StageError):
                self.s.confirm("qdrive", sha, credits)
        self.assertEqual(self.rec.calls, [])

    def test_matching_confirm_runs_the_solver_with_exactly_the_shown_credits(self):
        self.s.allow_spend = True
        p = self.s.plan("qdrive")
        self.s.confirm("qdrive", p["sha256"], p["credits"])
        argv = self.rec.calls[0]
        self.assertEqual(argv[1:4], ["-m", "src.quantum.solver", "qdrive"])
        self.assertEqual(argv[argv.index("--approve-credits") + 1], "1")
        self.assertEqual(argv[argv.index("--calib") + 1], self.s.calib)
        self.assertEqual(argv[argv.index("--out") + 1], self.s.solve)

    def test_graph_v1_costs_five_and_uses_its_own_name(self):
        self.s.allow_spend = True
        p = self.s.plan("graph-v1")
        self.assertEqual(p["credits"], 5)
        self.s.confirm("graph-v1", p["sha256"], 5)
        self.assertEqual(self.rec.calls[0][3], "graph-v1")

    def test_a_saved_result_is_free_and_sends_no_approval_even_without_allow_spend(self):
        run = os.path.join(self.s.solve, "qdrive")
        os.makedirs(run)
        p = self.s.plan("qdrive")
        with open(os.path.join(run, "payload.sha256"), "w") as f:
            f.write(p["sha256"] + "\n")
        with open(os.path.join(run, "raw_result_job.json"), "w") as f:
            f.write("{}")
        p = self.s.plan("qdrive")
        self.assertEqual((p["credits"], p["cached"]), (0, True))
        self.s.confirm("qdrive", p["sha256"], 0)
        self.assertNotIn("--approve-credits", self.rec.calls[0])

    def test_two_solves_cannot_run_at_once(self):
        self.s.allow_spend = True
        self.s.spawn = lambda argv, **kw: type("Hang", (FakeProc,), dict(wait=lambda self, timeout=None: time.sleep(5)))()
        p = self.s.plan("qdrive")
        self.s.confirm("qdrive", p["sha256"], p["credits"])
        with self.assertRaises(StageError):
            self.s.confirm("qdrive", p["sha256"], p["credits"])

    def test_jobs_submitted_without_a_saved_result_are_reported(self):
        run = os.path.join(self.s.solve, "qdrive")
        os.makedirs(run)
        for jid in ("aaaaaaaa-1", "bbbbbbbb-2"):
            with open(os.path.join(run, f"job_{jid}.json"), "w") as f:
                f.write("{}")
        with open(os.path.join(run, "raw_result_aaaaaaaa-1.json"), "w") as f:
            f.write("{}")
        self.assertEqual(self.s.plan("qdrive")["orphans"], ["bbbbbbbb-2"])
        self.assertIn("bbbbbbbb", self.s.state()["stages"][3].get("warn", ""))

    def test_payload_hash_matches_the_solvers_own(self):
        with open(os.path.join(self.s.calib, "payload_qdrive.json")) as f:
            params = json.load(f)["params"]
        self.assertEqual(self.s.plan("qdrive")["sha256"], payload_sha256({"engine": "qdrive-api-v1", "params": params}))


class Http(Base):
    def setUp(self):
        super().setUp()
        self.server, self.token, url = make_server(self.s)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def call(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Host": f"127.0.0.1:{self.port}"} | (headers or {})
        c.request(method, path, body=json.dumps(body) if body is not None else None, headers=h)
        r = c.getresponse()
        out = r.status, r.read()
        c.close()
        return out

    def test_page_carries_the_token_and_state_is_readable(self):
        code, page = self.call("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn(self.token.encode(), page)
        code, data = self.call("GET", "/api/state")
        self.assertEqual([s["id"] for s in json.loads(data)["stages"]], list(studio.STAGES))

    def test_posts_need_the_token(self):
        code, _ = self.call("POST", "/api/start", dict(stage="draw"))
        self.assertEqual(code, 403)
        code, _ = self.call("POST", "/api/solve/confirm", dict(engine="qdrive", sha256="x", credits=1), {"X-Token": "wrong"})
        self.assertEqual(code, 403)
        self.assertEqual(self.rec.calls, [])

    def test_foreign_host_header_is_refused(self):
        code, _ = self.call("GET", "/api/state", headers={"Host": "evil.example:80"})
        self.assertEqual(code, 403)

    def test_errors_come_back_as_messages(self):
        code, data = self.call("POST", "/api/start", dict(stage="targets"), {"X-Token": self.token})
        self.assertEqual(code, 400)
        self.assertIn("drawing", json.loads(data)["error"])

    def test_spend_refusal_is_a_403(self):
        self.with_targets()
        p = self.s.plan("qdrive")
        code, data = self.call("POST", "/api/solve/confirm", dict(engine="qdrive", sha256=p["sha256"], credits=p["credits"]), {"X-Token": self.token})
        self.assertEqual(code, 403)
        self.assertIn("allow-spend", json.loads(data)["error"])
        self.assertEqual(self.rec.calls, [])


if __name__ == "__main__":
    unittest.main()
