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
        owned = {"labels", "out", "run", "calib", "calibration", "circuit", "aer-run", "echo-run"} | {k["flag"][2:] for k in studio.knob_schema()}
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


class TomographySolve(Base):
    """Verify on Moth: the reference circuit through tomography-api-v2, behind the same hash-and-cost gate as the other Moth engines."""

    def setUp(self):
        super().setUp()
        self.with_targets()

    def tomo_dir(self):
        return os.path.join(self.s.solve, "tomography")

    def test_the_plan_shows_the_request_its_hash_and_one_credit_and_sends_nothing(self):
        p = self.s.plan("tomography")
        self.assertEqual((p["credits"], p["cached"], p["problems"], p["engine_id"]), (1, False, [], "tomography-api-v2"))
        self.assertEqual(len(p["sha256"]), 64)
        self.assertIn("tomography-api-v2", p["describe"])
        self.assertEqual(self.rec.calls, [])

    def test_a_preview_only_studio_refuses_to_spend(self):
        p = self.s.plan("tomography")
        with self.assertRaises(PermissionError):
            self.s.confirm("tomography", p["sha256"], 1)
        self.assertEqual(self.rec.calls, [])

    def test_a_matching_confirm_runs_the_verifier_with_exactly_the_shown_credit(self):
        self.s.allow_spend = True
        p = self.s.plan("tomography", dict(kappa=0.8))
        self.s.confirm("tomography", p["sha256"], 1, dict(kappa=0.8))
        argv = self.rec.calls[0]
        self.assertEqual(argv[1:3], ["-m", "src.quantum.verify_moth"])
        self.assertEqual(argv[argv.index("--approve-credits") + 1], "1")
        self.assertEqual(argv[argv.index("--out") + 1], self.tomo_dir())
        self.assertEqual(argv[argv.index("--aer-run") + 1], self.s.aer_dir)
        self.assertEqual(argv[argv.index("--kappa") + 1], "0.8")
        self.assertNotIn("--local", argv)

    def test_a_confirm_for_a_different_request_or_cost_is_refused(self):
        self.s.allow_spend = True
        p = self.s.plan("tomography")
        for sha, credits in (("0" * 64, 1), (p["sha256"], 0), (p["sha256"], 5)):
            with self.assertRaises(StageError):
                self.s.confirm("tomography", sha, credits)
        with self.assertRaises(StageError):                                                   # the hash follows the circuit: other settings, other request
            self.s.confirm("tomography", p["sha256"], 1, dict(kappa=0.8))
        self.assertEqual(self.rec.calls, [])

    def test_a_saved_result_is_free_and_sends_no_approval(self):
        p = self.s.plan("tomography")
        os.makedirs(self.tomo_dir())
        with open(os.path.join(self.tomo_dir(), "payload.sha256"), "w") as f:
            f.write(p["sha256"] + "\n")
        with open(os.path.join(self.tomo_dir(), "raw_result_job.json"), "w") as f:
            f.write("{}")
        p = self.s.plan("tomography")
        self.assertEqual((p["credits"], p["cached"]), (0, True))
        self.s.confirm("tomography", p["sha256"], 0)
        self.assertNotIn("--approve-credits", self.rec.calls[0])

    def test_jobs_submitted_without_a_saved_result_are_reported_but_local_rehearsals_are_not(self):
        os.makedirs(self.tomo_dir())
        for name in ("job_aaaaaaaa-1.json", "job_local-bbbb.json"):
            with open(os.path.join(self.tomo_dir(), name), "w") as f:
                f.write("{}")
        self.assertEqual(self.s.plan("tomography")["orphans"], ["aaaaaaaa-1"])

    def test_what_the_check_cannot_cover_is_refused(self):
        for show in (dict(engine="domain"), dict(source="oracle")):
            with self.assertRaises(StageError):
                self.s.plan("tomography", show)
        for shots in (5, 10 ** 9, "many"):
            with self.assertRaises(StageError):
                self.s.plan("tomography", None, dict(shots=shots))

    def write_state(self, **kw):
        os.makedirs(self.tomo_dir(), exist_ok=True)
        st = dict(kind="relief-tomography", labels_sha=studio.sha_file(self.s.labels_path), local=False, n_qubits=18,
                  score=dict(rms_single=0.01, rms_pairs=0.02, n_pair_values=225), aer_comparison=dict(max_abs_difference=0.004))
        st.update(kw)
        with open(os.path.join(self.tomo_dir(), "state.json"), "w") as f:
            json.dump(st, f)

    def solve_stage(self):
        return next(st for st in self.s.state()["stages"] if st["id"] == "solve")

    def test_a_verified_circuit_is_shown_with_its_numbers_but_does_not_complete_step_4(self):
        self.write_state()
        st = self.solve_stage()
        self.assertEqual(st["status"], "ready")                                                # tomography returns no shots, so it cannot drive Perform
        self.assertIn("verified the reference circuit", st["verified"])
        self.assertIn("0.004", st["verified"])
        self.assertTrue(st["results"]["tomography"]["cached"])

    def test_a_local_rehearsal_or_another_drawings_result_is_not_called_verified(self):
        for kw in (dict(local=True), dict(labels_sha="0" * 64)):
            self.write_state(**kw)
            self.assertNotIn("verified", self.solve_stage())


class QFacetSolve(Base):
    """QDrive-prepared facet register: one chained job per confirm behind the spend gate, and the Aer run that can execute the relief on the finished chain."""

    def setUp(self):
        super().setUp()
        self.with_targets()
        from src.capture.labels import load_labels
        from src.quantum import aer_relief as ar
        self.ar = ar
        self.spec = ar.build_spec(load_labels(self.s.labels_path))[2]

    def run_chain(self, jobs=None):
        """Advance the chain with the ideal local stand-in (what a finished chain looks like on disk), without spawning anything."""
        from src.quantum import facet_qdrive as fq
        for _ in range(jobs or len(fq.plan_chain(self.spec)["jobs"])):
            fq.send_next(self.spec, self.s.qfacets_dir, self.s.labels_path, local=True, log=lambda m: None)

    def test_the_plan_is_the_first_job_with_its_hash_and_one_credit_and_sends_nothing(self):
        p = self.s.plan("qdrive-facets")
        self.assertEqual((p["credits"], p["cached"], p["problems"]), (1, False, []))
        self.assertIn("job 1 of", p["describe"])
        self.assertEqual(len(p["sha256"]), 64)
        self.assertEqual(self.rec.calls, [])

    def test_each_confirm_is_one_job_behind_the_spend_gate(self):
        p = self.s.plan("qdrive-facets")
        with self.assertRaises(PermissionError):
            self.s.confirm("qdrive-facets", p["sha256"], 1)
        self.s.allow_spend = True
        with self.assertRaises(StageError):
            self.s.confirm("qdrive-facets", "0" * 64, 1)
        with self.assertRaises(StageError):
            self.s.confirm("qdrive-facets", p["sha256"], 5)
        self.s.confirm("qdrive-facets", p["sha256"], 1)
        argv = self.rec.calls[0]
        self.assertEqual(argv[1:4], ["-m", "src.quantum.facet_qdrive", "send"])
        self.assertEqual((argv[argv.index("--approve-credits") + 1], argv[argv.index("--jobs") + 1], argv[argv.index("--out") + 1]), ("1", "1", self.s.qfacets_dir))
        self.assertNotIn("--local", argv)

    def test_after_a_job_the_plan_offers_the_next_one_with_a_new_hash(self):
        first = self.s.plan("qdrive-facets")
        self.run_chain(1)
        second = self.s.plan("qdrive-facets")
        self.assertIn("job 2 of", second["describe"])
        self.assertIn("continuing job 1", second["describe"])
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_a_job_with_no_saved_result_blocks_the_plan_until_it_is_looked_at(self):
        from src.quantum import facet_qdrive as fq
        chain = fq.plan_chain(self.spec)
        st = fq.chain_state(self.s.qfacets_dir, chain, self.spec)
        st["steps"]["0"] = dict(status="submitted", job_id="abc")
        os.makedirs(self.s.qfacets_dir)
        fq.save_chain(self.s.qfacets_dir, st)
        self.s.allow_spend = True
        p = self.s.plan("qdrive-facets")
        self.assertTrue(p["problems"] and "abc" in p["problems"][0])
        with self.assertRaises(StageError):
            self.s.confirm("qdrive-facets", p["sha256"], 1)
        self.assertEqual(self.rec.calls, [])

    def test_a_finished_chain_is_free_to_load_and_shown_in_the_solve_step(self):
        self.run_chain()
        p = self.s.plan("qdrive-facets")
        self.assertEqual((p["credits"], p["cached"]), (0, True))
        self.s.confirm("qdrive-facets", p["sha256"], 0)                                      # no allow_spend needed: nothing is sent
        self.assertNotIn("--approve-credits", self.rec.calls[0])
        st = next(x for x in self.s.state()["stages"] if x["id"] == "solve")
        self.assertTrue(any("QDrive facet register" in x and "LOCAL rehearsal" in x and "complete" in x for x in st["extras"]))
        self.assertEqual(st["status"], "ready")                                              # a facet circuit alone is not yet a performance

    def test_the_aer_run_can_execute_the_relief_on_the_finished_chain_and_only_then(self):
        with self.assertRaises(StageError) as cm:
            self.s.plan("aer", None, dict(facets="qdrive"))
        self.assertIn("QDrive facet", str(cm.exception))
        self.run_chain(2)                                                                    # unfinished is not enough either
        with self.assertRaises(StageError):
            self.s.plan("aer", None, dict(facets="qdrive"))
        self.run_chain(len(__import__("src.quantum.facet_qdrive", fromlist=["x"]).plan_chain(self.spec)["jobs"]) - 2)
        ideal, qd = self.s.plan("aer"), self.s.plan("aer", None, dict(facets="qdrive"))
        self.assertNotEqual(ideal["sha256"], qd["sha256"])
        self.assertIn("facet register:", qd["describe"])
        self.assertEqual(qd["opts"]["facets"], "qdrive")
        self.s.confirm("aer", qd["sha256"], 0, None, dict(facets="qdrive"))
        argv = self.rec.calls[0]
        self.assertEqual(argv[argv.index("--facet-circuit") + 1], os.path.join(self.s.qfacets_dir, "circuit.qasm"))
        self.assertNotIn("--facet-circuit", self.s.plan("aer")["command"])
        with self.assertRaises(StageError):
            self.s.plan("aer", None, dict(facets="mystery"))

    def test_a_chain_of_other_relief_settings_cannot_feed_the_aer_run(self):
        self.run_chain()
        with self.assertRaises(StageError):
            self.s.plan("aer", dict(entangle=0.3), dict(facets="qdrive"))

    def test_a_chain_of_an_earlier_drawing_is_not_shown(self):
        self.run_chain()
        with open(self.s.labels_path) as f:
            labels = json.load(f)
        labels["glass"] = labels["glass"][:-1]
        with open(self.s.labels_path, "w") as f:
            json.dump(labels, f)
        st = next(x for x in self.s.state()["stages"] if x["id"] == "solve")
        self.assertEqual(st["extras"], [])


class EchoSolve(Base):
    """Dynamic relief: taps from the local echo (free) or Moth's otoc-echo-v1 (1 credit, behind the spend gate), circuits executed on Aer, and the Perform choice that plays them."""

    def setUp(self):
        super().setUp()
        self.with_targets()

    def fake_run(self, source="local", labels_sha="current", flags=None, consistent=True, aer=False):
        folder = self.s.echo_dir
        os.makedirs(folder, exist_ok=True)
        sha = studio.sha_file(self.s.labels_path) if labels_sha == "current" else labels_sha
        with open(os.path.join(folder, "state.json"), "w") as f:
            json.dump(dict(kind="relief-echo", sha256="e" * 64, labels_sha=sha, taps_source=source, depth=4, n_qubits=22, shots=1000, flags=flags or {}, consistent=consistent), f)
        with open(os.path.join(folder, "shots.npz"), "wb") as f:
            f.write(b"x")
        if aer:
            os.makedirs(self.s.aer_dir, exist_ok=True)
            with open(os.path.join(self.s.aer_dir, "state.json"), "w") as f:
                json.dump(dict(kind="relief-aer", sha256="f" * 64, labels_sha=sha, flags={}, n_qubits=22, shots=1000, score=dict(consistent=True)), f)
            with open(os.path.join(self.s.aer_dir, "shots.npz"), "wb") as f:
                f.write(b"x")

    def test_the_local_plan_is_free_and_runs_without_allow_spend(self):
        p = self.s.plan("echo-local")
        self.assertEqual((p["credits"], p["cached"], p["problems"], p["opts"]), (0, False, [], dict(depth=4, width=4, height=4, shots=200_000)))
        self.assertIn("LOCAL echo", p["describe"])
        self.assertEqual(self.rec.calls, [])
        self.s.confirm("echo-local", p["sha256"], 0)
        argv = self.rec.calls[0]
        self.assertEqual(argv[1:3], ["-m", "src.quantum.echo_relief"])
        self.assertEqual(argv[argv.index("--source") + 1], "local")
        self.assertEqual((argv[argv.index("--depth") + 1], argv[argv.index("--width") + 1], argv[argv.index("--out") + 1]), ("4", "4", self.s.echo_dir))
        for banned in ("--approve-credits", "--allow-spend"):
            self.assertNotIn(banned, argv)

    def test_the_options_shape_the_request_and_bad_ones_are_refused(self):
        a = self.s.plan("echo-local")["sha256"]
        for opts in (dict(depth=3), dict(width=3, height=4), dict(shots=50_000)):
            self.assertNotEqual(a, self.s.plan("echo-local", None, opts)["sha256"])
        for opts in (dict(depth=0), dict(depth=99), dict(width=0), dict(shots=5), dict(depth="deep")):
            with self.assertRaises(StageError):
                self.s.plan("echo-local", None, opts)
        big = self.s.plan("echo-local", None, dict(width=5, height=4))                          # 20 sites: more than the local echo can do
        self.assertTrue(big["problems"])
        with self.assertRaises(StageError):
            self.s.confirm("echo-local", big["sha256"], 0, None, dict(width=5, height=4))
        with self.assertRaises(StageError):
            self.s.plan("echo-local", dict(engine="domain"))
        self.assertEqual(self.rec.calls, [])

    def test_the_moth_plan_costs_one_credit_behind_the_gate_and_asks_for_it_by_hash(self):
        p = self.s.plan("echo-moth")
        self.assertEqual((p["credits"], p["problems"]), (1, []))
        self.assertIn("otoc-echo-v1", p["describe"])
        with self.assertRaises(PermissionError):
            self.s.confirm("echo-moth", p["sha256"], 1)
        self.s.allow_spend = True
        for sha, credits in (("0" * 64, 1), (p["sha256"], 0), (p["sha256"], 5)):
            with self.assertRaises(StageError):
                self.s.confirm("echo-moth", sha, credits)
        self.assertEqual(self.rec.calls, [])
        self.s.confirm("echo-moth", p["sha256"], 1)
        argv = self.rec.calls[0]
        self.assertEqual((argv[argv.index("--source") + 1], argv[argv.index("--approve-credits") + 1]), ("moth", "1"))

    def test_the_engine_lattice_may_be_bigger_than_the_local_one_but_not_beyond_24_sites(self):
        self.assertEqual(self.s.plan("echo-moth", None, dict(width=6, height=4))["problems"], [])
        self.assertTrue(self.s.plan("echo-moth", None, dict(width=5, height=5))["problems"])

    def test_a_saved_moth_result_is_free_to_reuse_and_unfinished_jobs_are_reported(self):
        p = self.s.plan("echo-moth")
        moth = os.path.join(self.s.echo_dir, "moth")
        os.makedirs(moth)
        with open(os.path.join(moth, "payload.sha256"), "w") as f:
            f.write(self.shaof(p) + "\n")
        with open(os.path.join(moth, "raw_result_x.json"), "w") as f:
            f.write("{}")
        p = self.s.plan("echo-moth")
        self.assertEqual(p["credits"], 0)
        self.assertTrue(p["local"])                                                             # the circuits are executed here
        self.s.confirm("echo-moth", p["sha256"], 0)
        self.assertNotIn("--approve-credits", self.rec.calls[-1])
        with open(os.path.join(moth, "job_zzz.json"), "w") as f:
            f.write("{}")
        self.assertEqual(self.s.plan("echo-moth")["orphans"], ["zzz"])

    def shaof(self, plan):
        from src.quantum import echo_relief as er
        return er.moth_job(plan["opts"]["width"], plan["opts"]["height"], plan["opts"]["depth"], plan["opts"]["width"] * plan["opts"]["height"] // 2)[1]

    def test_a_finished_run_completes_step_4_and_unlocks_playing_it(self):
        self.assertEqual(self.status()["final"], "blocked")
        self.fake_run(flags=dict(kappa=0.8))
        st = {x["id"]: x for x in self.s.state()["stages"]}
        self.assertEqual((st["solve"]["status"], st["final"]["status"], st["final"]["choices"]), ("done", "ready", ["echo"]))
        self.assertIn("Dynamic relief was executed on Aer", st["solve"]["note"])
        self.s.start("final")
        argv = self.rec.calls[-1]
        self.assertEqual(argv[1:3], ["-m", "src.show"])
        self.assertEqual((argv[argv.index("--echo-run") + 1], argv[argv.index("--kappa") + 1], argv[argv.index("--source") + 1]), (self.s.echo_dir, "0.8", "relief"))
        for banned in ("--aer-run", "--circuit", "--allow-spend"):
            self.assertNotIn(banned, argv)
        self.assertEqual(self.s.jobs["show"].meta["mode"], "echo")
        self.assertEqual(self.status()["final"], "done")

    def test_a_run_that_disagrees_with_its_reference_says_so(self):
        self.fake_run(consistent=False)
        self.assertIn("does NOT agree", next(x for x in self.s.state()["stages"] if x["id"] == "solve")["note"])

    def test_a_run_of_an_earlier_drawing_is_not_played(self):
        self.fake_run(labels_sha="0" * 64)
        st = {x["id"]: x for x in self.s.state()["stages"]}
        self.assertEqual((st["solve"]["status"], st["final"]["status"]), ("ready", "blocked"))
        self.assertIn("echo run was made from an earlier drawing", st["solve"]["warn"])
        with self.assertRaises(StageError):
            self.s.start("final", dict(**{"from": "echo"}))

    def test_with_the_aer_run_too_both_are_offered_aer_first_and_each_starts_its_own_show(self):
        self.fake_run(aer=True)
        final = next(x for x in self.s.state()["stages"] if x["id"] == "final")
        self.assertEqual(final["choices"], ["aer", "echo"])
        self.s.start("final")
        self.assertIn("--aer-run", self.rec.calls[-1])
        self.s.start("final", dict(**{"from": "echo"}))
        argv = self.rec.calls[-1]
        self.assertIn("--echo-run", argv)
        self.assertNotIn("--aer-run", argv)
        with self.assertRaises(StageError):
            self.s.start("final", dict(**{"from": "qdrive"}))


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


class AerSolve(Base):
    """Steps 4 and 5 on the quantum path: the relief circuit executed on Aer (local, free) and the show performing from its shots."""

    def setUp(self):
        super().setUp()
        self.with_targets()

    def fake_run(self, flags=None, labels_sha="current", consistent=True):
        """What src.quantum.aer_relief leaves behind, without running it."""
        folder = self.s.aer_dir
        os.makedirs(folder, exist_ok=True)
        sha = studio.sha_file(self.s.labels_path) if labels_sha == "current" else labels_sha
        with open(os.path.join(folder, "state.json"), "w") as f:
            json.dump(dict(kind="relief-aer", sha256="f" * 64, labels_sha=sha, flags=flags or {}, n_qubits=22, shots=1000, score=dict(consistent=consistent)), f)
        with open(os.path.join(folder, "shots.npz"), "wb") as f:
            f.write(b"x")

    def test_the_plan_is_free_local_and_names_the_circuit(self):
        p = self.s.plan("aer")
        self.assertEqual((p["credits"], p["cached"], p["problems"], p["orphans"]), (0, False, [], []))
        self.assertEqual(len(p["sha256"]), 64)
        self.assertIn("22 qubits", p["describe"])
        self.assertIn("no credits", p["describe"])
        self.assertEqual(self.rec.calls, [])

    def test_it_runs_without_allow_spend_and_never_asks_for_credits(self):
        self.assertFalse(self.s.allow_spend)
        p = self.s.plan("aer")
        self.assertTrue(self.s.confirm("aer", p["sha256"], 0)["started"])
        argv = self.rec.calls[0]
        self.assertEqual(argv[1:3], ["-m", "src.quantum.aer_relief"])
        self.assertEqual(argv[argv.index("--labels") + 1], self.s.labels_path)
        self.assertEqual(argv[argv.index("--out") + 1], self.s.aer_dir)
        self.assertEqual(argv[argv.index("--shots") + 1], str(p["shots"]))
        for banned in ("--approve-credits", "--allow-spend", "--calib"):
            self.assertNotIn(banned, argv)
        self.assertNotIn("MOTH", " ".join(argv))

    def test_a_confirm_for_a_different_circuit_is_refused(self):
        p = self.s.plan("aer")
        for sha, credits in (("0" * 64, 0), (p["sha256"], 1)):
            with self.assertRaises(StageError):
                self.s.confirm("aer", sha, credits)
        self.assertEqual(self.rec.calls, [])

    def test_relief_settings_change_the_circuit_and_reach_the_command(self):
        a, b = self.s.plan("aer"), self.s.plan("aer", dict(kappa=0.8, entangle=0.4))
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.s.confirm("aer", b["sha256"], 0, dict(kappa=0.8, entangle=0.4))
        argv = self.rec.calls[0]
        self.assertEqual((argv[argv.index("--kappa") + 1], argv[argv.index("--entangle") + 1]), ("0.8", "0.4"))
        self.assertNotEqual(a["sha256"], self.s.plan("aer", dict(), dict(shots=20_000))["sha256"])             # the shot count is part of what is executed

    def test_what_the_aer_run_cannot_cover_is_refused_up_front(self):
        for show in (dict(engine="domain"), dict(source="oracle")):
            with self.assertRaises(StageError):
                self.s.plan("aer", show)
        for shots in (5, 10 ** 9, "many"):
            with self.assertRaises(StageError):
                self.s.plan("aer", None, dict(shots=shots))
        self.s.engine = "domain"                                                                     # a studio started with --engine domain must say so, not run the wrong circuit
        with self.assertRaises(StageError):
            self.s.plan("aer")
        self.assertEqual(self.s.plan("aer", dict(engine="panel"))["problems"], [])
        self.assertEqual(self.rec.calls, [])

    def test_a_run_for_exactly_this_circuit_is_reported_as_saved(self):
        p = self.s.plan("aer")
        self.fake_run()
        with open(os.path.join(self.s.aer_dir, "state.json")) as f:
            st = json.load(f)
        st["sha256"] = p["sha256"]
        with open(os.path.join(self.s.aer_dir, "state.json"), "w") as f:
            json.dump(st, f)
        self.assertTrue(self.s.plan("aer")["cached"])

    def test_two_solves_cannot_run_at_once(self):
        self.s.spawn = lambda argv, **kw: type("Hang", (FakeProc,), dict(wait=lambda self, timeout=None: time.sleep(5)))()
        p = self.s.plan("aer")
        self.s.confirm("aer", p["sha256"], 0)
        with self.assertRaises(StageError):
            self.s.confirm("aer", p["sha256"], 0)

    def test_a_finished_run_completes_step_4_and_unlocks_performing_from_it(self):
        self.assertEqual((self.status()["solve"], self.status()["final"]), ("ready", "blocked"))
        self.fake_run(flags=dict(kappa=0.8, observations="line"))
        self.assertEqual((self.status()["solve"], self.status()["final"]), ("done", "ready"))
        final = next(st for st in self.s.state()["stages"] if st["id"] == "final")
        self.assertEqual(final["choices"], ["aer"])
        self.s.start("final")
        argv = self.rec.calls[-1]
        self.assertEqual(argv[1:3], ["-m", "src.show"])
        self.assertEqual(argv[argv.index("--source") + 1], "relief")
        self.assertEqual(argv[argv.index("--aer-run") + 1], self.s.aer_dir)
        self.assertEqual((argv[argv.index("--kappa") + 1], argv[argv.index("--observations") + 1]), ("0.8", "line"))   # performed with the settings that were executed
        self.assertNotIn("--circuit", argv)
        self.assertNotIn("--allow-spend", argv)
        self.assertEqual(self.s.jobs["show"].meta["mode"], "aer")
        self.assertEqual(self.status()["final"], "done")

    def test_a_run_from_an_earlier_drawing_is_not_performed(self):
        self.fake_run(labels_sha="0" * 64)
        st = {s["id"]: s for s in self.s.state()["stages"]}
        self.assertEqual((st["solve"]["status"], st["final"]["status"]), ("ready", "blocked"))
        self.assertIn("earlier drawing", st["solve"]["warn"])
        with self.assertRaises(StageError):
            self.s.start("final")

    def test_with_both_results_the_aer_run_is_first_and_the_qdrive_circuit_is_one_click_away(self):
        run = os.path.join(self.s.solve, "qdrive")
        os.makedirs(run)
        with open(os.path.join(run, "circuit.qasm"), "w") as f:
            f.write("OPENQASM 3.0;\n")
        with open(os.path.join(run, "payload.sha256"), "w") as f:
            f.write(self.s.plan("qdrive")["sha256"] + "\n")
        with open(os.path.join(run, "raw_result_x.json"), "w") as f:
            f.write("{}")
        self.assertEqual(next(st for st in self.s.state()["stages"] if st["id"] == "final")["choices"], ["qdrive"])
        self.fake_run()
        self.assertEqual(next(st for st in self.s.state()["stages"] if st["id"] == "final")["choices"], ["aer", "qdrive"])
        self.s.start("final")
        self.assertIn("--aer-run", self.rec.calls[-1])
        self.s.start("final", dict(**{"from": "qdrive"}))
        argv = self.rec.calls[-1]
        self.assertEqual(argv[argv.index("--source") + 1], "circuit")
        self.assertNotIn("--aer-run", argv)
        with self.assertRaises(StageError):
            self.s.start("final", dict(**{"from": "graph"}))

    def test_a_run_nobody_made_cannot_be_performed_from(self):
        with self.assertRaises(StageError):
            self.s.start("final", dict(**{"from": "aer"}))
        self.assertEqual(self.rec.calls, [])

    def test_over_http_the_plan_and_the_confirm_carry_the_show_options(self):
        server, token, url = make_server(self.s)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            def post(path, body):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                c.request("POST", path, json.dumps(body), {"Host": f"127.0.0.1:{port}", "X-Token": token})
                r = c.getresponse()
                out = r.status, json.loads(r.read())
                c.close()
                return out
            code, out = post("/api/solve/plan", dict(engine="aer", show=dict(kappa=0.8)))
            self.assertEqual((code, out["plan"]["flags"]), (200, dict(kappa=0.8)))
            code, out = post("/api/solve/confirm", dict(engine="aer", sha256=out["plan"]["sha256"], credits=0, show=dict(kappa=0.8)))
            self.assertEqual((code, out["started"]), (200, True))
            self.assertIn("--kappa", self.rec.calls[0])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
