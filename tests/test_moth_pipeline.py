"""The whole Moth path for the default scene (the six-pane bay window), against a FAKE Moth server on localhost: no real network, no real key, no credits.

    python -m unittest tests.test_moth_pipeline -v

Moth's servers are not always up, so this is the rehearsal of the day they are. A local HTTP server speaks the Atlas API as the client expects it
(POST /engines/<id>/process -> job_id, GET /jobs/<id>/status, GET /jobs/<id>/result, presigned file downloads that must arrive WITHOUT the
Authorization header) and refuses what the real API refuses (unknown params with a 422, graph-v1 over 20 qubits, a missing or wrong key). The REAL
client, solver, engine builders, parsers, show session and CLIs run against it, on the calibration of the default scene
(runs/calibration_bay_4x4, 20 qubits). The results the fake returns are exact moments of our own states, so the checks are about plumbing and shapes
(payload accepted, job recorded before polling, result parsed and scored, circuit loaded as a source, nothing paid twice), not about what Moth would
compute. If this passes and the servers come back, the first real call differs from these only in who answers.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import numpy as np

from src.capture.labels import DEFAULT_CALIB_DIR, bay_window_labels, fill_defaults, validate
from src.geometry.facets import build_facets
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import allocate
from src.quantum import moth_client, moth_engines as me, solver
from src.quantum import sampler_local as sl
from src.quantum.moth_client import MothClient
from src.quantum.relief_state import spec_from_facets
from src.quantum.sampler_mock import Sampler
from tests.make_selftest_fixtures import selftest_circuit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_KEY = "fake-key-for-the-local-server"
CALIB = os.path.join(ROOT, DEFAULT_CALIB_DIR)


class FakeMoth:
    """A localhost Atlas API. `results[engine_id](params, base_url)` makes the result envelope of a job; `posts` records every accepted POST."""

    def __init__(self, results):
        self.results, self.posts, self.rejected, self.file_auth_headers, self.jobs = results, [], [], [], {}
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, data, ctype="application/json"):
                body = data if isinstance(data, bytes) else json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                parts = self.path.strip("/").split("/")                       # api/v1/engines/<id>/process
                if self.headers.get("Authorization") != f"Bearer {FAKE_KEY}":
                    return self._send(401, dict(title="unauthorized"))
                if parts[:3] != ["api", "v1", "engines"] or parts[-1] != "process":
                    return self._send(404, dict(title="not found"))
                eid, params = parts[3], body.get("params")
                spec = moth_client.ENGINES.get(eid)
                if params is None or set(body) - {"params", "input_files"}:
                    outer.rejected.append((eid, "envelope"))
                    return self._send(422, dict(title="invalid", errors=[dict(location=["body"], message="expected {params, input_files?}")]))
                if spec and set(params) - spec["params"]:
                    outer.rejected.append((eid, sorted(set(params) - spec["params"])))
                    return self._send(422, dict(title="invalid params", errors=[dict(location=["params", k], message="unknown field") for k in sorted(set(params) - spec["params"])]))
                if eid == "graph-v1" and not 2 <= params.get("num_qubits", 0) <= 20:
                    outer.rejected.append((eid, "num_qubits"))
                    return self._send(422, dict(title="invalid params", errors=[dict(location=["params", "num_qubits"], message="out of range")]))
                job_id = str(uuid.uuid4())
                outer.jobs[job_id] = dict(engine=eid, params=params, input_files=body.get("input_files"), polls=0)
                outer.posts.append((eid, params, body.get("input_files")))
                self._send(200, dict(job_id=job_id))

            def do_GET(self):
                if self.path.startswith("/files/"):                              # presigned: no key expected, and one here would be a leak
                    outer.file_auth_headers.append(self.headers.get("Authorization"))
                    data = outer.files.get(self.path)
                    return self._send(200, data, "text/plain") if data is not None else self._send(404, dict(title="gone"))
                if self.headers.get("Authorization") != f"Bearer {FAKE_KEY}":
                    return self._send(401, dict(title="unauthorized"))
                parts = self.path.strip("/").split("/")                          # api/v1/jobs/<id>/status|result
                job = outer.jobs.get(parts[3]) if len(parts) == 5 else None
                if job is None:
                    return self._send(404, dict(title="no such job"))
                if parts[4] == "status":
                    job["polls"] += 1
                    return self._send(200, dict(status="running" if job["polls"] == 1 else "completed"))
                return self._send(200, outer.results[job["engine"]](job["params"], outer.base, outer))

        self.files = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        self._patches = [mock.patch.object(moth_client, "API", self.base + "/api/v1"), mock.patch.dict(os.environ, {moth_client.ENV_KEY: FAKE_KEY})]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self._patches:
            p.stop()
        self.server.shutdown()
        self.server.server_close()

    def publish(self, name, data):
        """A presigned file: the URL carries its own authority."""
        path = f"/files/{uuid.uuid4().hex}/{name}"
        self.files[path] = data.encode() if isinstance(data, str) else data
        return self.base + path


def default_scene():
    labels = validate(fill_defaults(bay_window_labels()))
    return labels, build_scene(labels)


def oracle_of_default_calibration():
    """(probabilities over the 20 qubits, sampler, layout, targets) of the default scene on its own calibration."""
    from src.quantum import qdrive_rounds as qr
    with open(os.path.join(CALIB, "targets.json")) as f:
        targets = json.load(f)
    labels, scene = default_scene()
    m = targets["meta"]
    s = m["sampler"]
    sampler = Sampler(scene, *m["grid"], kh=s["kh"], kf=s["kf"], J_in=s["J_in"], J_cross=s["J_cross"], J_pol=s["J_pol"], min_cover=labels["grid"]["min_cover"])
    layout = allocate(sampler.n, sampler.n_panels)
    p, n = qr.oracle_probabilities(labels, targets)
    return p, sampler, layout, targets


def graph_v1_result(params, base, fake):
    """What a healthy graph-v1 returns for the request, built from the exact moments of the oracle state the targets came from."""
    p, sampler, layout, targets = oracle_of_default_calibration()
    n = layout["n_qubits"]
    idx = np.arange(len(p))
    z = lambda q: 1 - 2 * ((idx >> q) & 1)
    bloch = {str(q): dict(X=0.0, Y=0.0, Z=float(p @ z(q))) for q in range(n)}
    rel = {f"{op['qubits'][0]},{op['qubits'][1]}": dict(ZZ=float(p @ (z(op['qubits'][0]) * z(op['qubits'][1]))), XX=0.0)
           for op in params["operations"] if op["type"] == "relationship"}
    order = np.argsort(p)[::-1][:20]
    top = [dict(bitstring="".join("1" if (i >> q) & 1 else "0" for q in range(n)), count=int(1024 * p[i]), probability=float(p[i])) for i in order]
    inline = dict(tomography=dict(bloch=bloch, relationships=rel), measurements=top, dominant_bitstring=top[0]["bitstring"], edge_agreement_score=0.5,
                  coupling_map=params["coupling_map"], shots=params["shots"], backend="emulator")
    return dict(outputs=None, result=inline)


def qdrive_result(params, base, fake):
    p, sampler, layout, targets = oracle_of_default_calibration()
    from qiskit import qasm3
    url = fake.publish("circuit", qasm3.dumps(selftest_circuit(sampler, layout)))
    return dict(outputs=[dict(slot="circuit", url=url, expires_at="2099-01-01T00:00:00Z", output_asset_id=str(uuid.uuid4()))], result=None)


def tomography_result(params, base, fake):
    """Exact single-qubit Bloch vectors and pair correlators of the circuit that was SENT (parsed back from its QASM, so the payload is what is checked)."""
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector
    sv = Statevector(qasm2.loads(params["circuit_qasm"])).data
    n = int(np.log2(len(sv)))
    from src.quantum.complementary import pauli_expectation
    bloch = {str(q): {c: pauli_expectation(sv, n, {q: c.lower()}) for c in "XYZ"} for q in params["qubit_list"]}
    rel = {f"{a},{b}": [[pauli_expectation(sv, n, {a: c1.lower(), b: c2.lower()}) for c2 in "XYZ"] for c1 in "XYZ"] for a, b in params.get("qubit_pair_list", [])}
    return dict(outputs=None, result=dict(bloch=bloch, relationships=rel, mutual_information={}))


def generic_result(params, base, fake):
    return dict(outputs=None, result=dict(ok=True))


RESULTS = {"graph-v1": graph_v1_result, "qdrive-api-v1": qdrive_result, "tomography-api-v2": tomography_result,
           "otoc-echo-v1": generic_result, "qpixl-v1": generic_result, "entanglement-shader-v1": generic_result}


class Calibration(unittest.TestCase):
    """The default calibration folder is the default scene's, at the qubit cap, and the payloads in it are sendable."""

    def test_calibration_belongs_to_the_default_scene(self):
        from src.targets.consistency import scene_problem
        _, scene = default_scene()
        with open(os.path.join(CALIB, "targets.json")) as f:
            meta = json.load(f)["meta"]
        self.assertEqual(meta["grid"], [4, 4])
        self.assertEqual(meta["n_qubits"], 20)                                       # 12 patches + 2 lamps + 6 polarity: the graph-v1 cap
        self.assertEqual(len(meta["qubit_map"]["polarity"]), 6)
        sampler = Sampler(scene, *meta["grid"])
        self.assertIsNone(scene_problem(meta, sampler.cells))
        self.assertIsNotNone(scene_problem(meta, Sampler(scene, 5, 4).cells))        # the first build's 5x4 grid is a different scene

    def test_both_payloads_pass_the_local_checks(self):
        for fname, eid in (("payload_graph_v1.json", "graph-v1"), ("payload_qdrive.json", "qdrive-api-v1")):
            with open(os.path.join(CALIB, fname)) as f:
                params = json.load(f)["params"]
            self.assertEqual(MothClient().prepare(eid, params).problems(), [], eid)


class Solver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "run")

    def tearDown(self):
        self.tmp.cleanup()

    def test_graph_v1_end_to_end_then_a_second_run_costs_nothing(self):
        with FakeMoth(RESULTS) as moth:
            state = solver.solve_graph_v1(CALIB, self.out, approve_credits=5)
            self.assertEqual(len(moth.posts), 1)
            self.assertEqual(moth.posts[0][0], "graph-v1")
            self.assertEqual(state["summary"]["n_achieved"], state["summary"]["n_requested"])      # every requested value came back
            self.assertLess(state["summary"]["rms"], 0.02)                                         # the fake answers with the exact moments
            d = os.path.join(self.out, "graph_v1")
            names = os.listdir(d)
            for need in ("payload.json", "payload.sha256", "state.json", "requested_vs_achieved.csv"):
                self.assertIn(need, names)
            self.assertTrue(any(n.startswith("job_") for n in names) and any(n.startswith("raw_result_") for n in names))
            solver.solve_graph_v1(CALIB, self.out, approve_credits=5)                               # same payload: served from disk
            self.assertEqual(len(moth.posts), 1)

    def test_qdrive_end_to_end_downloads_the_circuit_without_the_key_and_the_show_can_load_it(self):
        with FakeMoth(RESULTS) as moth:
            state = solver.solve_qdrive(CALIB, self.out, approve_credits=1)
            self.assertEqual(len(moth.posts), 1)
            self.assertEqual(moth.file_auth_headers, [None])                                       # presigned download: the key must not travel
            self.assertEqual(state["n_qubits"], 20)
            circuit = os.path.join(self.out, "qdrive", "circuit.qasm")
            self.assertTrue(os.path.exists(circuit))
            self.assertEqual(state["summary"]["n_achieved"], state["summary"]["n_requested"])
        from src.ui.session import Session
        s = Session(None, source="circuit", circuit=circuit, out_dir=os.path.join(self.tmp.name, "show"), run_dir=self.out, calib_dir=CALIB,
                    pool_size=4000, projector_size=(1280, 720))
        self.assertEqual(s.source_kind, "circuit")
        self.assertTrue(s.prov["from_moth"])                                                       # job_*.json beside it: the label may say Moth
        self.assertEqual(s.layout["n_qubits"], 20)

    def test_a_payload_for_another_scene_is_refused_before_anything_is_sent(self):
        # the first build's calibration (three panels, 5x4 grid) does not belong to the default scene: the show refuses to offer it for sending
        old = os.path.join(ROOT, "runs", "calibration_5x4")
        if not os.path.isdir(old):
            self.skipTest("the first build's calibration folder is not on disk")
        from src.ui.session import Session
        with FakeMoth(RESULTS) as moth:
            s = Session(None, out_dir=os.path.join(self.tmp.name, "show"), run_dir=self.out, calib_dir=old, allow_spend=True, projector_size=(1280, 720))
            with self.assertRaises(ValueError) as cm:
                s.resolve_plan("qdrive")
            self.assertIn("different scene", str(cm.exception))
            self.assertEqual(moth.posts, [])

    def test_a_new_payload_never_inherits_the_old_ones_result(self):
        run = os.path.join(self.out, "qdrive")
        from src.store.cache import cached_result, save_payload
        save_payload(run, {"engine": "e", "params": {"a": 1}}, "aaaa1111bbbb2222")
        with open(os.path.join(run, "raw_result_old.json"), "w") as f:
            json.dump({"result": "the old one"}, f)
        self.assertIsNotNone(cached_result(run, "aaaa1111bbbb2222"))
        save_payload(run, {"engine": "e", "params": {"a": 2}}, "cccc3333dddd4444")           # a different payload goes to the same folder (and its job then fails)
        self.assertIsNone(cached_result(run, "cccc3333dddd4444"))
        self.assertTrue(os.path.exists(os.path.join(run, "superseded", "aaaa1111bbbb", "raw_result_old.json")))      # what was paid for is kept

    def test_the_spend_gate_holds_over_real_http(self):
        with FakeMoth(RESULTS) as moth:
            self.assertIsNone(solver.solve_graph_v1(CALIB, self.out, approve_credits=None))        # preview: nothing sent
            with self.assertRaises(SystemExit):
                solver.solve_graph_v1(CALIB, self.out, approve_credits=4)                          # a cost that does not match
            self.assertEqual(moth.posts, [])


class ShowResolve(unittest.TestCase):
    def test_resolve_from_the_show_runs_the_whole_chain_and_switches_the_source(self):
        from src.ui.session import Session
        with tempfile.TemporaryDirectory() as tmp, FakeMoth(RESULTS) as moth:
            s = Session(None, out_dir=os.path.join(tmp, "show"), run_dir=os.path.join(tmp, "run"), calib_dir=CALIB, allow_spend=True, pool_size=4000,
                        projector_size=(1280, 720))
            plan = s.resolve_plan("qdrive")
            self.assertEqual((plan["credits"], plan["cached"]), (1, False))
            with self.assertRaises(ValueError):
                s.resolve_confirm("qdrive", "0" * 64, 1)                                           # a hash that is not the one shown
            self.assertEqual(moth.posts, [])
            s.resolve_confirm("qdrive", plan["sha256"], 1)
            for _ in range(600):
                if s.source_kind == "circuit":
                    break
                time.sleep(0.1)
            self.assertEqual(s.source_kind, "circuit")
            self.assertEqual(len(moth.posts), 1)
            self.assertTrue(s.prov["from_moth"])


class Engines(unittest.TestCase):
    """Every engine builder's payload is accepted by the (fake) API's own schema check, runs through the real client and parses back."""

    @classmethod
    def setUpClass(cls):
        _, cls.scene = default_scene()
        cls.fs = build_facets(cls.scene)
        cls.spec = spec_from_facets(cls.fs, cls.scene, entangle=0.8)

    def run_job(self, moth, eid, payload, out=None):
        job = MothClient().prepare(eid, payload["params"])
        self.assertEqual(job.problems(), [], f"{eid} would be refused locally")
        return MothClient(poll_interval=0.01).run(eid, payload["params"], approved=True, out_dir=out)

    def test_the_default_scene_fits_the_per_panel_engine(self):
        self.assertEqual((self.spec.F, self.spec.P, self.spec.n_sim, self.spec.n_full), (12, 6, 18, 22))

    def test_tomography_of_the_default_circuit_scores_zero_against_itself(self):
        payload = me.tomography_payload(self.spec)
        self.assertEqual(payload["params"]["qubit_list"], list(range(18)))
        with FakeMoth(RESULTS) as moth, tempfile.TemporaryDirectory() as tmp:
            out = self.run_job(moth, "tomography-api-v2", payload, tmp)
            parsed = me.parse_tomography(out["result"])
            score = me.score_tomography(parsed, self.spec)
            self.assertEqual(score["n_qubits"], 18)
            self.assertLess(score["rms_single"], 1e-6)
            self.assertEqual(len(score["visibility"]), 6)                                          # one per panel
            for v in score["visibility"]:
                self.assertAlmostEqual(v["measured"], v["isolated"], places=3)

    def test_the_other_engines_accept_their_payloads(self):
        payloads = [("qdrive-api-v1", me.qdrive_probe_payload()), ("qdrive-api-v1", dict(params=me.qdrive_facet_payloads(self.spec)[0]["params"])),
                    ("otoc-echo-v1", me.echo_payload()), ("qpixl-v1", me.qpixl_payload(np.linspace(0.05, 0.95, 16))), ("entanglement-shader-v1", me.shader_payload())]
        with FakeMoth(RESULTS) as moth:
            for eid, payload in payloads:
                self.run_job(moth, eid, payload)
            self.assertEqual(moth.rejected, [])
            self.assertEqual(len(moth.posts), len(payloads))

    def test_the_fake_really_does_refuse_what_the_real_api_refuses(self):
        with FakeMoth(RESULTS) as moth:
            c = MothClient()
            bad = dict(moth_client.PreparedJob("qdrive-api-v1", dict(machine="aer", n_qubits=2, shots=8, nonsense=1)).body)
            r = c._http.post(moth_client.API + "/engines/qdrive-api-v1/process", headers=c._headers(), json=bad)
            self.assertEqual(r.status_code, 422)
            r = c._http.post(moth_client.API + "/engines/qdrive-api-v1/process", headers=dict(Authorization="Bearer wrong"), json=bad)
            self.assertEqual(r.status_code, 401)

    def test_domain_payloads_for_the_default_scene(self):
        from src.geometry.domains import build_domains
        from src.quantum import domain_moth as dm
        from src.quantum.domain_state import leaf_domains, spec_from_domains
        ds = build_domains(self.scene, group_size=2)
        spec = spec_from_domains(ds, self.scene, entangle=0.8, pol_coupling=0.5, lock=1.5708)
        jobs = dm.qdrive_polarity_payloads(spec)
        self.assertTrue(jobs)
        patch_payload, patch = dm.patch_tomography_payload(spec, dm.connected_patch(ds, leaf_domains(ds)[0], 6))
        self.assertLessEqual(patch.n_sim, 20)
        with FakeMoth(RESULTS) as moth:
            self.run_job(moth, "qdrive-api-v1", dict(params=jobs[0]["params"]))
            self.run_job(moth, "tomography-api-v2", patch_payload)
            self.assertEqual(moth.rejected, [])


class CommandLines(unittest.TestCase):
    """Every command a README line tells you to type reaches its code with the default scene (previews and local rehearsals only: nothing is sent)."""

    def sh(self, *args, timeout=600):
        env = dict(os.environ, MOTH_API_KEY="not-a-real-key-and-nothing-is-sent")
        r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        self.assertEqual(r.returncode, 0, f"{' '.join(args)}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return r.stdout

    def test_solver_previews_send_nothing(self):
        for eng in ("graph-v1", "qdrive"):
            out = self.sh("src.quantum.solver", eng, "--out", tempfile.mkdtemp())
            self.assertIn("nothing was sent", out)

    def test_engine_builders(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("tomography-api-v2", self.sh("src.quantum.moth_engines", "plan"))
            for what in ("tomography", "qdrive-probe", "qdrive-facets", "echo", "qpixl", "shader"):
                self.assertIn("nothing was sent", self.sh("src.quantum.moth_engines", "build", what, "--out", tmp))
            for what in ("polarity-qdrive", "patch-tomography", "echo"):
                self.assertIn("nothing was sent", self.sh("src.quantum.domain_moth", "build", what, "--out", tmp))
            payload = os.path.join(tmp, "payload_tomography-api-v2_tomography.json")
            self.assertTrue(os.path.exists(payload))
            self.assertIn("nothing was sent", self.sh("src.quantum.moth_client", "preview", payload, "--engine", "tomography-api-v2"))

    def test_qdrive_in_rounds_plans_and_rehearses_locally_for_the_default_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.sh("src.quantum.qdrive_rounds", "plan", "--out", tmp)
            self.assertIn("2 component(s)", out)                                                   # patches + lamps (14 qubits) and polarity (6)
            out = self.sh("src.quantum.qdrive_rounds", "run", "--engine", "local", "--out", tmp)
            self.assertIn("achieved 65/65", out)
            circuit = os.path.join(tmp, "qdrive_local", "circuit.qasm")
            self.assertTrue(os.path.exists(circuit))
            from src.ui.session import Session
            s = Session(None, source="circuit", circuit=circuit, out_dir=os.path.join(tmp, "show"), run_dir=tmp, pool_size=2000, projector_size=(1280, 720))
            self.assertFalse(s.prov["from_moth"])                                                  # a local rehearsal is never called a Moth result


if __name__ == "__main__":
    unittest.main()
