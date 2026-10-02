"""moth_client safety properties, tested against a fake session: no network, no key, no credits.

    python -m unittest tests.test_moth_client -v
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import requests

from src.quantum import moth_client
from src.quantum.moth_client import MissingKey, MothClient, MothError, NotApproved

PARAMS = {"num_qubits": 2, "coupling_map": [[0, 1]], "operations": [], "shots": 8, "mode": "emu"}
FAKE_KEY = "sk-test-DO-NOT-LEAK-123"


class Resp:
    def __init__(self, data, ok=None, status=200, content=b""):
        ok = (status < 400) if ok is None else ok
        self._d, self.ok, self.status_code, self.text, self.content = data, ok, status, json.dumps(data), content

    def json(self):
        return self._d


class FakeSession:
    def __init__(self, post=None, gets=None):
        self.post_calls, self.get_calls, self._post, self._gets = [], [], post, list(gets or [])

    def post(self, url, headers=None, json=None, timeout=None):
        self.post_calls.append((url, headers, json))
        if isinstance(self._post, Exception):
            raise self._post
        return self._post

    def get(self, url, headers=None, timeout=None):
        self.get_calls.append((url, headers))
        return self._gets.pop(0)


class MothClientTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop(moth_client.ENV_KEY, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.dotenv = os.path.join(self.tmp.name, ".env")                    # never touch the real project .env in tests
        self.patch = mock.patch.object(moth_client, "DOTENV", self.dotenv)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()
        self.env.stop()

    def write_env(self, text, mode=0o600):
        with open(self.dotenv, "w") as f:
            f.write(text)
        os.chmod(self.dotenv, mode)

    def test_prepare_and_describe_need_no_key_and_never_print_it(self):
        os.environ[moth_client.ENV_KEY] = FAKE_KEY
        c = MothClient(session=FakeSession())
        text = c.prepare("graph-v1", PARAMS).describe()
        self.assertNotIn(FAKE_KEY, text)
        self.assertIn("5 credits", text)
        self.assertIn("/engines/graph-v1/process", text)
        self.assertEqual(c._http.post_calls, [])

    def test_key_from_dotenv_formats(self):
        for text in (f"MOTH_API_KEY={FAKE_KEY}\n", f'MOTH_API_KEY="{FAKE_KEY}"\n', f"# c\nexport MOTH_API_KEY='{FAKE_KEY}'\nOTHER=1\n",
                     f"OTHER=1\nMOTH_API_KEY={FAKE_KEY} # trailing\n"):
            self.write_env(text)
            self.assertEqual(moth_client.load_key(), (FAKE_KEY, ".env"), text)

    def test_environment_wins_over_dotenv_and_nothing_is_printed(self):
        self.write_env("MOTH_API_KEY=from-file\n")
        os.environ[moth_client.ENV_KEY] = FAKE_KEY
        self.assertEqual(moth_client.load_key(), (FAKE_KEY, "environment"))
        text = MothClient().prepare("graph-v1", PARAMS).describe()
        self.assertNotIn(FAKE_KEY, text)
        self.assertNotIn("from-file", text)
        self.assertIn("found in environment", text)

    def test_missing_everywhere_and_other_variables_ignored(self):
        self.assertEqual(moth_client.load_key(), (None, None))
        self.write_env("SOMETHING_ELSE=abc\nMOTH_API_KEY_OLD=abc\n")
        self.assertEqual(moth_client.load_key(), (None, None))

    def test_world_readable_dotenv_warns(self):
        self.write_env(f"MOTH_API_KEY={FAKE_KEY}\n", mode=0o644)
        with self.assertWarnsRegex(UserWarning, "chmod 600"):
            moth_client.load_key()

    def test_unapproved_submit_is_refused_before_any_http(self):
        s = FakeSession(post=Resp({"job_id": "j"}))
        with self.assertRaises(NotApproved):
            MothClient(api_key=FAKE_KEY, session=s).submit(MothClient().prepare("graph-v1", PARAMS))
        self.assertEqual(s.post_calls, [])

    def test_missing_key_is_refused_before_any_http(self):
        s = FakeSession(post=Resp({"job_id": "j"}))
        with self.assertRaises(MissingKey):
            MothClient(session=s).submit(MothClient().prepare("graph-v1", PARAMS), approved=True)
        self.assertEqual(s.post_calls, [])

    def test_post_is_never_retried(self):
        s = FakeSession(post=requests.ConnectionError("boom"))
        with self.assertRaises(requests.ConnectionError):
            MothClient(api_key=FAKE_KEY, session=s).submit(MothClient().prepare("graph-v1", PARAMS), approved=True)
        self.assertEqual(len(s.post_calls), 1)

    def test_full_run_records_job_first_and_downloads_without_the_key(self):
        s = FakeSession(post=Resp({"job_id": "job-1"}), gets=[
            Resp({"status": "running"}), Resp({"status": "completed"}),
            Resp({"outputs": [{"slot": "../evil/circuit", "url": "https://files.example/presigned?sig=abc"}]})])
        c = MothClient(api_key=FAKE_KEY, session=s, poll_interval=0)
        with tempfile.TemporaryDirectory() as d, mock.patch("src.quantum.moth_client.requests.get") as dl:
            dl.return_value = Resp({}, content=b"OPENQASM 3;")
            out = c.run("graph-v1", PARAMS, approved=True, out_dir=d)
            with open(os.path.join(d, "job_job-1.json")) as f:
                record = json.load(f)
            self.assertEqual((record["job_id"], record["estimated_credits"]), ("job-1", 5))
            saved = out["files"]["../evil/circuit"]
            self.assertEqual(os.path.dirname(saved), d)                     # slot name cannot escape out_dir
            with open(saved, "rb") as f:
                self.assertEqual(f.read(), b"OPENQASM 3;")
        self.assertEqual(s.post_calls[0][1]["Authorization"], f"Bearer {FAKE_KEY}")
        self.assertEqual(s.post_calls[0][2], {"params": PARAMS})
        _, kwargs = dl.call_args
        self.assertNotIn("headers", kwargs)                                  # presigned URL: no Authorization header

    def test_failed_job_surfaces_the_engine_error(self):
        s = FakeSession(post=Resp({"job_id": "j2"}), gets=[Resp({"status": "failed", "error": "too many qubits"})])
        with self.assertRaisesRegex(MothError, "too many qubits"):
            MothClient(api_key=FAKE_KEY, session=s, poll_interval=0).run("graph-v1", PARAMS, approved=True)

    def test_timeout_names_the_job_so_it_is_not_lost(self):
        s = FakeSession(post=Resp({"job_id": "j3"}), gets=[Resp({"status": "running"})] * 50)
        with self.assertRaisesRegex(TimeoutError, "j3"):
            MothClient(api_key=FAKE_KEY, session=s, poll_interval=0, timeout=-1).run("graph-v1", PARAMS, approved=True)

    def test_graph_v1_over_20_qubits_is_blocked_everywhere(self):
        job = MothClient().prepare("graph-v1", {**PARAMS, "num_qubits": 23})
        self.assertIn("BLOCKED", job.describe())
        s = FakeSession(post=Resp({"job_id": "j"}))
        with self.assertRaises(moth_client.InvalidPayload):
            MothClient(api_key=FAKE_KEY, session=s).submit(job, approved=True)
        self.assertEqual(s.post_calls, [])
        self.assertEqual(MothClient().prepare("graph-v1", {**PARAMS, "num_qubits": 19}).problems(), [])

    def test_unknown_param_and_qpu_mode_are_blocked(self):
        self.assertTrue(MothClient().prepare("graph-v1", {**PARAMS, "nonsense": 1}).problems())
        self.assertTrue(MothClient().prepare("graph-v1", {**PARAMS, "mode": "qpu"}).problems())
        self.assertEqual(MothClient().prepare("qdrive-api-v1", {"n_qubits": 19, "targets": [None], "seed": 1}).problems(), [])

    def test_inline_result_envelope_with_null_outputs_and_raw_saved_first(self):
        inline = {"tomography": {"bloch": []}, "measurements": {"top": []}}
        s = FakeSession(post=Resp({"job_id": "g1"}), gets=[Resp({"status": "processing"}), Resp({"status": "completed"}),
                                                           Resp({"outputs": None, "result": json.dumps(inline)})])   # string-encoded on purpose
        with tempfile.TemporaryDirectory() as d:
            out = MothClient(api_key=FAKE_KEY, session=s, poll_interval=0).run("graph-v1", PARAMS, approved=True, out_dir=d)
            self.assertEqual(out["inline"], inline)
            self.assertEqual(out["files"], {})
            self.assertTrue(os.path.exists(os.path.join(d, "raw_result_g1.json")))

    def test_422_problem_json_lists_every_violation(self):
        body = {"title": "Unprocessable", "errors": [{"location": ["params", "num_qubits"], "message": "must be <= 20"},
                                                     {"location": ["params", "x"], "message": "unknown"}]}
        s = FakeSession(post=Resp(body, ok=False, status=422))
        with self.assertRaisesRegex(MothError, r"params\.num_qubits: must be <= 20; params\.x: unknown"):
            MothClient(api_key=FAKE_KEY, session=s).submit(MothClient().prepare("graph-v1", PARAMS), approved=True)

    def test_post_503_says_not_recorded_and_is_not_retried(self):
        s = FakeSession(post=Resp({}, ok=False, status=503))
        with self.assertRaisesRegex(MothError, "NOT recorded"):
            MothClient(api_key=FAKE_KEY, session=s).submit(MothClient().prepare("graph-v1", PARAMS), approved=True)
        self.assertEqual(len(s.post_calls), 1)

    def test_get_retries_on_429_then_succeeds(self):
        s = FakeSession(gets=[Resp({}, ok=False, status=429), Resp({}, ok=False, status=503), Resp({"status": "completed"})])
        st = MothClient(api_key=FAKE_KEY, session=s, retry_sleep=0).status("j")
        self.assertEqual(st["status"], "completed")
        self.assertEqual(len(s.get_calls), 3)

    def test_failed_job_error_object_is_reported(self):
        err = {"type": "prep_failed", "message": "targets infeasible", "retryable": False}
        s = FakeSession(post=Resp({"job_id": "j9"}), gets=[Resp({"status": "failed", "error": err})])
        with self.assertRaisesRegex(MothError, r"prep_failed: targets infeasible \(retryable: False\)"):
            MothClient(api_key=FAKE_KEY, session=s, poll_interval=0).run("graph-v1", PARAMS, approved=True)

    def test_cli_send_requires_matching_credit_approval(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            with open(path, "w") as f:
                json.dump({"params": PARAMS}, f)
            with mock.patch("src.quantum.moth_client.requests.Session") as sess:
                with self.assertRaises(SystemExit):
                    moth_client.main(["send", path, "--engine", "graph-v1", "--approve-credits", "1"])
                with self.assertRaises(SystemExit):                          # unpriced engine cannot be sent
                    moth_client.main(["send", path, "--engine", "tessa-image-v1", "--approve-credits", "0"])
                sess.return_value.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
