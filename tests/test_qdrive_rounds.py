"""QDrive in rounds: cutting, chaining, resuming, failing, assembling piece-wise outputs. Nothing is sent to Moth.

    python -m unittest tests.test_qdrive_rounds -v

The engine here is an IDEAL fake: it returns a circuit that prepares exactly the state the plan wants after the step's last layer
(and, like the real service, refuses a chained job whose initial_circuit is not the asset it issued for the previous step). So these tests
check the plumbing (chaining, state file, scoring, resume, partial assembly, provenance), not how the real engine converges.
"""
import json
import os
import tempfile
import unittest
import uuid

import numpy as np

from src.quantum import qdrive_plan as qp
from src.quantum import qdrive_rounds as qr
from src.quantum import solver
from src.quantum.moth_client import MothError, NotApproved, PreparedJob, output_asset_id

PAIRS = [(0, 1), (1, 2), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4), (5, 6)]


def scene_dist():
    """7 qubits: an Ising chain with a hub on 0-4 (needs 3-body groups) times an independent pair 5-6."""
    n = 5
    idx = np.arange(2 ** n)
    z = lambda q: 1 - 2 * ((idx >> q) & 1)
    e = sum(j * z(a) * z(b) for (a, b), j in {(0, 1): .8, (1, 2): -.5, (2, 3): .9, (0, 4): .3, (1, 4): -.4, (2, 4): .2, (3, 4): .6}.items())
    a = np.exp(e); a /= a.sum()
    b = np.array([.45, .05, .05, .45])
    return np.kron(b, a)                         # little-endian: qubits 5,6 are the high bits


def make_plan(layers_per_round=2, **kw):
    p = scene_dist()
    return qr.plan_rounds(qp.OracleMoments(p, 7, PAIRS), p, 7, layers_per_round=layers_per_round, **kw), p


class FakeClient(qr.LocalClient):
    """The ideal local engine, as if it were Moth (is_local off) so the credit gate applies; can be told to fail or to return a wrong circuit."""
    is_local = False

    def __init__(self, rp, fail_at=None, wrong_at=None):
        super().__init__(rp)
        self.calls, self.fail_at, self.wrong_at = [], fail_at, wrong_at

    def run(self, engine_id, params, *, approved=False, out_dir=None, input_files=None):
        assert approved
        step = self._step(params)
        if step.index == 0:
            assert input_files is None and params["n_qubits"] == len(self.rp.comps[step.comp].qubits)
        self.calls.append(step.id)
        if self.fail_at == len(self.calls):
            job_id = str(uuid.uuid4())
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"job_{job_id}.json"), "w") as f:
                json.dump({"job_id": job_id}, f)
            raise MothError(f"job {job_id} failed: engine_timeout: The engine did not respond in time (retryable: True)")
        return super().run(engine_id, params, approved=approved, out_dir=out_dir, input_files=input_files)

    def circuit_for(self, step):
        prep = self.rp.comps[step.comp].prep
        if self.wrong_at == step.id:
            return qr.local_circuit(prep.layer_psi[0] if step.hi - 1 else np.eye(len(prep.layer_psi[0]))[0])
        return super().circuit_for(step)


def z_dist(qasm):
    from src.quantum import sampler_local as sl
    psi, _ = sl.statevector_from_qasm(qasm)
    return np.abs(psi) ** 2


class Cutting(unittest.TestCase):
    def test_rounds_cover_every_layer_once_and_chain(self):
        rp, _ = make_plan(1)
        for ci, comp in enumerate(rp.comps):
            steps = rp.steps_of(ci)
            self.assertEqual([(s.lo, s.hi) for s in steps], [(k, k + 1) for k in range(len(comp.prep.layers))])
            self.assertEqual([s.index for s in steps], list(range(len(steps))))

    def test_only_the_first_job_of_a_chain_states_n_qubits(self):
        rp, _ = make_plan(1)
        for s in rp.steps:
            params = rp.params(s)
            self.assertEqual("n_qubits" in params, s.index == 0)
            self.assertTrue(params["targets"][-1] is None)

    def test_rounds_alternate_components(self):
        rp, _ = make_plan(1)
        order = rp.order()
        self.assertEqual([s.index for s in order], sorted(s.index for s in order))

    def test_cut_targets_equal_the_single_job(self):
        rp, p = make_plan(2)
        for ci, comp in enumerate(rp.comps):
            joined = [t for s in rp.steps_of(ci) for t in rp.params(s)["targets"]]
            pairs = [q for q in PAIRS if q[0] in comp.qubits and q[1] in comp.qubits]
            whole = qp.build_state_prep_job(comp.qubits, pairs, qr.marginal(p, 7, comp.qubits)).params["targets"]
            self.assertEqual(joined, whole)

    def test_plan_changes_with_settings(self):
        self.assertNotEqual(make_plan(1)[0].sha, make_plan(2)[0].sha)
        self.assertNotEqual(make_plan(2)[0].sha, make_plan(2, coupling_map=False)[0].sha)

    def test_coupling_map_can_be_left_out(self):
        rp, _ = make_plan(2, coupling_map=False)
        self.assertTrue(all("coupling_map" not in rp.params(s) for s in rp.steps))


class Chaining(unittest.TestCase):
    def test_chained_job_rules_are_checked_before_sending(self):
        p = {"targets": [], "machine": "aer"}
        self.assertFalse(PreparedJob("qdrive-api-v1", p, {"initial_circuit": str(uuid.uuid4())}).problems())
        self.assertTrue(PreparedJob("qdrive-api-v1", p, {"initial_circuit": "job:abc/circuit"}).problems())
        self.assertTrue(PreparedJob("qdrive-api-v1", dict(p, n_qubits=3), {"initial_circuit": str(uuid.uuid4())}).problems())

    def test_output_asset_id(self):
        self.assertEqual(output_asset_id({"outputs": [{"slot": "circuit", "output_asset_id": "A"}]}), "A")
        self.assertIsNone(output_asset_id({"outputs": None}))


class Running(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = os.path.join(self.tmp.name, "qdrive")
        self.rp, self.p = make_plan(2)
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, client, **kw):
        return qr.run_rounds(self.rp, self.base, client, log=self.log.append, **kw)

    def test_preview_sends_nothing(self):
        c = FakeClient(self.rp)
        r = self.go(c)
        self.assertTrue(r["preview"])
        self.assertEqual(c.calls, [])
        self.assertFalse(os.path.exists(os.path.join(self.base, "rounds", "rounds_state.json")))

    def test_approval_must_match_the_queued_jobs(self):
        c = FakeClient(self.rp)
        with self.assertRaises(NotApproved):
            self.go(c, approve_credits=1, max_steps=3)
        self.assertEqual(c.calls, [])

    def test_full_run_assembles_the_exact_distribution(self):
        c = FakeClient(self.rp)
        r = self.go(c, approve_credits=len(self.rp.steps))
        self.assertTrue(r["ok"])
        self.assertEqual(sorted(c.calls), sorted(s.id for s in self.rp.steps))
        st = qr.load_state(self.base, self.rp)
        self.assertTrue(all(v["status"] == "completed" and v["tv"] < 1e-6 for v in st["steps"].values()))
        asm = qr.assemble(self.rp, self.base)
        self.assertTrue(asm["complete"])
        self.assertLess(0.5 * np.abs(z_dist(asm["text"]) - self.p).sum(), 1e-6)
        prov = qr.write_assembled(asm, self.rp, self.base)
        self.assertTrue(prov["from_moth"])
        self.assertTrue(os.path.exists(os.path.join(self.base, "circuit.qasm")))
        info = qr.circuit_provenance(os.path.join(self.base, "circuit.qasm"))
        self.assertTrue(info["from_moth"] and "chained jobs" in info["label"])

    def test_one_round_at_a_time_and_resume_never_resends(self):
        c = FakeClient(self.rp)
        self.go(c, approve_credits=2, max_steps=2)
        self.assertEqual(len(c.calls), 2)
        done = set(c.calls)
        c2 = FakeClient(self.rp)
        c2.assets = dict(c.assets)
        self.go(c2, approve_credits=len(self.rp.steps) - 2)
        self.assertFalse(done & set(c2.calls))
        self.assertEqual(len(c.calls) + len(c2.calls), len(self.rp.steps))

    def test_a_failed_job_stops_the_run_and_is_recorded(self):
        c = FakeClient(self.rp, fail_at=2)
        r = self.go(c, approve_credits=len(self.rp.steps))
        self.assertFalse(r["ok"])
        self.assertEqual(len(c.calls), 2)                             # stop on error: nothing else was sent
        rec = qr.load_state(self.base, self.rp)["steps"][r["failed"]]
        self.assertEqual(rec["status"], "failed")
        self.assertTrue(rec["job_id"])
        self.assertTrue(any("canary" in line for line in self.log))
        asm = qr.assemble(self.rp, self.base)
        self.assertFalse(asm["complete"])
        prov = qr.write_assembled(asm, self.rp, self.base)
        self.assertFalse(prov["from_moth"])
        self.assertFalse(os.path.exists(os.path.join(self.base, "circuit.qasm")))

    def test_failed_step_is_not_resent_without_retry(self):
        self.go(FakeClient(self.rp, fail_at=1), approve_credits=len(self.rp.steps))
        st = qr.load_state(self.base, self.rp)
        failed = next(k for k, v in st["steps"].items() if v["status"] == "failed")
        queue, blocked = qr.pending_steps(self.rp, st)
        self.assertIn(failed, blocked)
        self.assertNotIn(failed, [s.id for s in queue])
        self.assertTrue(all(s.comp != int(failed[1]) for s in queue))   # the chain behind a failed step is blocked too; the other component continues
        queue2, _ = qr.pending_steps(self.rp, st, retry_failed=True)
        self.assertIn(failed, [s.id for s in queue2])

    def test_retry_failed_completes_the_chain(self):
        c = FakeClient(self.rp, fail_at=1)
        self.go(c, approve_credits=len(self.rp.steps))
        queue, _ = qr.pending_steps(self.rp, qr.load_state(self.base, self.rp), retry_failed=True)
        c.fail_at = None
        r = self.go(c, approve_credits=len(queue), retry_failed=True)
        self.assertTrue(r["ok"])
        self.assertTrue(qr.assemble(self.rp, self.base)["complete"])

    def test_a_round_that_lands_far_from_its_target_halts_before_the_next(self):
        c = FakeClient(self.rp, wrong_at="c0s0")
        r = self.go(c, approve_credits=len(self.rp.steps), max_tv=0.05)
        self.assertEqual(r["halted"], "c0s0")
        self.assertEqual(c.calls, ["c0s0"])
        self.assertGreater(qr.load_state(self.base, self.rp)["steps"]["c0s0"]["tv"], 0.05)

    def test_stale_state_is_refused(self):
        self.go(FakeClient(self.rp), approve_credits=1, max_steps=1)
        other, _ = make_plan(1)
        with self.assertRaises(qr.StaleRounds):
            qr.load_state(self.base, other)
        qr.restart(self.base)
        self.assertEqual(qr.load_state(self.base, other)["steps"], {})

    def test_local_fill_marks_the_circuit_as_not_pure_moth(self):
        n0 = len(self.rp.steps_of(0))
        c = FakeClient(self.rp)
        self.assertTrue(self.go(c, approve_credits=n0, components=[0])["ok"])
        self.assertEqual({s[1] for s in c.calls}, {"0"})                 # only component 0 was sent
        self.assertFalse(qr.assemble(self.rp, self.base)["complete"])
        asm = qr.assemble(self.rp, self.base, local_fill=True)
        self.assertTrue(asm["complete"])
        self.assertEqual([p["source"] for p in asm["pieces"]], ["qdrive", "local"])
        self.assertLess(0.5 * np.abs(z_dist(asm["text"]) - self.p).sum(), 1e-6)
        prov = qr.write_assembled(asm, self.rp, self.base)
        info = qr.circuit_provenance(os.path.join(self.base, "circuit.qasm"))
        self.assertFalse(prov["from_moth"] or info["from_moth"])
        self.assertIn("not a pure Moth result", info["label"])

    def test_partial_assembly_is_not_the_shows_circuit(self):
        self.go(FakeClient(self.rp), approve_credits=len(self.rp.steps_of(0)), components=[0])
        prov = qr.write_assembled(qr.assemble(self.rp, self.base), self.rp, self.base)
        self.assertFalse(prov["complete"])
        self.assertTrue(os.path.exists(os.path.join(self.base, "circuit_partial.qasm")))
        self.assertFalse(os.path.exists(os.path.join(self.base, "circuit.qasm")))
        self.assertFalse(qr.circuit_provenance(os.path.join(self.base, "circuit_partial.qasm"))["from_moth"])


class LocalRehearsal(unittest.TestCase):
    def test_local_engine_needs_no_credits_and_is_never_called_moth(self):
        rp, p = make_plan(2)
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "qdrive_local")
            r = qr.run_rounds(rp, base, qr.LocalClient(rp), log=lambda m: None)
            self.assertTrue(r["ok"] and not r["preview"])
            asm = qr.assemble(rp, base)
            self.assertTrue(asm["complete"])
            self.assertLess(0.5 * np.abs(z_dist(asm["text"]) - p).sum(), 1e-6)
            prov = qr.write_assembled(asm, rp, base)
            info = qr.circuit_provenance(os.path.join(base, "circuit.qasm"))
            self.assertFalse(prov["from_moth"] or info["from_moth"])
            self.assertIn("LOCAL rehearsal", info["label"])


class Canary(unittest.TestCase):
    def test_canary_is_the_known_good_lab_job_and_obeys_the_gate(self):
        t = qr.CANARY_PARAMS["targets"]
        self.assertEqual(t[0], {"qubits": [0, 1], "expvals": {"ZI": 0.0, "IZ": 0.0, "ZZ": 1.0}})
        self.assertIsNone(qr.canary("/nonexistent", log=lambda m: None))             # preview: sends nothing
        with self.assertRaises(NotApproved):
            qr.canary("/nonexistent", approve_credits=5, log=lambda m: None)


class SolverFinish(unittest.TestCase):
    def test_finish_rounds_writes_what_the_show_reads(self):
        rp, p = make_plan(2)
        targets = dict(meta=dict(qubit_map=dict(n_qubits=7, patches=[0, 1, 2, 3], lamps=[4], polarity=[5, 6]), n_qubits=7),
                       bloch=[dict(qubit=q, label=f"q{q}", Z=float(qp.OracleMoments(p, 7, PAIRS)([q]))) for q in range(7)],
                       relationships=[dict(qubits=list(e), label=f"e{e}", kind="coplanar", ZZ=float(qp.OracleMoments(p, 7, PAIRS)(e))) for e in PAIRS])
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "qdrive")
            qr.run_rounds(rp, base, FakeClient(rp), approve_credits=len(rp.steps), log=lambda m: None)
            prov = solver.finish_rounds(rp, base, targets)
            self.assertTrue(prov["from_moth"])
            for name in ("circuit.qasm", "state.json", "requested_vs_achieved.csv", "provenance.json"):
                self.assertTrue(os.path.exists(os.path.join(base, name)), name)
            with open(os.path.join(base, "state.json")) as f:
                self.assertLess(json.load(f)["summary"]["rms"], 1e-6)


if __name__ == "__main__":
    unittest.main()
