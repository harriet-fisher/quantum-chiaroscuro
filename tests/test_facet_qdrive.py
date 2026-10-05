"""QDrive-prepared facet register: the plan, the chained jobs and their spend gate, the local rehearsal, and the Aer run that executes the relief on the
circuit that came back. A recording fake stands in for Moth; nothing here can reach the network.

    python -m unittest tests.test_facet_qdrive -v
"""
import json
import os
import tempfile
import unittest

import numpy as np

from src.capture.labels import bay_window_labels
from src.quantum import aer_relief as ar
from src.quantum import facet_qdrive as fq
from src.quantum.complementary import pauli_expectation
from src.quantum.moth_client import ENGINES, MothError, NotApproved
from src.quantum.relief_state import ReliefState, equal_up_to_phase, facet_block, facet_state, reference_circuit, statevector_of

LABELS = bay_window_labels()
SPEC = ar.build_spec(LABELS)[2]
PLAN = fq.plan_chain(SPEC)
TMP = tempfile.TemporaryDirectory()
LOG = lambda m: None


def tearDownModule():
    TMP.cleanup()


def folder(name):
    return os.path.join(TMP.name, name)


class FakeQDrive:
    """Records what is sent. Each job returns the ideal circuit for the couplings of the layers so far, like the local stand-in, but is NOT local."""
    is_local = False

    def __init__(self, fail_on=None):
        self.sent, self.fail_on = [], fail_on

    def prepare(self, engine_id, params, input_files=None):
        return fq.MothClient().prepare(engine_id, params, input_files)

    def run(self, engine_id, params, *, approved=False, out_dir=None, input_files=None):
        assert approved is True
        k = next(i for i, p in enumerate(PLAN["jobs"]) if p == params)
        self.sent.append((k, input_files))
        if self.fail_on == k:
            raise MothError("engine_timeout: retry the job")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"job_real-{k}.json"), "w") as f:
            json.dump({"job_id": f"real-{k}"}, f)
        done = (k + 1) * PLAN["layers_per_job"]
        with open(os.path.join(out_dir, "circuit"), "w") as f:
            f.write(fq.ideal_circuit(SPEC, None if done >= len(PLAN["layers"]) else done))
        asset = f"00000000-0000-4000-8000-00000000000{k}"
        return dict(job_id=f"real-{k}", result={"outputs": [{"slot": "circuit", "output_asset_id": asset}], "result": None}, inline=None, files={"circuit": os.path.join(out_dir, "circuit")})


class ThePlan(unittest.TestCase):
    def test_every_coupling_is_a_pair_target_and_every_target_carries_the_whole_reduced_state(self):
        covered = {tuple(t["qubits"]) for layer in PLAN["target_layers"] for t in layer}
        self.assertEqual(covered, {tuple(sorted((i, j))) for i, j, *_ in SPEC.edges} | {(q,) for q in range(SPEC.F) if all(q not in e[:2] for e in SPEC.edges)})
        psi = facet_state(SPEC)
        t = next(t for layer in PLAN["target_layers"] for t in layer if len(t["qubits"]) == 2)
        a, b = t["qubits"]
        self.assertAlmostEqual(t["expvals"]["ZZ"], pauli_expectation(psi, SPEC.F, {a: "z", b: "z"}), places=3)
        self.assertAlmostEqual(t["expvals"]["XI"], pauli_expectation(psi, SPEC.F, {a: "x"}), places=3)          # letter i acts on qubits[i]: singles are in the pair's words
        self.assertTrue(all(abs(v) <= 0.98 for layer in PLAN["target_layers"] for t in layer for v in t["expvals"].values()))

    def test_targets_that_share_a_qubit_are_in_successive_layers(self):
        for layer in PLAN["target_layers"]:
            qubits = [q for t in layer for q in t["qubits"]]
            self.assertEqual(len(qubits), len(set(qubits)))

    def test_jobs_are_small_chained_and_valid_for_the_engine(self):
        self.assertGreater(len(PLAN["jobs"]), 1)
        for k, p in enumerate(PLAN["jobs"]):
            self.assertTrue(set(p) <= ENGINES[fq.ENGINE]["params"])
            self.assertEqual("n_qubits" in p, k == 0)                        # a chained job's circuit carries its width
            self.assertEqual(p["targets"].count(None), min(fq.LAYERS_PER_JOB, len(PLAN["layers"]) - k * fq.LAYERS_PER_JOB))
            self.assertLess(sum(1 for t in p["targets"] if t), 25)
            self.assertEqual(fq.preview_job(PLAN, dict(steps={}), k).problems(), [])

    def test_the_plan_follows_the_circuit(self):
        self.assertEqual(PLAN["sha"], fq.plan_chain(SPEC)["sha"])
        self.assertNotEqual(PLAN["sha"], fq.plan_chain(ar.build_spec(LABELS, entangle=0.3)[2])["sha"])

    def test_the_hand_built_facet_block_is_the_facet_state_and_a_facet_circuit_replaces_it(self):
        self.assertTrue(equal_up_to_phase(statevector_of(facet_block(SPEC)), facet_state(SPEC), tol=1e-9))
        a = statevector_of(reference_circuit(SPEC))
        b = statevector_of(reference_circuit(SPEC, facet_circuit=facet_block(SPEC)))
        self.assertTrue(equal_up_to_phase(a, b, tol=1e-9))
        from qiskit import QuantumCircuit
        with self.assertRaises(ValueError):
            reference_circuit(SPEC, facet_circuit=QuantumCircuit(3))


class TheChain(unittest.TestCase):
    def test_a_preview_sends_nothing_and_changes_nothing(self):
        f, fake = folder("preview"), FakeQDrive()
        out = fq.send_next(SPEC, f, client=fake, log=LOG)
        self.assertTrue(out["preview"])
        self.assertEqual((fake.sent, os.path.exists(os.path.join(f, "chain.json"))), ([], False))

    def test_the_approval_must_equal_the_cost_of_the_jobs_queued(self):
        for credits, jobs in ((0, 1), (2, 1), (1, 2)):
            with self.assertRaises(NotApproved):
                fq.send_next(SPEC, folder("wrong"), client=FakeQDrive(), approve_credits=credits, jobs=jobs, log=LOG)

    def test_one_confirm_sends_one_job_and_the_next_continues_its_asset(self):
        f, fake = folder("chain"), FakeQDrive()
        out = fq.send_next(SPEC, f, client=fake, approve_credits=1, log=LOG)
        self.assertEqual((out["sent"], fake.sent[0][1]), ([0], None))
        self.assertFalse(out["state"]["complete"])
        self.assertTrue(os.path.exists(os.path.join(f, "circuit_partial.qasm")))
        self.assertFalse(os.path.exists(os.path.join(f, "circuit.qasm")))
        fq.send_next(SPEC, f, client=fake, approve_credits=1, log=LOG)
        self.assertEqual(fake.sent[1], (1, {"initial_circuit": "00000000-0000-4000-8000-000000000000"}))
        while True:
            out = fq.send_next(SPEC, f, client=fake, approve_credits=1, log=LOG)
            if out["state"]["complete"]:
                break
        self.assertEqual([k for k, _ in fake.sent], list(range(len(PLAN["jobs"]))))
        self.assertTrue(os.path.exists(os.path.join(f, "circuit.qasm")))
        self.assertFalse(os.path.exists(os.path.join(f, "circuit_partial.qasm")))
        st = fq.read_state(f)
        self.assertEqual((st["credits"], st["local"], st["from_moth"], st["jobs_done"]), (len(PLAN["jobs"]), False, True, len(PLAN["jobs"])))
        self.assertGreater(st["score"]["fidelity"], 0.999)
        self.assertTrue(fq.provenance_of(os.path.join(f, "circuit.qasm"))["from_moth"])
        again = FakeQDrive()
        fq.send_next(SPEC, f, client=again, approve_credits=1, log=LOG)                                 # finished: nothing is sent again
        self.assertEqual(again.sent, [])

    def test_a_job_that_died_is_recorded_and_the_next_confirm_resends_it(self):
        f = folder("died")
        with self.assertRaises(MothError):
            fq.send_next(SPEC, f, client=FakeQDrive(fail_on=0), approve_credits=1, log=LOG)
        rec = json.load(open(os.path.join(f, "chain.json")))["steps"]["0"]
        self.assertEqual(rec["status"], "failed")
        again = FakeQDrive()
        fq.send_next(SPEC, f, client=again, approve_credits=1, log=LOG)
        self.assertEqual([k for k, _ in again.sent], [0])

    def test_a_job_with_no_recorded_result_blocks_further_spending_until_looked_at(self):
        f = folder("orphan")
        os.makedirs(f)
        st = fq.chain_state(f, PLAN, SPEC)
        st["steps"]["0"] = dict(status="submitted", job_id="abc")
        fq.save_chain(f, st)
        fake = FakeQDrive()
        with self.assertRaises(MothError) as cm:
            fq.send_next(SPEC, f, client=fake, approve_credits=1, log=LOG)
        self.assertIn("abc", str(cm.exception))
        self.assertEqual(fake.sent, [])
        fq.send_next(SPEC, f, client=fake, approve_credits=1, retry=True, log=LOG)
        self.assertEqual([k for k, _ in fake.sent], [0])

    def test_a_chain_planned_for_other_settings_is_not_continued(self):
        f = folder("stale")
        fq.send_next(SPEC, f, client=FakeQDrive(), approve_credits=1, log=LOG)
        other = ar.build_spec(LABELS, entangle=0.3)[2]
        self.assertEqual(fq.chain_state(f, fq.plan_chain(other), other)["steps"], {})


class TheLocalRehearsal(unittest.TestCase):
    def test_fidelity_climbs_job_by_job_to_the_ideal_state_and_nothing_is_called_moth(self):
        f, fid = folder("local"), []
        for _ in PLAN["jobs"]:                                                                          # a fresh client every time, as separate processes would be
            out = fq.send_next(SPEC, f, local=True, log=LOG)
            fid.append(out["state"]["score"]["fidelity"])
        self.assertEqual(fid, sorted(fid))
        self.assertLess(fid[0], 0.9)
        self.assertGreater(fid[-1], 0.999999)
        st = fq.read_state(f)
        self.assertEqual((st["local"], st["from_moth"], st["credits"], st["complete"]), (True, False, 0, True))
        self.assertIn("not a Moth result", fq.provenance_of(os.path.join(f, "circuit.qasm"))["label"])

    def test_like_the_real_service_it_refuses_a_job_that_does_not_continue_the_previous_one(self):
        client = fq.LocalFacetClient(SPEC, PLAN)
        client.run(fq.ENGINE, PLAN["jobs"][0], out_dir=folder("contract"))
        with self.assertRaises(MothError):
            client.run(fq.ENGINE, PLAN["jobs"][1], out_dir=folder("contract"), input_files={"initial_circuit": "00000000-0000-4000-8000-000000000099"})


class TheAerRunOnThatCircuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for _ in PLAN["jobs"]:
            fq.send_next(SPEC, folder("done"), local=True, log=LOG)
        cls.chain = folder("done")
        cls.circuit = os.path.join(cls.chain, "circuit.qasm")
        cls.out = folder("aer_on_chain")
        cls.meta = ar.run(LABELS, None, cls.out, shots=100_000, seed=4, log=LOG, facet_circuit=cls.circuit)

    def test_the_run_records_which_facet_circuit_it_executed_and_how_good_it_was(self):
        f = self.meta["facets"]
        self.assertAlmostEqual(f["score"]["fidelity"], 1.0, places=6)
        self.assertFalse(f["from_moth"])
        self.assertIn("not a Moth result", f["label"])
        self.assertTrue(os.path.exists(os.path.join(self.out, "facet_circuit.qasm")))
        self.assertTrue(self.meta["score"]["consistent"])

    def test_the_hash_follows_the_facet_circuit_and_the_old_hash_is_unchanged_without_one(self):
        _, text, sha = ar.load_facet_circuit(self.circuit, SPEC)
        self.assertNotEqual(ar.plan_sha(SPEC, 100_000, 4, sha), ar.plan_sha(SPEC, 100_000, 4))
        self.assertEqual(ar.plan_sha(SPEC, 100_000, 4), ar.plan_sha(SPEC, 100_000, 4, None))
        self.assertEqual(ar.spec_fingerprint(SPEC), ar.spec_fingerprint(SPEC, None))

    def test_the_state_the_show_draws_from_is_built_on_that_circuit_and_labelled_with_its_origin(self):
        st = ar.AerReliefState(SPEC, self.out)
        self.assertIn("facet register", st.label)
        self.assertEqual((st.from_moth, st.untested), (False, True))
        self.assertEqual(len(st.draw(8, np.random.default_rng(0)).lit), SPEC.F + SPEC.n_classical)

    def test_a_circuit_from_moth_makes_the_show_say_so(self):
        moth = folder("moth_chain")
        os.makedirs(moth)
        for name in ("circuit.qasm",):
            with open(os.path.join(moth, name), "w") as f:
                f.write(open(self.circuit).read())
        with open(os.path.join(moth, "provenance.json"), "w") as f:
            json.dump(dict(kind="qdrive-facets", complete=True, local=False, from_moth=True, jobs=[dict(index=0, job_id="real-0")]), f)
        out = folder("aer_moth")
        ar.run(LABELS, None, out, shots=20_000, seed=4, log=LOG, facet_circuit=os.path.join(moth, "circuit.qasm"))
        st = ar.AerReliefState(SPEC, out)
        self.assertEqual((st.from_moth, st.untested), (True, False))
        from src.ui import caption
        prov = dict(label=st.label, quantum_backed=True, from_moth=True, untested=False, relief=True, executed="aer")
        self.assertIn("Moth's QDrive prepared", caption.provenance_line(prov))

    def test_the_run_is_checked_against_the_state_it_executed_not_the_ideal_one(self):
        from qiskit import QuantumCircuit, qasm3
        empty = os.path.join(folder("zero"), "circuit.qasm")
        os.makedirs(os.path.dirname(empty))
        with open(empty, "w") as f:
            f.write(qasm3.dumps(QuantumCircuit(SPEC.F)))                                                   # all facets |0>: far from the ideal state
        out = folder("aer_zero")
        meta = ar.run(LABELS, None, out, shots=100_000, seed=4, log=LOG, facet_circuit=empty)
        self.assertLess(meta["facets"]["score"]["fidelity"], 0.5)
        self.assertTrue(meta["score"]["consistent"])                                                       # the shots match THAT state ...
        with np.load(os.path.join(out, "shots.npz"), allow_pickle=False) as z:
            pool = tuple(z[f"z_{n}"] for n in "gwoxc")
        self.assertFalse(ar.score(SPEC, pool)["consistent"])                                               # ... and not the ideal one: the circuit really was executed
        a = ar.AerReliefState(SPEC, out)
        ideal = ReliefState(SPEC)
        rng1, rng2 = np.random.default_rng(1), np.random.default_rng(2)
        self.assertGreater(np.abs(np.mean([a.draw(32, rng1, g=0, world=0).lit for _ in range(300)], 0)
                                  - np.mean([ideal.draw(32, rng2, g=0, world=0).lit for _ in range(300)], 0)).max(), 0.1)

    def test_a_facet_circuit_that_is_not_the_one_the_run_used_is_refused(self):
        out = folder("tampered")
        ar.run(LABELS, None, out, shots=20_000, seed=4, log=LOG, facet_circuit=self.circuit)
        with open(os.path.join(out, "facet_circuit.qasm"), "a") as f:
            f.write("// edited\n")
        with self.assertRaises(ValueError) as cm:
            ar.AerReliefState(SPEC, out)
        self.assertIn("not the facet circuit", str(cm.exception))

    def test_a_circuit_of_the_wrong_width_or_that_is_not_qasm_is_refused_with_the_reason(self):
        bad = folder("bad")
        os.makedirs(bad)
        from qiskit import QuantumCircuit, qasm3
        for name, text in (("narrow.qasm", qasm3.dumps(QuantumCircuit(3))), ("junk.qasm", "this is not a circuit")):
            with open(os.path.join(bad, name), "w") as f:
                f.write(text)
            with self.assertRaises(ar.AerRunError):
                ar.run(LABELS, None, folder("bad_out"), shots=1000, log=LOG, facet_circuit=os.path.join(bad, name))


if __name__ == "__main__":
    unittest.main()
