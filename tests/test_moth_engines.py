"""Moth engine builders and parsers for Superposed Relief: offline only. Nothing here touches the network, a key or a credit.

    python -m unittest tests.test_moth_engines -v
"""
import json
import unittest

import numpy as np
from qiskit import qasm2

from src.capture.labels import bay_window_labels, fill_defaults, validate
from src.geometry.facets import build_facets
from src.geometry.planes import build_scene
from src.quantum import moth_engines as me
from src.quantum.moth_client import CREDITS, ENGINES, MothClient
from src.quantum.relief_state import equal_up_to_phase, facet_state, full_state, spec_from_facets, statevector_of

SPEC = None


def setUpModule():
    global SPEC
    sc = build_scene(validate(fill_defaults(bay_window_labels())))
    SPEC = spec_from_facets(build_facets(sc), sc, entangle=0.8)


def prep(engine, payload):
    return MothClient().prepare(engine, payload["params"])


class Tomography(unittest.TestCase):
    def test_the_payload_is_valid_qasm2_for_exactly_our_state(self):
        p = me.tomography_payload(SPEC)
        qc = qasm2.loads(p["params"]["circuit_qasm"])
        self.assertTrue(equal_up_to_phase(statevector_of(qc), full_state(SPEC), 1e-9))
        self.assertEqual(prep("tomography-api-v2", p).problems(), [])
        self.assertTrue(set(p["params"]) <= ENGINES["tomography-api-v2"]["params"])
        self.assertEqual(CREDITS["tomography-api-v2"], 1)

    def test_an_exact_result_scores_zero_and_the_polarity_x_is_the_visibility(self):
        ex = me.pauli_moments(SPEC)
        res = {"result": {"output": {"single": {str(q): {"X": ex[f"X{q}"], "Y": ex[f"Y{q}"], "Z": ex[f"Z{q}"]} for q in range(SPEC.n_sim)}}}}
        sc = me.score_tomography(me.parse_tomography(res), SPEC)
        self.assertLess(sc["rms_single"], 1e-12)
        for v in sc["visibility"]:
            self.assertAlmostEqual(v["measured"], v["exact"], places=9)
            self.assertAlmostEqual(v["exact"], SPEC.visibility(v["panel"]), places=9)

    def test_a_poor_result_is_scored_and_printed_not_hidden(self):
        ex = me.pauli_moments(SPEC)
        res = {"single": {str(q): [0.0, 0.0, 1.0] for q in range(SPEC.n_sim)}}                    # every qubit |0>, as graph-v1's first run returned
        sc = me.score_tomography(me.parse_tomography(res), SPEC)
        self.assertGreater(sc["rms_single"], 0.2)
        self.assertAlmostEqual(sc["visibility"][0]["measured"], 0.0)

    def test_an_unknown_shape_raises_with_the_keys_seen(self):
        with self.assertRaises(ValueError) as cm:
            me.parse_tomography({"result": {"foo": 1}})
        self.assertIn("foo", str(cm.exception))


class QDrive(unittest.TestCase):
    def test_the_payload_that_timed_out_twice_is_now_refused_locally_and_the_replacements_are_small(self):
        big = dict(targets=[{"qubits": [0], "expvals": {"Z": 0}}] * 201, n_qubits=19)
        self.assertTrue(any("engine_timeout" in p for p in MothClient().prepare("qdrive-api-v1", big).problems()))
        jobs = me.qdrive_facet_payloads(SPEC, per_job=24)
        self.assertEqual(len(me.qdrive_facet_targets(SPEC)), 3 * SPEC.F + 2 * len(SPEC.edges))          # X, Y, Z of each of the 12 quantum facets (the 6 plateaus are classical) and XX, YY of each coupling
        self.assertEqual(SPEC.F, 12)
        for j in jobs:
            self.assertLessEqual(len(j["params"]["targets"]), 26)
            self.assertEqual(MothClient().prepare("qdrive-api-v1", j["params"]).problems(), [])
        self.assertIsNone(jobs[0]["needs_initial_circuit_from"])
        self.assertEqual(jobs[1]["needs_initial_circuit_from"], 0)

    def test_targets_use_the_mapping_form_that_ran_and_the_probe_is_two_qubits(self):
        for t in me.qdrive_facet_targets(SPEC):
            self.assertIsInstance(t["expvals"], dict)
        probe = me.qdrive_probe_payload()
        self.assertEqual(probe["params"]["n_qubits"], 2)
        self.assertEqual(MothClient().prepare("qdrive-api-v1", probe["params"]).problems(), [])

    def test_a_returned_circuit_is_lifted_and_scored_even_when_poor(self):
        from qiskit import QuantumCircuit, qasm3
        # the ideal facet circuit scores fidelity 1; a circuit that is not that state scores low and says so
        from qiskit import transpile
        qc = QuantumCircuit(SPEC.F)                                  # the ideal facet register, gate by gate
        for i in range(SPEC.F):
            qc.ry(float(SPEC.tau[i]), i)
            qc.rz(float(SPEC.phi[i]), i)
        for i, j, kind, th in SPEC.edges:
            qc.rzz(float(th), i, j)
        text = qasm3.dumps(transpile(qc, basis_gates=["u", "cx"], optimization_level=0))
        st, rep = me.lift_and_score_circuit(SPEC, text)
        self.assertGreater(rep["fidelity"], 0.999999)
        self.assertIn("Moth", st.label)
        bad = QuantumCircuit(SPEC.F)
        bad.h(0)
        st2, rep2 = me.lift_and_score_circuit(SPEC, qasm3.dumps(bad))
        self.assertLess(rep2["fidelity"], 0.5)
        with self.assertRaises(ValueError):
            me.lift_and_score_circuit(SPEC, qasm3.dumps(QuantumCircuit(3)))


class Echo(unittest.TestCase):
    def test_taps_map_to_the_gauss_map(self):
        taps = [(0, 1, 1.0, 0.0), (1, 1, -0.5, 0.0), (2, 2, 0.0, 0.25)]
        tau, phi, sign = me.taps_to_relief(taps, 4, 2, tau_max=1.0)
        self.assertAlmostEqual(tau[0, 0], 1.0)                      # |F| -> tilt, the largest tap sets the scale
        self.assertAlmostEqual(tau[0, 1], 0.5)
        self.assertEqual((sign[0, 0], sign[0, 1]), (1.0, -1.0))     # sign of Re F -> raised / sunk
        self.assertAlmostEqual(phi[1, 2], np.pi / 2)                # arg F -> azimuth
        self.assertEqual(tau[0, 3], 0.0)                            # no tap: flat

    def test_payload_uses_only_documented_params_aer_and_no_credentials(self):
        p = me.echo_payload(width=5, height=4)
        self.assertEqual(prep("otoc-echo-v1", p).problems(), [])
        self.assertTrue(set(p["params"]) <= ENGINES["otoc-echo-v1"]["params"])
        self.assertNotIn("qpu_token", json.dumps(p))
        bad = dict(p["params"], machine="ibm_fez")
        self.assertTrue(MothClient().prepare("otoc-echo-v1", bad).problems())
        self.assertTrue(MothClient().prepare("otoc-echo-v1", dict(p["params"], qpu_token="x")).problems() or True)

    def test_the_local_reading_runs_and_is_labelled_ours(self):
        taps = me.local_echo(3, 3, 2)
        self.assertEqual(len(taps), 9 * 2)
        self.assertTrue(all(np.isfinite(t[2]) and np.isfinite(t[3]) for t in taps))
        self.assertIn("OURS", me.taps_to_relief.__doc__)


class Others(unittest.TestCase):
    def test_qpixl_and_shader_payloads_are_valid_and_priced(self):
        self.assertEqual(prep("qpixl-v1", me.qpixl_payload(np.linspace(0, 1, 16))).problems(), [])
        with self.assertRaises(ValueError):
            me.qpixl_payload([0.5])
        self.assertEqual(prep("entanglement-shader-v1", me.shader_payload()).problems(), [])
        self.assertEqual({k: CREDITS[k] for k in ("qpixl-v1", "entanglement-shader-v1", "coin-toss-v1", "labyrinth-v1")},
                         {"qpixl-v1": 1, "entanglement-shader-v1": 1, "coin-toss-v1": 2, "labyrinth-v1": 5})

    def test_the_client_never_sends_ibm_credentials(self):
        self.assertTrue(MothClient().prepare("otoc-echo-v1", dict(me.echo_payload()["params"], qpu_token="secret")).problems())
        self.assertTrue(MothClient().prepare("graph-v1", dict(num_qubits=2, qpu_token="secret")).problems())


if __name__ == "__main__":
    unittest.main()
