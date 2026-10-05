"""Offline tests (no network, no credits). Run: python qdrive_lab/test_offline.py   (also pytest-compatible)

1. The scorer reads Pauli expectations correctly on circuits whose answer is known (so a bad reading can never again be
   confused with a bad engine).
2. The project client's own pre-checks (PreparedJob.problems/warnings) against each payload variant we plan to send, so we know
   which probes the client would block before the server ever sees them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                  # noqa: E402
from qiskit import QuantumCircuit, qasm3            # noqa: E402
from qiskit.quantum_info import Statevector         # noqa: E402

import lab                                          # noqa: E402


def _sv(qc):
    return Statevector(qc), qc.num_qubits


def test_x_on_qubit0():
    qc = QuantumCircuit(3); qc.x(0)
    sv, n = _sv(qc)
    assert lab.pauli_expval(sv, n, "Z", [0]) == -1.0
    assert lab.pauli_expval(sv, n, "Z", [1]) == 1.0
    assert lab.pauli_expval(sv, n, "ZZ", [0, 1]) == -1.0


def test_bell_known_answers():
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)
    sv, n = _sv(qc)
    for word, want in (("ZZ", 1.0), ("XX", 1.0), ("YY", -1.0)):
        assert abs(lab.pauli_expval(sv, n, word, [0, 1]) - want) < 1e-9, word
    assert abs(lab.pauli_expval(sv, n, "Z", [0])) < 1e-9


def test_ghz_known_answers():
    qc = QuantumCircuit(4); qc.h(0)
    for i in range(3): qc.cx(i, i + 1)
    sv, n = _sv(qc)
    assert abs(lab.pauli_expval(sv, n, "XXXX", [0, 1, 2, 3]) - 1.0) < 1e-9
    assert abs(lab.pauli_expval(sv, n, "ZZ", [0, 3]) - 1.0) < 1e-9
    assert abs(lab.pauli_expval(sv, n, "Z", [2])) < 1e-9


def test_word_letter_order_matters():
    """letter i of the word acts on qubits[i]: |+> on q0 and |0> on q1 gives X on [0,1] word 'XZ' = 1 but 'ZX' = 0."""
    qc = QuantumCircuit(2); qc.h(0)
    sv, n = _sv(qc)
    assert abs(lab.pauli_expval(sv, n, "XZ", [0, 1]) - 1.0) < 1e-9
    assert abs(lab.pauli_expval(sv, n, "ZX", [0, 1])) < 1e-9


def test_qasm3_roundtrip_keeps_qubit_order():
    qc = QuantumCircuit(3); qc.x(2)
    back = qasm3.loads(qasm3.dumps(qc))
    sv = Statevector(back)
    assert lab.pauli_expval(sv, 3, "Z", [2]) == -1.0 and lab.pauli_expval(sv, 3, "Z", [0]) == 1.0


def test_project_loader_agrees_with_lab_scorer():
    from src.quantum import sampler_local as sl
    qc = QuantumCircuit(3); qc.h(0); qc.cx(0, 1); qc.x(2)
    text = qasm3.dumps(qc)
    src = sl.source_from_qasm(text, dict(n_qubits=3), label="t")
    single, zz = src.z_moments([(0, 1), (1, 2)])
    sv, n = _sv(qc)
    assert np.allclose(single, [lab.pauli_expval(sv, n, "Z", [q]) for q in range(3)], atol=1e-5)
    assert np.allclose(zz, [lab.pauli_expval(sv, n, "ZZ", p) for p in [(0, 1), (1, 2)]], atol=1e-5)


# ------------------------------------------------------------------ client pre-checks
def _problems(params):
    return lab._client.prepare(lab.ENGINE, params).problems()


def test_client_blocks_what_it_should_and_not_more():
    base = dict(n_qubits=2, machine="aer", update_method="spectral", targets=[dict(qubits=[0], expvals={"Z": 0.0}), None])
    assert _problems(base) == []
    assert any("unknown params" in p for p in _problems(dict(base, bogus=1)))
    big = dict(base, targets=[dict(qubits=[0], expvals={"Z": 0.0})] * 150)
    assert any("QDrive targets" in p for p in _problems(big))


def test_client_does_not_know_input_files_or_new_params():
    """Documented-but-unsupported by the client: initial_circuit chaining (input_files) and `ansatz` shape are not validated, so
    the lab sends those RAW to see the server's answer."""
    job = lab._client.prepare(lab.ENGINE, dict(machine="aer", targets=[None]))
    assert job.input_files is None


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:                              # noqa: BLE001
                fails += 1; print("FAIL", name, "->", type(e).__name__, e)
    print("\n%d failure(s)" % fails)
    sys.exit(1 if fails else 0)
