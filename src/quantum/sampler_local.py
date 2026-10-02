"""Local sampling of a pure quantum state: the quantum-backed replacement for sampler_mock.Sampler (handoff §6.2, §9.2, §9.4).

The state comes from a QASM3 circuit that QDrive returned (simulated here with Aer), or from a statevector. Every outcome
the renderer uses, i.e. the lamps (lighting world), the polarity bits and the patch bits, is one measurement of ONE
register, so correlations between them are whatever the state has, not something this code adds.

Conventions (pinned by tests/test_frames.py, because a real circuit would silently scramble the lighting if they slipped):
  * qubit q is bit q of the basis-state index (qiskit little-endian); layout comes from graph.allocate_qubits.allocate;
  * spin = +1 for bit 0, -1 for bit 1, i.e. Z = +1 is |0>: patch lit, lamp L1 light-from-right, L2 frontal, polarity raised;
  * measure_basis={qubit: "x"|"y"|"z"|(nx, ny, nz)} rotates before measuring. Measuring the polarity qubits in X while the lighting is
    read in Z is the complementary-observables variant (spec §3.6); nothing here claims it is quantum-specific by itself.
"""
import itertools
from dataclasses import dataclass

import numpy as np

H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
SDG_H = H @ np.array([[1, 0], [0, -1j]], complex)           # circuit order: sdg then h; rotates Y into Z
BASIS_GATE = {"z": None, "x": H, "y": SDG_H}


def bloch_unitary(n):
    """Single-qubit unitary U such that measuring Z after U measures n . sigma (n a unit Bloch vector): U = Ry(-theta) Rz(-phi).
    Outcome 0 (spin +1) means the qubit was along +n. Used for the CHSH settings, which lie between the X and Z axes."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    theta, phi = np.arccos(np.clip(n[2], -1, 1)), np.arctan2(n[1], n[0])
    ry = np.array([[np.cos(-theta / 2), -np.sin(-theta / 2)], [np.sin(-theta / 2), np.cos(-theta / 2)]], complex)
    rz = np.diag([np.exp(1j * phi / 2), np.exp(-1j * phi / 2)])
    return ry @ rz


def basis_gate(spec):
    """None for Z, else the 2x2 unitary. spec: "x" | "y" | "z" or a Bloch vector (3 numbers)."""
    if isinstance(spec, str):
        return BASIS_GATE[spec.lower()]
    return bloch_unitary(spec)


def apply_1q(psi, U, q, n):
    t = psi.reshape((2,) * n)
    axis = n - 1 - q                                           # qubit 0 is the last (fastest) axis
    return np.moveaxis(np.tensordot(U, t, axes=([1], [axis])), 0, axis).reshape(-1)


def spins(idx, qubits):
    """+1/-1 outcomes (Z = +1 for bit 0) of the given qubits for an array of basis-state indices."""
    return 1 - 2 * ((np.asarray(idx, dtype=np.int64)[:, None] >> np.asarray(qubits, dtype=np.int64)[None, :]) & 1)


@dataclass
class Pool:
    """M measurement shots of the whole register, split by role."""
    lamps: np.ndarray     # (M, 2) +/-1
    shots: np.ndarray     # (M, n_patches) +/-1
    pol: np.ndarray       # (M, n_panels) +/-1


class StateSource:
    def __init__(self, psi, layout, measure_basis=None, label="statevector"):
        self.n = int(np.log2(len(psi)))
        if 2 ** self.n != len(psi) or self.n != layout["n_qubits"]:
            raise ValueError(f"state has {len(psi)} amplitudes but the layout needs {layout['n_qubits']} qubits")
        self.layout, self.label = layout, label
        psi = np.asarray(psi, complex)
        psi = psi / np.linalg.norm(psi)
        self.basis = {q: (b.lower() if isinstance(b, str) else tuple(float(v) for v in b)) for q, b in (measure_basis or {}).items()}
        for q, b in self.basis.items():
            U = basis_gate(b)
            if U is not None:
                psi = apply_1q(psi, U, q, self.n)
        self.probs = np.abs(psi) ** 2
        self.probs /= self.probs.sum()

    def pool(self, M, rng):
        idx = rng.choice(len(self.probs), size=M, p=self.probs)
        L = self.layout
        return Pool(spins(idx, L["lamps"]), spins(idx, L["patches"]), spins(idx, L["polarity"]))

    def z_moments(self, pairs):
        """Exact <Z_q> for every qubit and <Z_a Z_b> for each (a, b) in pairs, in the measured basis (no sampling noise)."""
        sp = spins(np.arange(len(self.probs)), list(range(self.n))).astype(np.int8)
        single = self.probs @ sp
        return single, np.array([self.probs @ (sp[:, a] * sp[:, b]) for a, b in pairs])


# ------------------------------------------------------------------ loaders
def statevector_from_qasm(text):
    """Simulate a QASM3 (or QASM2) circuit to its final statevector with Aer. Final measurements are dropped."""
    from qiskit import qasm2, qasm3, transpile
    qc = (qasm2 if text.lstrip().startswith("OPENQASM 2") else qasm3).loads(text)
    qc = qc.remove_final_measurements(inplace=False)
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector")
        qc.save_statevector()
        res = sim.run(transpile(qc, sim)).result()
        return np.asarray(res.get_statevector(), complex), qc.num_qubits
    except ImportError:
        from qiskit.quantum_info import Statevector
        return np.asarray(Statevector(qc).data, complex), qc.num_qubits


def source_from_qasm(text, layout, measure_basis=None, label="QASM circuit"):
    psi, n = statevector_from_qasm(text)
    if n != layout["n_qubits"]:
        raise ValueError(f"circuit has {n} qubits, layout needs {layout['n_qubits']}")
    return StateSource(psi, layout, measure_basis, label)


def oracle_state(sampler, layout):
    """Pure state sum_s sqrt(P(s))|s> whose Z-basis statistics ARE the mock sampler's distribution, exactly (lamps uniform,
    patches Boltzmann given the lamps, polarity from its exact chain law). A known-answer reference for testing the frame
    pipeline and its conventions, and an upper bound on what an engine could reach on Z-type targets. It is built classically
    here: it is NOT something Moth returned."""
    np_, P, n = sampler.n, sampler.n_panels, layout["n_qubits"]
    if layout["patches"] != list(range(np_)):
        raise ValueError("oracle assumes patches occupy qubits 0..n-1")
    S = spins(np.arange(2 ** np_), list(range(np_))).astype(float)          # (2^np, np), +1 for bit 0
    pair = 0.5 * np.einsum("ij,ij->i", S @ sampler.J, S)
    wpol = sampler.polarity_weights(); wpol = wpol / wpol.sum()
    pol_states = sampler.polarity_states()                                  # tuples of +/-1, +1 raised
    full = np.zeros(2 ** n)
    base = np.arange(2 ** np_)
    for L1, L2 in itertools.product([-1, 1], repeat=2):
        h = sampler.hub1 * L1 + sampler.hub2 * L2
        w = np.exp(sampler.beta * (S @ h + pair)); w /= w.sum()
        lamp_bits = (((1 - L1) // 2) << layout["lamps"][0]) | (((1 - L2) // 2) << layout["lamps"][1])
        for pk, ps in enumerate(pol_states):
            pol_bits = sum(((1 - v) // 2) << q for v, q in zip(ps, layout["polarity"]))
            full[base | lamp_bits | pol_bits] = 0.25 * w * wpol[pk]
    return np.sqrt(full)
