"""A target-correlation preparation for the polarity register: the ground state of a geometry-set Ising model on the domain graph.

The closed-form preparation of the relief (H on every polarity qubit, one ZZ layer) needs no engine. This module makes the register's state something
that is *specified by correlations* instead: the ground state of

        H = - J sum_{(a,b) in domain graph} s_ab Z_a Z_b  -  h sum_d X_d            s_ab = +1 coplanar, crease_sign across a crease

(s is the geometry: depths agree within a plane, and are set by `crease_sign` across a shared edge). With h = 0 the ground states are the classical
Ising ground states, and their number measures how much the geometry leaves undecided: a balanced signed graph has 2 per connected component (the
global flip), a frustrated one has more. With h > 0 the quantum ground state is the symmetric superposition of them, softened by the field: a
superposition of competing depth interpretations, which is what the polarity qubits were for.

Two ways to prepare it, both as circuits, so a device could run them:
  * a QAOA-style ansatz fitted here (`optimize`): p layers of  exp(-i theta_e ZZ / 2) with theta_e = -2 gamma_l J s_e, then exp(-i beta_l X / 2), starting
    from |+>^D. The angles are four to six numbers, found by minimising the energy on the exact state (D <= 20) or on the matrix-product state;
  * a circuit returned by a Moth engine for the target moments (domain_moth.qdrive_polarity_payloads), attached as `spec.pol_circuit`.

Everything here is local; `report` says how close the fitted state is to the exact ground state when D is small enough to know it.
"""
import numpy as np
from scipy import optimize
from scipy.sparse.linalg import LinearOperator, eigsh

from src.quantum import complementary as comp


def signed_edges(ds):
    """[(a, b, s)] of the domain graph: +1 coplanar, ds.params['crease_sign'] across a crease."""
    cs = float(ds.params["crease_sign"])
    return [(int(a), int(b), 1.0 if kind == "coplanar" else cs) for a, b, kind, _ in ds.edges]


# ------------------------------------------------------------------ the classical and exact quantum ground states
def classical_ground(D, edges, J=1.0):
    """Minimum of E(z) = -J sum s z_a z_b over all 2^D spin configurations and how many configurations attain it. D <= 24."""
    if D > 24:
        raise ValueError("classical_ground enumerates 2^D configurations: D <= 24")
    idx = np.arange(2 ** D, dtype=np.int64)
    E = np.zeros(2 ** D)
    bits = lambda q: ((idx >> q) & 1)
    for a, b, s in edges:
        E -= J * s * (1 - 2 * bits(a)) * (1 - 2 * bits(b))
    e0 = float(E.min())
    return dict(energy=e0, degeneracy=int((np.abs(E - e0) < 1e-9).sum()), configurations=2 ** D)


def transverse_ising_ground(D, edges, J=1.0, h=0.5):
    """(E0, psi0) of H = -J sum s ZZ - h sum X by Lanczos. D <= 18."""
    if D > 18:
        raise ValueError("exact diagonalisation is for D <= 18")
    N = 2 ** D
    idx = np.arange(N, dtype=np.int64)
    diag = np.zeros(N)
    for a, b, s in edges:
        diag -= J * s * (1 - 2 * ((idx >> a) & 1)) * (1 - 2 * ((idx >> b) & 1))

    def matvec(v):
        out = diag * v
        for d in range(D):
            out = out - h * v[idx ^ (1 << d)]
        return out

    w, v = eigsh(LinearOperator((N, N), matvec=matvec, dtype=float), k=1, which="SA")
    psi = v[:, 0] * np.sign(v[np.argmax(np.abs(v[:, 0])), 0])
    return float(w[0]), psi.astype(complex)


# ------------------------------------------------------------------ the ansatz
def layers_for(edges, J, gammas, betas, D):
    """pol_layers (relief_state's format) of the QAOA-style ansatz: exp(-i theta_e ZZ / 2), theta_e = -2 gamma J s_e, then exp(-i beta X / 2)."""
    return [dict(zz=[float(-2.0 * g * J * s) for _, _, s in edges], rx=[float(b)] * D) for g, b in zip(gammas, betas)]


def _circuit(D, edges, layers):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(D)
    for d in range(D):
        qc.h(d)
    for layer in layers:
        for (a, b, _), th in zip(edges, layer["zz"]):
            qc.rzz(float(th), a, b)
        for d, b in enumerate(layer["rx"]):
            if b:
                qc.rx(float(b), d)
    return qc


def energy(D, edges, layers, J=1.0, h=0.5, backend="auto"):
    """<H> of the ansatz state, exact (statevector) for D <= 20, else from the matrix-product state's reduced pair and single states."""
    qc = _circuit(D, edges, layers)
    if backend == "exact" or (backend == "auto" and D <= 20):
        from qiskit.quantum_info import Statevector
        psi = np.asarray(Statevector(qc).data, complex)
        e = -J * sum(s * comp.pauli_expectation(psi, D, {a: "z", b: "z"}) for a, b, s in edges)
        return e - h * sum(comp.pauli_expectation(psi, D, {d: "x"}) for d in range(D))
    from src.quantum.mps import MPS
    m = MPS.from_circuit(qc, 1e-9, 48)
    e = -J * sum(s * m.pauli_expectation({a: "z", b: "z"}) for a, b, s in edges)
    return e - h * sum(m.pauli_expectation({d: "x"}) for d in range(D))


def optimize_layers(D, edges, J=1.0, h=0.5, p=2, restarts=4, backend="auto", seed=0, maxiter=120):
    """Fit gamma_l, beta_l (l < p) by minimising the energy. Returns dict(gammas, betas, energy, evaluations, layers)."""
    rng = np.random.default_rng(seed)
    evals = [0]

    def f(x):
        evals[0] += 1
        return energy(D, edges, layers_for(edges, J, x[:p], x[p:], D), J, h, backend)

    best = None
    for r in range(restarts):
        x0 = np.concatenate([np.linspace(0.15, 0.45, p) * (1 + 0.3 * rng.standard_normal(p)), np.linspace(0.6, 0.3, p) * (1 + 0.3 * rng.standard_normal(p))]) if r == 0 \
            else rng.uniform(0, 0.8, 2 * p)
        res = optimize.minimize(f, x0, method="COBYLA", options=dict(maxiter=maxiter, rhobeg=0.2))
        if best is None or res.fun < best.fun:
            best = res
    g, b = best.x[:p], best.x[p:]
    return dict(gammas=[float(v) for v in g], betas=[float(v) for v in b], energy=float(best.fun), evaluations=evals[0], layers=layers_for(edges, J, g, b, D))


def ground_state_spec(spec, ds, J=1.0, h=0.5, p=2, restarts=4, backend="auto", seed=0):
    """(new spec, report): `spec` with its polarity preparation replaced by the fitted ground-state circuit. The facets attach to it exactly as before
    (CZ from each domain's polarity qubit), so every other property of the relief is unchanged. report: the fitted energy, the exact ground energy and the
    fidelity to the exact ground state when D <= 18, the classical ground energy and its degeneracy when D <= 24."""
    import copy
    D = spec.P
    edges = signed_edges(ds)
    assert [(a, b) for a, b, _ in edges] == [tuple(e) for e in spec.pol_graph], "the spec's polarity graph must be the domain graph"
    fit = optimize_layers(D, edges, J, h, p, restarts, backend, seed)
    out = copy.copy(spec)
    out.pol_layers = fit["layers"]
    out.pol_circuit = None
    report = dict(D=D, edges=len(edges), J=J, h=h, layers=p, gammas=fit["gammas"], betas=fit["betas"], energy=fit["energy"], evaluations=fit["evaluations"])
    if D <= 24:
        report["classical"] = classical_ground(D, edges, J)
    if D <= 18:
        e0, psi0 = transverse_ising_ground(D, edges, J, h)
        from qiskit.quantum_info import Statevector
        psi = np.asarray(Statevector(_circuit(D, edges, fit["layers"])).data, complex)
        report.update(exact_energy=e0, fidelity_to_exact=float(abs(np.vdot(psi0, psi)) ** 2),
                      energy_gap_fraction=float((fit["energy"] - e0) / abs(e0)) if e0 else None)
    return out, report
