"""A paper-based MOCK of a Motte-model engine, used to test hypotheses about QDrive against lab data and to dry-run plans (free, local).

Hypothesis H_stale: a layer (the targets before one `null`) derives EVERY gate from the state at the START of the layer
("tomography once per layer"), then applies the gates one after another. Alternative H_fresh: each gate is derived from the
current state. Each gate is the smallest rotation U = exp(-i sum_a th_a P_a) on the target's qubit group that moves that group's
reduced state to match the target Pauli words (least squares from th=0). fraction f<1 applies only part of the step.
This is NOT QDrive's algorithm; it only encodes what the paper says in its simplest form.
"""
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0, -1.0]).astype(complex)
P1 = dict(I=I2.astype(complex), X=X, Y=Y, Z=Z)


def pauli(word):
    m = np.eye(1, dtype=complex)
    for ch in word:
        m = np.kron(m, P1[ch])
    return m


def rdm(psi, n, qubits):
    """Reduced density matrix of `qubits` (local index: qubits[0] = most significant), psi little-endian (qubit q = bit q)."""
    t = psi.reshape((2,) * n)                                     # axis a <-> qubit n-1-a
    axes = [n - 1 - q for q in qubits]
    rest = [a for a in range(n) if a not in axes]
    m = np.transpose(t, axes + rest).reshape(2 ** len(qubits), -1)
    return m @ m.conj().T


def apply_unitary(psi, n, qubits, U):
    t = psi.reshape((2,) * n)
    axes = [n - 1 - q for q in qubits]
    k = len(qubits)
    rest = [a for a in range(n) if a not in axes]
    m = np.transpose(t, axes + rest).reshape(2 ** k, -1)
    m = U @ m
    inv = np.argsort(axes + rest)
    return np.transpose(m.reshape((2,) * n), inv).reshape(-1)


_GEN = {}
def generators(k):
    if k not in _GEN:
        _GEN[k] = [pauli("".join(w)) for w in itertools.product("IXYZ", repeat=k) if set(w) != {"I"}]
    return _GEN[k]


def ideal_gate(rho, words, k, f=1.0, restarts=4, seed=0, lam=0.02):
    """Smallest-rotation unitary on k qubits bringing <P_w> of rho to the targets; U**f by scaling the angles.
    theta=0 is a stationary point for Z-type words on computational states (all gradients vanish), so the search starts from small
    seeded random angles and keeps the best of a few restarts (lowest misfit, then smallest rotation)."""
    G = generators(k); Pw = [(pauli(w), v) for w, v in words.items()]
    def resid(th):
        H = sum(t * g for t, g in zip(th, G))
        U = expm(-1j * H); r = U @ rho @ U.conj().T
        return np.concatenate([[np.real(np.trace(P @ r)) - v for P, v in Pw], np.sqrt(lam) * th])   # misfit + small rotation penalty
    rng = np.random.default_rng(seed); best = None
    for i in range(restarts):
        th0 = np.zeros(len(G)) if i == 0 else 0.4 * rng.standard_normal(len(G))
        sol = least_squares(resid, th0, method="lm", xtol=1e-9, ftol=1e-9, max_nfev=300)
        key = (round(float(np.sum(sol.fun ** 2)), 6), float(np.linalg.norm(sol.x)))
        if best is None or key < best[0]:
            best = (key, sol.x)
    H = sum(t * g for t, g in zip(best[1] * f, G))
    return expm(-1j * H)


class MotteMock:
    def __init__(self, n, stale=True, f=1.0, psi=None):
        self.n = n; self.stale = stale; self.f = f
        self.psi = psi if psi is not None else np.eye(1, 2 ** n, 0, dtype=complex).ravel()

    def run(self, targets):
        layer = []
        for t in list(targets) + [None]:
            if t is not None:
                layer.append(t); continue
            snap = self.psi.copy()
            for tg in layer:
                q = tg["qubits"]; src = snap if self.stale else self.psi
                U = ideal_gate(rdm(src, self.n, q), tg["expvals"], len(q), self.f)
                self.psi = apply_unitary(self.psi, self.n, q, U)
            layer = []
        return self

    def moment(self, word, qubits):
        """<Pauli word> with letter i on qubits[i], read from the reduced state of just those qubits (cheap for any n)."""
        return float(np.real(np.trace(pauli(word) @ rdm(self.psi, self.n, list(qubits)))))
