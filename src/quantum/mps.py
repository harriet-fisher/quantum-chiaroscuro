"""A small matrix-product-state toolkit on top of Qiskit Aer's MPS simulator: exact *conditional* sampling, reduced density matrices and bond dimensions.

Why it exists. A frame of the domain relief needs one polarity outcome for the whole frame and then K photons *given* that outcome (the exact engine
does this with a 2^n table). Aer's MPS sampler cannot condition on a measured subset, so Aer only evolves the state (`save_matrix_product_state`) and
the sampling is done here, exactly, from the tensors. The same tensors give the bond dimension used by the classical-baseline benchmark.

Conventions (pinned by tests against the exact statevector):
  * site k holds logical qubit k (Aer returns the tensors in qubit order); build the circuit with the qubits already in the site order you want,
    see `permute_circuit`;
  * tensor A[k] has shape (2, chi_left, chi_right) with the Vidal Gamma and the right Lambda merged, so the state is sum_s A[0][s0] A[1][s1] ... ;
  * bit 0 is spin +1 (|0>), as everywhere in the repo;
  * `rdm(sites)` is returned with sites[0] the MOST significant bit (the convention of complementary.reduced_pair).
"""
import time

import numpy as np

PAULI = {"x": np.array([[0, 1], [1, 0]], complex), "y": np.array([[0, -1j], [1j, 0]], complex), "z": np.diag([1.0, -1.0]).astype(complex),
         "i": np.eye(2, dtype=complex)}


def permute_circuit(qc, site_of):
    """The same circuit with logical qubit q placed on wire site_of[q] (a permutation), so Aer's MPS has the sites in that order."""
    from qiskit import QuantumCircuit
    n = qc.num_qubits
    out = QuantumCircuit(n, name=qc.name)
    for inst in qc.data:
        out.append(inst.operation, [out.qubits[site_of[qc.find_bit(q).index]] for q in inst.qubits])
    return out


class MPS:
    def __init__(self, tensors, build_seconds=None, lam=None):
        self.A = [np.asarray(t, complex) for t in tensors]
        self.n = len(self.A)
        self.build_seconds = build_seconds
        self.lam = lam                                              # Schmidt coefficients at every bond when the MPS came from Aer

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_circuit(cls, qc, truncation=1e-12, max_bond=None):
        """Evolve `qc` (no measurements) on Aer's MPS simulator and keep the tensors. truncation: discarded Schmidt weight per bond."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        t0 = time.time()
        c = qc.copy()
        c.save_matrix_product_state()
        opts = dict(method="matrix_product_state", matrix_product_state_truncation_threshold=truncation)
        if max_bond:
            opts["matrix_product_state_max_bond_dimension"] = int(max_bond)
        sim = AerSimulator(**opts)
        # no transpile against the simulator's target: it caps the circuit at 63 qubits, which the MPS method itself does not; the gates used here
        # (h, rx, ry, rz, z, cz, rzz) are all native to Aer
        try:
            res = sim.run(c, shots=1).result()
        except Exception:                                           # a gate Aer does not know natively: let Qiskit decompose it (small circuits only)
            res = sim.run(transpile(c, sim, optimization_level=0), shots=1).result()
        gam, lam = res.data(0)["matrix_product_state"]
        n = len(gam)
        tensors = []
        for k in range(n):
            g = np.stack([np.asarray(gam[k][0]), np.asarray(gam[k][1])])          # (2, chi_l, chi_r)
            if k < n - 1:
                g = g * np.asarray(lam[k])[None, None, :]
            tensors.append(g)
        return cls(tensors, build_seconds=time.time() - t0, lam=[np.asarray(l, float) for l in lam])

    @classmethod
    def from_statevector(cls, psi, n):
        """Exact MPS of a flat little-endian vector (site k = qubit k). For tests and small cases."""
        t = np.asarray(psi, complex).reshape((2,) * n)               # axis a is qubit n-1-a
        t = np.transpose(t, list(range(n))[::-1])                    # axis k is now qubit k
        tensors, rest = [], t.reshape(1, -1)
        chi = 1
        for k in range(n - 1):
            m = rest.reshape(chi * 2, -1)
            u, s, vh = np.linalg.svd(m, full_matrices=False)
            keep = int((s > 1e-14 * s[0]).sum())
            u, s, vh = u[:, :keep], s[:keep], vh[:keep]
            tensors.append(np.transpose(u.reshape(chi, 2, keep), (1, 0, 2)))
            rest = s[:, None] * vh
            chi = keep
        tensors.append(np.transpose(rest.reshape(chi, 2, 1), (1, 0, 2)))
        return cls(tensors)

    # ------------------------------------------------------------------ cheap operations on the tensors
    def apply_1q(self, site, U):
        """A new MPS with the single-qubit unitary U applied at `site`: only that tensor changes (the other tensors are shared), the bonds do not."""
        A = list(self.A)
        A[site] = np.tensordot(np.asarray(U, complex), A[site], axes=([1], [0]))
        return MPS(A)

    def apply_many(self, unitaries):
        """{site: U} applied at once."""
        A = list(self.A)
        for site, U in unitaries.items():
            A[site] = np.tensordot(np.asarray(U, complex), A[site], axes=([1], [0]))
        return MPS(A)

    @staticmethod
    def combine(states, coeffs):
        """The (unnormalised) superposition sum_k coeffs[k] |states[k]> as an MPS: the tensors are stacked block-diagonally, so the bond dimension is the SUM
        of the bond dimensions (no truncation, exact). A lamp qubit read in a rotated basis leaves the rest of the wall in exactly such a superposition of
        its conditioned states, one per lamp value."""
        n = states[0].n
        if any(s.n != n for s in states):
            raise ValueError("states must have the same number of sites")
        out = []
        for k in range(n):
            ts = [s.A[k] for s in states]
            if n == 1:
                out.append(sum(c * t for c, t in zip(coeffs, ts)))
                continue
            L = sum(t.shape[1] for t in ts)
            R = sum(t.shape[2] for t in ts)
            T = np.zeros((2, L if k else 1, R if k < n - 1 else 1), complex)
            l0 = r0 = 0
            for c, t in zip(coeffs, ts):
                l1, r1 = l0 + t.shape[1], r0 + t.shape[2]
                if k == 0:
                    T[:, :, r0:r1] += c * t
                elif k == n - 1:
                    T[:, l0:l1, :] += t
                else:
                    T[:, l0:l1, r0:r1] = t
                l0, r0 = (l1, r1) if k else (0, r1)
                if k == n - 1:
                    l0 = l1
            out.append(T)
        return MPS(out)

    # ------------------------------------------------------------------ properties
    def bond_dims(self):
        return [int(self.A[k].shape[2]) for k in range(self.n - 1)]

    def max_bond(self):
        b = self.bond_dims()
        return max(b) if b else 1

    def entropies(self):
        """Entanglement entropy (bits) across every bond, from the Schmidt coefficients Aer returned."""
        out = []
        for l in self.lam or []:
            p = np.abs(l) ** 2
            p = p[p > 1e-16] / p.sum()
            out.append(float(-(p * np.log2(p)).sum()))
        return out

    def norm2(self):
        E = np.ones((1, 1), complex)
        for A in self.A:
            E = sum(A[s].conj().T @ E @ A[s] for s in (0, 1))
        return float(E[0, 0].real)

    def amplitude(self, bits):
        v = np.ones((1, 1), complex)
        for k, b in enumerate(bits):
            v = v @ self.A[k][int(b)]
        return complex(v[0, 0])

    def to_statevector(self):
        """Flat little-endian vector; only for small n."""
        v = np.ones((1, 1), complex)                                 # (basis index so far, chi)
        for k in range(self.n):
            A = self.A[k]                                           # (2, l, r)
            v = np.einsum("ia,sab->isb", v, A).reshape(-1, A.shape[2])
        t = v.reshape((2,) * self.n)                                # axis k is site k, site 0 most significant
        return np.transpose(t, list(range(self.n))[::-1]).reshape(-1)

    # ------------------------------------------------------------------ reduced states and expectations
    def rdm(self, sites, between=None):
        """Reduced density matrix of `sites` (all other sites traced), shape (2^m, 2^m), sites[0] most significant.

        between={site: 2x2 matrix M}: instead of the plain trace over a traced site, contract it as sum_{s,t} M[t, s] ket(s) bra(t)*, that is
        Tr_rest[ A |psi><psi| B^dagger ] with M = B^dagger A on that site. With M = identity this is the ordinary reduced state; with a diagonal M it
        gives the off-diagonal blocks that a lamp qubit coupled by exp(-i theta Z_L Z_B / 2) would have, without putting the lamp in the MPS."""
        return self._rdm(self, sites, between)

    def cross_rdm(self, other, sites):
        """Tr_rest[ |self><other| ] on `sites`, (2^m, 2^m), unnormalised: the ket tensors come from self, the bra tensors from other. The off-diagonal block
        of a reduced state between two branches that differ on the traced sites too (a lamp that picks the facets' light axis coherently)."""
        return self._rdm(other, sites, None)

    def _rdm(self, bra, sites, between):
        sites = list(sites)
        between = between or {}
        order = sorted(range(len(sites)), key=lambda j: sites[j])
        sorted_sites = [sites[j] for j in order]
        E = np.ones((1, 1, 1, 1), complex)                          # (chi_ket, chi_bra, d, d')
        for k in range(self.n):
            A, B = self.A[k], bra.A[k]
            T = np.tensordot(A, E, axes=([1], [0]))                  # (s, b, e, i, j): the ket tensor absorbed; two matmuls per site, not a 3-way einsum
            if k in sorted_sites:
                Ec = np.tensordot(T, B.conj(), axes=([2], [1]))      # (s, b, i, j, t, c): s on the ket, t on the bra stay open
                d = E.shape[2] * 2
                E = np.transpose(Ec, (1, 5, 2, 0, 3, 4)).reshape(A.shape[2], B.shape[2], d, d)
            else:
                if k in between:
                    T = np.tensordot(np.asarray(between[k], complex), T, axes=([1], [0]))        # M[t, s] on the ket index: (t, b, e, i, j)
                Ec = np.tensordot(T, B.conj(), axes=([0, 2], [0, 1]))                          # (b, i, j, c), summed over s (or t) and e
                E = np.transpose(Ec, (0, 3, 1, 2))
        rho = E[0, 0]                                                # basis index: sorted sites, first = most significant
        m = len(sites)
        rho = rho.reshape((2,) * (2 * m))
        perm = [order.index(j) for j in range(m)]                   # axis for sites[j] in the sorted layout
        rho = np.transpose(rho, perm + [m + p for p in perm]).reshape(2 ** m, 2 ** m)
        return rho if (between or bra is not self) else rho / np.trace(rho).real

    def pauli_expectation(self, ops):
        """<psi|prod sigma_{ops[k]}(site k)|psi> / <psi|psi> for ops = {site: 'x'|'y'|'z'}."""
        sites = sorted(ops)
        rho = self.rdm(sites)
        M = np.array([[1.0 + 0j]])
        for k in sites:
            M = np.kron(M, PAULI[ops[k]])
        return float(np.trace(rho @ M).real)

    # ------------------------------------------------------------------ sampling
    def sampler(self, measured, fixed=None):
        """Exact sampler of the `measured` sites given `fixed` = {site: bit}; every other site is traced out."""
        return _Sampler(self, list(measured), dict(fixed or {}))


class _Sampler:
    def __init__(self, mps, measured, fixed):
        self.m, self.measured, self.fixed = mps, set(measured), fixed
        self.sites = sorted(measured)
        n = mps.n
        R = [None] * (n + 1)
        R[n] = np.ones((1, 1), complex)
        for k in range(n - 1, -1, -1):
            A = mps.A[k]
            allowed = (fixed[k],) if k in fixed else (0, 1)
            R[k] = sum(A[s] @ R[k + 1] @ A[s].conj().T for s in allowed)
        self.R = R
        self.p_fixed = float(np.trace(R[0]).real)                   # probability of the fixed outcomes (relative to the norm)

    def draw(self, rng, shots=1):
        """(shots, len(sites)) array of 0/1 outcomes, columns in sorted site order."""
        mps, R = self.m, self.R
        out = np.zeros((shots, len(self.sites)), np.int8)
        col = {s: j for j, s in enumerate(self.sites)}
        for r in range(shots):
            L = np.ones((1, 1), complex)
            for k in range(mps.n):
                A = mps.A[k]
                if k in self.fixed:
                    s = self.fixed[k]
                    L = A[s].conj().T @ L @ A[s]
                elif k in self.measured:
                    Ls = [A[s].conj().T @ L @ A[s] for s in (0, 1)]
                    w = np.array([np.trace(Ls[s] @ R[k + 1]).real for s in (0, 1)])
                    w = np.clip(w, 0, None)
                    s = int(rng.random() * w.sum() > w[0])
                    out[r, col[k]] = s
                    L = Ls[s]
                else:
                    L = sum(A[s].conj().T @ L @ A[s] for s in (0, 1))
                t = np.trace(L @ R[k + 1]).real
                if t > 0:
                    L = L / t
        return out
