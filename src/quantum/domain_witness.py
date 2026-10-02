"""Witnesses for the domain relief: is the entanglement really between *edges* (domains across the geometric graph), and between the light and the depth?

Everything is computed from the simulated state (exact statevector, or the MPS the Aer simulator returned) and says so; a state with no
non-classical signature is reported as such. The controls are the same as for the per-panel relief: control B (every polarity qubit dephased in Z,
so the depth is a classical mixture, which leaves the one- and two-body Z statistics alone) and independent noise.

  visibility     <X_B> of every domain: the amplitude with which the domain's raised and sunk depth can interfere (isolated value prod cos tau)
  edge pairs     for every edge of the domain graph (and so every crease between panels): negativity of the two polarity qubits' reduced state
                 (> 0 certifies entanglement between the two domains) and the best CHSH value over all settings (> 2 violates a local model)
  lamp - depth   lamp qubit L1 against a locked polarity qubit: best CHSH over all settings, and a Mermin test on the lamp plus three locked
                 domains from different panels with numerically optimised settings (local bound 4 for 4 parties, quantum maximum 8)
  frustration    cycles of the signed domain graph, with and without the lamp lock (geometry/domains.frustration)
  bonds          MPS backend: the largest bond dimension and entanglement entropy of the state
"""
import numpy as np
from scipy import optimize

from src.geometry.domains import frustration
from src.quantum import complementary as comp
from src.quantum.mps import MPS, permute_circuit
from src.quantum.relief_state import reference_circuit

CHSH_BOUND = 2.0


# ------------------------------------------------------------------ reduced states from either backend
def reduced_set(psi, n, qubits):
    """Reduced density matrix of `qubits` of a flat little-endian n-qubit vector; qubits[0] is the most significant bit."""
    t = np.asarray(psi, complex).reshape((2,) * n)
    m = len(qubits)
    M = np.moveaxis(t, [n - 1 - q for q in qubits], list(range(m))).reshape(2 ** m, -1)
    rho = M @ M.conj().T
    return rho / np.trace(rho).real


class _Exact:
    def __init__(self, psi, n, label):
        self.psi, self.n, self.label = psi, n, label

    def rdm(self, qubits):
        return reduced_set(self.psi, self.n, list(qubits))


class _Mps:
    def __init__(self, mps, site_of, label):
        self.mps, self.site_of, self.label = mps, site_of, label

    def rdm(self, qubits):
        return self.mps.rdm([int(self.site_of[q]) for q in qubits])


def prepared_state(state):
    """The prepared register (facets then polarity), before any lamp or observation rotation, as an object with .rdm(qubits)."""
    sp = state.spec
    if getattr(state, "backend", "exact") == "mps":
        if getattr(state, "_prep", None) is None:
            m = MPS.from_circuit(permute_circuit(reference_circuit(sp), state.site_of), state.truncation, state.max_bond)
            state._prep = _Mps(m, state.site_of, "MPS")
            state._prep_mps = m
        return state._prep
    return _Exact(state.Psi.reshape(-1), sp.n_sim, "exact statevector")


def lamp_state(state):
    """An object with .rdm(qubits) for the prepared register plus lamp qubit L1 (logical qubit n_sim) coupled to the locked polarity qubits.

    Exact backend: the lamp is one more qubit of the statevector. MPS backend: no lamp qubit (its long-range gates are what is slow). The lamp is a
    uniform control, |+>, and the coupling exp(-i theta_d Z_L Z_Bd / 2) is diagonal in it, so the joint state is (|0>_L U_+|Phi> + |1>_L U_-|Phi>) / sqrt 2
    with U_z = prod_d Rz(z theta_d) on the polarity qubits; the blocks of any reduced state follow from the prepared MPS with operators inserted on the
    traced polarity qubits (MPS.rdm(between=...)) and the kept ones rotated afterwards. Exact in both cases (tested against each other)."""
    from qiskit import QuantumCircuit
    sp = state.spec
    lock = sp.lock_array()
    if lock is None:
        return None
    n = sp.n_sim
    if getattr(state, "backend", "exact") == "mps":
        return _LampMps(prepared_state(state), sp, lock, state.site_of)
    qc = QuantumCircuit(n + 1)
    qc.compose(reference_circuit(sp), qubits=list(range(n)), inplace=True)
    qc.h(n)
    for d in range(sp.P):
        if lock[d]:
            qc.rzz(float(lock[d]), n, sp.F + d)
    from qiskit.quantum_info import Statevector
    return _Exact(np.asarray(Statevector(qc).data, complex), n + 1, "exact statevector")


class _LampMps:
    """rdm over {lamp (logical qubit n_sim)} + polarity qubits, from the prepared MPS (see lamp_state). Facets cannot be kept."""

    def __init__(self, prep, spec, lock, site_of):
        self.prep, self.sp, self.lock, self.site_of = prep, spec, lock, site_of

    def rdm(self, qubits):
        sp, n = self.sp, self.sp.n_sim
        qubits = list(qubits)
        kept = [q for q in qubits if q != n]
        if any(q < sp.F for q in kept):
            raise ValueError("the lamp state is available for polarity qubits only")
        lamp_pos = qubits.index(n) if n in qubits else None
        Rz = lambda a: np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])
        zs = (1, -1)
        out_blocks = {}
        for iz, z in enumerate(zs):
            for iw, zp in enumerate(zs):
                between = {}
                for d in range(sp.P):
                    q = sp.F + d
                    if self.lock[d] and q not in kept:
                        # operator B^dagger A on a traced locked qubit: exp(-i theta (z - z') Z / 2)
                        between[int(self.site_of[q])] = Rz(self.lock[d] * (z - zp))
                sigma = self.prep.mps.rdm([int(self.site_of[q]) for q in kept], between=between)
                m = len(kept)
                Uz = np.array([[1.0 + 0j]])
                Uw = np.array([[1.0 + 0j]])
                for q in kept:
                    d = q - sp.F
                    a = self.lock[d]
                    Uz = np.kron(Uz, Rz(a * z))
                    Uw = np.kron(Uw, Rz(a * zp))
                out_blocks[(iz, iw)] = Uz @ sigma @ Uw.conj().T
        m = len(kept)
        dim = 2 ** m
        rho = np.zeros((2 * dim, 2 * dim), complex)                  # (lamp, kept polarity qubits), lamp most significant
        for (iz, iw), blk in out_blocks.items():
            rho[iz * dim:(iz + 1) * dim, iw * dim:(iw + 1) * dim] = 0.5 * blk
        rho = rho / np.trace(rho).real
        if lamp_pos is None:
            return rho.reshape(2, dim, 2, dim).trace(axis1=0, axis2=2)
        # reorder the (lamp, kept...) basis to the order requested in `qubits`
        order_now = [n] + kept
        perm = [order_now.index(q) for q in qubits]
        t = rho.reshape((2,) * (2 * (m + 1)))
        t = np.transpose(t, perm + [m + 1 + p for p in perm])
        return t.reshape(2 ** (m + 1), 2 ** (m + 1))


# ------------------------------------------------------------------ quantities of a reduced state
def negativity(rho):
    """Sum of |negative eigenvalues| of the partial transpose of a two-qubit state (qubit order: first listed = most significant)."""
    r = rho.reshape(2, 2, 2, 2)
    pt = np.transpose(r, (0, 3, 2, 1)).reshape(4, 4)
    ev = np.linalg.eigvalsh((pt + pt.conj().T) / 2)
    return float(-ev[ev < 0].sum())


def dephase(rho, positions, m):
    """Dephase the qubits at `positions` (indices into the m-qubit state, 0 = most significant) in Z: the classical mixture of their bit values."""
    idx = np.arange(2 ** m)
    bit = lambda p: (idx >> (m - 1 - p)) & 1
    keep = np.ones((2 ** m, 2 ** m), bool)
    for p in positions:
        keep &= bit(p)[:, None] == bit(p)[None, :]
    return rho * keep


def mermin_value(rho, axes):
    """Re <prod_j (A_j + i A'_j)> for the m-qubit state rho, A_j = a_j.sigma, A'_j = a'_j.sigma; axes is (m, 2, 3) of orthonormal pairs."""
    M = np.array([[1.0 + 0j]])
    for a, b in axes:
        A = sum(a[k] * comp.PAULI["xyz"[k]] for k in range(3))
        B = sum(b[k] * comp.PAULI["xyz"[k]] for k in range(3))
        M = np.kron(M, A + 1j * B)
    return float(np.trace(rho @ M).real)


def _frame(params):
    """(a, a') orthonormal pair from three Euler angles."""
    th, ph, ps = params
    cz, sz = np.cos(ps), np.sin(ps)
    Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    R = Rz(ph) @ Ry(th) @ Rz(ps)
    return R[:, 0], R[:, 1]


def max_mermin(rho, m, restarts=24, seed=0):
    """Largest Mermin value over local measurement frames (numerical; restarts from random frames). Returns (value, axes)."""
    rng = np.random.default_rng(seed)
    f = lambda x: -mermin_value(rho, [_frame(x[3 * j:3 * j + 3]) for j in range(m)])
    best = (-np.inf, None)
    for _ in range(restarts):
        r = optimize.minimize(f, rng.uniform(0, 2 * np.pi, 3 * m), method="BFGS", options=dict(gtol=1e-7))
        if -r.fun > best[0]:
            best = (-float(r.fun), r.x)
    return best[0], [_frame(best[1][3 * j:3 * j + 3]) for j in range(m)]


# ------------------------------------------------------------------ the certificate
def domain_certificate(state, mermin=True, lamp=True):
    """Everything above for `state` (an exact ReliefState or an MPSReliefState of a domain spec), with provenance and the two controls."""
    sp = state.spec
    prep = prepared_state(state)
    out = dict(provenance=state.label, backend=getattr(state, "backend", "exact"), n_qubits=sp.n_sim, facets=sp.F, domains=sp.P)
    vis = []
    for d in range(sp.P):
        rho = prep.rdm([sp.F + d])
        x = float(np.trace(rho @ comp.PAULI["x"]).real)
        vis.append(dict(domain=d, visibility=x, isolated=sp.visibility(d)))
    out["visibility"] = vis
    names = {}
    edges = []
    for (a, b) in sp.pol_graph:
        rho = prep.rdm([sp.F + a, sp.F + b])
        T, _, _ = comp.correlation_matrix(rho)
        edges.append(dict(a=a, b=b, negativity=negativity(rho), chsh=comp.max_chsh(T),
                          negativity_dephased=negativity(dephase(rho, [0, 1], 2))))
    out["edges"] = edges
    out["summary"] = dict(
        edges=len(edges), entangled_edges=sum(1 for e in edges if e["negativity"] > 1e-6),
        max_negativity=max([e["negativity"] for e in edges], default=0.0), max_chsh=max([e["chsh"] for e in edges], default=0.0),
        dephased_entangled_edges=sum(1 for e in edges if e["negativity_dephased"] > 1e-9))
    if getattr(state, "backend", "exact") == "mps":
        pm = getattr(state, "_prep_mps", None)
        if pm is not None:
            out["bonds"] = dict(max_bond=pm.max_bond(), max_entropy_bits=max(pm.entropies() or [0.0]), bond_dims=pm.bond_dims())
    if lamp and sp.lock_array() is not None:
        out["lamp"] = _lamp_block(state, mermin)
    return out


def _lamp_block(state, mermin):
    sp = state.spec
    ls = lamp_state(state)
    n = sp.n_sim
    lock = sp.lock_array()
    pairs = []
    for d in range(sp.P):
        if lock[d]:
            rho = ls.rdm([n, sp.F + d])
            T, _, _ = comp.correlation_matrix(rho)
            rho_d = dephase(rho, [1], 2)
            Td, _, _ = comp.correlation_matrix(rho_d)
            pairs.append(dict(domain=d, chsh=comp.max_chsh(T), chsh_dephased=comp.max_chsh(Td), negativity=negativity(rho)))
    block = dict(pairs=pairs, best_chsh=max([p["chsh"] for p in pairs], default=0.0), local_bound=CHSH_BOUND)
    block["violating_pairs"] = sum(1 for p in pairs if p["chsh"] > CHSH_BOUND + 1e-9)
    if mermin:
        # one locked domain per panel (the most visible one), so the three leaves are not neighbours on one strip
        by_panel = {}
        dom_panel = {int(sp.panel_of[i]): int(sp.panel_of[i] if sp.facet_panel is None else sp.facet_panel[i]) for i in range(sp.F)}
        for d in range(sp.P):
            if lock[d]:
                p = dom_panel.get(d, 0)
                v = sp.visibility(d)
                if p not in by_panel or v > by_panel[p][1]:
                    by_panel[p] = (d, v)
        leaves = [d for d, _ in by_panel.values()][:3]
        if len(leaves) >= 2:
            group = [n] + [sp.F + d for d in leaves]
            rho = ls.rdm(group)
            val, _ = max_mermin(rho, len(group))
            rho_dep = dephase(rho, list(range(1, len(group))), len(group))
            val_dep, _ = max_mermin(rho_dep, len(group), restarts=8)
            bound, top = comp.mermin_bounds(len(group))
            block["mermin"] = dict(parties=len(group), leaves=leaves, value=val, value_dephased=val_dep, local_bound=bound, quantum_max=top,
                                   violated=bool(val > bound + 1e-6))
    return block


def graph_frustration(ds, spec=None):
    locked = None if spec is None or spec.lock_array() is None else [d for d in range(spec.P) if spec.lock_array()[d]]
    return dict(without_lamp=frustration(ds), with_lamp=frustration(ds, lock=True, locked=locked))


def describe_domain_certificate(cert, ds=None, spec=None):
    """Plain-language lines for the science panel."""
    s = cert["summary"]
    lines = [f"state: {cert['provenance']} ({cert['n_qubits']} qubits: {cert['facets']} facets + {cert['domains']} domain polarity qubits, {cert['backend']} backend)"]
    v = [r["visibility"] for r in cert["visibility"]]
    lines.append(f"domain visibility <X_B>: {min(v):.3f} to {max(v):.3f} (isolated value {np.mean([r['isolated'] for r in cert['visibility']]):.3f}; neighbours lower it)")
    lines.append(f"{s['entangled_edges']} of {s['edges']} edges of the domain graph carry entanglement between their two domains (largest negativity {s['max_negativity']:.3f}, "
                 f"best CHSH {s['max_chsh']:.3f}); control B (dephased depth): {s['dephased_entangled_edges']} entangled edges, as it must be for a classical mixture")
    if "bonds" in cert:
        lines.append(f"matrix-product state: largest bond dimension {cert['bonds']['max_bond']}, largest bond entropy {cert['bonds']['max_entropy_bits']:.2f} bits")
    if "lamp" in cert:
        L = cert["lamp"]
        lines.append(f"lamp - depth: {L['violating_pairs']} of {len(L['pairs'])} locked domains violate CHSH with the lamp (best {L['best_chsh']:.3f}, local bound 2); dephased depth: "
                     f"{max([p['chsh_dephased'] for p in L['pairs']], default=0.0):.3f}")
        if "mermin" in L:
            m = L["mermin"]
            lines.append(f"Mermin test, lamp + {m['parties'] - 1} domains from different panels: {m['value']:.3f} against a local bound {m['local_bound']} (quantum maximum {m['quantum_max']}): "
                         + ("VIOLATED" if m["violated"] else "not violated") + f"; dephased depth {m['value_dephased']:.3f}")
    if ds is not None:
        fr = graph_frustration(ds, spec)
        lines.append(f"frustration: {fr['without_lamp']['frustrated_cycles']} frustrated cycles among the domains alone, {fr['with_lamp']['frustrated_cycles']} once the lamp is joined to them")
    return lines
