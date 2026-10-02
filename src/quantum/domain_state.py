"""The domain relief as a quantum state: spec builder, exact and matrix-product-state backends, frames.

    spec_from_domains(ds, scene, ...)  -> ReliefSpec     (the circuit; nothing in it is random)
    make_state(spec)                   -> ReliefState    exact statevector, up to EXACT_MAX qubits
                                          MPSReliefState  Aer matrix-product-state evolution + exact conditional sampling, any size the entanglement allows

What changes against the per-panel relief (quantum/relief_state.py), which it extends rather than replaces:
  * one polarity qubit per depth *domain* (geometry/domains.py), so the tilt budget is per domain and V does not shrink with the size of the wall;
  * the polarity qubits are coupled along the domain graph, exp(-i theta ZZ / 2) on |+>|+>, a graph-state-like Ising layer: entanglement across the
    geometric edges, including the creases between panels (theta > 0 within a plane, crease_sign * theta across a crease);
  * optional: facets pushed toward the equator (`tau_mix`) with ZZ couplings on the facet adjacency graph: the graph-state regime, where the lamp
    direction is the measurement basis and the one-body normal survives only as far as the coupling angle leaves it (transverse_shrink);
  * optional: lamp L1 coupled to the polarity qubits with the crater-gauge signs: the light side and the depth are entangled;
  * a transverse field on the polarity qubits (`pol_field`) and further layers (quantum/ground_state.py prepares the ground state of the Ising layer).

A frame is exactly what it was: draw the lamp world w and the observation g (uniform control registers), measure the polarity qubits once (the
frame's depth decision), then K photons of the facets *given* that decision. The MPS backend does this exactly, not by post-selection.
"""
import numpy as np

from src.quantum import sampler_local as sl
from src.quantum.mps import MPS, permute_circuit
from src.quantum.relief_state import (DEFAULT_OBSERVATIONS, WORLDS, LightRig, ReliefDraw, ReliefSpec, ReliefState, reference_circuit, zz_rotation)

EXACT_MAX = 20
MPS_DEFAULTS = dict(truncation=1e-8, max_bond=32)     # a 57-qubit wall builds in <1 s at chi <= 32 (norm deficit ~4e-4); chi 64 takes ~5 s, uncapped minutes
DOMAIN_OBSERVATIONS = [(0.0, 0.0), (float(np.deg2rad(45)), float(np.pi / 2)), (float(np.pi / 2), 0.0), (float(np.pi / 2), float(np.pi / 2))]


def leaf_domains(ds, panels=None):
    """One crater-gauge-signed domain per panel: the most visible one, the one nearest the panel's middle on a tie. These are the domains the lamp
    is locked to by default (and the three leaves of the Mermin test): a lamp locked to every domain is decohered by the records the others
    hold of it (monogamy), so no Bell-type violation could survive."""
    out = []
    for k in (range(max(ds.domain_panel) + 1) if panels is None else panels):
        cand = [d for d in range(ds.n_domains) if ds.domain_panel[d] == k and ds.lock_sign[d] != 0]
        if not cand:
            continue
        pc = np.mean([ds.domain_centroid[d] for d in range(ds.n_domains) if ds.domain_panel[d] == k], axis=0)
        out.append(max(cand, key=lambda d: (round(ds.visibility(d), 6), -float(np.hypot(*(np.array(ds.domain_centroid[d]) - pc))))))
    return out


def spec_from_domains(ds, scene, entangle=0.8, pol_coupling=0.8, lock=0.0, tau_mix=0.0, pol_field=0.0, rig=None, observations=None, compensate=True,
                      pol_layers=None, order="loop", lock_domains="leaves"):
    """The circuit of a DomainSet.

    entangle      theta of the ZZ couplings between touching facets (+ coplanar, - across a crease); 0 leaves the facets a product state
    pol_coupling  theta of the Ising layer on the polarity qubits (sign: + coplanar, ds.params['crease_sign'] across a crease)
    lock          theta of the lamp-L1 -- polarity coupling, times each domain's crater-gauge sign (0: the lamp is a plain coin)
    lock_domains  "leaves" (one domain per panel, see leaf_domains), "all", or a list of domain indices
    tau_mix       0..1: tau_i -> (1 - mix) tau_i + mix pi/2, the facets move toward the equator (the graph-state regime; V_d falls toward 0)
    pol_field     transverse field angle on every polarity qubit after the Ising layer (0: none, which keeps the circuit IQP-shaped)
    pol_layers    replaces the layer built from the arguments above (used by the ground-state preparation)
    order         MPS site order: "loop" (domains in the order they run around each loop: a ring has cut width 2, so the bond dimension stays small)
                  or "sweep" (sorted along the longer canvas axis: every loop crosses every cut, which is far worse)"""
    fs = ds.facets
    nq = fs.n_quantum
    tau, phi = fs.qtau.copy(), fs.qphi.copy()
    if tau_mix:
        tau = (1 - tau_mix) * tau + tau_mix * (np.pi / 2)
    edges = []
    if entangle:
        for i, j, kind, _ in fs.edges:
            if i < nq and j < nq:
                edges.append((i, j, "zz", float(entangle) if kind == "coplanar" else -float(entangle)))
    if compensate:
        phi = phi - zz_rotation(tau, edges)
    graph = [(a, b) for a, b, _, _ in ds.edges]
    cs = float(ds.params["crease_sign"])
    if pol_layers is None:
        pol_layers = []
        if (pol_coupling or pol_field) and graph:
            zz = [float(pol_coupling) * (1.0 if kind == "coplanar" else cs) for _, _, kind, _ in ds.edges]
            pol_layers = [dict(zz=zz, rx=[float(pol_field)] * ds.n_domains if pol_field else None)]
    obs = observations or (DOMAIN_OBSERVATIONS if lock else DEFAULT_OBSERVATIONS)
    classical = [f.panel for f in fs.facets[nq:]]
    panel_of_facet = np.array([f.panel for f in fs.facets[:nq]], int)
    lock_vec = 0.0
    if lock:
        chosen = leaf_domains(ds) if lock_domains == "leaves" else (range(ds.n_domains) if lock_domains == "all" else list(lock_domains))
        lock_vec = np.zeros(ds.n_domains)
        for d in chosen:
            lock_vec[d] = float(lock) * ds.lock_sign[d]
    spec = ReliefSpec(tau, phi, ds.domain_of, [p.angle_deg for p in scene.panels], edges, rig or LightRig(), list(obs),
                      [f"d{ds.domain_of[i]}.{i}" for i in range(nq)], n_classical=len(classical), classical_panel=classical,
                      n_groups=ds.n_domains, facet_panel=panel_of_facet, pol_graph=graph, pol_layers=pol_layers,
                      lamp_lock=lock_vec)
    order = []
    for d in (range(ds.n_domains) if order == "loop" else ds.sweep_order()):
        order.append(nq + d)
        order.extend(ds.facets_of(d))
    spec.site_order = order
    return spec


def make_state(spec, label="local reference circuit", backend="auto", truncation=None, max_bond=None):
    """ReliefState (exact) when the register is small enough, else the MPS backend."""
    if backend == "exact" or (backend == "auto" and spec.n_sim <= EXACT_MAX):
        return ReliefState(spec, label=label)
    if backend not in ("auto", "mps"):
        raise ValueError("backend must be 'auto', 'exact' or 'mps'")
    return MPSReliefState(spec, label=label, truncation=MPS_DEFAULTS["truncation"] if truncation is None else truncation,
                          max_bond=MPS_DEFAULTS["max_bond"] if max_bond is None else max_bond)


class MPSReliefState:
    """The relief state evolved by Aer's matrix-product-state simulator and sampled exactly from the tensors. The interface is the part of
    ReliefState the show uses (spec, label, draw). The lamp read in X/Y (`lamp_mode`) needs the lamp register inside the state and is exact-only."""
    backend = "mps"

    def __init__(self, spec, label="local reference circuit", truncation=MPS_DEFAULTS["truncation"], max_bond=MPS_DEFAULTS["max_bond"]):
        self.spec, self.label = spec, label
        self.truncation, self.max_bond = truncation, max_bond
        F, P = spec.F, spec.P
        order = spec.site_order or list(range(F + P))
        if sorted(order) != list(range(F + P)):
            raise ValueError("site_order must be a permutation of the F + P qubits")
        self.site_of = np.empty(F + P, int)
        self.site_of[order] = np.arange(F + P)
        self._mps, self._psamp, self._pol_mps = {}, {}, None
        self.bond_log = []

    # ---- circuits (logical qubits: facets 0..F-1, polarity F..F+P-1), conditioned on the lamp world w and observation g
    def _conditioned_circuit(self, g, w):
        from qiskit import QuantumCircuit
        sp = self.spec
        F, P = sp.F, sp.P
        qc = reference_circuit(sp)
        lock = sp.lock_array()
        if lock is not None:
            zl = WORLDS[w][0]                                       # lamp L1 read in Z as +/-1: exp(-i theta z_L Z_B / 2) is a z-rotation of the polarity qubit
            for d in range(P):
                if lock[d]:
                    qc.rz(float(lock[d] * zl), F + d)
        gam, chi = sp.observations[g]
        for d in range(P):
            if abs(chi) > 1e-12:
                qc.rz(-float(chi), F + d)
            if abs(gam) > 1e-12:
                qc.ry(-float(gam), F + d)
        for i in range(F):
            a = sp.rig.panel_local(w, sp.panel_angles[sp.light_panel(i)])
            qc.rz(-float(np.arctan2(a[1], a[0])), i)
            qc.ry(-float(np.arccos(np.clip(a[2], -1, 1))), i)
        return qc

    def mps(self, g, w):
        key = (g, w)
        if key not in self._mps:
            m = MPS.from_circuit(permute_circuit(self._conditioned_circuit(g, w), self.site_of), self.truncation, self.max_bond)
            self.bond_log.append((m.max_bond(), m.build_seconds))
            self._mps[key] = m
        return self._mps[key]

    def warm(self):
        """Build the 16 conditioned states (4 observations x 4 lamp worlds) now instead of on first use. Returns the seconds it took."""
        import time
        t0 = time.time()
        for g in range(len(self.spec.observations)):
            for w in range(4):
                self.mps(g, w)
        return time.time() - t0

    def accuracy(self):
        """dict(max_bond, norm_deficit): the worst bond dimension reached and the largest 1 - <psi|psi> from truncation over the states built so far."""
        if not self._mps:
            return dict(max_bond=0, norm_deficit=0.0)
        return dict(max_bond=max(m.max_bond() for m in self._mps.values()), norm_deficit=float(max(abs(1 - m.norm2()) for m in self._mps.values())))

    def _site(self, q):
        return int(self.site_of[q])

    def _pol_sites(self):
        return [self._site(self.spec.F + d) for d in range(self.spec.P)]

    def _fac_sites(self):
        return [self._site(i) for i in range(self.spec.F)]

    # ---- the polarity register alone (control B needs its Z-basis distribution)
    def _polarity_z_sampler(self):
        if self._pol_mps is None:
            from qiskit import QuantumCircuit
            sp = self.spec
            if sp.pol_circuit is not None:
                qc = sp.pol_circuit
            else:
                qc = QuantumCircuit(sp.P)
                for d in range(sp.P):
                    qc.h(d)
                for layer in sp.pol_layers:
                    if layer.get("zz") is not None:
                        for (a, b), th in zip(sp.pol_graph, layer["zz"]):
                            qc.rzz(float(th), a, b)
                    if layer.get("rx") is not None:
                        for d, b in enumerate(layer["rx"]):
                            if b:
                                qc.rx(float(b), d)
            self._pol_mps = MPS.from_circuit(qc, self.truncation, self.max_bond)
        return self._pol_mps

    def _facet_branch(self, g, w, s_bits):
        """Facet state of the classical polarity branch s (control B): psi_f, Z on the facets of every domain with s_d = 1, then the light rotation."""
        from qiskit import QuantumCircuit
        sp = self.spec
        F = sp.F
        qc = QuantumCircuit(F)
        for i in range(F):
            qc.ry(float(sp.tau[i]), i)
            qc.rz(float(sp.phi[i]), i)
        for i, j, kind, th in sp.edges:
            if kind == "zz":
                qc.rzz(float(th), i, j)
        for i in range(F):
            if s_bits[sp.panel_of[i]]:
                qc.z(i)
            a = sp.rig.panel_local(w, sp.panel_angles[sp.light_panel(i)])
            qc.rz(-float(np.arctan2(a[1], a[0])), i)
            qc.ry(-float(np.arccos(np.clip(a[2], -1, 1))), i)
        return MPS.from_circuit(qc, self.truncation, self.max_bond)

    # ---- frames
    def draw(self, K, rng, g=None, control="coherent", lamp_mode="z", world=None):
        if lamp_mode != "z":
            raise ValueError("reading the lamp in X needs the lamp register inside the state: exact backend only (at most %d qubits)" % EXACT_MAX)
        sp = self.spec
        F, P = sp.F, sp.P
        g = int(rng.integers(len(sp.observations))) if g is None else int(g)
        gam, chi = sp.observations[g]
        w = int(rng.integers(4)) if world is None else int(world)
        if control == "noise":
            o_bits = rng.integers(0, 2, P)
            lit = rng.binomial(K, 0.5, F) / K
        elif control == "dephased":
            s_bits = self._polarity_z_sampler().sampler(range(P)).draw(rng, 1)[0]
            rot = np.abs(sl.bloch_unitary((np.sin(gam) * np.cos(chi), np.sin(gam) * np.sin(chi), np.cos(gam)))) ** 2     # r[o, s]
            o_bits = np.array([int(rng.random() < rot[1, int(s_bits[d])]) for d in range(P)])
            branch = self._facet_branch(g, w, s_bits)
            x = branch.sampler(range(F)).draw(rng, K)
            lit = (x == 0).mean(axis=0)
        else:
            m = self.mps(g, w)
            ps = self._psamp.get((g, w))
            if ps is None:
                ps = self._psamp[(g, w)] = m.sampler(self._pol_sites())
            o_sorted = ps.draw(rng, 1)[0]                           # columns in sorted site order
            psites = sorted(self._pol_sites())
            bit_of_site = {s: int(b) for s, b in zip(psites, o_sorted)}
            o_bits = np.array([bit_of_site[self._site(F + d)] for d in range(P)])
            fs = m.sampler(self._fac_sites(), fixed=bit_of_site)
            x = fs.draw(rng, K)                                     # (K, F) in sorted facet-site order
            fsites = sorted(self._fac_sites())
            col = {s: j for j, s in enumerate(fsites)}
            lit = np.array([(x[:, col[self._site(i)]] == 0).mean() for i in range(F)])
        if sp.n_classical:
            lit = np.concatenate([lit, ReliefState._classical_lit(self, K, rng, w, lamp_mode, control)])
        pol = tuple(int(1 - 2 * int(b)) for b in o_bits)
        lean = tuple(float(p * np.cos(gam)) for p in pol)
        return ReliefDraw(lit, WORLDS[w], pol, int(K), w, g, float(gam), float(chi), lean, control, lamp_mode)
