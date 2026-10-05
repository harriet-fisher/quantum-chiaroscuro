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


def seam_facets(ds):
    """Quantum facets that touch a facet of another panel across a crease, and the pairs that do."""
    fs = ds.facets
    nq = fs.n_quantum
    pairs = [(i, j) for i, j, kind, _ in fs.edges if kind == "crease" and i < nq and j < nq]
    return sorted({i for p in pairs for i in p}), pairs


MAX_LEAVES = 3          # the parity game is the four-party Mermin game: the lamp and three leaf domains


def leaf_domains(ds, prefer="visible", candidates=None, max_leaves=MAX_LEAVES):
    """The leaf domains: ONE crater-gauge-signed domain per PLANE (a bay-window wing is one plane even when it is two stacked window panes), at most
    `max_leaves` of them. prefer='visible': the most visible domain of the plane, the one nearest the plane's middle on a tie. prefer='seam': the one with
    the most facets on a crease (then the most visible), so the game is played on the seam graph. A lamp locked to every domain is decohered by the
    records the others hold of it (monogamy), so the lock goes to these only. Of more than `max_leaves` planes the leaves are spread across the wall
    (the left-most, the middle and the right-most by x). `candidates` restricts the domains considered (default: all). Returned in plane order.

    Per plane and not per panel: the six panes of the built-in bay window give the three leaves left, centre and right, as the three-panel scene
    always did; one leaf per pane would lock the lamp to six qubits and the Mermin violation would be gone."""
    seam, _ = seam_facets(ds)
    seam = set(seam)
    planes = ds.domain_plane if ds.domain_plane is not None else ds.domain_panel
    allowed = set(range(ds.n_domains) if candidates is None else (int(c) for c in candidates))
    out = []
    for plane in sorted(set(planes)):
        mine = [d for d in range(ds.n_domains) if planes[d] == plane]
        cand = [d for d in mine if d in allowed and ds.lock_sign[d] != 0]
        if not cand:
            continue
        pc = np.mean([ds.domain_centroid[d] for d in mine], axis=0)
        near = lambda d: -float(np.hypot(*(np.array(ds.domain_centroid[d]) - pc)))
        if prefer == "seam":
            out.append(max(cand, key=lambda d: (sum(1 for i in ds.facets_of(d) if i in seam), round(ds.visibility(d), 6), near(d))))
        else:
            out.append(max(cand, key=lambda d: (round(ds.visibility(d), 6), near(d))))
    if len(out) > max_leaves:
        by_x = sorted(out, key=lambda d: ds.domain_centroid[d][0])
        keep = {by_x[i] for i in np.unique(np.round(np.linspace(0, len(by_x) - 1, max_leaves)).astype(int))}
        out = [d for d in out if d in keep]
    return out


def spec_from_domains(ds, scene, entangle=0.8, pol_coupling=0.8, lock=0.0, tau_mix=0.0, pol_field=0.0, rig=None, observations=None, compensate=True,
                      pol_layers=None, order="sweep", lock_domains="leaves", crease_coupling=None, seam_coupling=None, seam_mix=0.0, leaf_tau=None,
                      floquet=None, leaf_prefer="visible"):
    """The circuit of a DomainSet.

    entangle      theta of the ZZ couplings between touching facets (+ coplanar, - across a crease); 0 leaves the facets a product state
    pol_coupling  theta of the Ising layer on the polarity qubits (sign: + coplanar, ds.params['crease_sign'] across a crease)
    lock          theta of the lamp-L1 -- polarity coupling, times each domain's crater-gauge sign (0: the lamp is a plain coin)
    lock_domains  "leaves" (one domain per plane, at most three: see leaf_domains), "all", or a list of domain indices
    tau_mix       0..1: tau_i -> (1 - mix) tau_i + mix pi/2, the facets move toward the equator (the graph-state regime; V_d falls toward 0)
    pol_field     transverse field angle on every polarity qubit after the Ising layer (0: none, which keeps the circuit IQP-shaped)
    pol_layers    replaces the layer built from the arguments above (used by the ground-state preparation)
    order         MPS site order. "sweep" (default): domains sorted along the longer canvas axis. "loop": domains in the order they run around each loop.
                  MEASURED on the bay window (69 qubits, bond cap 32, norm deficit): sweep 1.5e-3, loop 5.2e-3. The intuition that a ring has cut width 2
                  so loop order should win is wrong here (the loops are coupled to each other along the seams and the glass strips run beside the outer
                  ones); an earlier version also never honoured this argument (a local variable shadowed it), so "sweep" is what every earlier number used.

    Seams (the creases between panels), where the entanglement between EDGES lives:
    crease_coupling  theta of the polarity Ising coupling across a crease only (default: pol_coupling); stronger there than along a boundary
    seam_coupling    theta of the facet ZZ couplings that touch across a crease and among seam facets (default: entangle); pi/2 is a CZ up to local phases
    seam_mix         0..1: the SEAM facets only move toward the equator (tau -> (1 - mix) tau + mix pi/2): the graph-state regime on the seam and the
                     visible relief regime everywhere else. A domain holding a seam facet loses its visibility V (it is cos(tau) of that facet), so the game
                     leaves should avoid it unless the point is to see what that costs (leaf_prefer='seam' picks them ON the seam).
    leaf_tau         tilt of the facets of the leaf domains (default: ds.params['tau']); smaller raises their V, so the parity game wins more rounds
    floquet          (steps, theta_zz, theta_x): kicked-Ising dynamics on the facet graph after the preparation. Every step entangles further; steps = 0 is static."""
    fs = ds.facets
    nq = fs.n_quantum
    tau, phi = fs.qtau.copy(), fs.qphi.copy()
    if leaf_tau is not None:
        leaves_ = leaf_domains(ds, prefer=leaf_prefer)
        for d in leaves_:
            for i in ds.facets_of(d):
                tau[i] = min(tau[i] * float(leaf_tau) / float(ds.params["tau"]), 1.1)
    seam, seam_pairs = seam_facets(ds)
    seam_set = set(seam)
    if seam_mix:
        for i in seam:
            tau[i] = (1 - seam_mix) * tau[i] + seam_mix * (np.pi / 2)
    if tau_mix:
        tau = (1 - tau_mix) * tau + tau_mix * (np.pi / 2)
    edges = []
    th_seam = entangle if seam_coupling is None else seam_coupling
    for i, j, kind, _ in fs.edges:
        if i >= nq or j >= nq:
            continue
        on_seam = kind == "crease" or (i in seam_set and j in seam_set)
        th = float(th_seam if on_seam else entangle)
        if th:
            edges.append((i, j, "zz", th if kind == "coplanar" else -th))
    if compensate:
        phi = phi - zz_rotation(tau, edges)
    graph = [(a, b) for a, b, _, _ in ds.edges]
    kinds = [kind for _, _, kind, _ in ds.edges]
    cs = float(ds.params["crease_sign"])
    if pol_layers is None:
        pol_layers = []
        if (pol_coupling or crease_coupling or pol_field) and graph:
            cc = pol_coupling if crease_coupling is None else crease_coupling
            zz = [float(pol_coupling) if kind == "coplanar" else cs * float(cc) for kind in kinds]
            pol_layers = [dict(zz=zz, rx=[float(pol_field)] * ds.n_domains if pol_field else None)]
    obs = observations or (DOMAIN_OBSERVATIONS if lock else DEFAULT_OBSERVATIONS)
    classical = [f.panel for f in fs.facets[nq:]]
    panel_of_facet = np.array([f.panel for f in fs.facets[:nq]], int)
    lock_vec, leaves = 0.0, None
    if lock:
        chosen = leaf_domains(ds, prefer=leaf_prefer) if lock_domains == "leaves" else (range(ds.n_domains) if lock_domains == "all" else list(lock_domains))
        lock_vec = np.zeros(ds.n_domains)
        for d in chosen:
            lock_vec[d] = float(lock) * ds.lock_sign[d]
        leaves = leaf_domains(ds, prefer=leaf_prefer, candidates=chosen)
    spec = ReliefSpec(tau, phi, ds.domain_of, [p.angle_deg for p in scene.panels], edges, rig or LightRig(), list(obs),
                      [f"d{ds.domain_of[i]}.{i}" for i in range(nq)], n_classical=len(classical), classical_panel=classical,
                      n_groups=ds.n_domains, facet_panel=panel_of_facet, pol_graph=graph, pol_layers=pol_layers,
                      lamp_lock=lock_vec, pol_kinds=kinds, seam_pairs=seam_pairs, seam_facets=seam, leaves=leaves)
    if floquet and floquet[0]:
        spec.floquet = dict(steps=int(floquet[0]), theta_zz=float(floquet[1]), theta_x=float(floquet[2]))
    order_ = []
    for d in (range(ds.n_domains) if order == "loop" else ds.sweep_order()):
        order_.append(nq + d)
        order_.extend(ds.facets_of(d))
    spec.site_order = order_
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
    """The relief state evolved by Aer's matrix-product-state simulator and sampled exactly from the tensors. The interface is the part of ReliefState
    the show uses (spec, label, draw), plus what the parity game needs: the lamp read along any axis and a depth-sphere axis per polarity qubit.

    Aer builds ONE state, the prepared register (facets, polarity, couplings, Floquet steps). Everything else a frame needs is a single-qubit gate and
    so only touches one tensor: the lamp lock (a z-rotation of a locked polarity qubit by the lamp's Z value), the light rotation of every facet for the
    lamp world, and the observation rotation of every polarity qubit. A lamp qubit read along a rotated axis leaves the wall in a superposition of its
    conditioned states (MPS.combine: the bond dimensions add, exactly). Two semantics, kept apart on purpose:
      * lamp_mode 'x' / 'y' (the experiment key): the lamp CONTROLS the facets' light axis coherently, so the facets record the light side and dephase
        whatever coherence the lamp had with the depth qubits (exactly the exact engine's lamp_mode);
      * game_axis (the parity game): the lamp is read first and its outcome then CHOOSES the light classically (feed-forward), so the facets hold no record
        of it. A lamp read in Z is the same thing, which is why ordinary frames and game rounds share one state."""
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
        self._mps, self._derived, self._psamp, self._pol_mps = {}, {}, {}, None
        self.bond_log = []

    # ---- the prepared register and the single-qubit gates around it
    def prep(self):
        if "prep" not in self._mps:
            m = MPS.from_circuit(permute_circuit(reference_circuit(self.spec), self.site_of), self.truncation, self.max_bond)
            self.bond_log.append((m.max_bond(), m.build_seconds))
            self._mps["prep"] = m
        return self._mps["prep"]

    def lock_unitaries(self, b1):
        """{site: Rz} of the lamp lock for lamp L1 = b1 read in Z: exp(-i theta_d z Z_B / 2) with z = 1 - 2 b1."""
        lock = self.spec.lock_array()
        out = {}
        if lock is not None:
            z = 1 - 2 * int(b1)
            for d in range(self.spec.P):
                if lock[d]:
                    a = float(lock[d] * z)
                    out[self._site(self.spec.F + d)] = np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])
        return out

    def light_unitaries(self, w):
        sp = self.spec
        return {self._site(i): sl.bloch_unitary(sp.rig.panel_local(w, sp.panel_angles[sp.light_panel(i)])) for i in range(sp.F)}

    def base(self, w):
        """The prepared register with the lock and the light rotation of lamp world w = b1 + 2 b2 (no observation rotation)."""
        key = ("base", w)
        if key not in self._derived:
            self._derived[key] = self.prep().apply_many({**self.lock_unitaries(w & 1), **self.light_unitaries(w)})
        return self._derived[key]

    def lockstate(self, b1):
        key = ("lock", b1)
        if key not in self._derived:
            self._derived[key] = self.prep().apply_many(self.lock_unitaries(b1))
        return self._derived[key]

    @staticmethod
    def _axis(gam, chi):
        return (np.sin(gam) * np.cos(chi), np.sin(gam) * np.sin(chi), np.cos(gam))

    def obs_unitaries(self, g, overrides=None):
        """{site: U} rotating each polarity qubit to read along the depth-sphere axis of observation g; `overrides` = {domain: (gamma, chi)} sets
        individual domains (the parity game's per-party settings)."""
        sp = self.spec
        Ug = sl.bloch_unitary(self._axis(*sp.observations[g]))
        out = {self._site(sp.F + d): Ug for d in range(sp.P)}
        for d, (gm, ch) in (overrides or {}).items():
            out[self._site(sp.F + d)] = sl.bloch_unitary(self._axis(gm, ch))
        return out

    def mps(self, g, w):
        key = (g, w)
        if key not in self._derived:
            self._derived[key] = self.base(w).apply_many(self.obs_unitaries(g))
        return self._derived[key]

    def warm(self):
        """Build the prepared state now instead of on first use. Returns the seconds it took."""
        import time
        t0 = time.time()
        self.prep()
        return time.time() - t0

    def accuracy(self):
        """dict(max_bond, norm_deficit): the bond dimension reached and the norm lost to truncation (every later gate is single-qubit, so this is all of it)."""
        if "prep" not in self._mps:
            return dict(max_bond=0, norm_deficit=0.0)
        m = self._mps["prep"]
        return dict(max_bond=m.max_bond(), norm_deficit=float(abs(1 - m.norm2())))

    def _site(self, q):
        return int(self.site_of[q])

    def _pol_sites(self):
        return [self._site(self.spec.F + d) for d in range(self.spec.P)]

    def _fac_sites(self):
        return [self._site(i) for i in range(self.spec.F)]

    def game_lamp_states(self, axis):
        """{m: (unnormalised MPS, probability)}: lamp L1 read along `axis` (a Bloch vector), the lock applied, NO light rotation yet (feed-forward)."""
        U = sl.bloch_unitary(axis)
        st = [self.lockstate(0), self.lockstate(1)]
        out = {}
        for m in (0, 1):
            phi = MPS.combine(st, [U[m, 0] / np.sqrt(2), U[m, 1] / np.sqrt(2)])
            out[m] = (phi, phi.norm2())
        return out

    # ---- the lamp read along an axis
    def lamp_outcomes(self, lamp_axes, zbits=(0, 0)):
        """{(m1, m2): (unnormalised MPS before the observation rotation, probability)}. lamp_axes = (n1, n2), each a Bloch vector or None (read in Z, with
        the classical bit zbits[q] standing for the outcome). Qubit q read along n projects the lamp |+> onto <m|U_n: the wall is left in
        sum_b U_n[m, b] / sqrt 2 |Psi_b>."""
        U = [None if a is None else sl.bloch_unitary(a) for a in lamp_axes]
        outs = {}
        axis_q = [q for q in (0, 1) if U[q] is not None]
        import itertools
        for ms in itertools.product((0, 1), repeat=len(axis_q)):
            m = list(zbits)
            for q, mq in zip(axis_q, ms):
                m[q] = mq
            terms = []
            for b1 in ((0, 1) if U[0] is not None else (zbits[0],)):
                for b2 in ((0, 1) if U[1] is not None else (zbits[1],)):
                    c = 1.0 + 0j
                    for q, b in ((0, b1), (1, b2)):
                        if U[q] is not None:
                            c *= U[q][m[q], b] / np.sqrt(2)
                    terms.append((b1 + 2 * b2, c))
            st = MPS.combine([self.base(w) for w, _ in terms], [c for _, c in terms]) if len(terms) > 1 else \
                MPS([terms[0][1] * t if k == 0 else t for k, t in enumerate(self.base(terms[0][0]).A)])
            outs[tuple(m)] = (st, st.norm2())
        return outs

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

    def _z_mps(self, w):
        """The prepared register with only the light rotation on the facets (the polarity qubits stay in the Z basis): control B measures them in Z."""
        key = ("z", w)
        if key not in self._derived:
            self._derived[key] = self.prep().apply_many(self.light_unitaries(w))
        return self._derived[key]

    def _reweighted_polarity(self, r_row):
        """The polarity register's MPS with every site tensor scaled by sqrt(r[o_d, s]): its Z-distribution is the posterior of the classical polarity s
        given the frame's observation outcome o, p(s|o) proportional to |beta_s|^2 prod_d r[o_d, s_d]."""
        base = self._polarity_z_sampler()
        A = [np.stack([np.sqrt(r_row[d][s]) * base.A[d][s] for s in (0, 1)]) for d in range(self.spec.P)]
        return MPS(A)

    # ---- frames
    LAMP_BASES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0)}

    def draw(self, K, rng, g=None, control="coherent", lamp_mode="z", world=None, lamp_axes=None, obs_axes=None, game_axis=None):
        """One frame. lamp_mode 'z' (a light world is drawn), 'x' / 'y' (both lamp qubits read in that basis: the light directions interfere, exactly as
        in the exact engine) or lamp_axes=(n1, n2) (each a Bloch vector or None for Z): both lamp qubits then CONTROL the light coherently. game_axis = a Bloch vector reads L1
        along it with feed-forward (the outcome then picks the light classically, L2 stays a Z coin): the parity game's lamp. obs_axes = {domain:
        (gamma, chi)} overrides the observation of single polarity qubits. The returned draw carries lamp_outcome = (m1, m2)."""
        sp = self.spec
        F, P = sp.F, sp.P
        g = int(rng.integers(len(sp.observations))) if g is None else int(g)
        gam, chi = sp.observations[g]
        if lamp_mode in self.LAMP_BASES and lamp_axes is None:
            lamp_axes = (self.LAMP_BASES[lamp_mode], self.LAMP_BASES[lamp_mode])
        game = game_axis is not None and control == "coherent"
        axis_mode = (not game) and lamp_axes is not None and any(a is not None for a in lamp_axes) and control == "coherent"
        w = int(rng.integers(4)) if world is None else int(world)
        lamp_outcome = (w & 1, w >> 1)
        if control == "noise":
            o_bits = rng.integers(0, 2, P)
            lit = rng.binomial(K, 0.5, F) / K
        elif control == "dephased":
            # Control B, the same semantics as the exact engine: the polarity is a classical mixture s; the frame's observation outcome o is read on it
            # (r[o, s]); the K photons share o but each carries its OWN hidden s ~ p(s|o). Keeping one s for the whole frame would show a full bevel the
            # observation never decided, which the coherent state cannot.
            rot = np.abs(sl.bloch_unitary(self._axis(gam, chi))) ** 2     # r[o, s]
            s0 = self._polarity_z_sampler().sampler(range(P)).draw(rng, 1)[0]
            o_bits = np.array([int(rng.random() < rot[1, int(s0[d])]) for d in range(P)])
            post = self._reweighted_polarity([rot[int(o_bits[d])] for d in range(P)]).sampler(range(P)).draw(rng, K)            # (K, P): hidden s per photon
            zm = self._z_mps(w)
            fsites = sorted(self._fac_sites())
            col = {s_: j for j, s_ in enumerate(fsites)}
            psites = self._pol_sites()
            x = np.zeros((K, F), np.int8)
            groups = {}
            for k in range(K):
                groups.setdefault(tuple(int(b) for b in post[k]), []).append(k)
            for sbits, ks in groups.items():
                xs = zm.sampler(self._fac_sites(), fixed={psites[d]: sbits[d] for d in range(P)}).draw(rng, len(ks))
                for j, k in enumerate(ks):
                    x[k] = [xs[j, col[self._site(i)]] for i in range(F)]
            lit = (x == 0).mean(axis=0)
        else:
            if game or axis_mode or obs_axes:
                if game:
                    outs = self.game_lamp_states(game_axis)
                    pm = np.array([max(outs[0][1], 0.0), max(outs[1][1], 0.0)])
                    m1 = int(rng.choice(2, p=pm / pm.sum()))
                    w = m1 + 2 * (w >> 1)                           # the outcome picks the light side; L2 stays a Z coin
                    lamp_outcome = (m1, w >> 1)
                    base_state = outs[m1][0].apply_many(self.light_unitaries(w))
                elif axis_mode:
                    outs = self.lamp_outcomes(lamp_axes, (w & 1, w >> 1))
                    keys = list(outs)
                    pm = np.array([max(outs[k][1], 0.0) for k in keys])
                    lamp_outcome = keys[int(rng.choice(len(keys), p=pm / pm.sum()))]
                    base_state = outs[lamp_outcome][0]
                    w = lamp_outcome[0] + 2 * lamp_outcome[1]
                else:
                    base_state = self.base(w)
                m = base_state.apply_many(self.obs_unitaries(g, obs_axes))
                ps = m.sampler(self._pol_sites())
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
        zmode = not axis_mode
        if sp.n_classical:
            lit = np.concatenate([lit, ReliefState._classical_lit(self, K, rng, w, "z" if zmode else "x", control)])
        pol = tuple(int(1 - 2 * int(b)) for b in o_bits)
        lean = tuple(float(p * np.cos(gam)) for p in pol)
        d = ReliefDraw(lit, WORLDS[w] if zmode else (0, 0), pol, int(K), w, g, float(gam), float(chi), lean, control, "z" if zmode else (lamp_mode if lamp_mode in self.LAMP_BASES else "x"))
        d.lamp_outcome = tuple(int(v) for v in lamp_outcome)
        d.obs_axes = dict(obs_axes) if obs_axes else {}
        return d
