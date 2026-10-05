"""Superposed Relief: the quantum state, its circuit, and how a frame is drawn from it (v2 handoff sections 2 and 4.2).

Qubits (little-endian, as everywhere in this repo: qubit q is bit q of the basis-state index; spin +1 is bit 0):
    facets    one per facet, state = the facet's surface normal, Bloch b = (sin t cos p, sin t sin p, cos t)      [simulated]
    polarity  one per panel, |+>, controls whether the panel's relief is raised (h) or sunk (-h)                     [simulated]
    lamp      2 qubits, 4 light directions; they control the axis along which each facet is measured                 [in the circuit]
    observe   2 qubits, 4 observation axes (gamma on the depth sphere) for the polarity qubits                      [in the circuit]

The state of the simulated register (facets + polarity), exactly:
        |Psi> = 2^(-P/2) sum_s |s>_B (x)_p P_p^{s_p} |psi_f>,      |psi_f> = E (x)_i Ry(tau_i) Rz(phi_i) |0>
with P_p = Z on every facet of panel p ("turn every bump into a hollow": Z maps Bloch (x, y, z) -> (-x, -y, z)), built as H on
each polarity qubit, the facet preparation, and a layer of CZ(B_p, f). E is an optional layer of two-qubit couplings:
    'zz'  exp(-i theta ZZ/2)        diagonal: commutes with every Z string, keeps the bevel exactly flat at the depth sphere's equator
    'xy'  exp(-i theta (XX + YY)/2) parity-preserving exchange (flips two spins): an opt-in experiment, it leaves a residual bevel there
Both leave the interference visibility V_p = <X_Bp> = <psi_f|P_p|psi_f> = prod_{i in p} cos(tau_i) exactly unchanged (tests pin it),
so entanglement between facets can be added without touching the scaling law. With 'zz' only, the whole circuit (single-qubit layer,
commuting diagonal gates, single-qubit measurement rotations) is an IQP circuit with geometry-set angles.

A frame (handoff 4.2.3). L (lamp) and G (observation) are uniform control registers: a frame draws them, the facets are rotated so
that measuring Z measures sigma . l for that facet's light direction l (Lambert's cosine law IS the Born rule: P(lit) = (1 + b.l)/2),
each polarity qubit is rotated to read along the depth-sphere axis (gamma, chi), and the whole simulated register is measured once;
K photons = K shots, all from the one state. The outcome of a polarity qubit is the frame's depth decision (a bump, a hollow, or
something between, by where gamma landed). Everything a frame shows comes out of that one measurement.

Honest status: at ~18 simulated qubits all of this is classically simulable, and it is simulated here with an exact statevector.
L and G are uniform controls read in Z (physical randomness on a QPU, pseudo-random here): equivalent to coin flips unless the lamp
register is read in another basis (`lamp_mode`), in which case light directions interfere. The quantum content is the facet register,
the polarity superposition and the controlled operations between them.
"""
import itertools
from dataclasses import dataclass, field

import numpy as np

from src.quantum import sampler_local as sl

# lamp world index w = bit(L1) + 2 bit(L2); bit 0 means +1 (L1: light from the right, L2: frontal), as in the rest of the repo
WORLDS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
WORLD_NAMES = ["right, frontal", "left, frontal", "right, grazing", "left, grazing"]
# (gamma, chi) of the 4 observe-register values. The reduced state of a polarity qubit is (V, 0, 0) on the depth sphere, so chi picks which
# coherence of the raised and sunk branches a frame samples: cos(chi) reads it in the polarity outcome (P(o=0) = (1 + V sin(gamma) cos(chi))/2), and
# the facets are left in |psi> + e^{i chi} P|psi>, a different bevel signature for each chi. With 'zz' the bevel is exactly flat on the equator at
# chi = 0 and 180 only; at chi = 90 the sin(chi) cross-term leaves a small outcome-dependent residual (about +-0.03 on the bay window), so the
# ring puts its equatorial observation at chi = 180.
LINE_OBSERVATIONS = [(float(np.deg2rad(g)), 0.0) for g in (0, 30, 60, 90)]          # the original line of longitude, chi = 0 throughout
RING_OBSERVATIONS = [(0.0, 0.0), (float(np.deg2rad(60)), 0.0), (float(np.deg2rad(60)), float(np.pi / 2)), (float(np.pi / 2), float(np.pi))]
DEFAULT_OBSERVATIONS = RING_OBSERVATIONS      # decided; between, +X side; between, +Y side (outcome-symmetric); undecided, -X side
OBSERVATION_SETS = dict(line=LINE_OBSERVATIONS, ring=RING_OBSERVATIONS)


def parse_observations(text):
    """'line' | 'ring' | 'g,c;g,c;g,c;g,c' (degrees) -> [(gamma, chi)] in radians. Exactly four: the observe register is two qubits."""
    if text is None:
        return None
    t = str(text).strip().lower()
    if t in OBSERVATION_SETS:
        return list(OBSERVATION_SETS[t])
    try:
        obs = [tuple(float(np.deg2rad(float(v))) for v in pair.split(",")) for pair in t.split(";")]
    except ValueError:
        obs = []
    if len(obs) != 4 or any(len(o) != 2 for o in obs):
        raise ValueError("observations: 'line', 'ring', or four 'gamma,chi' pairs in degrees separated by ';' (the observe register has 2 qubits)")
    return obs
X_, Y_, Z_ = (np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.diag([1.0, -1.0]).astype(complex))


@dataclass
class LightRig:
    """The four light worlds in image coordinates (x right, y down, z toward the viewer): from the right or left, frontal or grazing,
    slightly from above. `vector` is in the world, `panel_local` in a panel's own frame (z = panel normal)."""
    frontal: float = 0.45
    grazing: float = 0.85
    elevation: float = -0.25

    def vector(self, world):
        s1, s2 = WORLDS[world] if isinstance(world, (int, np.integer)) else world
        lx = s1 * (self.frontal if s2 > 0 else self.grazing)
        ly = self.elevation
        return np.array([lx, ly, np.sqrt(max(1.0 - lx * lx - ly * ly, 1e-12))])

    def panel_local(self, world, angle_deg):
        th = np.deg2rad(angle_deg)
        x, y, z = self.vector(world)
        return np.array([x * np.cos(th) - z * np.sin(th), y, x * np.sin(th) + z * np.cos(th)])


@dataclass
class ReliefSpec:
    """Everything that defines the relief circuit; nothing in it is random."""
    tau: np.ndarray
    phi: np.ndarray
    panel_of: np.ndarray
    panel_angles: list                       # degrees, one per panel (side panels are tilted; the Bloch vectors are panel-local)
    edges: list = field(default_factory=list)  # [(i, j, "xy" | "zz", theta)]
    rig: LightRig = field(default_factory=LightRig)
    observations: list = field(default_factory=lambda: list(DEFAULT_OBSERVATIONS))
    facet_names: list = None
    # classical plateau facets (tilt ~0): no qubit, lit by the deterministic Lambert value; appended after the F quantum facets in every draw
    n_classical: int = 0
    classical_panel: list = field(default_factory=list)
    # domain relief (all optional; the defaults reproduce the per-panel relief exactly). `panel_of` is then the facet -> polarity-qubit map:
    n_groups: int = None                              # number of polarity qubits; default one per panel
    facet_panel: np.ndarray = None                    # the panel whose frame holds each facet's light axis; default panel_of
    pol_graph: list = field(default_factory=list)     # [(a, b)] polarity-qubit pairs: the domain graph the Ising couplings live on
    pol_layers: list = field(default_factory=list)    # [dict(zz=[theta per pol_graph edge] or None, rx=[beta per polarity qubit] or None)]
    lamp_lock: object = 0.0                           # coupling theta_d of lamp qubit L1 to polarity qubit d: exp(-i theta_d Z_L1 Z_Bd / 2); scalar = all
    site_order: list = None                           # logical qubits in matrix-product-state site order (a locality hint for the MPS backend only)
    pol_circuit: object = None                        # a Qiskit circuit on the P polarity qubits that REPLACES the H + layers preparation (e.g. one QDrive returned)
    floquet: dict = None                              # dict(steps, theta_zz, theta_x): kicked-Ising steps on the facet register after its preparation, before the polarity attaches
    pol_kinds: list = None                            # "coplanar" | "crease" for every edge of pol_graph (geometry labels, used by the witnesses)
    seam_pairs: list = None                           # facet pairs (i, j) that touch across a crease
    seam_facets: list = None                          # facets on a crease
    leaves: list = None                               # domain relief: the polarity qubits that play the parity game with the lamp (one per plane, at most three)

    def __post_init__(self):
        self.tau = np.asarray(self.tau, float)
        self.phi = np.asarray(self.phi, float)
        self.panel_of = np.asarray(self.panel_of, int)
        if self.facet_panel is not None:
            self.facet_panel = np.asarray(self.facet_panel, int)

    @property
    def F(self):
        return len(self.tau)

    @property
    def P(self):
        """Polarity qubits: one per panel, or one per depth domain when n_groups is given."""
        return len(self.panel_angles) if self.n_groups is None else int(self.n_groups)

    def light_panel(self, i):
        return int(self.panel_of[i] if self.facet_panel is None else self.facet_panel[i])

    def lock_array(self):
        """Per polarity qubit coupling to lamp L1, or None when there is none."""
        lk = np.broadcast_to(np.asarray(self.lamp_lock, float), (self.P,)).copy()
        return lk if np.any(lk != 0) else None

    @property
    def n_sim(self):
        return self.F + self.P

    @property
    def n_full(self):
        return self.F + self.P + 4

    def layout(self):
        F, P = self.F, self.P
        return dict(facets=list(range(F)), polarity=list(range(F, F + P)), lamp=[F + P, F + P + 1], observe=[F + P + 2, F + P + 3],
                    n_sim=F + P, n_full=F + P + 4)

    def facets_of(self, p):
        return [int(i) for i in np.nonzero(self.panel_of == p)[0]]

    def bloch(self):
        return np.stack([np.sin(self.tau) * np.cos(self.phi), np.sin(self.tau) * np.sin(self.phi), np.cos(self.tau)], axis=1)

    def visibility(self, p=None):
        """prod cos(tau) over a panel's facets (or every panel's, as an array): the interference visibility between its depth maps."""
        if p is None:
            return np.array([self.visibility(k) for k in range(self.P)])
        return float(np.prod(np.cos(self.tau[self.panel_of == p])))

    def key(self):
        """A hashable fingerprint, used to cache states."""
        lk = self.lock_array()
        return (self.tau.round(9).tobytes(), self.phi.round(9).tobytes(), self.panel_of.tobytes(), tuple(self.panel_angles),
                tuple((i, j, k, round(float(t), 9)) for i, j, k, t in self.edges), self.rig.frontal, self.rig.grazing, self.rig.elevation,
                tuple((round(g, 9), round(c, 9)) for g, c in self.observations), self.n_classical, tuple(self.classical_panel), self.P,
                tuple(map(tuple, self.pol_graph)), tuple((None if l.get("zz") is None else tuple(np.round(l["zz"], 9)),
                                                         None if l.get("rx") is None else tuple(np.round(l["rx"], 9))) for l in self.pol_layers),
                None if lk is None else tuple(np.round(lk, 9)),
                None if self.facet_panel is None else self.facet_panel.tobytes(),
                None if not self.floquet else tuple(sorted(self.floquet.items())),
                None if self.pol_circuit is None else hash(str(self.pol_circuit.qasm() if hasattr(self.pol_circuit, "qasm") else repr(self.pol_circuit.data))))


def spec_from_facets(fs, scene, entangle=0.0, rig=None, observations=None, coupling="zz", compensate=True):
    """The relief circuit for a FacetSet. `entangle` scales the facet couplings (0 builds the product relief).

    coupling="zz" (default): diagonal phases exp(-i theta ZZ/2), theta = +entangle between facets of one plane and -entangle across a crease,
    the geometry's own sign pattern. Together with the polarity CZs the whole circuit is IQP-shaped (single-qubit layer, commuting diagonal
    gates, single-qubit measurement rotations), preserves the visibility law and the stabilizer, and keeps the bevel EXACTLY flat at the
    equator of the depth sphere for every outcome.
    coupling="xy": parity-preserving exchange exp(-i theta (XX+YY)/2) between coplanar facets of a panel. It keeps the visibility law and the
    stabilizer but NOT the flat equator (it leaves a residual bevel), so it is an experiment, not the default.

    compensate: a diagonal coupling rotates each facet's transverse Bloch vector about z by sum_j atan2(sin(theta_ij) cos(tau_j), cos(theta_ij))
    (exact, because diagonal gates leave every Z population alone), so the preparation azimuths are pre-rotated by the opposite angle and every
    facet's one-body state keeps the azimuth the geometry gave it. Its transverse length shrinks by prod |cos theta + i sin theta cos tau_j|,
    the entanglement's price; its tilt, and so the visibility, is untouched."""
    if coupling not in ("zz", "xy"):
        raise ValueError("coupling must be 'zz' or 'xy'")
    nq = fs.n_quantum
    edges = []
    if entangle:
        for i, j, kind, _ in fs.edges:
            if i >= nq or j >= nq:                                  # classical plateau facets carry no qubit, so no coupling
                continue
            same_panel = fs.facets[i].panel == fs.facets[j].panel
            if coupling == "xy":
                if kind == "coplanar" and same_panel:
                    edges.append((i, j, "xy", float(entangle)))
                elif kind == "crease":
                    edges.append((i, j, "zz", -float(entangle)))
            else:
                edges.append((i, j, "zz", float(entangle) if kind == "coplanar" else -float(entangle)))
    tau, phi = fs.qtau.copy(), fs.qphi.copy()
    if compensate:
        phi = phi - zz_rotation(tau, edges)
    classical = [f.panel for f in fs.facets[nq:]]
    return ReliefSpec(tau, phi, fs.qpanel_of, [p.angle_deg for p in scene.panels], edges, rig or LightRig(),
                      list(observations or DEFAULT_OBSERVATIONS), [f"{f.kind[0]}{f.idx}" for f in fs.facets[:nq]],
                      n_classical=len(classical), classical_panel=classical)


def zz_rotation(tau, edges):
    """Per facet: the azimuth rotation (radians) the diagonal couplings give its transverse Bloch vector, exact for any set of 'zz' edges."""
    rot = np.zeros(len(tau))
    for i, j, kind, th in edges:
        if kind == "zz":
            rot[i] += np.arctan2(np.sin(th) * np.cos(tau[j]), np.cos(th))
            rot[j] += np.arctan2(np.sin(th) * np.cos(tau[i]), np.cos(th))
    return rot


def transverse_shrink(tau, edges):
    """Per facet: prod |cos theta + i sin theta cos tau_j| over its diagonal couplings, the factor by which its transverse Bloch length shrinks."""
    sh = np.ones(len(tau))
    for i, j, kind, th in edges:
        if kind == "zz":
            sh[i] *= np.hypot(np.cos(th), np.sin(th) * np.cos(tau[j]))
            sh[j] *= np.hypot(np.cos(th), np.sin(th) * np.cos(tau[i]))
    return sh


# ------------------------------------------------------------------ exact state
def _apply_2q(psi, U, a, b, n):
    """4x4 unitary U (basis order |bit_a bit_b>, a the more significant) on qubits a and b of an n-qubit flat vector."""
    t = psi.reshape((2,) * n)
    ax, bx = n - 1 - a, n - 1 - b
    t = np.tensordot(U.reshape(2, 2, 2, 2), t, axes=([2, 3], [ax, bx]))
    return np.moveaxis(t, [0, 1], [ax, bx]).reshape(-1)


def _xy_gate(theta):
    c, s = np.cos(theta), -1j * np.sin(theta)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, s, c, 0], [0, 0, 0, 1]], complex)


def facet_state(spec):
    """|psi_f> = E (x) Ry(tau_i) Rz(phi_i)|0>, the raised relief, as a flat 2^F vector (facet i = bit i)."""
    F = spec.F
    vec = np.array([1 + 0j])
    for i in range(F):
        a = np.array([np.cos(spec.tau[i] / 2), np.exp(1j * spec.phi[i]) * np.sin(spec.tau[i] / 2)])
        vec = np.kron(a, vec)                                   # facet i is more significant than 0..i-1
    for i, j, kind, theta in spec.edges:
        if kind == "xy":
            vec = _apply_2q(vec, _xy_gate(theta), i, j, F)
    zz = [(i, j, th) for i, j, kind, th in spec.edges if kind == "zz"]
    if zz:
        bits = (np.arange(2 ** F)[:, None] >> np.arange(F)[None, :]) & 1
        z = 1 - 2 * bits
        phase = sum(-th / 2 * z[:, i] * z[:, j] for i, j, th in zz)
        vec = vec * np.exp(1j * phase)
    fl = spec.floquet
    if fl and fl.get("steps"):
        idx = np.arange(2 ** F)
        z = 1 - 2 * ((idx[:, None] >> np.arange(F)[None, :]) & 1)
        phase = sum(-np.sign(th) * fl["theta_zz"] / 2 * z[:, i] * z[:, j] for i, j, kind, th in spec.edges if kind == "zz")
        kick = _RX(fl["theta_x"])
        for _ in range(int(fl["steps"])):
            vec = vec * np.exp(1j * phase) if not np.isscalar(phase) else vec
            for q in range(F):
                vec = sl.apply_1q(vec, kick, q, F)
    return vec


_RX = lambda b: np.array([[np.cos(b / 2), -1j * np.sin(b / 2)], [-1j * np.sin(b / 2), np.cos(b / 2)]], complex)


def polarity_amplitudes(spec):
    """beta[s]: the amplitudes of the polarity register before it is attached to the facets. |+>^P (uniform 2^(-P/2)) for the per-panel relief;
    for the domain relief, H on every polarity qubit and then the layers: exp(-i theta ZZ / 2) on each edge of the domain graph (Ising) and
    exp(-i beta X / 2) on each qubit (a transverse field), as many layers as the spec has. All of it is a circuit (see reference_circuit)."""
    P = spec.P
    if spec.pol_circuit is not None:                                # little-endian like everything here: polarity qubit d is bit d
        from qiskit.quantum_info import Statevector
        return np.asarray(Statevector(spec.pol_circuit).data, complex)
    beta = np.full(2 ** P, 2.0 ** (-P / 2), complex)
    if not spec.pol_layers:
        return beta
    z = 1 - 2 * ((np.arange(2 ** P)[:, None] >> np.arange(P)[None, :]) & 1)
    for layer in spec.pol_layers:
        if layer.get("zz") is not None:
            for (a, b), th in zip(spec.pol_graph, layer["zz"]):
                beta = beta * np.exp(-0.5j * th * z[:, a] * z[:, b])
        if layer.get("rx") is not None:
            for d, b in enumerate(layer["rx"]):
                if b:
                    beta = sl.apply_1q(beta, _RX(b), d, P)
    return beta


def parity_signs(spec):
    """sign[s, x] = (-1)^(sum_p s_p parity_p(x)): the matrix of CZ(B_p, f) layers, shape (2^P, 2^F)."""
    F, P = spec.F, spec.P
    x = np.arange(2 ** F)
    bits = (x[:, None] >> np.arange(F)[None, :]) & 1
    par = np.stack([bits[:, spec.panel_of == p].sum(axis=1) & 1 for p in range(P)], axis=1)         # (2^F, P)
    s = np.arange(2 ** P)
    sb = (s[:, None] >> np.arange(P)[None, :]) & 1                                                  # (2^P, P)
    return 1 - 2 * ((sb @ par.T) & 1)


def lift_polarity(psi_f, spec):
    """Superpose raised and sunk: Psi[s, x] = beta[s] (-1)^(sum_p s_p parity_p(x)) psi_f[x], with beta = 2^(-P/2) for the per-panel relief.
    This is the polarity preparation followed by CZ(B_p, f) for every facet f of polarity group p, and works for ANY facet state (a Moth
    circuit's included). Returns shape (2^P, 2^F)."""
    return polarity_amplitudes(spec)[:, None] * parity_signs(spec) * np.asarray(psi_f)[None, :]


def full_state(spec, psi_f=None):
    """The simulated register as a flat vector of 2^(F+P) amplitudes (polarity qubits above the facets)."""
    return lift_polarity(facet_state(spec) if psi_f is None else np.asarray(psi_f, complex), spec).reshape(-1)


def _rot(spec_axis):
    return sl.bloch_unitary(spec_axis)


def _apply_axis_rotations(T, n, rotations):
    """rotations: {qubit: 2x2 unitary} on a flat length-2^n vector (or a stack: leading axes are untouched)."""
    flat = T.reshape(-1, 2 ** n)
    out = np.empty_like(flat)
    for r in range(len(flat)):
        v = flat[r]
        for q, U in rotations.items():
            v = sl.apply_1q(v, U, q, n)
        out[r] = v
    return out.reshape(T.shape)


# ------------------------------------------------------------------ frames
@dataclass
class ReliefDraw:
    """One frame: the measurement outcomes of one run of the circuit, K photons deep. Compatible with the older FrameDraw fields."""
    lit: np.ndarray                  # (F,) fraction of the K photons that landed (bit 0), per facet
    lamp: tuple                      # (L1, L2) in +/-1: the light world drawn (lamp read in Z)
    pol: tuple                       # per panel +1 / -1: the polarity qubit's outcome, 0 = raised-ish, 1 = sunk-ish
    K: int
    world: int = 0                   # lamp register value 0..3
    obs: int = 0                     # observe register value
    gamma: float = 0.0               # depth-sphere polar angle of the observation (radians)
    chi: float = 0.0                 # and azimuth
    lean: tuple = ()                 # per panel in [-1, 1]: how raised (+) or sunk (-) the frame reads, (1 - 2 outcome) cos(gamma)
    control: str = "coherent"        # "coherent" | "dephased" | "noise"
    lamp_mode: str = "z"
    shot_ids: np.ndarray = None      # unused, kept for compatibility with the pool-based drawers


def depth_word(lean, tol=0.25):
    if abs(lean) < tol:
        return "undecided"
    return ("raised" if lean > 0 else "sunk") if abs(lean) > 0.8 else ("leaning raised" if lean > 0 else "leaning sunk")


class ReliefState:
    """A relief circuit's state and everything drawn from it. `psi_f` replaces the reference facet state (e.g. the statevector of a
    Moth QDrive circuit); the polarity superposition is always lifted locally."""

    def __init__(self, spec, psi_f=None, label="local reference circuit"):
        self.spec, self.label = spec, label
        self.psi_f = facet_state(spec) if psi_f is None else np.asarray(psi_f, complex)
        if len(self.psi_f) != 2 ** spec.F:
            raise ValueError(f"facet state has {len(self.psi_f)} amplitudes, the spec has {spec.F} facets")
        self.psi_f = self.psi_f / np.linalg.norm(self.psi_f)
        self.beta = polarity_amplitudes(spec)
        self._signs = parity_signs(spec)
        self.Psi = self.beta[:, None] * self._signs * self.psi_f[None, :]   # (2^P, 2^F)
        self._cells, self._cdf, self._deph = {}, {}, {}

    # ---- exact distributions
    def _facet_rot(self, world):
        sp = self.spec
        return {i: sl.bloch_unitary(sp.rig.panel_local(world, sp.panel_angles[sp.light_panel(i)])) for i in range(sp.F)}

    def _pol_rot(self, g):
        gam, chi = self.spec.observations[g]
        a = (np.sin(gam) * np.cos(chi), np.sin(gam) * np.sin(chi), np.cos(gam))
        return sl.bloch_unitary(a)

    def _lamp_mode_matrix(self, mode):
        if mode == "z":
            return None
        U = {"x": sl.H, "y": sl.SDG_H}[mode]
        return np.kron(U, U)                                               # lamp qubit 1 is the more significant factor of world = b1 + 2 b2? see below

    def cell(self, g, lamp_mode="z", control="coherent"):
        """probs[w, o, x] = P(lamp outcome w, polarity outcomes o, facet bits x | observation g), float64, summing to 1.
        w and o are the register values (bit p of o is polarity qubit p's bit; bit 0 = raised-ish); x is the facet bits (bit i = facet i)."""
        key = (g, lamp_mode, control)
        if key in self._cells:
            return self._cells[key]
        sp = self.spec
        F, P = sp.F, sp.P
        rot_b = self._pol_rot(g)
        n = F + P
        if control == "dephased":
            probs = self._cell_dephased(g)
        else:
            lock = sp.lock_array()
            T = np.empty((4, 2 ** n), complex)
            if lock is None:
                Psi = self.Psi.reshape(-1)
                for p in range(P):
                    Psi = sl.apply_1q(Psi, rot_b, F + p, n)                    # observe the depth sphere
            for w in range(4):
                if lock is not None:                                           # lamp L1 is coupled to the polarity qubits: exp(-i theta Z_L1 Z_B / 2),
                    Psi = (self.Psi * self._lock_phase(w, lock)[:, None]).reshape(-1)    # which for L1 = z is a z-rotation of every polarity qubit
                    for p in range(P):
                        Psi = sl.apply_1q(Psi, rot_b, F + p, n)
                v = Psi.copy()
                for i, U in self._facet_rot(w).items():
                    v = sl.apply_1q(v, U, i, n)                           # light axis for facet i in world w
                T[w] = v / 2.0                                              # lamp |+>|+> controls the rotation: amplitude 1/2 per world
            if lamp_mode != "z":
                T = self._lamp_readout(T, lamp_mode)
            probs = (np.abs(T) ** 2).reshape(4, 2 ** P, 2 ** F)
        probs = probs / probs.sum()
        self._cells[key] = probs
        return probs

    def _lock_phase(self, w, lock):
        """(2^P,) phase of exp(-i theta_d Z_L1 Z_Bd / 2) on the polarity basis state s, for the lamp world w (Z_L1 = WORLDS[w][0])."""
        P = self.spec.P
        z = 1 - 2 * ((np.arange(2 ** P)[:, None] >> np.arange(P)[None, :]) & 1)
        return np.exp(-0.5j * WORLDS[w][0] * (z @ lock))

    @staticmethod
    def _lamp_readout(T, mode):
        """Read the two lamp qubits in X or Y instead of Z: world index w = b1 + 2 b2, so qubit L1 is the fast index."""
        U = {"x": sl.H, "y": sl.SDG_H}[mode]
        M = np.kron(U, U)                                                  # (b2, b1) ordering: kron(U_L2, U_L1) with L1 the least significant
        return np.tensordot(M, T, axes=([1], [0]))

    def _cell_dephased(self, g):
        """Control B: every polarity qubit is replaced by its classical mixture (it is read in Z first, then observed along the axis).
        Same one-body statistics as the coherent state, no cross terms between the raised and sunk depth maps."""
        sp = self.spec
        F, P = sp.F, sp.P
        rot_b = self._pol_rot(g)
        # P(outcome o | polarity value s) for one qubit: |<o| R |s>|^2
        r = np.abs(rot_b) ** 2                                             # r[o, s]
        probs = np.zeros((4, 2 ** P, 2 ** F))
        for s in range(2 ** P):
            sign = self._signs[s] * self.psi_f                              # branch s: P_s |psi_f>
            po = np.ones(2 ** P)
            for o in range(2 ** P):
                for p in range(P):
                    po[o] *= r[(o >> p) & 1, (s >> p) & 1]
            for w in range(4):
                v = sign.copy()
                for i, U in self._facet_rot(w).items():
                    v = sl.apply_1q(v, U, i, F)
                px = np.abs(v) ** 2
                px /= px.sum()
                probs[w, :, :] += 0.25 * abs(self.beta[s]) ** 2 * po[:, None] * px[None, :]
        return probs

    def marginals(self, g, w=None, o=None, lamp_mode="z", control="coherent"):
        """Exact per-facet P(lit) (bit 0), conditioned on the lamp outcome w and the polarity outcomes o when given."""
        pr = self.cell(g, lamp_mode, control)
        if w is not None:
            pr = pr[w:w + 1]
        if o is not None:
            pr = pr[:, o:o + 1]
        pj = pr.sum(axis=(0, 1))
        pj = pj / pj.sum()
        bits = (np.arange(2 ** self.spec.F)[:, None] >> np.arange(self.spec.F)[None, :]) & 1
        return pj @ (1 - bits)

    def polarity_probs(self, g, control="coherent"):
        return self.cell(g, "z", control).sum(axis=(0, 2))

    # ---- drawing
    def _cond_cdf(self, g, w, o, lamp_mode, control):
        key = (g, w, o, lamp_mode, control)
        c = self._cdf.get(key)
        if c is None:
            p = self.cell(g, lamp_mode, control)[w, o]
            tot = p.sum()
            c = self._cdf[key] = (np.cumsum(p / tot), float(tot))
        return c

    def draw(self, K, rng, g=None, control="coherent", lamp_mode="z", world=None):
        """One frame. g (the observe register) and world (the lamp register) are drawn uniformly unless given (tests and experiments
        only: the show never sets them). control: 'coherent' (the state), 'dephased' (control B) or 'noise' (control A, fair coins)."""
        sp = self.spec
        g = int(rng.integers(len(sp.observations))) if g is None else int(g)
        gam, chi = sp.observations[g]
        if control == "noise":
            w = int(rng.integers(4)) if world is None else int(world)
            o = int(rng.integers(2 ** sp.P))
            lit = rng.binomial(K, 0.5, sp.F) / K
        else:
            pr = self.cell(g, lamp_mode, control)
            pw = pr.sum(axis=(1, 2))
            w = int(rng.choice(4, p=pw / pw.sum())) if world is None else int(world)
            po = pr[w].sum(axis=1)
            o = int(rng.choice(2 ** sp.P, p=po / po.sum()))
            cdf, _ = self._cond_cdf(g, w, o, lamp_mode, control)
            x = np.minimum(np.searchsorted(cdf, rng.random(K)), len(cdf) - 1)
            lit = (((x[:, None] >> np.arange(sp.F)[None, :]) & 1) == 0).mean(axis=0)
        if sp.n_classical:
            lit = np.concatenate([lit, self._classical_lit(K, rng, w, lamp_mode, control)])
        pol = tuple(1 - 2 * ((o >> p) & 1) for p in range(sp.P))
        lean = tuple(float(pv * np.cos(gam)) for pv in pol)
        return ReliefDraw(lit, WORLDS[w] if lamp_mode == "z" else (0, 0), pol, int(K), w, g, float(gam), float(chi), lean, control, lamp_mode)

    def _classical_lit(self, K, rng, w, lamp_mode, control):
        """The plateau facets (tilt ~0, no qubit): K photons at the Lambert value of the flat panel, (1 + l_z)/2 in the panel's frame. They do
        not take part in the lamp register's interference, so with the lamp read in X they use the value averaged over the four lights."""
        sp = self.spec
        out = np.empty(sp.n_classical)
        for c, panel in enumerate(sp.classical_panel):
            if control == "noise":
                p = 0.5
            elif lamp_mode == "z":
                p = (1 + sp.rig.panel_local(w, sp.panel_angles[panel])[2]) / 2
            else:
                p = float(np.mean([(1 + sp.rig.panel_local(k, sp.panel_angles[panel])[2]) / 2 for k in range(4)]))
            out[c] = rng.binomial(K, p) / K
        return out


# ------------------------------------------------------------------ qiskit circuits (hardware-ready, cross-checked in tests)
def facet_block(spec, edges=None):
    """The facet register's preparation as a Qiskit circuit on F qubits: Ry(tau) Rz(phi) on every facet, the couplings (`edges` replaces spec.edges, e.g. a
    prefix of them), then the Floquet steps. Its statevector is facet_state(spec) up to a global phase."""
    from qiskit import QuantumCircuit
    F = spec.F
    edges = spec.edges if edges is None else edges
    qc = QuantumCircuit(F, name="facets")
    for i in range(F):
        qc.ry(float(spec.tau[i]), i)
        qc.rz(float(spec.phi[i]), i)
    for i, j, kind, th in edges:
        if kind == "xy":
            qc.rxx(float(th), i, j)
            qc.ryy(float(th), i, j)
    for i, j, kind, th in edges:
        if kind == "zz":
            qc.rzz(float(th), i, j)
    fl = spec.floquet
    if fl and fl.get("steps"):                                      # kicked-Ising dynamics on the facet graph: entanglement grows with every step
        for _ in range(int(fl["steps"])):
            for i, j, kind, th in edges:
                if kind == "zz":
                    qc.rzz(float(np.sign(th) * fl["theta_zz"]), i, j)
            for i in range(F):
                qc.rx(float(fl["theta_x"]), i)
    return qc


def reference_circuit(spec, measure=False, facet_circuit=None):
    """The simulated register as a Qiskit circuit (F facet qubits, then P polarity qubits): H on each polarity qubit, Ry(tau) Rz(phi) on each
    facet, the couplings, then CZ(B_p, f) for every facet of panel p. Its statevector equals full_state(spec) up to a global phase.
    `facet_circuit` (F qubits, no measurements, e.g. one QDrive returned) replaces the facet block; the polarity superposition is lifted on top as ever."""
    from qiskit import QuantumCircuit
    F, P = spec.F, spec.P
    qc = QuantumCircuit(F + P, name="superposed_relief")
    if spec.pol_circuit is not None:
        qc.compose(spec.pol_circuit, qubits=list(range(F, F + P)), inplace=True)
    else:
        for p in range(P):
            qc.h(F + p)
        for layer in spec.pol_layers:                               # the domain relief's Ising layers on the polarity register
            if layer.get("zz") is not None:
                for (a, b), th in zip(spec.pol_graph, layer["zz"]):
                    qc.rzz(float(th), F + a, F + b)
            if layer.get("rx") is not None:
                for d, b in enumerate(layer["rx"]):
                    if b:
                        qc.rx(float(b), F + d)
    if facet_circuit is not None and facet_circuit.num_qubits != F:
        raise ValueError(f"the facet circuit has {facet_circuit.num_qubits} qubits, the facet register has {F}")
    qc.compose(facet_block(spec) if facet_circuit is None else facet_circuit, qubits=list(range(F)), inplace=True)
    for i in range(F):
        qc.cz(F + int(spec.panel_of[i]), i)
    if measure:
        qc.measure_all()
    return qc


def full_circuit(spec, lamp_mode="z", facet_circuit=None):
    """The whole performance circuit, ready for a QPU or Aer: relief register, lamp register (H, then per-world controlled rotations of every
    facet to its light axis), observe register (H, then per-value controlled rotations of every polarity qubit to its depth-sphere axis),
    and measurement of everything into registers facets / pol / lamp / obs."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import RYGate, RZGate
    F, P = spec.F, spec.P
    fq, pq, lq, oq = QuantumRegister(F, "f"), QuantumRegister(P, "b"), QuantumRegister(2, "lamp"), QuantumRegister(2, "obs")
    cf, cp, cl, co = ClassicalRegister(F, "cf"), ClassicalRegister(P, "cb"), ClassicalRegister(2, "cl"), ClassicalRegister(2, "co")
    qc = QuantumCircuit(fq, pq, lq, oq, cf, cp, cl, co, name="standing_light_v2")
    qc.compose(reference_circuit(spec, facet_circuit=facet_circuit), qubits=list(fq) + list(pq), inplace=True)
    qc.h(lq), qc.h(oq)
    lock = spec.lock_array()
    if lock is not None:                                            # lamp L1 coupled to the polarity qubits: the light side and the depth are entangled
        for d in range(P):
            if lock[d]:
                qc.rzz(float(lock[d]), lq[0], pq[d])
    for g, (gam, chi) in enumerate(spec.observations):
        if abs(chi) > 1e-12 or abs(gam) > 1e-12:
            for p in range(P):
                if abs(chi) > 1e-12:
                    qc.append(RZGate(-chi).control(2, ctrl_state=g), [*oq, pq[p]])
                if abs(gam) > 1e-12:
                    qc.append(RYGate(-gam).control(2, ctrl_state=g), [*oq, pq[p]])
    for w in range(4):
        for i in range(F):
            a = spec.rig.panel_local(w, spec.panel_angles[spec.light_panel(i)])
            th, ph = float(np.arccos(np.clip(a[2], -1, 1))), float(np.arctan2(a[1], a[0]))
            qc.append(RZGate(-ph).control(2, ctrl_state=w), [*lq, fq[i]])
            qc.append(RYGate(-th).control(2, ctrl_state=w), [*lq, fq[i]])
    if lamp_mode != "z":
        for q in lq:
            if lamp_mode == "x":
                qc.h(q)
            else:
                qc.sdg(q), qc.h(q)
    qc.measure(fq, cf), qc.measure(pq, cp), qc.measure(lq, cl), qc.measure(oq, co)
    return qc


def statevector_of(qc):
    """Final statevector of a circuit without measurements (numpy flat vector, little-endian)."""
    from qiskit.quantum_info import Statevector
    return np.asarray(Statevector(qc).data, complex)


def equal_up_to_phase(a, b, tol=1e-9):
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    return abs(abs(np.vdot(a, b)) - np.linalg.norm(a) * np.linalg.norm(b)) < tol
