"""Witnesses for Superposed Relief: what makes the flat-looking, undecided panel not flat (v2 handoff section 2.8).

An undecided panel (observation axis gamma = 90 deg) shows no bevel at all: the first-order shading is symmetric. The evidence that
both depth maps are present lives in joint statistics, and this module measures it, with provenance. Everything is computed from a
state (exact, or sampled the way a device would); a state with no non-classical signature is reported as such.

The measurements, for panel p with polarity qubit B_p and facets F_p:
  visibility   V_p = <X_Bp>. For the reference state V_p = <psi_f|P_p|psi_f> = prod_{i in F_p} cos(tau_i): the amplitude with which the
               raised and sunk depth maps interfere. Dephased depth gives 0, independent noise 0.
  stabilizer   S_p = <X_Bp (x) Z^{F_p}> = 1 EXACTLY for the ideal state (and for any parity-preserving coupling between facets): the
               polarity qubit read in X and the parity of the facets read in Z are perfectly correlated, although each alone is only V_p.
               One measurement setting (B in X, facets in Z); dephased depth gives 0.
  fidelity     F = |<Psi_ref|rho|Psi_ref>| across the cut B_p | rest. For any state separable across that cut F <= lambda_max^2 = (1 + V_p)/2
               (the largest eigenvalue of the reduced state of B_p), so F > (1 + V_p)/2 CERTIFIES entanglement between the polarity
               qubit and the rest (Bourennane et al. witness). F is the probability of reading all zeros after un-computing the reference
               circuit, so a device can measure it with one more circuit. The dephased control has F = 2^-P, independent noise 2^-n.
  CHSH         exact best Bell value of any qubit pair over all settings (Horodecki), for context. At the tilts used here the pair
               (B_p, facet) does not violate it (the other facets act as an environment); that is reported, not hidden.
"""
import itertools

import numpy as np

from src.quantum import complementary as comp
from src.quantum import sampler_local as sl
from src.quantum.relief_state import ReliefState, full_state

LOCAL_BOUND_CHSH = 2.0


# ------------------------------------------------------------------ exact quantities from a state vector
def parity_visibility(psi, spec):
    """[<X_Bp>] per panel from the flat (2^(F+P)) state."""
    return np.array([comp.pauli_expectation(psi, spec.n_sim, {spec.F + p: "x"}) for p in range(spec.P)])


def stabilizers(psi, spec):
    """[<X_Bp Z^{F_p}>] per panel."""
    out = []
    for p in range(spec.P):
        ops = {spec.F + p: "x", **{i: "z" for i in spec.facets_of(p)}}
        out.append(comp.pauli_expectation(psi, spec.n_sim, ops))
    return np.array(out)


def fidelity(psi, psi_ref):
    return float(abs(np.vdot(psi_ref, psi)) ** 2 / (np.vdot(psi, psi).real * np.vdot(psi_ref, psi_ref).real))


def entanglement_bound(V):
    """lambda_max^2 across the cut B_p | rest: a state with fidelity above this to the reference cannot be separable across that cut."""
    return (1.0 + abs(V)) / 2.0


def pair_chsh(psi, spec):
    """Exact best CHSH value over every (polarity, facet) and (facet, facet) pair: [(kind, a, b, S_max)], most non-classical first."""
    n = spec.n_sim
    rows = []
    for a, b in itertools.combinations(range(n), 2):
        T, _, _ = comp.correlation_matrix(comp.reduced_pair(psi, n, a, b))
        kind = "polarity-facet" if (a >= spec.F) != (b >= spec.F) else ("polarity-polarity" if a >= spec.F else "facet-facet")
        rows.append((kind, a, b, comp.max_chsh(T)))
    return sorted(rows, key=lambda r: -r[3])


# ------------------------------------------------------------------ a witness run as a device would do it
def sample_witness_run(state, rng, shots=20000, control="coherent"):
    """Measure every polarity qubit in X and every facet in Z (one setting, `shots` runs) and estimate, per panel, V_p, the facet parity
    <Z^{F_p}> and the stabilizer. control: 'coherent' (the state), 'dephased' (polarity classical), 'noise' (fair coins).
    Returns dict of arrays with standard errors."""
    sp = state.spec
    F, P = sp.F, sp.P
    n = sp.n_sim
    if control == "noise":
        xs = rng.choice([-1, 1], (shots, P))
        zs = rng.choice([-1, 1], (shots, F))
    else:
        if control == "coherent" and hasattr(state, "measure_witness"):      # a state that is a circuit executed elsewhere (Aer): its own measurement, not numpy's
            xs, zs = state.measure_witness(shots, rng)
        elif control == "coherent":
            psi = state.Psi.reshape(-1).copy()
            for p in range(P):
                psi = sl.apply_1q(psi, sl.H, F + p, n)
            pr = np.abs(psi) ** 2
            idx = rng.choice(len(pr), size=shots, p=pr / pr.sum())
            xs = sl.spins(idx, list(range(F, F + P)))
            zs = sl.spins(idx, list(range(F)))
        else:                                                    # dephased: B is a classical bit, so reading it in X is a fair coin
            pf = np.abs(state.psi_f) ** 2
            idx = rng.choice(len(pf), size=shots, p=pf / pf.sum())
            zs = sl.spins(idx, list(range(F)))
            xs = rng.choice([-1, 1], (shots, P))
    par = np.stack([zs[:, sp.facets_of(p)].prod(axis=1) for p in range(P)], axis=1)
    stab = xs * par

    def est(a):
        return a.mean(axis=0), a.std(axis=0) / np.sqrt(len(a))
    (vis, vis_se), (par_m, par_se), (st_m, st_se) = est(xs), est(par), est(stab)
    return dict(shots=shots, control=control, visibility=vis, visibility_se=vis_se, facet_parity=par_m, facet_parity_se=par_se,
                stabilizer=st_m, stabilizer_se=st_se)


# ------------------------------------------------------------------ the certificate shown in the science panel
def certificate(state, rng=None, shots=20000, chsh=True):
    """A list of per-panel certificates for `state` (a ReliefState) with the provenance of the state, for the coherent state and for the
    two controls. Each entry says plainly whether the entanglement witness is violated.

    The exact numbers come from the statevector; `sampled` repeats the visibility and stabilizer by sampling with error bars. For the
    controls the fidelity is analytic (dephased: 2^-P, independent noise: 2^-n_sim) because they are mixtures, not states."""
    rng = rng or np.random.default_rng(0)
    sp = state.spec
    psi = state.Psi.reshape(-1)
    ref = full_state(sp)
    F_coh = fidelity(psi, ref)
    V = parity_visibility(psi, sp)
    S = stabilizers(psi, sp)
    out = dict(provenance=state.label, n_qubits=sp.n_sim, panels=[], controls={},
               sampled_by=f"the circuit executed on {state.executed_by}" if hasattr(state, "measure_witness") else "numpy sampling of the reference state")
    for p in range(sp.P):
        bound = entanglement_bound(sp.visibility(p))
        out["panels"].append(dict(panel=p, visibility=float(V[p]), visibility_product_law=sp.visibility(p), stabilizer=float(S[p]),
                                  fidelity=F_coh, bound=bound, entangled=bool(F_coh > bound + 1e-9)))
    sampled = {c: sample_witness_run(state, rng, shots, c) for c in ("coherent", "dephased", "noise")}
    out["sampled"] = {c: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for c, r in sampled.items()}
    for c, f in (("dephased", 2.0 ** -sp.P), ("noise", 2.0 ** -sp.n_sim)):
        out["controls"][c] = dict(fidelity=f, panels=[dict(panel=p, bound=entanglement_bound(sp.visibility(p)), entangled=bool(f > entanglement_bound(sp.visibility(p))))
                                                      for p in range(sp.P)])
    if chsh:
        rows = pair_chsh(psi, sp)
        best = {}
        for kind, a, b, s in rows:
            best.setdefault(kind, (a, b, s))
        out["chsh_best"] = {k: dict(a=a, b=b, S_max=float(s), violates=bool(s > LOCAL_BOUND_CHSH + 1e-9)) for k, (a, b, s) in best.items()}
    return out


def describe_certificate(cert):
    """Plain-language lines for the operator's science panel."""
    lines = [f"state: {cert['provenance']} ({cert['n_qubits']} simulated qubits, exact statevector)"]
    if cert.get("sampled_by") and not cert["sampled_by"].startswith("numpy"):
        lines.append(f"sampled quantities (visibility, stabilizer with error bars) were measured by {cert['sampled_by']}; the exact values are the reference statevector's")
    for r in cert["panels"]:
        verdict = "ENTANGLED (witness violated)" if r["entangled"] else "no entanglement certified"
        lines.append(f"panel {r['panel']}: visibility <X_B> = {r['visibility']:.3f} (product law {r['visibility_product_law']:.3f}); stabilizer <X_B Z^F> = "
                     f"{r['stabilizer']:.3f}; fidelity {r['fidelity']:.3f} vs bound {r['bound']:.3f}: {verdict}")
    for c, e in cert["controls"].items():
        ok = any(r["entangled"] for r in e["panels"])
        lines.append(f"control {c}: fidelity {e['fidelity']:.3g}: " + ("entangled?!" if ok else "not violated, as it must be for a classical mixture"))
    for kind, r in cert.get("chsh_best", {}).items():
        lines.append(f"best CHSH over {kind} pairs: {r['S_max']:.3f} (classical bound 2): " + ("violated" if r["violates"] else "not violated"))
    return lines


# ------------------------------------------------------------------ scaling law (handoff section 5)
def visibility_law(tau):
    """prod cos(tau_i) and its Gaussian approximation exp(-sum tau_i^2 / 2)."""
    tau = np.asarray(tau, float)
    return float(np.prod(np.cos(tau))), float(np.exp(-(tau ** 2).sum() / 2))


def scaled_tilts(kappa, N):
    """Equal tilts that hold sum tau^2 = kappa^2 as N grows: tau = kappa / sqrt(N)."""
    return np.full(N, kappa / np.sqrt(N))


def visibility_table(kappa=0.99, Ns=(8, 12, 16, 18, 50, 200, 1000), tau_fixed=0.35):
    """Rows (N, V with fixed tau, V with tau = kappa/sqrt(N)); the second column is flat, the first collapses."""
    return [(int(N), float(np.cos(tau_fixed) ** N), visibility_law(scaled_tilts(kappa, N))[0]) for N in Ns]
