#!/usr/bin/env python3
"""Reference simulation for Superposed Relief (numpy only, exact statevector, <= ~22 qubits).

Checks, with known answers, the claims the handoff relies on:
  1. Lambert's cosine law is the Born rule: P(lit | facet normal n, light l) = (1 + n.l) / 2.
  2. Flipping every bump to a hollow is the parity operator P = Z^(x)N on the facets (Z maps Bloch (x,y,z) -> (-x,-y,z)).
  3. Superposed relief = (|0>_B |psi> + |1>_B P|psi>)/sqrt2 = CZ(B, every facet) applied to |+>_B |psi>.
     Branch visibility <X_B> = <psi|P|psi> = prod(cos tau_i) for a product relief; it collapses as N grows unless sum(tau_i^2) is held fixed.
  4. Observation axis gamma on B slides the facets from "decided raised" through "undecided" to "decided sunk"; bevel contrast is monotone, odd about gamma=90 deg, zero there.
Run: python superposed_relief_refsim.py
"""
import numpy as np

def ket(theta, phi=0.0):                       # Bloch polar theta, azimuth phi
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])

def kron_all(vs):
    out = np.array([1 + 0j])
    for v in vs: out = np.kron(out, v)
    return out

def facet_rotation_to_axis(l):                 # unitary U so that measuring Z after U measures l.sigma
    th, ph = np.arccos(l[2]), np.arctan2(l[1], l[0])
    ry = np.array([[np.cos(-th/2), -np.sin(-th/2)], [np.sin(-th/2), np.cos(-th/2)]], complex)
    return ry @ np.diag([np.exp(1j*ph/2), np.exp(-1j*ph/2)])

def build(tau, phi):                           # product raised-relief psi_i = Ry(tau_i) Rz(phi_i)|0>, i.e. Bloch (sin t cos p, sin t sin p, cos t)
    return kron_all([ket(t, p) for t, p in zip(tau, phi)])

def parity_op_apply(psi, n):                   # P = Z^(x)n on a flat statevector (qubit 0 most significant)
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    return psi * (1 - 2 * (bits.sum(1) % 2))

def superposed(psi, n):                        # (|0>|psi> + |1>P|psi>)/sqrt2, B = leading qubit
    return np.concatenate([psi, parity_op_apply(psi, n)]) / np.sqrt(2)

def facet_lit_probs(state, n_total, facet_axes, B_axis_gamma=None):
    """Per-facet P(lit)=P(outcome 0) after rotating each facet to its light axis, and B read along (sin g, 0, cos g)."""
    t = state.reshape((2,) * n_total).astype(complex)
    for q in range(1, n_total):
        U = facet_rotation_to_axis(facet_axes[q - 1])
        t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q)
    if B_axis_gamma is not None:
        UB = facet_rotation_to_axis(np.array([np.sin(B_axis_gamma), 0, np.cos(B_axis_gamma)]))
        t = np.moveaxis(np.tensordot(UB, t, axes=([1], [0])), 0, 0)
    p = np.abs(t) ** 2
    return t, p

def facet_marginals(p, n_total, b_outcome):
    pb = p[b_outcome]; pb = pb / pb.sum()
    return np.array([pb.sum(axis=tuple(a for a in range(n_total - 1) if a != q)) [0] for q in range(n_total - 1)])

if __name__ == "__main__":
    # --- 1. Lambert = Born
    n = np.array([0.3, -0.2, 0.0]); n[2] = np.sqrt(1 - n[0]**2 - n[1]**2); l = np.array([0.6, 0.0, 0.8])
    th, ph = np.arccos(n[2]), np.arctan2(n[1], n[0])
    U = facet_rotation_to_axis(l); p0 = abs((U @ ket(th, ph))[0]) ** 2
    print(f"1. Born P(lit) = {p0:.6f}   (1 + n.l)/2 = {(1 + n @ l) / 2:.6f}")
    # --- 2. Z flips slope direction
    z = np.diag([1, -1]); b = lambda k: np.real([k.conj() @ s @ k for s in (np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), z)])
    k = ket(th, ph); print(f"2. Bloch(psi) = {np.round(b(k),4)}   Bloch(Z psi) = {np.round(b(z @ k),4)}  (x,y flip, z kept)")
    # --- 3. visibility and the sqrt(N) rule
    N = 8; tau = np.full(N, 0.35); phi = np.zeros(N); psi = build(tau, phi)
    sup = superposed(psi, N); XB = 2 * np.real(np.vdot(sup[:2**N], sup[2**N:]))   # <X_B> = 2 Re<0-branch|1-branch>
    print(f"3. <X_B> = {XB:.5f}   prod cos(tau) = {np.prod(np.cos(tau)):.5f}   <psi|P|psi> = {np.real(np.vdot(psi, parity_op_apply(psi, N))):.5f}")
    for n_f in (8, 18, 50, 200, 1000):
        kappa2 = 8 * 0.35**2                                                       # sum tau^2 fixed at the N=8 value
        print(f"   N={n_f:5d}: tau=0.35 each -> visibility {np.cos(0.35)**n_f:.2e};  tau=kappa/sqrt(N) (sum tau^2={kappa2:.2f}) -> visibility {np.cos(np.sqrt(kappa2 / n_f))**n_f:.4f}")
    # --- 4. observation axis sweeps raised -> undecided -> sunk
    tau = np.array([.55, .55, .35, .35, .35, .35, .55, .55]); phi = np.array([0, 0, 0, 0, np.pi, np.pi, np.pi, np.pi])   # slope azimuth: left facets 0, right facets pi
    N = 8; sup = superposed(build(tau, phi), N); beta = np.deg2rad(50)
    axes = [np.array([np.sin(beta), 0, np.cos(beta)])] * N                                   # one light, from +x
    print("4. bevel contrast (lit-prob of facets facing the light minus facing away), B outcome 0, light from +x")
    for g in np.linspace(0, np.pi, 7):
        t, p = facet_lit_probs(sup, N + 1, axes, g)
        m = facet_marginals(p, N + 1, 0)
        print(f"   gamma={np.degrees(g):5.0f} deg  contrast = {m[:4].mean() - m[4:].mean():+.3f}   P(outcome0) = {p[0].sum():.3f}")
