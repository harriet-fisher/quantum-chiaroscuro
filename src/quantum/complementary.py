#!/usr/bin/env python3
"""Polarity-complementary experiment (spec §3.6, §9.4; handoff §3.2, §9.4).   STATUS: UNTESTED against a Moth circuit.

    python -m src.quantum.complementary [--out runs/complementary] [--circuit QDRIVE.qasm] [--sweep]

The claim to test (spec §3.6): lit/dark is read in Z and raised/hollow in X, so the room cannot have a definite light state and a
definite relief state at once, and with entanglement in both bases no assignment of pre-existing classical values reproduces the
joint statistics. graph-v1 only returns Z-basis bitstrings, so this runs on a state simulated LOCALLY: a QDrive circuit when one
exists (--circuit), otherwise the stand-in below. As of this writing QDrive has returned no circuit for this scene (small lab jobs
have run, see docs/qdrive-field-notes.md), so everything below uses the stand-in and says so in every output.

What the stand-in is.  Take the classical mock's distribution as a pure state (src.quantum.sampler_local.oracle_state without its
polarity chain): lamps uniform, patches Boltzmann given the lamps. Put each panel's polarity qubit in |+> and apply a
controlled phase CP(phi) between it and the lamp that governs that panel's light (its hub coupling decides: the side panels
follow L1, the flat centre follows L2). Controlled phases are diagonal, so the Z statistics of every patch and lamp are EXACTLY
those of the Z-only baseline: the lighting stays as coherent as before. What changes is only what the polarity qubits do:

    read in Z   a fair coin, independent of everything                      (polarity carries no information)
    read in X   +1 when its lamp bit is 0, else +1 with probability cos^2(phi/2): at phi = pi, locked to the light

So the relief sign lives in a phase: the same circuit shows incoherent relief if you forget the rotation and light-locked relief if
you read polarity in X. The state is entangled across the lamp-polarity pairs, which is what the witness measures.

The witness (a Bell test that needs no assumptions about the circuit).  For any two qubits the best CHSH value over ALL
measurement settings follows from their 3x3 correlation matrix T (Horodecki): S_max = 2 sqrt(s1^2 + s2^2), s1, s2 its two
largest singular values; S_max > 2 means no classical (local hidden variable) account of that pair exists. scan_pairs() computes
it exactly for every pair, and chsh_estimate() repeats it by sampling with the optimal settings (what a real run would do, with
error bars). Diagonal (Z-only) statistics cannot violate it. Note the distinction: the Z-only baseline's measured STATISTICS are classically
reproducible, but the sqrt(P) oracle STATE behind them is not classical in other bases (one patch pair in it already scores 2.35,
and its polarity pair sits exactly on the boundary at 2.000), so S_max of the baseline is reported for contrast, not as a "classical" label.

What the first run found (stand-in, 19 qubits).  Polarity read in X is locked to its lamp (correlation 1.00) and the lighting is
unchanged, but the CHSH violation survives only for pairs whose lamp the patches have NOT already recorded: the centre panel's
partner L2 gives S_max = 2.74, while the side panels' partner L1, which the patches follow strongly (kh = 1.6), gives exactly 2.00,
because the patch register holds a copy of L1 and that decoheres the pair. Strong lighting coherence and strong polarity
complementarity compete; --sweep draws the curve. S_max <= 2 for a pair does not make the whole state classical (the pair is
entangled with the patches, not with a classical record), it only means this pair test cannot show it.

Honest limits.  Exact numbers come from a statevector, shots only add noise. A violation says the STATE is non-classical; it is
not a quantum-advantage claim (20 qubits is classically simulable, as is this whole script). The audience cannot see the
violation in any single frame: it lives in the statistics across measurement settings. And the experiment says nothing about what
QDrive will return for these targets until a real circuit goes through --circuit.
"""
import argparse
import itertools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline.coherence import coherence_metrics
from src.capture.labels import DEFAULT_CALIB_DIR, bay_window_labels, fill_defaults, load_labels, validate
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import allocate
from src.quantum import sampler_local as sl
from src.quantum.sampler_mock import Sampler
from src.texture.frames import FrameGenerator

STATUS = "UNTESTED: local stand-in state, no Moth circuit has been run through this"
COUPLINGS = ("hub", "lamp1", "lamp2", "none")
PAULI = {"x": np.array([[0, 1], [1, 0]], complex), "y": np.array([[0, -1j], [1j, 0]], complex), "z": np.diag([1.0, -1.0]).astype(complex)}
CHSH_CLASSICAL, CHSH_TSIRELSON = 2.0, 2 * np.sqrt(2)
WORLDS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]


# ------------------------------------------------------------------ the stand-in state
def z_structure(sampler, layout):
    """Amplitudes sqrt(P) over (patches, lamps): lamps uniform, patches Boltzmann given the lamps. Length 2^(n_patches + 2), qubit
    order of graph.allocate_qubits (patches 0..n-1, then L1, L2)."""
    n = sampler.n
    if layout["patches"] != list(range(n)) or layout["lamps"] != [n, n + 1]:
        raise ValueError("expects patches on qubits 0..n-1 followed by the two lamps")
    S = sl.spins(np.arange(2 ** n), list(range(n))).astype(float)
    pair = 0.5 * np.einsum("ij,ij->i", S @ sampler.J, S)
    zp = np.zeros(2 ** (n + 2))
    base = np.arange(2 ** n)
    for L1, L2 in itertools.product([-1, 1], repeat=2):
        w = np.exp(sampler.beta * (S @ (sampler.hub1 * L1 + sampler.hub2 * L2) + pair))
        w /= w.sum()
        zp[base | (((1 - L1) // 2) << n) | (((1 - L2) // 2) << (n + 1))] = 0.5 * np.sqrt(w)
    return zp


def partners(sampler, layout, coupling="hub"):
    """For each panel, the qubits its polarity qubit is phase-coupled to. 'hub': the lamp with the larger mean hub coupling over the
    panel's patches (geometry decides which light governs the panel); 'lamp1' / 'lamp2': that lamp for all panels ('lamp2', the frontal-vs-
    grazing lamp the patches record least, keeps the pair coherent); 'none': nothing (control: a product state)."""
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {COUPLINGS}")
    L1, L2 = layout["lamps"]
    out = []
    for k in range(sampler.n_panels):
        if coupling == "none":
            out.append([])
        elif coupling in ("lamp1", "lamp2"):
            out.append([L1 if coupling == "lamp1" else L2])
        else:
            mine = [i for i, c in enumerate(sampler.cells) if c["k"] == k]
            h1 = np.abs(sampler.hub1[mine]).mean() if mine else 0.0
            h2 = np.abs(sampler.hub2[mine]).mean() if mine else 0.0
            out.append([L1] if h1 >= h2 else [L2])
    return out


def complementary_state(sampler, layout, coupling="hub", phi=np.pi):
    """The stand-in: Z-only lighting structure x polarity qubits in |+> with CP(phi) to their partners. See the module docstring."""
    zp = z_structure(sampler, layout)
    P, size = sampler.n_panels, len(zp)
    idx = np.arange(size)
    part = partners(sampler, layout, coupling)
    bit = lambda q: (idx >> q) & 1
    full = np.zeros(size * 2 ** P, complex)
    for pb in range(2 ** P):
        phase = np.zeros(size)
        for k in range(P):
            if (pb >> k) & 1:
                for u in part[k]:
                    phase += bit(u)
        full[pb * size:(pb + 1) * size] = zp * np.exp(1j * phi * phase) / np.sqrt(2 ** P)
    return full


# ------------------------------------------------------------------ reduced states and the witness
def reduced_pair(psi, n, a, b):
    """4x4 density matrix of qubits (a, b), basis index 2*bit_a + bit_b (bit 0 = Z +1)."""
    t = np.asarray(psi, complex).reshape((2,) * n)
    M = np.moveaxis(t, [n - 1 - a, n - 1 - b], [0, 1]).reshape(4, -1)
    return M @ M.conj().T


def correlation_matrix(rho):
    """T[i, j] = <sigma_i (x) sigma_j> for i, j in x, y, z, plus the two local Bloch vectors."""
    ops = [PAULI[c] for c in "xyz"]
    T = np.array([[np.trace(rho @ np.kron(A, B)).real for B in ops] for A in ops])
    ra = np.array([np.trace(rho @ np.kron(A, np.eye(2))).real for A in ops])
    rb = np.array([np.trace(rho @ np.kron(np.eye(2), B)).real for B in ops])
    return T, ra, rb


def max_chsh(T):
    s = np.linalg.svd(T, compute_uv=False)
    return float(2 * np.sqrt(s[0] ** 2 + s[1] ** 2))


def chsh_settings(T):
    """Optimal CHSH axes for correlation matrix T: ((a, a'), (b, b')), unit Bloch vectors with
    S = a.T.b + a.T.b' + a'.T.b - a'.T.b' = max_chsh(T)."""
    U, s, Vt = np.linalg.svd(T)
    V = Vt.T
    alpha = np.arctan2(s[1], s[0])
    return (U[:, 0], U[:, 1]), (np.cos(alpha) * V[:, 0] + np.sin(alpha) * V[:, 1], np.cos(alpha) * V[:, 0] - np.sin(alpha) * V[:, 1])


def sample_correlation(psi, n, a, b, va, vb, M, rng):
    """Estimate <(va.sigma)_a (vb.sigma)_b> from M shots with the two qubits measured along va and vb. Returns (E, standard error)."""
    st = np.asarray(psi, complex)
    for q, v in ((a, va), (b, vb)):
        st = sl.apply_1q(st, sl.bloch_unitary(v), q, n)
    p = np.abs(st) ** 2
    p /= p.sum()
    idx = rng.choice(len(p), size=M, p=p)
    s = sl.spins(idx, [a, b])
    E = float((s[:, 0] * s[:, 1]).mean())
    return E, float(np.sqrt(max(1 - E * E, 1e-12) / M))


def chsh_estimate(psi, n, a, b, M, rng):
    """CHSH by sampling with the optimal settings of the pair. Returns dict(S, se, sigmas_above_2, exact)."""
    T, _, _ = correlation_matrix(reduced_pair(psi, n, a, b))
    (a0, a1), (b0, b1) = chsh_settings(T)
    parts = [sample_correlation(psi, n, a, b, x, y, M, rng) for x, y in ((a0, b0), (a0, b1), (a1, b0), (a1, b1))]
    S = parts[0][0] + parts[1][0] + parts[2][0] - parts[3][0]
    se = float(np.sqrt(sum(p[1] ** 2 for p in parts)))
    return dict(S=float(S), se=se, sigmas_above_2=float((S - CHSH_CLASSICAL) / se), exact=max_chsh(T), shots_per_setting=M)


# ------------------------------------------------------------------ multipartite witness (Mermin)
def pauli_expectation(psi, n, ops):
    """<psi| prod_q sigma_{ops[q]} |psi> for ops = {qubit: 'x'|'y'|'z'}."""
    phi = np.asarray(psi, complex)
    for q, c in ops.items():
        phi = sl.apply_1q(phi, PAULI[c], q, n)
    return float(np.vdot(psi, phi).real)


def mermin_terms(group, axes):
    """The terms of M_m = Re prod_j (X_j + i Y_j) = sum over even-size subsets S of (-1)^{|S|/2} Y_S X_rest, as (sign, {qubit: axis}).
    axes[q] = (X-like letter, Y-like letter, sign of the Y-like operator) of that qubit in the GHZ frame the state was built in; a Hadamard
    conjugation sends X -> Z and Y -> -Y, so a polarity qubit has ("z", "y", -1)."""
    terms = []
    for r in range(0, len(group) + 1, 2):
        for S in itertools.combinations(group, r):
            sign = (-1) ** (r // 2)
            for q in S:
                sign *= axes[q][2]
            terms.append((sign, {q: axes[q][1] if q in S else axes[q][0] for q in group}))
    return terms


def mermin_bounds(m):
    """(local-hidden-variable bound, quantum maximum) of the m-party Mermin operator."""
    return 2 ** (m // 2), 2 ** (m - 1)


def mermin_exact(psi, n, group, axes):
    return float(sum(sign * pauli_expectation(psi, n, ops) for sign, ops in mermin_terms(group, axes)))


def mermin_estimate(psi, n, group, axes, M, rng):
    """The same value by sampling M shots per term (each term is its own measurement context). Returns dict(value, se, sigmas_above_bound)."""
    val, var = 0.0, 0.0
    for sign, ops in mermin_terms(group, axes):
        st = np.asarray(psi, complex)
        for q, c in ops.items():
            st = sl.apply_1q(st, sl.bloch_unitary({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[c]), q, n)
        p = np.abs(st) ** 2
        p /= p.sum()
        sp = sl.spins(rng.choice(len(p), size=M, p=p), list(group)).prod(axis=1)
        E = float(sp.mean())
        val += sign * E
        var += max(1 - E * E, 1e-12) / M
    bound, top = mermin_bounds(len(group))
    se = float(np.sqrt(var))
    return dict(value=float(val), se=se, local_bound=bound, quantum_max=top, sigmas_above_bound=float((val - bound) / se))


def lamp_groups(layout, part):
    """GHZ-frame groups of the stand-in: each lamp together with the polarity qubits phase-coupled to it, and the Pauli axes to use.
    The lamp is read in X/Y, a polarity qubit in Z/-Y (Hadamard-conjugate of the GHZ frame, because its phase coupling is a CZ)."""
    groups = {}
    for k, q in enumerate(layout["polarity"]):
        for u in part[k]:
            groups.setdefault(u, []).append(q)
    out = []
    for u, qs in sorted(groups.items()):
        axes = {u: ("x", "y", 1), **{q: ("z", "y", -1) for q in qs}}
        out.append(dict(lamp_qubit=u, polarity_qubits=qs, group=[u] + qs, axes=axes))
    return out


def role(layout, q):
    if q in layout["patches"]:
        return "patch"
    return "lamp" if q in layout["lamps"] else "polarity"


def scan_pairs(psi, layout):
    """Exact S_max for every pair of qubits, most non-classical first."""
    n = layout["n_qubits"]
    rows = []
    for a, b in itertools.combinations(range(n), 2):
        T, ra, rb = correlation_matrix(reduced_pair(psi, n, a, b))
        rows.append(dict(a=a, b=b, kind=f"{role(layout, a)}-{role(layout, b)}", S_max=max_chsh(T),
                         violates=bool(max_chsh(T) > CHSH_CLASSICAL + 1e-9)))
    return sorted(rows, key=lambda r: -r["S_max"])


def bloch_length(psi, n, q):
    """|r| of qubit q's reduced state: 1 = pure and definite along some axis, 0 = maximally entangled with the rest."""
    t = np.asarray(psi, complex).reshape((2,) * n)
    M = np.moveaxis(t, n - 1 - q, 0).reshape(2, -1)
    rho = M @ M.conj().T
    return float(np.linalg.norm([np.trace(rho @ PAULI[c]).real for c in "xyz"]))


# ------------------------------------------------------------------ frames and their statistics
def polarity_lamp_correlation(pool):
    """corr(polarity of panel k, lamp bit j) over the pool: (n_panels, 2). The Z-read stand-in gives ~0, the X-read gives the lock."""
    out = np.zeros((pool.pol.shape[1], 2))
    for k in range(pool.pol.shape[1]):
        for j in range(2):
            a, b = pool.pol[:, k].astype(float), pool.lamps[:, j].astype(float)
            out[k, j] = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
    return out


def make_pool(psi, layout, pol_basis, M, rng):
    basis = {q: pol_basis for q in layout["polarity"]} if pol_basis != "z" else None
    return sl.StateSource(psi, layout, basis).pool(M, rng)


def describe_pool(sampler, pool):
    m = coherence_metrics(sampler, pool.lamps, pool.shots, pool.pol)
    m["polarity_lamp_correlation"] = polarity_lamp_correlation(pool).round(3).tolist()
    return m


def fig_frames(scene, sampler, rows, out, K, seed):
    fig, axs = plt.subplots(len(rows), len(WORLDS), figsize=(4.1 * len(WORLDS), 2.55 * len(rows) + 0.5), facecolor="#111", squeeze=False)
    for r, (name, pool) in enumerate(rows):
        gen = FrameGenerator(scene, sampler, pool)
        for c, w in enumerate(WORLDS):
            ax = axs[r, c]; ax.axis("off")
            try:
                img, d = gen.frame(K, np.random.default_rng(seed + c), lamp=w)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{name}\n{'R' if w[0] > 0 else 'L'}{'F' if w[1] > 0 else 'G'}  pol {''.join('R' if p > 0 else 'S' for p in d.pol)}",
                             color="w", fontsize=7.5)
            except ValueError:
                ax.text(.5, .5, "this world is not in the data", color="#888", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(f"{STATUS}\nSame four lighting worlds, K={K}, identical compose code. pol: R raised, S sunk per panel (left, centre, right)", color="w", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=95, facecolor=fig.get_facecolor()); plt.close(fig)


def fig_witness(scans, sweep, out):
    fig, axs = plt.subplots(1, 2 if sweep else 1, figsize=(13 if sweep else 7, 4.8), facecolor="#111", squeeze=False)
    ax = axs[0, 0]; ax.set_facecolor("#16171a")
    names = list(scans)
    for i, name in enumerate(names):
        top = scans[name][:10]
        ax.bar(np.arange(len(top)) + i * 0.4 / len(names) * 2 - 0.2, [t["S_max"] for t in top], 0.4, label=name,
               color=["#9a9ca3", "#f2a33a", "#5fd38d", "#4c9be8"][i % 4])
    ax.axhline(CHSH_CLASSICAL, color="#ff6b5e", lw=1.2); ax.text(9.9, CHSH_CLASSICAL - 0.07, "classical bound 2", color="#ff6b5e", fontsize=8, ha="right")
    ax.axhline(CHSH_TSIRELSON, color="#5fd38d", lw=1.0, ls="--"); ax.text(9.9, CHSH_TSIRELSON + 0.02, "quantum maximum 2.83", color="#5fd38d", fontsize=8, ha="right")
    ax.set_ylim(1.6, 3.0); ax.set_ylabel("best CHSH value of the pair, S_max", color="w"); ax.set_xlabel("ten most non-classical qubit pairs", color="w")
    ax.tick_params(colors="w"); ax.legend(facecolor="#222", labelcolor="w", fontsize=8)
    ax.set_title("Exact CHSH witness over all qubit pairs", color="w", fontsize=10)
    if sweep:
        ax2 = axs[0, 1]; ax2.set_facecolor("#16171a")
        sc = [s["scale"] for s in sweep]
        for k, colour in zip(range(len(sweep[0]["pairs"])), ["#f2a33a", "#c77dff", "#5fd38d", "#ff8fab"]):
            ax2.plot(sc, [s["pairs"][k]["S_max"] for s in sweep], "o-", color=colour, label=f"S_max: panel {sweep[0]['pairs'][k]['panel']} polarity with its lamp")
        ax2.axhline(CHSH_CLASSICAL, color="#ff6b5e", lw=1.2)
        ax2.set_ylim(1.6, 3.0); ax2.set_xlabel("lamp-to-patch coupling, x the mock's (kh, kf)", color="w"); ax2.set_ylabel("S_max", color="w"); ax2.tick_params(colors="w")
        ax3 = ax2.twinx()
        ax3.plot(sc, [s["lighting_coherence"] for s in sweep], "s--", color="#4c9be8", label="light follows lamp L1 (coherence metric)")
        ax3.plot(sc, [s["mermin_lamp2"] / s["mermin_lamp2_max"] for s in sweep], "^:", color="#ffd60a", label="Mermin value / maximum, lamp2 coupling")
        ax3.axhline(0.5, color="#ffd60a", lw=0.8, ls=":"); ax3.text(sc[-1], 0.51, "Mermin classical bound", color="#ffd60a", fontsize=7, ha="right")
        ax3.set_ylim(0, 1.05); ax3.set_ylabel("lighting coherence  /  Mermin fraction", color="#4c9be8"); ax3.tick_params(colors="#4c9be8")
        h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, facecolor="#222", labelcolor="w", fontsize=7, loc="lower right")
        ax2.set_title("Trade-off: the more the patches reveal the lamp, the weaker the polarity entanglement", color="w", fontsize=9)
    fig.suptitle(STATUS, color="w", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


def tradeoff(scene, NX, NY, min_cover, coupling, phi, scales, pool_size, rng):
    """Re-run the stand-in with the lamp-to-patch couplings (kh, kf) scaled: S_max of each polarity-lamp pair against how coherent the
    lighting is. pairs: [{polarity_qubit, lamp_qubit, panel, S_max}]."""
    rows = []
    for s in scales:
        sm = Sampler(scene, NX, NY, kh=1.6 * s, kf=1.3 * s, min_cover=min_cover)
        lay = allocate(sm.n, sm.n_panels)
        psi = complementary_state(sm, lay, coupling, phi)
        n, part = lay["n_qubits"], partners(sm, lay, coupling)
        pairs = [dict(panel=k, polarity_qubit=q, lamp_qubit=u, S_max=max_chsh(correlation_matrix(reduced_pair(psi, n, q, u))[0]))
                 for k, q in enumerate(lay["polarity"]) for u in part[k]]
        pool = make_pool(psi, lay, "z", pool_size, rng)
        m = coherence_metrics(sm, pool.lamps, pool.shots, pool.pol)
        psi2 = complementary_state(sm, lay, "lamp2", phi)
        g = lamp_groups(lay, partners(sm, lay, "lamp2"))[0]
        rows.append(dict(scale=float(s), pairs=pairs, lighting_coherence=float(m["light_left_right"]),
                         mermin_lamp2=mermin_exact(psi2, n, g["group"], g["axes"]), mermin_lamp2_max=mermin_bounds(len(g["group"]))[1]))
    return rows


# ------------------------------------------------------------------ QDrive payload (preview only)
def write_payload_preview(a, layout, sampler, report):
    """Build the QDrive program for the stand-in's targets from the calibration's Z targets. Writes the file and costs it; sends nothing."""
    from src.quantum.moth_client import PreparedJob, estimate_credits
    from src.store.jsonfmt import dumps_compact
    from src.targets.payloads import payload_sha256, qdrive_complementary_payload
    path = os.path.join(a.calib, "targets.json")
    with open(path) as f:
        t = json.load(f)
    if t["meta"]["qubit_map"] != {k: layout[k] for k in ("patches", "lamps", "polarity", "n_qubits")}:
        report["payload"] = "skipped: the calibration targets do not match this scene's qubit layout"
        return
    with open(os.path.join(a.calib, "payload_qdrive.json")) as f:
        cmap = json.load(f)["params"]["coupling_map"]
    cp = a.coupling if a.coupling != "none" else "hub"
    p = qdrive_complementary_payload(t["bloch"], t["relationships"], layout["n_qubits"], cmap, layout["polarity"], partners(sampler, layout, cp), a.phi)
    out = os.path.join(a.out, "payload_qdrive_complementary.json")
    with open(out, "w") as f:
        f.write(dumps_compact(p) + "\n")
    job = PreparedJob("qdrive-api-v1", p["params"])
    report["payload"] = dict(file=out, coupling=cp, sha256=payload_sha256({"engine": "qdrive-api-v1", "params": p["params"]}), credits_if_sent=estimate_credits("qdrive-api-v1"),
                             local_problems=job.problems(), sent=False)
    print(f"\nQDrive payload for this design: {out}  ({len(p['params']['targets'])} target entries, {len(p['params']['coupling_map'])} edges, "
          f"{report['payload']['credits_if_sent']} credit if sent). NOT SENT. Local check: {job.problems() or 'nothing known to be wrong'}. "
          f"Target syntax for mixed Pauli words is unverified.")
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


# ------------------------------------------------------------------ driver
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/complementary")
    ap.add_argument("--calib", default=DEFAULT_CALIB_DIR, help="directory with targets.json (grid)")
    ap.add_argument("--labels")
    ap.add_argument("--circuit", help="a QDrive QASM3 circuit to analyse in place of / beside the stand-in")
    ap.add_argument("--coupling", choices=COUPLINGS, default="hub")
    ap.add_argument("--phi", type=float, default=np.pi, help="controlled-phase angle; pi locks X-read polarity to the lamp")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--pool", type=int, default=60000)
    ap.add_argument("--chsh-shots", type=int, default=20000, help="shots per CHSH setting for the sampled estimate")
    ap.add_argument("--sweep", action="store_true", help="also run the lamp-coupling trade-off (slower)")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)

    with open(os.path.join(a.calib, "targets.json")) as f:
        NX, NY = json.load(f)["meta"]["grid"]
    labels = load_labels(a.labels) if a.labels else validate(fill_defaults(bay_window_labels()))
    scene = build_scene(labels)
    sampler = Sampler(scene, NX, NY, min_cover=labels["grid"]["min_cover"])
    layout = allocate(sampler.n, sampler.n_panels)
    n = layout["n_qubits"]
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    couplings = list(dict.fromkeys([a.coupling, "lamp2"])) if a.coupling not in ("none",) else [a.coupling]
    states = {"Z-only baseline (oracle, polarity chain in Z)": sl.oracle_state(sampler, layout)}
    stand_in = {}
    for cp in couplings:
        name = f"stand-in complementary (coupling={cp}, phi={a.phi:.3g})"
        states[name], stand_in[name] = complementary_state(sampler, layout, cp, a.phi), cp
    provenance = {k: "classical sqrt(P) oracle built locally: NOT a Moth result" if k.startswith("Z-only") else STATUS for k in states}
    if a.circuit:
        psi, nq = sl.statevector_from_qasm(open(a.circuit).read())
        if nq != n:
            raise SystemExit(f"{a.circuit} has {nq} qubits but this scene's layout needs {n}")
        states[f"circuit {os.path.basename(a.circuit)}"] = psi
        provenance[f"circuit {os.path.basename(a.circuit)}"] = "a circuit simulated locally; if it came from QDrive this is the real test, otherwise see its provenance"

    report = dict(status=STATUS, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n_qubits=n, qubit_map=layout, couplings=couplings,
                  phi=a.phi, partners={cp: partners(sampler, layout, cp) for cp in couplings}, seed=a.seed, states={})
    pools, scans, frame_rows = {}, {}, []
    for name, psi in states.items():
        scan = scan_pairs(psi, layout)
        scans[name] = scan
        entry = dict(provenance=provenance[name], pairs_violating_chsh=sum(r["violates"] for r in scan), best_pairs=scan[:6],
                     polarity_bloch_length=[round(bloch_length(psi, n, q), 4) for q in layout["polarity"]])
        for basis in ("z", "x"):
            pool = make_pool(psi, layout, basis, a.pool, rng)
            pools[(name, basis)] = pool
            entry[f"polarity_read_in_{basis}"] = describe_pool(sampler, pool)
            if basis == "x" or not name.startswith("stand-in") or name.endswith(f"coupling={couplings[0]}, phi={a.phi:.3g})"):
                tag = name.split(" (")[0] + (f" [{stand_in[name]}]" if name in stand_in else "")
                frame_rows.append((f"{tag}: polarity read in {basis.upper()}", pool))
        if name in stand_in:
            part = partners(sampler, layout, stand_in[name])
            entry["sampled_chsh"] = [dict(polarity_qubit=q, lamp_qubit=u, **chsh_estimate(psi, n, q, u, a.chsh_shots, rng))
                                     for k, q in enumerate(layout["polarity"]) for u in part[k]]
            entry["mermin"] = []
            for g in lamp_groups(layout, part):
                if len(g["group"]) >= 3:
                    bound, top = mermin_bounds(len(g["group"]))
                    entry["mermin"].append(dict(lamp_qubit=g["lamp_qubit"], polarity_qubits=g["polarity_qubits"], parties=len(g["group"]),
                                                exact=mermin_exact(psi, n, g["group"], g["axes"]), local_bound=bound, quantum_max=top,
                                                sampled=mermin_estimate(psi, n, g["group"], g["axes"], a.chsh_shots, rng)))
        report["states"][name] = entry

    sweep = None
    if a.sweep and a.coupling != "none":
        sweep = tradeoff(scene, NX, NY, labels["grid"]["min_cover"], a.coupling, a.phi, [0.0, 0.25, 0.5, 1.0, 1.5, 2.0], 30000, rng)
        report["tradeoff"] = sweep
    fig_frames(scene, sampler, frame_rows, os.path.join(a.out, "complementary_frames.png"), a.K, a.seed)
    fig_witness(scans, sweep, os.path.join(a.out, "complementary_witness.png"))
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    write_payload_preview(a, layout, sampler, report)

    print(STATUS)
    for name, e in report["states"].items():
        print(f"\n{name}\n  provenance: {e['provenance']}")
        print(f"  qubit pairs with S_max > 2: {e['pairs_violating_chsh']} of {len(scans[name])}; best: " +
              ", ".join(f"{r['kind']} ({r['a']},{r['b']}) {r['S_max']:.3f}" for r in e["best_pairs"][:3]))
        print(f"  polarity qubits' Bloch length (0 = maximally entangled): {e['polarity_bloch_length']}")
        for basis in ("z", "x"):
            m = e[f"polarity_read_in_{basis}"]
            print(f"  polarity read in {basis.upper()}: agreement {m['polarity_agreement']:+.3f}, correlation with lamps (L1, L2) per panel {m['polarity_lamp_correlation']}")
        for m in e.get("mermin", []):
            sm = m["sampled"]
            print(f"  Mermin, lamp {m['lamp_qubit']} + polarity {m['polarity_qubits']} ({m['parties']} parties): exact {m['exact']:.3f}, sampled {sm['value']:.3f} +- {sm['se']:.3f} "
                  + (f"({sm['sigmas_above_bound']:.0f} sigma above the classical bound {m['local_bound']}; quantum maximum {m['quantum_max']})"
                     if sm["value"] > m["local_bound"] else f"(within the classical bound {m['local_bound']}: no violation)"))
        for c in e.get("sampled_chsh", []):
            print(f"  sampled CHSH, polarity {c['polarity_qubit']} x lamp {c['lamp_qubit']}: S = {c['S']:.3f} +- {c['se']:.3f} "
                  + (f"({c['sigmas_above_2']:.1f} sigma above 2; exact {c['exact']:.3f})" if c["S"] > 2 + 2 * c["se"] else f"(no violation; exact {c['exact']:.3f})"))
    if sweep:
        print(f"\ntrade-off, coupling={a.coupling}: lamp-to-patch coupling scale -> S_max per polarity-lamp pair, and how coherent the lighting is")
        for s in sweep:
            print(f"  x{s['scale']:<5} " + "  ".join(f"panel {p['panel']}-L{p['lamp_qubit'] - layout['lamps'][0] + 1}: {p['S_max']:.3f}" for p in s["pairs"]) +
                  f"   lighting coherence {s['lighting_coherence']:+.3f}   lamp2 Mermin {s['mermin_lamp2']:.2f} of {s['mermin_lamp2_max']}")
    print(f"\nwrote {a.out}/report.json, complementary_frames.png, complementary_witness.png")


if __name__ == "__main__":
    main()
