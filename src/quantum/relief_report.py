#!/usr/bin/env python3
"""Figures and numbers for Superposed Relief, computed from the code (v2 handoff section 7, "claims slide").

    python -m src.quantum.relief_report [--out runs/relief] [--labels labels.json] [--entangle 0.8]

Writes   relief_contrast_vs_visibility.png   V against the visible bevel signal: the tension between interference and shading, per-panel vs domain relief
         domain_frames.png, report["domain"]   with --domain: the domain relief's frames, certificate, frustration and budget
         relief_frames.png    decided -> undecided frames in four light worlds, then control B and control A through the identical compose code
         relief_complementarity.png   bevel contrast and P(outcome 0) against the observation angle (the which-depth / visibility trade-off)
         relief_scaling.png   visibility against N for a fixed tilt and for tilt = kappa/sqrt(N), exact simulation points and the closed form
         report.json          the witness certificate with provenance, the budget, and every number printed
Everything is a local exact simulation of a circuit; nothing here touches Moth.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from src.capture.labels import bay_window_labels, fill_defaults, load_labels, validate
from src.geometry.facets import build_facets, panel_budget
from src.geometry.planes import build_scene
from src.quantum import relief_witness as rw
from src.quantum.complementary import pauli_expectation
from src.quantum.relief_state import WORLD_NAMES, LightRig, ReliefSpec, ReliefState, full_state, spec_from_facets
from src.texture.relief_compose import ReliefComposer


class _FixedRig(LightRig):
    def vector(self, world):
        b = np.deg2rad(50)
        return np.array([np.sin(b), 0, np.cos(b)])

    def panel_local(self, world, angle):
        return self.vector(world)


def contact_sheet(scene, state, composer, out, K=48, seed=3):
    rng = np.random.default_rng(seed)
    obs = state.spec.observations
    if len(obs) == 4 and state.spec.pol_graph:                              # domain relief: its own four observations, named by what they are
        rows = [(f"obs {g}: gamma {np.degrees(gm):.0f} chi {np.degrees(ch):.0f}", g, "coherent") for g, (gm, ch) in enumerate(obs)]
        rows += [("control B: dephased depth", 0, "dephased"), ("control A: independent noise", 0, "noise")]
    else:
        deg = lambda g: f"gamma {np.degrees(obs[g][0]):.0f} chi {np.degrees(obs[g][1]):.0f}"
        rows = [(f"decided ({deg(0)})", 0, "coherent"), (f"between ({deg(1)})", 1, "coherent"), (f"between ({deg(2)})", 2, "coherent"), (f"undecided ({deg(3)})", 3, "coherent"),
                ("control B: dephased depth", 0, "dephased"), ("control A: independent noise", 0, "noise")]
    s = 0.27
    th, tw = int(scene.H * s), int(scene.W * s)
    sheet = Image.new("L", (4 * tw, len(rows) * th), 0)
    d = ImageDraw.Draw(sheet)
    for r, (name, g, control) in enumerate(rows):
        for w in range(4):
            dr = state.draw(K, rng, g=g, world=w, control=control)
            img = composer.compose(dr.lit)
            assert np.all(img[scene.impenetrable] == 0)
            sheet.paste(Image.fromarray((img * 255).astype(np.uint8)).resize((tw, th)), (w * tw, r * th))
            depth = "".join("R" if p > 0 else "S" for p in dr.pol) if len(dr.pol) <= 6 else f"{sum(1 for p in dr.pol if p > 0)} raised / {sum(1 for p in dr.pol if p < 0)} sunk"
            d.text((w * tw + 4, r * th + 3), f"{name} | light {WORLD_NAMES[w]} | depth {depth}", fill=255)
    sheet.save(out)


def complementarity(out):
    tau = np.array([.55, .55, .35, .35, .35, .35, .55, .55])
    phi = np.array([0, 0, 0, 0, np.pi, np.pi, np.pi, np.pi])
    gam = np.deg2rad(np.linspace(0, 180, 37))
    st = ReliefState(ReliefSpec(tau, phi, np.zeros(8, int), [0.0], [], _FixedRig(), [(g, 0.0) for g in gam]))
    c, p0 = [], []
    for g in range(len(gam)):
        m = st.marginals(g, w=0, o=0)
        c.append(m[:4].mean() - m[4:].mean())
        p0.append(st.polarity_probs(g)[0])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(np.degrees(gam), c, "o-", color="#f2a33a", label="bevel contrast (outcome 0)")
    ax.plot(np.degrees(gam), np.array(p0) - 0.5, "s--", color="#4c9be8", label="P(outcome 0) - 1/2 = V sin(gamma)/2")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(90, color="#888", lw=0.8, ls=":")
    ax.set_xlabel("observation angle gamma on the depth sphere (deg): 0 decided raised, 90 undecided, 180 decided sunk")
    ax.set_ylabel("")
    ax.legend(fontsize=8)
    ax.set_title("The more decided the depth, the less the two depths interfere", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return dict(gamma_deg=[float(v) for v in np.degrees(gam)], contrast=[float(v) for v in c], p_outcome0=[float(v) for v in p0], visibility=float(np.prod(np.cos(tau))))


def scaling(out, kappa=0.99):
    Ns = (4, 6, 8, 10, 12, 14, 16, 18)
    sim_fixed, sim_scaled = [], []
    for N in Ns:
        for tau, store in ((np.full(N, 0.35), sim_fixed), (rw.scaled_tilts(kappa, N), sim_scaled)):
            sp = ReliefSpec(tau, np.zeros(N), np.zeros(N, int), [0.0], [], _FixedRig(), [(0.0, 0.0)])
            store.append(pauli_expectation(full_state(sp), sp.n_sim, {N: "x"}))
    big = np.array([4, 8, 18, 50, 200, 1000])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.semilogy(big, np.cos(0.35) ** big, "-", color="#ff6b5e", label="fixed tilt 0.35: cos^N (closed form)")
    ax.semilogy(big, [rw.visibility_law(rw.scaled_tilts(kappa, int(n)))[0] for n in big], "-", color="#5fd38d", label="tilt = kappa/sqrt(N) (closed form)")
    ax.semilogy(Ns, sim_fixed, "o", color="#ff6b5e", label="exact simulation")
    ax.semilogy(Ns, sim_scaled, "o", color="#5fd38d")
    ax.set_xscale("log")
    ax.set_xlabel("facets N")
    ax.set_ylabel("interference visibility V")
    ax.legend(fontsize=8)
    ax.set_title("Depth maps interfere only if the relief is spread thin", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return dict(N=list(Ns), simulated_fixed_tilt=[float(v) for v in sim_fixed], simulated_scaled_tilt=[float(v) for v in sim_scaled],
                table=rw.visibility_table(kappa))


def contrast_vs_visibility(out, kappa=1.0, K=48, light_x=0.65):
    """Analytic, per facet. Interference visibility V = prod cos(tau) over a polarity group, and the visible bevel signal, the difference in lit probability
    between a facet tilted toward the light and one tilted away: sin(tau) * light_x, against the photon noise 0.5 / sqrt(K) of a K-photon frame.
    Per-panel relief (tau = kappa/sqrt(N), one polarity qubit for all N facets): V stays put, the signal per facet falls like 1/sqrt(N) and sinks under the noise.
    Domain relief (tau fixed, group_size m facets per polarity qubit): V = cos(tau)^m and the signal per facet are both independent of the size of the wall."""
    Ns = np.unique(np.round(np.logspace(np.log10(4), 3, 40)).astype(int))
    noise = 0.5 / np.sqrt(K)
    V_panel = np.array([np.prod(np.cos(np.full(N, kappa / np.sqrt(N)))) for N in Ns])
    sig_panel = np.sin(kappa / np.sqrt(Ns)) * light_x
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].semilogx(Ns, V_panel, "-", color="#4c9be8", label="per-panel: visibility V")
    ax[0].semilogx(Ns, sig_panel / noise, "--", color="#4c9be8", label="per-panel: bevel signal / photon noise per facet")
    for tau, c in ((0.5, "#5fd38d"),):
        m = 2
        ax[0].semilogx(Ns, np.full(len(Ns), np.cos(tau) ** m), "-", color=c, label=f"domain (tau {tau}, {m} facets per qubit): V")
        ax[0].semilogx(Ns, np.full(len(Ns), np.sin(tau) * light_x / noise), "--", color=c, label="domain: bevel signal / photon noise per facet")
    ax[0].axhline(1, color="#888", ls=":")
    ax[0].set_xlabel("facets N in the wall"), ax[0].set_ylabel(f"V, and signal / noise (K = {K} photons)")
    ax[0].set_title("per-panel: the bevel sinks under the photon noise as N grows", fontsize=10), ax[0].legend(fontsize=7)
    ax[1].plot(V_panel, sig_panel / noise, "-", color="#4c9be8", label=f"per-panel, N = 4 ... 1000 (kappa {kappa})")
    for N in (4, 18, 100, 1000):
        k = int(np.argmin(np.abs(Ns - N)))
        ax[1].annotate(f"N={Ns[k]}", (V_panel[k], sig_panel[k] / noise), fontsize=7)
    pts = []
    for tau, mk in ((0.3, "^"), (0.5, "o"), (0.7, "s")):
        for m in (1, 2, 3, 4, 6, 8):
            pts.append((np.cos(tau) ** m, np.sin(tau) * light_x / noise, tau, m))
        sel = [p for p in pts if p[2] == tau]
        ax[1].plot([p[0] for p in sel], [p[1] for p in sel], mk + "-", color="#f2a33a", alpha=0.3 + tau, label=f"domain, tau {tau}, facets per qubit m = 1 ... 8")
    ax[1].axhline(1, color="#888", ls=":")
    ax[1].set_xlabel("interference visibility V of a polarity group"), ax[1].set_ylabel("bevel signal / photon noise per facet")
    ax[1].set_title("the trade-off: what the domain size buys", fontsize=10), ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return dict(K=K, light_x=light_x, N=[int(n) for n in Ns], V_panel=[float(v) for v in V_panel], signal_over_noise_panel=[float(v) for v in sig_panel / noise],
                domain=[dict(tau=float(t), m=int(m), V=float(v), signal_over_noise=float(s_)) for v, s_, t, m in pts])


def domain_report(scene, out, rounds=400, **overrides):
    """Frames, the certificate, the parity game and the budget of the domain relief on `scene` (a matrix-product state beyond 20 qubits), at the show's own defaults
    (ui.session.DOMAIN_DEFAULTS) unless overridden."""
    from src.geometry.domains import build_domains, frustration
    from src.quantum import domain_witness as dw
    from src.quantum.domain_state import MPSReliefState, make_state, spec_from_domains
    from src.quantum.parity_game import ParityGame
    from src.texture.relief_compose import make_composer
    from src.ui.session import DOMAIN_DEFAULTS
    dp = dict(DOMAIN_DEFAULTS, **overrides)
    ds = build_domains(scene, seg_len=dp["seg_len"], group_size=dp["group_size"], tau=dp["tau"], crease_sign=dp["crease_sign"])
    spec = spec_from_domains(ds, scene, entangle=dp.get("entangle", 0.8), pol_coupling=dp["pol_coupling"], lock=dp["lock"], tau_mix=dp["tau_mix"], pol_field=dp["pol_field"],
                             crease_coupling=dp["crease_coupling"], seam_coupling=dp["seam_coupling"], seam_mix=dp["seam_mix"], leaf_tau=dp["leaf_tau"],
                             leaf_prefer=dp["leaf_prefer"], floquet=(dp["floquet_steps"], dp["floquet_zz"], dp["floquet_x"]))
    state = make_state(spec, backend="mps")
    state.warm()
    contact_sheet(scene, state, make_composer(scene, ds.facets), os.path.join(out, "domain_frames.png"))
    cert = dw.domain_certificate(state)
    cert["lines"] = dw.describe_domain_certificate(cert, ds, spec)
    game = None
    if spec.lock_array() is not None:
        g = ParityGame(state, restarts=12)
        rng = np.random.default_rng(0)
        for _ in range(rounds):
            g.play(rng, K=2)
        game = dict(g.stats.summary(), mermin=g.M, parties=g.parties, text=g.describe())
    return dict(params=dp, facets=spec.F, domains=spec.P, simulated_qubits=spec.n_sim, circuit_qubits=spec.n_full, backend="mps", accuracy=state.accuracy(),
                domain_edges=len(ds.edges), creases=sum(1 for e in ds.edges if e[2] == "crease"),
                frustration=dict(without_lamp=frustration(ds), with_lamp=frustration(ds, lock=True, locked=[d for d in range(spec.P) if spec.lock_array() is not None and spec.lock_array()[d]])),
                certificate=cert, game=game)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/relief")
    ap.add_argument("--labels")
    ap.add_argument("--entangle", type=float, default=0.8)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--domain", action="store_true", help="also report the domain relief (matrix-product state beyond 20 qubits): frames, certificate, budget")
    ap.add_argument("--seg-len", type=float, default=150.0)
    ap.add_argument("--group-size", type=int, default=1)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    labels = load_labels(a.labels) if a.labels else validate(fill_defaults(bay_window_labels()))
    scene = build_scene(labels)
    fs = build_facets(scene, kappa=a.kappa)
    spec = spec_from_facets(fs, scene, entangle=a.entangle)
    state = ReliefState(spec)
    contact_sheet(scene, state, ReliefComposer(scene, fs), os.path.join(a.out, "relief_frames.png"))
    cvv = contrast_vs_visibility(os.path.join(a.out, "relief_contrast_vs_visibility.png"), a.kappa)
    comp = complementarity(os.path.join(a.out, "relief_complementarity.png"))
    sc = scaling(os.path.join(a.out, "relief_scaling.png"))
    cert = rw.certificate(state, np.random.default_rng(0), 40000)
    report = dict(status="LOCAL exact simulation of a circuit; no Moth engine involved", facets=spec.F, polarity=spec.P, simulated_qubits=spec.n_sim,
                  circuit_qubits=spec.n_full, couplings=len(spec.edges), kappa=a.kappa, budget={str(k): v for k, v in panel_budget(fs).items()},
                  complementarity=comp, scaling=sc, contrast_vs_visibility=cvv, certificate=dict(cert, lines=rw.describe_certificate(cert)))
    if a.domain:
        report["domain"] = domain_report(scene, a.out, seg_len=a.seg_len, group_size=a.group_size, entangle=a.entangle)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(report["status"])
    print(f"{spec.F} facet qubits + {spec.P} polarity = {spec.n_sim} simulated; {spec.n_full} in the full circuit; {len(spec.edges)} diagonal couplings")
    print("\n".join(report["certificate"]["lines"]))
    if a.domain:
        d = report["domain"]
        print(f"\ndomain relief: {d['facets']} facets + {d['domains']} domain qubits = {d['simulated_qubits']} simulated ({d['backend']}), {d['domain_edges']} edges ({d['creases']} creases)")
        print("\n".join(d["certificate"]["lines"]))
        if d["game"]:
            print(d["game"]["text"])
    print(f"wrote {a.out}/relief_frames.png, relief_complementarity.png, relief_scaling.png, relief_contrast_vs_visibility.png, report.json" + (", domain_frames.png" if a.domain else ""))


if __name__ == "__main__":
    main()
