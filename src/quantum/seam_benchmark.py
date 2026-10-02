"""Seams, the parity game and time: where the entanglement between edges lives, what it costs a classical simulator, and whether the game survives it.

    python -m src.quantum.seam_benchmark [--out runs/seams] [--max-seconds 120] [--quick]

Four measurements, all on matrix-product states with a FIXED bond-dimension cap (32) so that "how badly the classical approximation does" is one number,
the norm deficit (1 - <psi|psi>, the weight the truncation threw away: about the infidelity of the best chi-32 classical stand-in):

  seam sweep     bay window, seam facets moved toward the equator (seam_mix 0 .. 1): the deficit, the entanglement of a seam facet with the rest, the entanglement
                 of the creases' polarity qubits, and the parity game's predicted win rate with its leaves away from the seam and ON it. The question item 3
                 asks: does the depth-consistency game survive the regime that is hard to simulate? (It does not: a leaf holding an equatorial seam facet has
                 visibility 0.)
  tiled walls    n x n panels, every shared edge a seam: the seams now form a 2D network. The deficit against the number of qubits for the relief seam and for the
                 equatorial seam. A bay window's seams are two lines, quasi-1D, and an MPS handles 1D; the tiled wall is where the cut width grows.
  Floquet        kicked-Ising steps on the facet graph: mean bond entropy and the deficit against the number of steps. Static states do not get harder with time;
                 this one does.
  game           the win rate of the Mermin game against the classical 75%, for the default design, from the Mermin value of the state.
Every point runs in its own process with a time limit; a point that times out is reported as such.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

CAP = 32
KW = dict(lock=1.5708, pol_coupling=0.3, crease_coupling=1.0, seam_coupling=1.0, leaf_tau=0.3)      # the domain engine's defaults (seams strong, boundaries weak)


def tiled_labels(n, panel=300):
    """n x n square panels, every shared edge a crease (alternating plane tilt), a glass pane in each: the seams form a 2D network."""
    from src.capture.labels import GRID_DEFAULT, SCHEMA, default_window
    from src.mask.build_mask import scale_poly
    W = n * panel
    panels, glass = [], []
    for r in range(n):
        for c in range(n):
            k = r * n + c
            poly = [[float(c * panel), float(r * panel)], [float((c + 1) * panel), float(r * panel)], [float((c + 1) * panel), float((r + 1) * panel)], [float(c * panel), float((r + 1) * panel)]]
            ang = 25.0 if (r + c) % 2 == 0 else -25.0
            p = dict(id=k, name=f"p{r}{c}", plane_id=k, angle_deg=ang, polygon=poly)
            p.update(default_window(ang, poly))
            panels.append(p)
            glass.append({"polygon": [list(pt) for pt in scale_poly(poly, 0.45)]})
    return dict(schema=SCHEMA, source="synthetic", image=dict(file=None, width=W, height=W), canvas=dict(width=W, height=W), grid=dict(GRID_DEFAULT),
                panels=panels, glass=glass, off_limits=[])


def _bay_scene():
    from src.capture.labels import bay_window_labels, fill_defaults, validate
    from src.geometry.planes import build_scene
    return build_scene(validate(fill_defaults(bay_window_labels())))


def _tiled_scene(n):
    from src.capture.labels import fill_defaults, validate
    from src.geometry.planes import build_scene
    return build_scene(validate(fill_defaults(tiled_labels(n))))


def _prep(spec, cap=CAP):
    from src.quantum.domain_state import MPSReliefState
    st = MPSReliefState(spec, max_bond=cap)
    t = time.time()
    m = st.prep()
    return st, m, time.time() - t


def job_tiled(n, seam_mix, cap=CAP):
    from src.geometry.domains import build_domains
    from src.quantum.domain_state import spec_from_domains
    sc = _tiled_scene(n)
    ds = build_domains(sc, seg_len=150, group_size=1)
    sp = spec_from_domains(ds, sc, seam_mix=seam_mix, **{**KW, "lock": 0.0})
    st, m, secs = _prep(sp, cap)
    return dict(n=n, seam_mix=seam_mix, qubits=sp.n_sim, facets=sp.F, seam_facets=len(sp.seam_facets), max_bond=m.max_bond(), deficit=float(abs(1 - m.norm2())),
                mean_entropy=float(np.mean(m.entropies())), seconds=secs)


def job_bay(seam_mix, leaf_prefer, cap=CAP):
    from src.geometry.domains import build_domains
    from src.quantum import domain_witness as dw, parity_game as pg
    from src.quantum.domain_state import spec_from_domains
    sc = _bay_scene()
    ds = build_domains(sc, seg_len=150, group_size=1)
    sp = spec_from_domains(ds, sc, seam_mix=seam_mix, leaf_prefer=leaf_prefer, **KW)
    st, m, secs = _prep(sp, cap)
    cert = dw.domain_certificate(st, mermin=False, lamp=False)
    game = pg.ParityGame(st, restarts=8)
    return dict(seam_mix=seam_mix, leaf_prefer=leaf_prefer, qubits=sp.n_sim, deficit=float(abs(1 - m.norm2())), mean_entropy=float(np.mean(m.entropies())),
                crease_negativity=cert["creases"]["max_negativity"], creases_entangled=cert["creases"]["entangled"], creases=cert["creases"]["edges"],
                seam_facet_entropy=cert["seams"]["seam_facet_entropy"], other_facet_entropy=cert["seams"]["other_facet_entropy"],
                seam_stabilizer=float(np.mean([r["stabilizer"] for r in cert["seams"]["domain_stabilizers"]])), mermin=game.M, predicted_win=game.predicted,
                leaves=game.leaves, seconds=secs)


def job_floquet(steps, zz, xx, cap=CAP):
    from src.geometry.domains import build_domains
    from src.quantum.domain_state import spec_from_domains
    sc = _bay_scene()
    ds = build_domains(sc, seg_len=150, group_size=1)
    sp = spec_from_domains(ds, sc, floquet=(steps, zz, xx), **KW)
    st, m, secs = _prep(sp, cap)
    return dict(steps=steps, theta_zz=zz, theta_x=xx, qubits=sp.n_sim, max_bond=m.max_bond(), deficit=float(abs(1 - m.norm2())), mean_entropy=float(np.mean(m.entropies())),
                max_entropy=float(max(m.entropies())), seconds=secs)


JOBS = dict(tiled=job_tiled, bay=job_bay, floquet=job_floquet)


def _worker(kind, args, q):
    try:
        q.put(JOBS[kind](*args))
    except Exception as e:                                              # noqa: BLE001 - reported as a failed point
        q.put(dict(error=f"{type(e).__name__}: {e}"))


def run(kind, args, max_seconds):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(kind, args, q))
    t = time.time()
    p.start()
    p.join(max_seconds)
    if p.is_alive():
        p.terminate()
        p.join()
        return dict(kind=kind, args=list(args), timed_out=True, seconds=time.time() - t)
    try:
        return dict(kind=kind, args=list(args), **q.get(timeout=5))
    except Exception:                                                   # noqa: BLE001
        return dict(kind=kind, args=list(args), error="no result")


def run_all(max_seconds=120, quick=False):
    out = dict(cap=CAP, defaults=KW, max_seconds=max_seconds)
    mixes = (0.0, 0.5, 1.0) if quick else (0.0, 0.25, 0.5, 0.75, 1.0)
    out["seam_sweep"] = []
    for mix in mixes:
        for lp in ("visible", "seam"):
            r = run("bay", (mix, lp), max_seconds)
            out["seam_sweep"].append(r)
            print(f"  bay seam_mix {mix:.2f} leaves {lp:7s}: " + (f"deficit {r['deficit']:.1e}, seam-facet entropy {r['seam_facet_entropy']:.2f} bits, crease negativity {r['crease_negativity']:.3f}, "
                  f"Mermin {r['mermin']:.2f}, predicted win {r['predicted_win']:.3f}" if "deficit" in r else str(r)), flush=True)
    out["tiled"] = []
    for mix in (0.0, 1.0):
        for n in ((1, 2) if quick else (1, 2, 3, 4)):
            r = run("tiled", (n, mix), max_seconds)
            out["tiled"].append(r)
            print(f"  tiled {n}x{n} seam_mix {mix:.0f}: " + (f"{r['qubits']} qubits ({r['seam_facets']} seam facets), deficit {r['deficit']:.1e}, mean entropy {r['mean_entropy']:.2f}, {r['seconds']:.1f}s"
                  if "deficit" in r else str(r)), flush=True)
            if r.get("timed_out") or r.get("error"):
                break
    out["floquet"] = []
    for (zz, xx) in ((0.7, 0.5), (1.0, 0.9)):
        for T in ((0, 2, 4) if quick else (0, 1, 2, 3, 4, 6, 8, 12)):
            r = run("floquet", (T, zz, xx), max_seconds)
            out["floquet"].append(r)
            print(f"  floquet ({zz}, {xx}) steps {T:2d}: " + (f"mean entropy {r['mean_entropy']:.2f} bits, deficit {r['deficit']:.1e}" if "deficit" in r else str(r)), flush=True)
    return out


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    sw = [r for r in res["seam_sweep"] if "deficit" in r]
    for lp, c, lab in (("visible", "#5fd38d", "game leaves away from the seam"), ("seam", "#ff6b5e", "game leaves ON the seam")):
        rows = [r for r in sw if r["leaf_prefer"] == lp]
        ax[0].plot([r["seam_mix"] for r in rows], [r["predicted_win"] for r in rows], "o-", color=c, label=lab)
    ax[0].axhline(0.75, color="#888", ls="--", label="classical maximum 0.75")
    ax[0].set_xlabel("seam_mix: how far the seam facets sit toward the equator"), ax[0].set_ylabel("predicted parity-game win rate")
    ax[0].set_title("does the game survive the hard seam?", fontsize=10), ax[0].legend(fontsize=7)
    rows = [r for r in sw if r["leaf_prefer"] == "visible"]
    ax[1].plot([r["seam_mix"] for r in rows], [r["deficit"] for r in rows], "s-", color="#f2a33a", label="classical cost: norm deficit at chi 32")
    ax[1].set_yscale("log"), ax[1].set_xlabel("seam_mix"), ax[1].set_ylabel("norm deficit")
    b = ax[1].twinx()
    b.plot([r["seam_mix"] for r in rows], [r["seam_facet_entropy"] for r in rows], "o--", color="#4c9be8", label="seam facet entropy (bits)")
    b.plot([r["seam_mix"] for r in rows], [r["crease_negativity"] * 4 for r in rows], "^--", color="#a070e0", label="4 x crease negativity")
    b.set_ylabel("bits  /  4 x negativity")
    ax[1].set_title("what the seam regime costs a classical simulator", fontsize=10)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    for mix, c, lab in ((0.0, "#4c9be8", "relief seam"), (1.0, "#ff6b5e", "equatorial seam")):
        rows = [r for r in res["tiled"] if "deficit" in r and r["seam_mix"] == mix]
        ax[2].plot([r["qubits"] for r in rows], [max(r["deficit"], 1e-16) for r in rows], "o-", color=c, label=lab)
    ax[2].set_xscale("log"), ax[2].set_yscale("log"), ax[2].set_xlabel("qubits (n x n panels, every shared edge a seam)"), ax[2].set_ylabel("norm deficit at chi 32")
    ax[2].set_title("tiled walls: the seams become a 2D network", fontsize=10), ax[2].legend(fontsize=7)
    for (zz, xx), c in (((0.7, 0.5), "#4c9be8"), ((1.0, 0.9), "#ff6b5e")):
        rows = [r for r in res["floquet"] if "deficit" in r and r["theta_zz"] == zz]
        ax[3].plot([r["steps"] for r in rows], [r["mean_entropy"] for r in rows], "o-", color=c, label=f"mean bond entropy (zz {zz}, x {xx})")
    ax[3].set_xlabel("kicked-Ising steps"), ax[3].set_ylabel("mean bond entropy (bits)")
    d = ax[3].twinx()
    for (zz, xx), c in (((0.7, 0.5), "#4c9be8"), ((1.0, 0.9), "#ff6b5e")):
        rows = [r for r in res["floquet"] if "deficit" in r and r["theta_zz"] == zz]
        d.plot([r["steps"] for r in rows], [r["deficit"] for r in rows], "s--", color=c, alpha=0.6)
    d.set_yscale("log"), d.set_ylabel("norm deficit at chi 32 (dashed)")
    ax[3].set_title("entanglement grows with time", fontsize=10), ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/seams")
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = run_all(a.max_seconds, a.quick)
    with open(os.path.join(a.out, "seams.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    figure(res, os.path.join(a.out, "seams.png"))
    print(f"wrote {a.out}/seams.json and seams.png")


if __name__ == "__main__":
    main()
