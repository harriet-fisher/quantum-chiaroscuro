"""Where does classical simulation stop being cheap? A benchmark to run for every design, in place of the visibility-versus-N plot.

    python -m src.quantum.classical_baselines [--out runs/baselines] [--max-seconds 60] [--quick]

Three classical methods, each with the family of states it can and cannot handle:

  two-branch sampler   exact, O(N) per sample, for a PRODUCT relief with ONE polarity qubit (the per-panel relief without couplings): the state is a
                       two-term superposition of product states, so a frame is drawn by rejection from a mixture of two product distributions. Checked against
                       the exact engine for small N, timed to N = 5000. It stops working the moment the polarity qubits are coupled (2^D branches).
  excitation truncation  a state near |0..0> has Poisson-binomial excitation statistics (diagonal gates leave Z populations alone), so keeping only the
                       components with <= k excitations has fidelity P(<= k). `k99` is the k that reaches 0.99. The per-panel relief holds sum tau^2 fixed
                       (tau = kappa/sqrt(N)), so its mean excitation number stays ~kappa^2/4 at every N and k99 stays 2 or 3: easy at any size. A relief at fixed
                       tau, or one pushed toward the equator, has mean excitation number ~N p and k99 ~ N p.
  matrix-product state  bond dimension chi (Aer, truncation 1e-8, chi capped) and build time against the number of qubits, for
                       * the real bay window cut finer and finer (loops, so cut width 2: chi stays small),
                       * an L x L lattice of domains in the relief regime (tau 0.5, theta 0.5) and
                       * the same lattice in the graph-state regime (facets at the equator, theta = pi/2): chi grows ~ 2^L, which is where an MPS stops being cheap.
Each point runs in its own process with a time limit, so one expensive size cannot hang the benchmark; a point that hits the limit or the bond cap is
reported as such, not dropped. Nothing here claims advantage: it measures how large each design can be before THESE methods stop being cheap.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from src.quantum.relief_state import LightRig, ReliefSpec, ReliefState, reference_circuit

CAP = 128
TRUNC = 1e-8


# ------------------------------------------------------------------ method 1: the two-branch sampler (product relief, one polarity qubit)
class TwoBranchSampler:
    """Exact frames of a product relief with ONE polarity qubit (facets 0..N-1), without any coupling. Light world w and observation g are given."""

    def __init__(self, spec, g=0, w=0):
        from src.quantum import sampler_local as sl
        if spec.P != 1 or spec.edges or spec.pol_layers:
            raise ValueError("the two-branch sampler is for one polarity qubit and a product facet state")
        self.spec, N = spec, spec.F
        gam, chi = spec.observations[g]
        r = sl.bloch_unitary((np.sin(gam) * np.cos(chi), np.sin(gam) * np.sin(chi), np.cos(gam)))        # r[o, s] = <o|R|s>
        self.r = r
        a = np.empty((N, 2), complex)
        b = np.empty((N, 2), complex)
        for i in range(N):
            U = sl.bloch_unitary(spec.rig.panel_local(w, spec.panel_angles[spec.light_panel(i)]))
            psi = np.array([np.cos(spec.tau[i] / 2), np.exp(1j * spec.phi[i]) * np.sin(spec.tau[i] / 2)])
            a[i] = U @ psi                                        # <x|U|psi_i>
            b[i] = U @ (np.diag([1, -1]) @ psi)                   # <x|U Z|psi_i>
        self.a, self.b = a, b
        self.V = float(np.prod(np.cos(spec.tau)))

    def draw(self, rng, K):
        """(o, lit): the polarity outcome and the K-photon lit fractions per facet, one frame."""
        r, a, b = self.r, self.a, self.b
        N = len(a)
        # P(o) = (|r_o0|^2 + |r_o1|^2 + 2 Re(conj(r_o0) r_o1) V) / 2
        po = np.array([(abs(r[o, 0]) ** 2 + abs(r[o, 1]) ** 2 + 2 * (np.conj(r[o, 0]) * r[o, 1]).real * self.V) / 2 for o in (0, 1)])
        o = int(rng.random() * po.sum() > po[0])
        pa, pb = np.abs(a) ** 2, np.abs(b) ** 2                   # each row sums to 1
        cum_a, cum_b = np.cumsum(pa, axis=1), np.cumsum(pb, axis=1)
        out = np.zeros((K, N), np.int8)
        done = 0
        while done < K:
            m = max(4 * (K - done), 16)
            from_a = rng.random(m) < 0.5
            ua, ub = rng.random((m, N)), rng.random((m, N))
            xa = (ua > cum_a[:, 0]).astype(np.int8)
            xb = (ub > cum_b[:, 0]).astype(np.int8)
            x = np.where(from_a[:, None], xa, xb)
            idx = np.arange(N)
            with np.errstate(divide="ignore", invalid="ignore"):                 # products of thousands of amplitudes underflow: work with logs
                rho = np.exp(np.log(b[idx[None, :], x]).sum(axis=1) - np.log(a[idx[None, :], x]).sum(axis=1))
            amp = np.abs(r[o, 0] + r[o, 1] * rho) ** 2 / (1 + np.abs(rho) ** 2)        # |r0 A + r1 B|^2 / (|A|^2 + |B|^2), in [0, 1]
            acc = np.where(np.isfinite(amp), amp, 0.0)
            keep = x[rng.random(m) < acc]
            take = min(len(keep), K - done)
            out[done:done + take] = keep[:take]
            done += take
        return o, (out == 0).mean(axis=0)


def two_branch_check(N=5, seed=0, K=1, frames=40000):
    """Total variation distance between the two-branch sampler and the exact engine on a small product relief (polarity outcome and facets jointly)."""
    rng = np.random.default_rng(seed)
    tau = np.full(N, 0.8 / np.sqrt(N))
    phi = rng.uniform(-3, 3, N)
    spec = ReliefSpec(tau, phi, np.zeros(N, int), [20.0], [], LightRig(), [(np.deg2rad(60), 0.0)])
    st, tb = ReliefState(spec), TwoBranchSampler(spec, g=0, w=1)
    exact = st.cell(0)[1]                                         # [o, x]
    exact = exact / exact.sum()
    emp = np.zeros_like(exact)
    for _ in range(frames):
        o, lit = tb.draw(rng, 1)
        x = int(sum((0 if lit[i] > 0.5 else 1) << i for i in range(N)))
        emp[o, x] += 1
    emp /= emp.sum()
    return float(0.5 * np.abs(exact - emp).sum())


def two_branch_timing(Ns=(10, 100, 1000, 5000), K=48, seed=0, frames=20):
    rng = np.random.default_rng(seed)
    rows = []
    for N in Ns:
        spec = ReliefSpec(np.full(N, 0.99 / np.sqrt(N)), rng.uniform(-3, 3, N), np.zeros(N, int), [20.0], [], LightRig(), [(np.deg2rad(60), 0.0)])
        tb = TwoBranchSampler(spec, 0, 1)
        t = time.time()
        for _ in range(frames):
            tb.draw(rng, K)
        rows.append(dict(N=N, seconds_per_frame=(time.time() - t) / frames))
    return rows


# ------------------------------------------------------------------ method 2: excitation truncation
def excitation_stats(tau, target=0.99):
    """Poisson-binomial statistics of the number of excitations of facets with tilts tau (in the preparation basis): mean, P(vacuum) and the smallest
    k with P(<= k) >= target, which is the fidelity of the state truncated to <= k excitations."""
    p = np.sin(np.asarray(tau, float) / 2) ** 2
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for q in p:
        pmf[1:] = pmf[1:] * (1 - q) + pmf[:-1] * q
        pmf[0] *= (1 - q)
    cdf = np.cumsum(pmf)
    return dict(mean=float(p.sum()), vacuum=float(pmf[0]), k_target=int(np.searchsorted(cdf, target)), target=target, N=len(p))


# ------------------------------------------------------------------ method 3: families of specs and the MPS measurements
def panel_family(N, kappa=1.0):
    return ReliefSpec(np.full(N, kappa / np.sqrt(N)), np.zeros(N), np.zeros(N, int), [0.0], [], LightRig(), [(0.0, 0.0)])


def grid_family(L, regime="relief", m=2):
    """An L x L lattice of domains: one polarity qubit and m facets each, ZZ(theta_pol) on the lattice edges. Sites run row by row, each domain's polarity qubit
    then its facets, so a cut crosses L lattice edges."""
    D = L * L
    F = D * m
    relief = regime == "relief"
    tau = np.full(F, 0.5 if relief else np.pi / 2)
    phi = np.random.default_rng(1).uniform(-3, 3, F)
    panel_of = np.repeat(np.arange(D), m)
    graph = [(r * L + c, r * L + c + 1) for r in range(L) for c in range(L - 1)] + [(r * L + c, (r + 1) * L + c) for r in range(L - 1) for c in range(L)]
    th = 0.5 if relief else np.pi / 2
    edges = [] if relief else [(d * m + k, d * m + k + 1, "zz", np.pi / 2) for d in range(D) for k in range(m - 1)]
    spec = ReliefSpec(tau, phi, panel_of, [0.0], edges, LightRig(), [(0.0, 0.0)], n_groups=D, facet_panel=np.zeros(F, int), pol_graph=graph,
                      pol_layers=[dict(zz=[th] * len(graph), rx=None)])
    order = []
    for d in range(D):
        order.append(F + d)
        order.extend(range(d * m, (d + 1) * m))
    spec.site_order = order
    return spec


def bay_family(seg_len):
    from src.capture.labels import bay_window_labels, fill_defaults, validate
    from src.geometry.domains import build_domains
    from src.geometry.planes import build_scene
    from src.quantum.domain_state import spec_from_domains
    scene = build_scene(validate(fill_defaults(bay_window_labels())))
    ds = build_domains(scene, seg_len=seg_len, group_size=2)
    return spec_from_domains(ds, scene, lock=0.0)


def _measure(job):
    """Worker: build the family member, evolve it as a matrix-product state, report chi, discarded weight and seconds."""
    family, size, *rest = job
    cap = rest[0] if rest else CAP
    from src.quantum.mps import MPS, permute_circuit
    spec = {"panel": lambda s: panel_family(s), "bay": lambda s: bay_family(s), "grid-relief": lambda s: grid_family(s, "relief"),
            "grid-graph": lambda s: grid_family(s, "graph")}[family](size)
    n = spec.n_sim
    order = spec.site_order or list(range(n))
    site_of = np.empty(n, int)
    site_of[order] = np.arange(n)
    t = time.time()
    m = MPS.from_circuit(permute_circuit(reference_circuit(spec), site_of), TRUNC, cap)
    return dict(family=family, size=size, cap=cap, qubits=n, max_bond=m.max_bond(), hit_cap=bool(m.max_bond() >= cap), norm_deficit=float(abs(1 - m.norm2())),
                seconds=time.time() - t, max_entropy_bits=max(m.entropies() or [0.0]))


def _worker(job, q):
    try:
        q.put(_measure(job))
    except Exception as e:                                              # noqa: BLE001 - reported as a failed point
        q.put(dict(family=job[0], size=job[1], error=f"{type(e).__name__}: {e}"))


def run_point(family, size, max_seconds, cap=None):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=((family, size) if cap is None else (family, size, cap), q))
    t = time.time()
    p.start()
    p.join(max_seconds)
    if p.is_alive():
        p.terminate()
        p.join()
        return dict(family=family, size=size, timed_out=True, seconds=time.time() - t)
    try:
        return q.get(timeout=5)
    except Exception:                                                   # noqa: BLE001
        return dict(family=family, size=size, error="no result")


def statevector_timing(sizes=(12, 16, 20, 22)):
    from qiskit.quantum_info import Statevector
    rows = []
    for n in sizes:
        spec = panel_family(n - 1)
        t = time.time()
        Statevector(reference_circuit(spec))
        rows.append(dict(qubits=n, seconds=time.time() - t, memory_mb=2 ** n * 16 / 1e6))
    return rows


def run_all(max_seconds=60, quick=False):
    out = dict(cap=CAP, truncation=TRUNC, max_seconds=max_seconds)
    out["two_branch_tv"] = two_branch_check()
    out["two_branch_timing"] = two_branch_timing((10, 100, 1000) if quick else (10, 100, 1000, 5000))
    out["excitations"] = {
        "per-panel relief, tau = kappa/sqrt(N), kappa 1": [dict(excitation_stats(np.full(N, 1 / np.sqrt(N))), N=N) for N in (8, 18, 100, 1000)],
        "fixed tau 0.5": [dict(excitation_stats(np.full(N, 0.5)), N=N) for N in (8, 18, 100, 1000)],
        "equator tau pi/2": [dict(excitation_stats(np.full(N, np.pi / 2)), N=N) for N in (8, 18, 100, 1000)],
    }
    out["statevector"] = statevector_timing((12, 16) if quick else (12, 16, 20, 22))
    sizes = dict(panel=(8, 18, 100) if quick else (8, 18, 100, 400), bay=(500, 220, 150) if quick else (500, 220, 150, 100, 60, 40),
                 **{"grid-relief": (2, 3, 4) if quick else (2, 3, 4, 5, 6, 7, 8), "grid-graph": (2, 3, 4) if quick else (2, 3, 4, 5, 6, 7)})
    out["mps"] = []
    for fam, ss in sizes.items():
        stop = False
        for s in ss:
            if stop:
                break
            row = run_point(fam, s, max_seconds)
            out["mps"].append(row)
            print(f"  {fam:12s} size {s:4d}: " + (f"{row['qubits']:4d} qubits, chi {row['max_bond']:4d}{' (CAP)' if row.get('hit_cap') else ''}, "
                  f"norm deficit {row['norm_deficit']:.1e}, {row['seconds']:.1f}s" if "max_bond" in row else str(row)), flush=True)
            if row.get("timed_out") or row.get("error") or row.get("hit_cap"):
                stop = True                                                  # bigger members of this family are past the point of the question
    # the accuracy a capped bond dimension buys on the design the show actually runs (bay window, 150 px facets, 2 facets per qubit)
    out["bay_accuracy"] = []
    for cap in ((8, 16, 32) if quick else (8, 16, 32, 64)):
        row = run_point("bay", 150, max_seconds, cap)
        out["bay_accuracy"].append(row)
        print(f"  bay 150 px, bond cap {cap:3d}: " + (f"norm deficit {row['norm_deficit']:.1e}, {row['seconds']:.1f}s" if "norm_deficit" in row else str(row)), flush=True)
    return out


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    colors = {"panel": "#4c9be8", "bay": "#5fd38d", "grid-relief": "#f2a33a", "grid-graph": "#ff6b5e"}
    labels = {"panel": "per-panel relief (1 polarity qubit)", "bay": "bay window, finer and finer", "grid-relief": "lattice, relief regime (tau 0.5)",
              "grid-graph": "lattice, graph-state regime (equator, CZ)"}
    for fam in colors:
        rows = [r for r in res["mps"] if r["family"] == fam and "max_bond" in r]
        if rows:
            ax[0].plot([r["qubits"] for r in rows], [r["max_bond"] for r in rows], "o-", color=colors[fam], label=labels[fam])
            capped = [r for r in rows if r["hit_cap"]]
            ax[0].plot([r["qubits"] for r in capped], [r["max_bond"] for r in capped], "x", color="k", ms=10)
        to = [r for r in res["mps"] if r["family"] == fam and (r.get("timed_out") or r.get("error"))]
        for r in to:
            ax[0].annotate(f"{fam}\nstopped", (rows[-1]["qubits"] if rows else 1, rows[-1]["max_bond"] if rows else 1), fontsize=7)
    ax[0].axhline(res["cap"], color="#888", ls=":")
    ax[0].set_yscale("log"), ax[0].set_xscale("log")
    ax[0].set_xlabel("qubits"), ax[0].set_ylabel("MPS bond dimension (x: cap reached)")
    ax[0].legend(fontsize=7), ax[0].set_title(f"matrix-product state (truncation {res['truncation']:g}, cap {res['cap']}): where it stops being cheap", fontsize=8)
    for k, rows in res["excitations"].items():
        ax[1].plot([r["N"] for r in rows], [r["k_target"] for r in rows], "o-", label=k)
    ax[1].set_xscale("log"), ax[1].set_yscale("log")
    ax[1].set_xlabel("facets N"), ax[1].set_ylabel("k99: excitations to keep for fidelity 0.99")
    ax[1].legend(fontsize=7), ax[1].set_title("excitation truncation: easy only near |0...0>", fontsize=10)
    ba = [r for r in res.get("bay_accuracy", []) if "norm_deficit" in r]
    if ba:
        ax[3].loglog([r["cap"] for r in ba], [max(r["norm_deficit"], 1e-16) for r in ba], "o-", color="#5fd38d")
        ax[3].set_xlabel("bond dimension cap"), ax[3].set_ylabel("norm deficit (discarded weight)")
        ax[3].set_title("accuracy a bond cap buys: the bay window, 69 qubits", fontsize=9)
    sv = res["statevector"]
    ax[2].semilogy([r["qubits"] for r in sv], [r["seconds"] for r in sv], "o-", color="#888", label="exact statevector (seconds)")
    tb = res["two_branch_timing"]
    ax[2].semilogy([r["N"] for r in tb], [r["seconds_per_frame"] for r in tb], "s-", color="#4c9be8", label="two-branch sampler, per frame (product relief)")
    ax[2].set_xscale("log"), ax[2].set_xlabel("qubits / facets"), ax[2].set_ylabel("seconds")
    ax[2].legend(fontsize=7), ax[2].set_title(f"exact simulation vs the two-branch sampler (TV to exact {res['two_branch_tv']:.3f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/baselines")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--quick", action="store_true", help="small sizes, for a smoke test")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = run_all(a.max_seconds, a.quick)
    with open(os.path.join(a.out, "baselines.json"), "w") as f:
        json.dump(res, f, indent=1)
    figure(res, os.path.join(a.out, "classical_baselines.png"))
    print(f"wrote {a.out}/baselines.json and classical_baselines.png")


if __name__ == "__main__":
    main()
