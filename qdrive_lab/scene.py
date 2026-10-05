"""Rebuild the living-room scene's exact Z-statistics oracle state from runs/harrietlivingroom (free, local)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.capture.labels import load_labels
from src.geometry.planes import build_scene
from src.quantum.sampler_mock import Sampler
from src.quantum import sampler_local as sl

RUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "harrietlivingroom")


def load():
    t = json.load(open(os.path.join(RUN, "calib", "targets.json")))
    m = t["meta"]; s = m["sampler"]
    labels = load_labels(os.path.join(RUN, "labels.json"))
    NX, NY = m["grid"]
    sampler = Sampler(build_scene(labels), NX, NY, kh=s["kh"], kf=s["kf"], J_in=s["J_in"], J_cross=s["J_cross"], J_pol=s["J_pol"],
                      min_cover=labels["grid"]["min_cover"])
    layout = dict(m["qubit_map"])
    psi = sl.oracle_state(sampler, layout)
    return t, sampler, layout, psi


def probs(psi):
    p = np.abs(psi) ** 2
    return p / p.sum()


def zword_moment(p, n, qubits):
    """<prod Z_q> for qubits (little-endian index: qubit q is bit q), from a probability vector over 2^n basis states."""
    idx = np.arange(len(p))
    sign = np.ones(len(p))
    for q in qubits:
        sign *= 1 - 2 * ((idx >> q) & 1)
    return float(p @ sign)

if __name__ == "__main__":
    t, sampler, layout, psi = load()
    p = probs(psi); n = layout["n_qubits"]
    print("n", n, "norm", float(np.linalg.norm(psi)), "patches", sampler.n, "pol edges", sampler.pol_edges)
    worst = 0
    for r in t["relationships"]:
        ex = zword_moment(p, n, r["qubits"]); worst = max(worst, abs(ex - r["ZZ"]))
    print("max |oracle ZZ - stored target| over", len(t["relationships"]), "pairs:", round(worst, 4), "(stored targets are MC estimates, se ~0.003)")
    # independence of the two components?
    A = list(range(0, 11)); B = list(range(11, 19))
    cross = max(abs(zword_moment(p, n, [a, b]) - zword_moment(p, n, [a]) * zword_moment(p, n, [b])) for a in A for b in B)
    print("max cross-component covariance <Z_a Z_b> - <Z_a><Z_b>:", round(cross, 6))
