"""The src/ port must reproduce mock/standing_light_mock.py exactly (bit for bit).

    python -m tests.check_port_matches_mock                     # numeric comparison against the original module
    python -m tests.check_port_matches_mock --figs DIR_A DIR_B  # also compare fig*.png pixels between two directories
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

from src.capture.labels import bay_window_labels, fill_defaults, load_labels, save_labels, validate
from src.geometry.planes import build_scene
from src.quantum.sampler_mock import Sampler
from src.texture.compose import render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_original():
    spec = importlib.util.spec_from_file_location("standing_light_mock", os.path.join(ROOT, "mock", "standing_light_mock.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def check(name, ok, failures):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        failures.append(name)


def compare_numeric(labels, tag, failures):
    print(f"[{tag}]")
    orig = load_original()
    scene = build_scene(labels)
    for new, old, name in [(scene.region, orig.REGION, "region"), (scene.blocked, orig.GLASS, "glass"),
                           (scene.frame, orig.FRAME, "frame"), (scene.impenetrable, orig.IMPENETRABLE, "impenetrable"),
                           (scene.t, orig.T, "depth t"), (scene.lo, orig.LO, "window lo"),
                           (scene.hi, orig.HI, "window hi"), (scene.relief, orig.R, "relief R")]:
        check(f"mask/field {name} identical", same(new, old), failures)

    new_s, old_s = Sampler(scene, 6, 4), orig.Sampler(6, 4)
    check("patch cells identical (box, panel)", [(c["i"], c["j"], c["box"], c["k"]) for c in new_s.cells] ==
          [(c["i"], c["j"], c["box"], c["k"]) for c in old_s.cells], failures)
    for a, b, name in [(new_s.J, old_s.J, "couplings J"), (new_s.hub1, old_s.hub1, "hub1"), (new_s.hub2, old_s.hub2, "hub2")]:
        check(f"sampler {name} identical", same(a, b), failures)
    check("qubit count identical", new_s.n_qubits == old_s.n_qubits, failures)

    pol_ok = all(new_s.sample_polarity(np.random.default_rng(s)) == old_s.sample_polarity(np.random.default_rng(s)) for s in range(50))
    check("polarity draws identical (50 seeds)", pol_ok, failures)

    # every frame the five mock figures render, same seeds and parameters as mock.main()
    frames = [((l), (-1, -1, -1), 40, 3) for l in [(1, 1), (-1, 1), (1, -1), (-1, -1)]]
    frames += [((1, 1), p, 40, 5) for p in [(-1, -1, -1), (1, 1, 1), (-1, 1, -1), (1, -1, 1)]]
    frames += [((1, 1), (-1, -1, -1), K, 11) for K in [1, 3, 8, 32, 128]]
    rng = np.random.default_rng(2026)
    for _ in range(12):
        lamp = tuple(rng.choice([-1, 1], 2)); pol = old_s.sample_polarity(rng)
        K = int(rng.choice([1, 4, 16, 64])); frames.append((lamp, pol, K, int(rng.integers(1e9))))
    bad = 0
    for lamp, pol, K, seed in frames:
        a = render(scene, new_s, lamp, pol, K, np.random.default_rng(seed))
        b = orig.render(old_s, lamp, pol, K, np.random.default_rng(seed))
        bad += not same(a, b)
        if not same(a, b):
            print(f"       max |diff| = {np.abs(a - b).max():.3g} at lamp={lamp} pol={pol} K={K}")
    check(f"{len(frames)} rendered frames identical", bad == 0, failures)
    check("impenetrable pixels exactly 0 in the last frame", bool(np.all(a[scene.impenetrable] == 0)), failures)


def compare_figs(dir_a, dir_b, failures):
    import matplotlib.image as mpimg
    print(f"[figures] {dir_a}  vs  {dir_b}")
    for name in sorted(f for f in os.listdir(dir_a) if f.startswith("fig") and f.endswith(".png")):
        pb = os.path.join(dir_b, name)
        if not os.path.exists(pb):
            check(f"{name} exists in {dir_b}", False, failures)
            continue
        check(f"{name} pixels identical", same(mpimg.imread(os.path.join(dir_a, name)), mpimg.imread(pb)), failures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs=2, metavar=("DIR_A", "DIR_B"))
    args = ap.parse_args()
    failures = []

    labels = validate(fill_defaults(bay_window_labels()))
    compare_numeric(labels, "built-in bay window labels", failures)

    path = os.path.join(ROOT, "runs", "_roundtrip_labels.json")
    save_labels(labels, path)
    compare_numeric(load_labels(path), "labels.json written to disk and read back", failures)
    os.remove(path)

    if args.figs:
        compare_figs(*args.figs, failures)
    print("\nPASS: port reproduces the mock exactly" if not failures else f"\nFAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
