#!/usr/bin/env python3
"""Generate a stream of lighting frames from a quantum state (handoff §9.2, §9.4) and write them with a provenance manifest.

    python -m src.quantum.make_frames --source circuit runs/bay_run/qdrive/circuit.qasm --n 48 --K 8 --out runs/bay_run/frames
    python -m src.quantum.make_frames --source circuit CIRCUIT --polarity-basis x      # complementary-polarity experiment
    python -m src.quantum.make_frames --source oracle  --out runs/oracle_frames        # known-answer reference, NOT from Moth
    python -m src.quantum.make_frames --source mock    --out runs/mock_frames          # classical stand-in

Sources:
  circuit  a QASM3 circuit returned by QDrive, simulated locally (Aer). One register measurement supplies the lamps (lighting
           world), the polarity bits and the patch bits of every frame, so their correlations are the circuit's own.
  oracle   sqrt(P) state whose Z statistics equal the classical mock's distribution exactly. Built locally; used to check the
           pipeline and as the best a Z-type target set can do. It is labelled in the manifest and never called a Moth result.
  mock     classical Gibbs sampler (src/quantum/sampler_mock.py).
graph-v1 cannot be a source: it returns tomography and the 20 most likely bitstrings, not a state or circuit.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image

from src.capture.labels import DEFAULT_CALIB_DIR, bay_window_labels, fill_defaults, load_labels, validate
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import allocate
from src.quantum import sampler_local as sl
from src.quantum.ab_compare import mock_pool
from src.quantum.sampler_mock import Sampler
from src.texture.frames import FrameGenerator


def contact_sheet(imgs, cols=6, scale=0.35):
    h, w = imgs[0].shape
    th, tw = int(h * scale), int(w * scale)
    rows = -(-len(imgs) // cols)
    sheet = Image.new("L", (cols * tw, rows * th), 0)
    for i, im in enumerate(imgs):
        sheet.paste(Image.fromarray((im * 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["circuit", "oracle", "mock", "graph-v1"], required=True)
    ap.add_argument("circuit", nargs="?", help="QASM3 file (for --source circuit)")
    ap.add_argument("--calib", default=DEFAULT_CALIB_DIR, help="directory with targets.json (grid and qubit layout)")
    ap.add_argument("--labels", help="labels.json (default: built-in bay window)")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--K", type=int, default=8, help="shots averaged per frame (hardness)")
    ap.add_argument("--pool", type=int, default=100000, help="register measurements drawn once and reused")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--hold-lamp", nargs=2, type=int, metavar=("L1", "L2"), help="fix the lighting world (+1/-1 each)")
    ap.add_argument("--polarity-basis", choices=["z", "x", "y"], default="z", help="basis the polarity qubits are measured in")
    ap.add_argument("--out", default="runs/frames")
    a = ap.parse_args(argv)

    if a.source == "graph-v1":
        raise SystemExit("graph-v1 returns tomography and the 20 most likely bitstrings, not a state or circuit, so it cannot generate "
                         "frames. Use the QDrive circuit (--source circuit) once QDrive has returned one.")
    with open(os.path.join(a.calib, "targets.json")) as f:
        meta = json.load(f)["meta"]
    NX, NY = meta["grid"]
    labels = load_labels(a.labels) if a.labels else validate(fill_defaults(bay_window_labels()))
    scene = build_scene(labels)
    sampler = Sampler(scene, NX, NY, min_cover=labels["grid"]["min_cover"])
    layout = allocate(sampler.n, sampler.n_panels)
    rng = np.random.default_rng(a.seed)
    basis = {q: a.polarity_basis for q in layout["polarity"]} if a.polarity_basis != "z" else None

    provenance = dict(source=a.source, polarity_basis=a.polarity_basis)
    if a.source == "circuit":
        if not a.circuit:
            raise SystemExit("--source circuit needs the path to a QASM3 file")
        text = open(a.circuit).read()
        src = sl.source_from_qasm(text, layout, basis, label=f"circuit {a.circuit}")
        pool = src.pool(a.pool, rng)
        provenance.update(circuit=a.circuit, circuit_sha256=hashlib.sha256(text.encode()).hexdigest(), simulated="locally with Aer statevector",
                          quantum_backed=True)
    elif a.source == "oracle":
        src = sl.StateSource(sl.oracle_state(sampler, layout), layout, basis, label="oracle")
        pool = src.pool(a.pool, rng)
        provenance.update(note="classical sqrt(P) oracle built locally from the mock model: NOT a Moth result", quantum_backed=False)
    else:
        if basis:
            raise SystemExit("--polarity-basis needs a quantum state; the mock has no basis to rotate")
        pool = mock_pool(sampler, a.pool // 4, rng)
        provenance.update(note="classical Gibbs sampler", quantum_backed=False)

    gen = FrameGenerator(scene, sampler, pool)
    os.makedirs(a.out, exist_ok=True)
    imgs, manifest = [], []
    hold = tuple(a.hold_lamp) if a.hold_lamp else None
    for i, (img, d) in enumerate(gen.frames(a.n, a.K, rng, lamp=hold)):
        assert np.all(img[scene.impenetrable] == 0), "impenetrable pixels must be exactly black"
        Image.fromarray(np.round(img * 255).astype(np.uint8)).save(os.path.join(a.out, f"frame_{i:04d}.png"))
        imgs.append(img)
        manifest.append(dict(frame=i, lamp=list(d.lamp), polarity=list(d.pol), K=d.K, shot_ids=d.shot_ids[:3].tolist()))
    contact_sheet(imgs).save(os.path.join(a.out, "contact_sheet.png"))
    worlds = {}
    for m in manifest:
        worlds[str(tuple(m["lamp"]))] = worlds.get(str(tuple(m["lamp"])), 0) + 1
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n=a.n, K=a.K, pool=a.pool, seed=a.seed, grid=[NX, NY],
                       n_qubits=layout["n_qubits"], qubit_map=layout, held_lamp=hold, provenance=provenance, lighting_worlds=worlds,
                       frames=manifest), f, indent=1)
    print(f"{a.n} frames (K={a.K}) -> {a.out}  [{provenance.get('note', 'quantum-backed: ' + str(provenance.get('quantum_backed')))}]")
    print("lighting worlds drawn:", worlds, "| glass pixels exactly 0 in every frame: True")


if __name__ == "__main__":
    main()
