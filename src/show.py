#!/usr/bin/env python3
"""Standing Light performance mode (handoff §10): operator panel, projector output window and audience screen.

    python -m src.show                                       # the built-in bay window, Superposed Relief (local reference circuit)
    python -m src.show --labels runs/scene/labels.json       # shapes drawn with the pen tool (projector space, no warp needed)
    python -m src.show --source relief --circuit FACETS.qasm # a Moth QDrive circuit for the 15 facet qubits; polarity lifted locally
    python -m src.show --source oracle                       # CLASSICAL rehearsal (the older pool-based sources: oracle, mock, circuit, complementary)

In the relief source nothing chooses a look: the light, each panel's depth and the observation axis are drawn inside the circuit. The
operator panel's science section holds the controls (A independent noise, B dephased depth), the lamp-in-X experiment and the witness run.

Then: open the printed /output URL in a browser window on the PROJECTOR display and press F (or use its button); open /audience on a
companion screen; drive everything from the operator page. Press ? there for the keys. Nothing is sent to Moth unless the server is
started with --allow-spend AND a re-solve is confirmed in the dialog that shows the payload hash and credit cost.
"""
import argparse
import os
import sys
import threading
import webbrowser

from src.capture.labels import load_labels
from src.ui.operator_panel import make_server
from src.ui.session import SOURCES, Session


def parse_size(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        assert w > 0 and h > 0
        return w, h
    except (ValueError, AssertionError):
        raise SystemExit(f"--projector-size must look like 1920x1080, got {text!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="labels.json from the pen tool (default: the built-in synthetic bay window)")
    ap.add_argument("--calibration", help="calibration.json (default: <out>/calibration.json if present, else scale-only)")
    ap.add_argument("--source", choices=SOURCES, default="relief")
    ap.add_argument("--circuit", help="QASM3 file: for --source relief the FACET register (one qubit per facet); for --source circuit a whole classical-family register")
    ap.add_argument("--kappa", type=float, default=1.0, help="relief tilt budget per panel: sum of tau^2 = kappa^2 (visibility ~ exp(-kappa^2/2))")
    ap.add_argument("--n-dirs", type=int, default=4, help="relief: slope-orientation facets per panel (4 gives 5 facets, 15 qubits for the bay window)")
    ap.add_argument("--entangle", type=float, default=0.8, help="relief: strength of the diagonal facet couplings (0 = product relief)")
    ap.add_argument("--contrast", type=float, default=1.15, help="relief: projection tone, a stretch of the lit field about mid-grey")
    ap.add_argument("--engine", choices=["panel", "domain"], default="panel",
                    help="relief engine: panel (one polarity qubit per panel, exact, 12 facet qubits) or domain (one per depth domain, coupled along the geometry graph; "
                         "dozens of qubits as a matrix-product state; --source relief only)")
    ap.add_argument("--seg-len", type=float, default=150.0, help="domain engine: facet length along a polygon edge, px (smaller: more qubits)")
    ap.add_argument("--group-size", type=int, default=1, help="domain engine: facets sharing one polarity qubit (tilt budget per domain; 1 keeps V = cos(tau), the most coherent)")
    ap.add_argument("--tau", type=float, default=0.5, help="domain engine: tilt of each facet, radians (V per domain = cos(tau)^group-size)")
    ap.add_argument("--pol-coupling", type=float, default=0.3, help="domain engine: Ising angle between neighbouring domains' polarity qubits along a boundary (weak: the entanglement lives on the seams)")
    ap.add_argument("--crease-coupling", type=float, default=1.0, help="domain engine: the same Ising angle across a crease between panels (strong)")
    ap.add_argument("--seam-coupling", type=float, default=1.0, help="domain engine: ZZ angle between facets that touch across a crease and among seam facets (pi/2: CZ up to local phases)")
    ap.add_argument("--seam-mix", type=float, default=0.0, help="domain engine: 0..1 moves ONLY the seam facets toward the equator (graph-state regime on the seams, visible relief elsewhere)")
    ap.add_argument("--leaf-tau", type=float, default=0.3, help="domain engine: tilt of the facets of the game's leaf domains (smaller: more visibility, more rounds won)")
    ap.add_argument("--leaf-prefer", choices=["visible", "seam"], default="visible", help="domain engine: leaves of the parity game and of the lamp lock: the most visible domains, or those ON the seam")
    ap.add_argument("--evolve-steps", type=int, default=0, help="domain engine: kicked-Ising steps on the facet graph before the polarity attaches (entanglement grows with each)")
    ap.add_argument("--evolve-zz", type=float, default=0.7), ap.add_argument("--evolve-x", type=float, default=0.5)
    ap.add_argument("--game", action="store_true", help="domain engine: start with the parity game on (some looks are rounds of a Mermin game; the win rate must beat 75%%)")
    ap.add_argument("--game-fraction", type=float, default=0.5, help="the fraction of looks that are rounds")
    ap.add_argument("--lock", type=float, default=1.5708, help="domain engine: coupling of lamp L1 to the polarity qubits with the crater-gauge signs (0: the lamp is a plain coin)")
    ap.add_argument("--tau-mix", type=float, default=0.0, help="domain engine: 0..1 moves the facets toward the equator (graph-state regime; visibility falls)")
    ap.add_argument("--pol-field", type=float, default=0.0, help="domain engine: transverse-field angle on the polarity qubits (0 keeps the circuit IQP-shaped)")
    ap.add_argument("--crease-sign", type=float, default=-1.0, help="domain engine: sign of the polarity coupling across a crease (-1: depths differ across a shared edge)")
    ap.add_argument("--backend", choices=["auto", "exact", "mps"], default="auto", help="domain engine: exact statevector up to 20 qubits, else matrix-product state")
    ap.add_argument("--polarity-basis", choices=["z", "x"], help="basis polarity is read in (default: x for complementary, else z)")
    ap.add_argument("--coupling", choices=["hub", "lamp1", "lamp2"], default="lamp2", help="complementary stand-in: which lamp each polarity qubit follows")
    ap.add_argument("--run", default="runs/first_run", help="run directory with engine results (for the demo slides and re-solve)")
    ap.add_argument("--calib", default="runs/calibration_5x4", help="directory with targets.json and the engine payloads")
    ap.add_argument("--out", help="where calibration.json and photo-test snapshots go (default: next to --labels, else runs/show)")
    ap.add_argument("--projector-size", help="initial projector pixel size WxH; the output window reports its real size once open")
    ap.add_argument("--pool", type=int, default=60000, help="measurement outcomes drawn once and reused")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--port", type=int, default=0, help="default: any free port")
    ap.add_argument("--lan", action="store_true", help="also serve the read-only pages (/output, /audience) to other machines on this network")
    ap.add_argument("--allow-spend", action="store_true", help="let the re-solve dialog submit to Moth (still needs an explicit confirm showing the cost)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)

    labels = load_labels(a.labels) if a.labels else None
    if labels is None:                                    # the built-in bay window, on the grid the calibration targets were made for (19 qubits, under the 20 cap)
        import json
        from src.capture.labels import bay_window_labels
        labels = bay_window_labels()
        try:
            with open(os.path.join(a.calib, "targets.json")) as f:
                labels["grid"]["nx"], labels["grid"]["ny"] = json.load(f)["meta"]["grid"]
        except (OSError, KeyError, ValueError):
            pass
    out = a.out or (os.path.dirname(os.path.abspath(a.labels)) if a.labels else "runs/show")
    from src.projector.calibrate import load_calibration
    cal = load_calibration(a.calibration) if a.calibration else None
    print("building the scene and the first frames (a few seconds)...")
    try:
        session = Session(labels, cal, source=a.source, pol_basis=a.polarity_basis, circuit=a.circuit, coupling=a.coupling, calib_dir=a.calib,
                          run_dir=a.run, out_dir=out, projector_size=parse_size(a.projector_size) if a.projector_size else None, seed=a.seed,
                          pool_size=a.pool, allow_spend=a.allow_spend, log=lambda m: print("  " + m),
                          kappa=a.kappa, n_dirs=a.n_dirs, entangle=a.entangle, contrast=a.contrast, engine=a.engine,
                          game=a.game, game_fraction=a.game_fraction,
                          domain=dict(seg_len=a.seg_len, group_size=a.group_size, tau=a.tau, pol_coupling=a.pol_coupling, lock=a.lock, tau_mix=a.tau_mix,
                                      pol_field=a.pol_field, crease_sign=a.crease_sign, backend=a.backend, crease_coupling=a.crease_coupling,
                                      seam_coupling=a.seam_coupling, seam_mix=a.seam_mix, leaf_tau=a.leaf_tau, leaf_prefer=a.leaf_prefer,
                                      floquet_steps=a.evolve_steps, floquet_zz=a.evolve_zz, floquet_x=a.evolve_x))
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"cannot start: {e}")
    server, token, base = make_server(session, port=a.port, lan=a.lan)
    session.start_cycler()
    p = session.prov
    print(f"\nStanding Light is running.   source: {p['label']}" + ("" if p["quantum_backed"] else "   [CLASSICAL REHEARSAL: no quantum calls]"))
    if session.family == "relief":
        print(f"  {session._budget_text()}")
    print(f"  operator panel   {base}/")
    print(f"  projector window {base}/output      (open on the projector display, press F)")
    print(f"  audience screen  {base}/audience")
    print(f"  qubits in the circuit: {session.budget['total']}   Moth submit from this panel: {'ENABLED (--allow-spend), still asks first' if a.allow_spend else 'off (preview only)'}")
    print("Ctrl+C to stop.")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(base + "/")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        session.stop_cycler()
        server.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
