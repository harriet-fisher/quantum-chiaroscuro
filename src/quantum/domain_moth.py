"""Moth engines for the domain relief: what each can be asked, as payloads (BUILDING A PAYLOAD SENDS NOTHING).

    python -m src.quantum.domain_moth plan
    python -m src.quantum.domain_moth build polarity-qdrive|patch-tomography|echo --out runs/moth     # writes payload_<engine>_<what>.json, prints the exact request

  polarity-qdrive   qdrive-api-v1 (1 credit per job): the polarity register as a *target-correlation* state: the <X>, <Y>, <Z> of every domain qubit and the
                    <XX>, <YY>, <ZZ> of every edge of the domain graph, taken from the fitted ground state (ground_state.py). The returned circuit goes back in
                    as spec.pol_circuit (lift_polarity_circuit) and every other part of the relief is unchanged. Jobs are cut to <= 24 targets: two jobs of 201
                    targets died with engine_timeout.
  patch-tomography  tomography-api-v2 (1 credit): verification of a PATCH of the wall (a few connected domains, <= ~20 qubits, the sub-circuit with its
                    boundary couplings cut: an open-boundary design in its own right, NOT the reduced state of the full wall). Scored against the exact moments
                    of the same patch by moth_engines.score_tomography. The qubit cap of the engine is not documented.
  echo              otoc-echo-v1 (1 credit): dynamic relief. The echo taps F(site, depth) of a kicked-Ising lattice modulate each facet's tilt and slope direction
                    frame by frame (`dynamic_specs`); `local_echo` is our local reading of the engine's description, not its definition.
"""
import argparse
import copy
import os

import numpy as np

from src.quantum import moth_engines as me
from src.quantum.relief_state import ReliefSpec, polarity_amplitudes, reference_circuit, zz_rotation
from src.store.jsonfmt import dumps_compact


# ------------------------------------------------------------------ patches of the wall
def patch_spec(spec, domains):
    """The sub-circuit on the given domains: their facets and polarity qubits renumbered from 0, only the couplings among them, no classical plateaus and no
    lamp lock outside. The couplings to everything else are CUT, so this is a smaller design, not the reduced state of the wall."""
    domains = sorted(int(d) for d in domains)
    dmap = {d: k for k, d in enumerate(domains)}
    fac = [i for i in range(spec.F) if int(spec.panel_of[i]) in dmap]
    fmap = {i: k for k, i in enumerate(fac)}
    edges = [(fmap[i], fmap[j], kind, th) for i, j, kind, th in spec.edges if i in fmap and j in fmap]
    graph_idx = [e for e, (a, b) in enumerate(spec.pol_graph) if a in dmap and b in dmap]
    graph = [(dmap[spec.pol_graph[e][0]], dmap[spec.pol_graph[e][1]]) for e in graph_idx]
    layers = []
    for layer in spec.pol_layers:
        layers.append(dict(zz=None if layer.get("zz") is None else [layer["zz"][e] for e in graph_idx],
                           rx=None if layer.get("rx") is None else [layer["rx"][d] for d in domains]))
    lock = spec.lock_array()
    out = ReliefSpec(spec.tau[fac].copy(), spec.phi[fac].copy(), np.array([dmap[int(spec.panel_of[i])] for i in fac], int), list(spec.panel_angles), edges, spec.rig,
                     list(spec.observations), None, n_groups=len(domains), facet_panel=None if spec.facet_panel is None else spec.facet_panel[fac],
                     pol_graph=graph, pol_layers=layers, lamp_lock=0.0 if lock is None else lock[domains])
    return out


def connected_patch(ds, start, size):
    """`size` domains grown breadth-first from `start` along the domain graph."""
    adj = {d: [] for d in range(ds.n_domains)}
    for a, b, *_ in ds.edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, queue = [start], [start]
    while queue and len(seen) < size:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in seen and len(seen) < size:
                seen.append(v)
                queue.append(v)
    return seen


# ------------------------------------------------------------------ polarity register through QDrive
def polarity_moments(spec):
    """{(qubits tuple, word): value} of the polarity register alone (before the facets attach): X, Y, Z of every domain qubit and XX, YY, ZZ of every edge."""
    from src.quantum import complementary as comp
    D = spec.P
    out = {}
    if D <= 20:
        psi = polarity_amplitudes(spec)
        ev = lambda ops: comp.pauli_expectation(psi, D, ops)
    else:
        from src.quantum.mps import MPS
        from qiskit import QuantumCircuit
        qc = spec.pol_circuit
        if qc is None:
            qc = QuantumCircuit(D)
            for d in range(D):
                qc.h(d)
            for layer in spec.pol_layers:
                for (a, b), th in zip(spec.pol_graph, layer["zz"]):
                    qc.rzz(float(th), a, b)
                if layer.get("rx") is not None:
                    for d, b in enumerate(layer["rx"]):
                        if b:
                            qc.rx(float(b), d)
        m = MPS.from_circuit(qc, 1e-9, 48)
        ev = lambda ops: m.pauli_expectation(ops)
    for d in range(D):
        for c in "xyz":
            out[((d,), c.upper())] = ev({d: c})
    for a, b in spec.pol_graph:
        for c in "xyz":
            out[((a, b), c.upper() * 2)] = ev({a: c, b: c})
    return out


def qdrive_polarity_targets(spec):
    """The polarity register's moments as QDrive targets, largest magnitude first."""
    rows = [(abs(v), {"qubits": list(q), "expvals": {w: round(float(v), 4)}}) for (q, w), v in polarity_moments(spec).items()]
    return [t for _, t in sorted(rows, key=lambda r: -r[0])]


def qdrive_polarity_payloads(spec, per_job=24, update_method="spectral", seed=7, shots=1024):
    """The polarity-register program cut into jobs of at most `per_job` targets (job k > 0 chains on job k-1's circuit: that flow is UNVERIFIED)."""
    targets = qdrive_polarity_targets(spec)
    edges = [list(e) for e in sorted({tuple(sorted(e)) for e in spec.pol_graph})]
    jobs = []
    for k in range(0, len(targets), per_job):
        body = dict(machine="aer", n_qubits=spec.P, coupling_map=edges, update_method=update_method, shots=shots, tomography=0, sample=False, seed=seed,
                    targets=targets[k:k + per_job] + [None])
        jobs.append(dict(params=body, needs_initial_circuit_from=None if k == 0 else (k // per_job) - 1))
    return jobs


def lift_polarity_circuit(spec, qasm_text):
    """Whatever circuit QDrive returns for the polarity register becomes the polarity preparation of the relief (the facets attach to it as before).
    Returns (new spec, report) with the fidelity to the intended state and the moment errors when D <= 20. Reported even when poor."""
    from qiskit import qasm3
    qc = qasm3.loads(qasm_text)
    if qc.num_qubits != spec.P:
        raise ValueError(f"circuit has {qc.num_qubits} qubits, the polarity register has {spec.P}")
    new = copy.copy(spec)
    new.pol_circuit = qc
    report = dict(n_qubits=qc.num_qubits)
    if spec.P <= 20:
        ref = polarity_amplitudes(spec)
        got = polarity_amplitudes(new)
        report["fidelity"] = float(abs(np.vdot(ref, got)) ** 2 / (np.vdot(got, got).real * np.vdot(ref, ref).real))
    return new, report


# ------------------------------------------------------------------ verification of a patch
def patch_tomography_payload(spec, domains, shots=4096, provider="aer"):
    """(payload, patch spec): the tomography request for a patch of the wall, with the pairs that carry the physics."""
    patch = patch_spec(spec, domains)
    if patch.n_sim > 24:
        raise ValueError(f"a patch of {patch.n_sim} qubits is too big to verify exactly; choose fewer domains")
    return me.tomography_payload(patch, shots=shots, provider=provider), patch


# ------------------------------------------------------------------ dynamic relief from the echo engine
def facet_sites(ds, scene, width, height):
    """The lattice site of every quantum facet: its centroid's cell in a width x height grid over the canvas."""
    out = []
    for f in ds.facets.facets[:ds.facets.n_quantum]:
        cx = min(width - 1, int(f.centroid[0] / scene.W * width))
        cy = min(height - 1, int(f.centroid[1] / scene.H * height))
        out.append(cy * width + cx)
    return out


def dynamic_specs(spec, sites, taps, depth, n_sites, tau_gain=0.8, phi_gain=0.5):
    """One spec per echo depth: facet i's tilt is its geometry tilt times (0.4 + tau_gain s_i(d)), s = |F|/max |F| at its lattice site, and its slope
    direction is rotated by phi_gain arg F. The mapping is OURS (taps_to_relief's Gauss-map reading), not the engine's definition. The azimuth
    compensation for the diagonal couplings is recomputed for every frame."""
    tau_t, phi_t, _ = me.taps_to_relief(taps, n_sites, depth, tau_max=1.0)
    base_phi = spec.phi + zz_rotation(spec.tau, spec.edges)                     # undo the compensation to get the geometry's own azimuths
    out = []
    for d in range(depth):
        s = np.array([tau_t[d, k] for k in sites])
        a = np.array([phi_t[d, k] for k in sites])
        tau = np.minimum(spec.tau * (0.4 + tau_gain * s), 1.1)
        phi = base_phi + phi_gain * a
        sp = copy.copy(spec)
        sp.tau, sp.phi = tau, phi - zz_rotation(tau, spec.edges)
        out.append(sp)
    return out


# ------------------------------------------------------------------ CLI
PLAN = [
    ("qdrive-api-v1", 1, "polarity register as a target-correlation state (ground state of the geometry-set Ising model), <= 24 targets per job"),
    ("tomography-api-v2", 1, "verify a PATCH of the wall (<= ~20 qubits, boundary couplings cut) against its exact moments"),
    ("otoc-echo-v1", 1, "dynamic relief: echo taps modulate each facet's tilt and slope direction frame by frame"),
]


def _build(args):
    from src.capture.labels import bay_window_labels, fill_defaults, load_labels, validate
    from src.geometry.domains import build_domains
    from src.geometry.planes import build_scene
    from src.quantum.domain_state import spec_from_domains
    labels = load_labels(args.labels) if args.labels else validate(fill_defaults(bay_window_labels()))
    scene = build_scene(labels)
    ds = build_domains(scene, seg_len=args.seg_len, group_size=args.group_size)
    return scene, ds, spec_from_domains(ds, scene, entangle=args.entangle, pol_coupling=args.pol_coupling, lock=args.lock)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    b = sub.add_parser("build")
    b.add_argument("what", choices=["polarity-qdrive", "patch-tomography", "echo"])
    b.add_argument("--out", default="runs/moth")
    b.add_argument("--labels")
    b.add_argument("--seg-len", type=float, default=150.0)
    b.add_argument("--group-size", type=int, default=2)
    b.add_argument("--entangle", type=float, default=0.8)
    b.add_argument("--pol-coupling", type=float, default=0.5)
    b.add_argument("--lock", type=float, default=1.5708)
    b.add_argument("--patch-size", type=int, default=6, help="patch-tomography: number of connected domains")
    b.add_argument("--ground", action="store_true", help="polarity-qdrive: target the fitted ground state instead of the closed-form Ising layer")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        for eid, cr, role in PLAN:
            print(f"{eid:22s}{cr:>4d} credit  {role}")
        print("\nNothing here sends anything. Preview: python -m src.quantum.moth_client preview FILE --engine ID")
        return
    from src.quantum.moth_client import MothClient
    os.makedirs(a.out, exist_ok=True)
    scene, ds, spec = _build(a)
    if a.what == "polarity-qdrive":
        if a.ground:
            from src.quantum.ground_state import ground_state_spec
            spec, rep = ground_state_spec(spec, ds)
            print(f"fitted ground state: energy {rep['energy']:.4f}" + (f", exact {rep['exact_energy']:.4f}, fidelity {rep['fidelity_to_exact']:.3f}" if "exact_energy" in rep else ""))
        jobs = qdrive_polarity_payloads(spec)
        items = [("qdrive-api-v1", {"params": jobs[0]["params"]})]
        print(f"{len(jobs)} job(s) of <= 24 targets for {spec.P} domain qubits; only job 0 is self-contained (the chaining flow is unverified)")
    elif a.what == "patch-tomography":
        from src.quantum.domain_state import leaf_domains
        payload, patch = patch_tomography_payload(spec, connected_patch(ds, leaf_domains(ds)[0], a.patch_size))
        print(f"patch of {patch.F} facets + {patch.P} domain qubits = {patch.n_sim} qubits")
        items = [("tomography-api-v2", payload)]
    else:
        items = [("otoc-echo-v1", me.echo_payload())]
    c = MothClient()
    for eid, payload in items:
        path = os.path.join(a.out, f"payload_{eid}_domain-{a.what}.json")
        with open(path, "w") as f:
            f.write(dumps_compact(payload) + "\n")
        print(f"wrote {path}\n{c.prepare(eid, payload['params']).describe(max_ops=3)[:1500]}\n(nothing was sent)\n")


if __name__ == "__main__":
    main()
