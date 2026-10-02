"""Superposed Relief (v2 handoff, section 6 acceptance tests plus the properties found while building it).

    python -m unittest tests.test_relief -v
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.capture.labels import bay_window_labels, fill_defaults, validate
from src.geometry.facets import build_facets, panel_budget
from src.geometry.planes import build_scene
from src.quantum import relief_witness as rw
from src.quantum import sampler_local as sl
from src.quantum.complementary import pauli_expectation
from src.quantum.relief_state import (LightRig, ReliefSpec, ReliefState, equal_up_to_phase, facet_state, full_circuit, full_state, lift_polarity,
                                      reference_circuit, spec_from_facets, statevector_of, transverse_shrink)
from src.texture.relief_compose import ReliefComposer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = FACETS = SPEC = STATE = COMPOSER = None


def setUpModule():
    global SCENE, FACETS, SPEC, STATE, COMPOSER
    SCENE = build_scene(validate(fill_defaults(bay_window_labels())))
    FACETS = build_facets(SCENE)
    SPEC = spec_from_facets(FACETS, SCENE, entangle=0.8)
    STATE = ReliefState(SPEC)
    COMPOSER = ReliefComposer(SCENE, FACETS)


class FixedRig(LightRig):
    """One light for every world and panel: the reference simulation's 50 degree light from +x."""
    def vector(self, world):
        b = np.deg2rad(50)
        return np.array([np.sin(b), 0, np.cos(b)])

    def panel_local(self, world, angle):
        return self.vector(world)


def toy_spec(gammas=(0, 30, 60, 90, 120, 150, 180)):
    tau = np.array([.55, .55, .35, .35, .35, .35, .55, .55])
    phi = np.array([0, 0, 0, 0, np.pi, np.pi, np.pi, np.pi])
    return ReliefSpec(tau, phi, np.zeros(8, int), [0.0], [], FixedRig(), [(np.deg2rad(g), 0.0) for g in gammas])


class Refsim(unittest.TestCase):
    def test_refsim_runs_and_prints_the_handoffs_numbers(self):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "superposed_relief_refsim.py")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("0.963095", out.stdout)
        m = re.search(r"<X_B> = ([0-9.]+)\s+prod cos\(tau\) = ([0-9.]+)", out.stdout)
        self.assertAlmostEqual(float(m.group(1)), float(m.group(2)), places=5)

    def test_the_engine_reproduces_the_refsim_bevel_table(self):
        """Handoff section 2.5 / refsim check 4: contrast +.332 +.238 +.122 0 -.122 -.238 -.332 and P(outcome 0) .500 .603 .678 .706 ..."""
        st = ReliefState(toy_spec())
        want_c = [0.332, 0.238, 0.122, 0.0, -0.122, -0.238, -0.332]
        want_p = [0.500, 0.603, 0.678, 0.706, 0.678, 0.603, 0.500]
        for g in range(7):
            m = st.marginals(g, w=0, o=0)
            self.assertAlmostEqual(m[:4].mean() - m[4:].mean(), want_c[g], places=3)
            self.assertAlmostEqual(st.polarity_probs(g)[0], want_p[g], places=3)

    def test_lambert_is_the_born_rule(self):
        """One facet, one light: P(lit) = (1 + n.l)/2 for the decided raised relief (gamma = 0, outcome 0)."""
        sp = ReliefSpec([0.6], [0.4], [0], [0.0], [], FixedRig(), [(0.0, 0.0)])
        n = sp.bloch()[0]
        l = FixedRig().vector(0)
        self.assertAlmostEqual(ReliefState(sp).marginals(0, w=0, o=0)[0], (1 + n @ l) / 2, places=9)


class Conventions(unittest.TestCase):
    """The qubit order, the spin sign and the relief flip are pinned: a real circuit would silently scramble the picture if they slipped."""

    def test_a_tilted_to_the_south_pole_facet_is_bit_one_little_endian(self):
        sp = ReliefSpec([np.pi, 0.0], [0.0, 0.0], [0, 0], [0.0], [], FixedRig(), [(0.0, 0.0)])
        psi = facet_state(sp)
        self.assertAlmostEqual(abs(psi[1]) ** 2, 1.0)                  # facet 0 = |1>, facet 1 = |0>: index 1
        self.assertAlmostEqual(sl.spins(np.array([1]), [0])[0, 0], -1)  # bit 1 is spin -1

    def test_parity_operator_flips_every_slope_and_keeps_the_tilt(self):
        """Z maps Bloch (x, y, z) -> (-x, -y, z): the sunk relief is P|psi>, built as CZ(B, f)."""
        sp = ReliefSpec([0.5, 0.9, 0.3], [0.2, 2.0, -1.0], [0, 0, 0], [0.0], [], FixedRig(), [(0.0, 0.0)])
        psi_f = facet_state(sp)
        Psi = lift_polarity(psi_f, sp)                                 # (2^P, 2^F)
        sunk = Psi[1] * np.sqrt(2)
        sunk /= np.linalg.norm(sunk)
        for q in range(3):
            def bloch(v):
                return [pauli_expectation(v, 3, {q: c}) for c in "xyz"]
            raised = bloch(psi_f)
            flipped = bloch(sunk)
            np.testing.assert_allclose(flipped, [-raised[0], -raised[1], raised[2]], atol=1e-12)

    def test_qiskit_reference_circuit_equals_the_numpy_state(self):
        sp = ReliefSpec([0.5, 0.4, 0.6, 0.3], [0.2, -1.0, 2.5, 3.0], [0, 0, 1, 1], [35.0, -35.0],
                        [(0, 1, "xy", 0.7), (2, 3, "xy", 0.5), (1, 2, "zz", -0.6)])
        a = statevector_of(reference_circuit(sp))
        b = full_state(sp)
        self.assertTrue(equal_up_to_phase(a, b, tol=1e-10))
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-10)

    def test_the_whole_circuit_sampled_by_aer_matches_the_exact_distribution(self):
        """Lamp and observation registers inside the circuit (what a QPU would run) vs the numpy pipeline, as a total-variation distance."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        sp = ReliefSpec([0.5, 0.4, 0.6, 0.3], [0.2, -1.0, 2.5, 3.0], [0, 0, 1, 1], [35.0, -35.0], [(0, 1, "xy", 0.7), (1, 2, "zz", -0.6)])
        shots = 200_000
        sim = AerSimulator(method="statevector")
        counts = sim.run(transpile(full_circuit(sp), sim), shots=shots, seed_simulator=3).result().get_counts()
        emp = {}
        for key, c in counts.items():
            co, cl, cb, cf = key.split(" ")                             # registers print last-added first
            emp[(int(co, 2), int(cl, 2), int(cb, 2), int(cf, 2))] = c
        st = ReliefState(sp)
        tv = 0.0
        for g in range(4):
            pr = st.cell(g)
            for w in range(4):
                for o in range(2 ** sp.P):
                    for x in range(2 ** sp.F):
                        tv += abs(pr[w, o, x] / 4 - emp.get((g, w, o, x), 0) / shots) / 2
        self.assertLess(tv, 0.03)


class Facets(unittest.TestCase):
    def test_every_projectable_pixel_has_exactly_one_facet_and_glass_has_none(self):
        self.assertTrue(np.all(FACETS.label[SCENE.frame] >= 0))
        self.assertTrue(np.all(FACETS.label[~SCENE.frame] == -1))

    def test_the_budget_is_the_handoffs(self):
        self.assertEqual((SPEC.F, SPEC.P, SPEC.n_sim, SPEC.n_full), (12, 3, 15, 19))          # 4 band facets x 3 panels; the 3 plateaus are classical
        self.assertEqual((FACETS.n, FACETS.n_quantum, SPEC.n_classical), (15, 12, 3))
        for k, b in panel_budget(FACETS).items():
            self.assertAlmostEqual(b["sum_tau2"], FACETS.kappa ** 2, places=6)
            self.assertAlmostEqual(b["visibility"], np.prod(np.cos(FACETS.tau[FACETS.panel_of == k])), places=9)

    def test_opposite_slopes_are_separate_facets_and_the_plateau_is_flat(self):
        for k in range(SCENE.n_panels):
            band = [FACETS.facets[i] for i in FACETS.of_panel(k) if FACETS.facets[i].kind == "band"]
            self.assertEqual(len(band), 4)
            self.assertTrue(all(f.tau > 0.3 for f in band))
            flat = [FACETS.facets[i] for i in FACETS.of_panel(k) if FACETS.facets[i].kind == "interior"]
            self.assertTrue(all(f.tau < 0.05 for f in flat))


class SeamsAndPlateaus(unittest.TestCase):
    def test_band_facets_of_neighbouring_panels_touch_across_the_shared_edge(self):
        """The slope used to be differenced across the seam, which halved it there, so the seam pixels were 'flat' and only the plateaus touched:
        the facet graph had no cross-panel edge between tilted facets and the panels were exactly unentangled."""
        cross = [(i, j) for i, j, k, n in FACETS.edges if FACETS.facets[i].panel != FACETS.facets[j].panel]
        self.assertGreaterEqual(len(cross), 2)
        for i, j in cross:
            self.assertTrue(FACETS.facets[i].tau > 0.3 and FACETS.facets[j].tau > 0.3, (i, j))
        self.assertTrue(any(kind == "zz" for i, j, kind, _ in SPEC.edges if SPEC.panel_of[i] != SPEC.panel_of[j]))

    def test_the_panels_are_now_entangled_with_each_other(self):
        psi = STATE.Psi.reshape(-1)
        n = SPEC.n_sim
        keep = SPEC.facets_of(0) + [SPEC.F]
        t = psi.reshape((2,) * n)
        m = np.moveaxis(t, [n - 1 - q for q in keep], list(range(len(keep)))).reshape(2 ** len(keep), -1)
        ev = np.clip(np.linalg.eigvalsh(m @ m.conj().T), 1e-15, 1)
        self.assertGreater(float(-(ev * np.log2(ev)).sum()), 1e-3)

    def test_plateau_facets_carry_no_qubit_and_are_lit_by_the_lambert_value(self):
        self.assertTrue(all(f.classical for f in FACETS.facets[FACETS.n_quantum:]))
        self.assertTrue(all(not f.classical for f in FACETS.facets[:FACETS.n_quantum]))
        rng = np.random.default_rng(0)
        d = STATE.draw(100000, rng, g=0, world=1)
        for c, panel in enumerate(SPEC.classical_panel):
            want = (1 + SPEC.rig.panel_local(1, SPEC.panel_angles[panel])[2]) / 2
            self.assertAlmostEqual(d.lit[SPEC.F + c], want, delta=0.01)

    def test_keeping_the_plateau_qubits_is_still_possible(self):
        fk = build_facets(SCENE, interior="keep")
        self.assertEqual((fk.n, fk.n_quantum), (15, 15))
        self.assertEqual(spec_from_facets(fk, SCENE).F, 15)


class Couplings(unittest.TestCase):
    def test_diagonal_couplings_keep_every_facets_azimuth_and_shrink_its_transverse_length_by_the_closed_form(self):
        psi_f = facet_state(SPEC)
        shrink = transverse_shrink(SPEC.tau, SPEC.edges)
        for q in range(SPEC.F):
            bx, by, bz = (pauli_expectation(psi_f, SPEC.F, {q: c}) for c in "xyz")
            self.assertAlmostEqual(bz, np.cos(FACETS.tau[q]), places=9)                      # populations untouched: tilt, hence visibility, untouched
            if FACETS.tau[q] > 0.2:
                self.assertAlmostEqual(np.hypot(bx, by), np.sin(FACETS.tau[q]) * shrink[q], places=9)
                d = (np.arctan2(by, bx) - FACETS.phi[q] + np.pi) % (2 * np.pi) - np.pi
                self.assertAlmostEqual(d, 0.0, places=9)                                     # the geometry's normal is where the geometry put it

    def test_the_default_circuit_is_iqp_shaped(self):
        """Single-qubit layer, then commuting diagonal gates (CZ, RZZ), then (in the full circuit) single-qubit measurement rotations."""
        names = {inst.operation.name for inst in reference_circuit(SPEC).data}
        self.assertEqual(names, {"h", "ry", "rz", "rzz", "cz"})

    def test_the_bevel_is_exactly_flat_at_the_equator_for_every_polarity_outcome_with_diagonal_couplings(self):
        xs = [i for i, f in enumerate(FACETS.facets) if f.kind == "band" and abs(np.cos(f.phi)) > 0.9]
        sg = np.array([np.sign(np.cos(FACETS.facets[i].phi)) for i in xs])
        for o in range(2 ** SPEC.P):
            m = STATE.marginals(3, w=1, o=o)
            self.assertLess(abs(float((m[xs] * sg).mean())), 2e-3, o)

    def test_exchange_couplings_are_an_opt_in_that_gives_up_the_flat_equator(self):
        """The xy exchange keeps the visibility law and the stabilizer but leaves a residual bevel at 90 degrees: documented, not the default."""
        sp = spec_from_facets(FACETS, SCENE, entangle=0.6, coupling="xy")
        np.testing.assert_allclose(rw.stabilizers(full_state(sp), sp), 1.0, atol=1e-9)
        st = ReliefState(sp)
        xs = [i for i, f in enumerate(FACETS.facets) if f.kind == "band" and abs(np.cos(f.phi)) > 0.9]
        sg = np.array([np.sign(np.cos(FACETS.facets[i].phi)) for i in xs])
        self.assertGreater(abs(float((st.marginals(3, w=1, o=7)[xs] * sg).mean())), 0.05)


class Scaling(unittest.TestCase):
    def test_visibility_law_with_a_fixed_budget_across_n(self):
        """tau_i = kappa / sqrt(N): V stays ~constant from N = 8 to 18 (and equals prod cos); a fixed tau collapses (handoff section 5)."""
        kappa, vs = 0.99, []
        for N in (8, 10, 12, 14, 16, 18):
            sp = ReliefSpec(rw.scaled_tilts(kappa, N), np.zeros(N), np.zeros(N, int), [0.0], [], FixedRig(), [(0.0, 0.0)])
            v_state = pauli_expectation(full_state(sp), sp.n_sim, {N: "x"})
            self.assertAlmostEqual(v_state, np.prod(np.cos(sp.tau)), places=9)
            vs.append(v_state)
        self.assertLess(max(vs) / min(vs) - 1, 0.05)
        fixed = [np.cos(0.35) ** N for N in (8, 18)]
        self.assertLess(fixed[1] / fixed[0], 0.6)

    def test_law_holds_with_entanglers_xy_within_a_panel_and_zz_across(self):
        for p in range(SPEC.P):
            v = pauli_expectation(full_state(SPEC), SPEC.n_sim, {SPEC.F + p: "x"})
            self.assertAlmostEqual(v, SPEC.visibility(p), places=9)
        self.assertGreater(len(SPEC.edges), 10)

    def test_the_stabilizer_is_one_for_any_parity_preserving_coupling(self):
        np.testing.assert_allclose(rw.stabilizers(full_state(SPEC), SPEC), 1.0, atol=1e-9)


class Observation(unittest.TestCase):
    def test_bevel_contrast_is_monotone_odd_and_zero_at_ninety_degrees(self):
        """Handoff test 5, with the observation forced to each value (tests only; the show never sets it)."""
        st = ReliefState(toy_spec((0, 30, 60, 90)))
        c = [abs(st.marginals(g, w=0, o=0)[:4].mean() - st.marginals(g, w=0, o=0)[4:].mean()) for g in range(4)]
        self.assertTrue(all(a > b for a, b in zip(c, c[1:])))
        self.assertLess(c[3], 1e-9)

    def test_sampled_frames_at_ninety_degrees_have_no_bevel_on_the_real_scene(self):
        rng = np.random.default_rng(1)
        left_facing = [i for i, f in enumerate(FACETS.facets) if f.kind == "band" and abs(np.cos(f.phi)) > 0.9]
        diffs = []
        for _ in range(400):
            d = STATE.draw(32, rng, g=3, world=1)                       # light from the left
            sgn = np.array([np.sign(np.cos(FACETS.facets[i].phi)) for i in left_facing])
            diffs.append(float((d.lit[left_facing] * sgn).mean()))
        diffs = np.array(diffs)
        self.assertLess(abs(diffs.mean()), 4 * diffs.std() / np.sqrt(len(diffs)) + 0.02)

    def test_decided_frames_have_a_definite_bevel_with_the_sign_of_the_outcome(self):
        rng = np.random.default_rng(2)
        facing = [i for i in FACETS.of_panel(1) if FACETS.facets[i].kind == "band" and abs(np.cos(FACETS.facets[i].phi)) > 0.9]
        for o_wanted in (+1, -1):
            vals = []
            for _ in range(300):
                d = STATE.draw(64, rng, g=0, world=1)
                if d.pol[1] == o_wanted:
                    vals.append(sum(d.lit[i] * np.sign(np.cos(FACETS.facets[i].phi)) for i in facing))
            self.assertGreater(len(vals), 50)
            # light from the left: a raised bevel facing left is bright, a sunk one dark; the sign flips with the outcome
            self.assertEqual(np.sign(np.mean(vals)), -o_wanted)

    def test_unconditioned_facet_statistics_are_the_same_coherent_or_dephased_but_the_joint_with_the_polarity_outcome_differs(self):
        """No-signalling: dephasing the polarity qubit cannot change the facets' own statistics. What it removes is the correlation between
        the facets and the polarity OUTCOME (handoff test 6, stated precisely). Independent noise differs in both."""
        st = ReliefState(toy_spec((90,)))
        coh, deph = st.cell(0, control="coherent"), st.cell(0, control="dephased")
        np.testing.assert_allclose(coh.sum(axis=(0, 1)), deph.sum(axis=(0, 1)), atol=1e-12)            # facet bits, outcome summed away
        # the polarity outcome itself carries the visibility: read on the equator it is biased, P(o=0) = (1 + V)/2, where a classical mixture gives 1/2
        self.assertAlmostEqual(coh.sum(axis=(0, 2))[0], (1 + st.spec.visibility(0)) / 2, places=9)
        self.assertAlmostEqual(deph.sum(axis=(0, 2))[0], 0.5, places=9)
        tv = 0.5 * np.abs(coh - deph).sum()
        self.assertGreater(tv, 0.05)
        self.assertLess(tv, 0.5)
        noise = np.full_like(coh, 1 / coh.size)
        self.assertGreater(0.5 * np.abs(coh.sum(axis=(0, 1)) - noise.sum(axis=(0, 1))).sum(), 0.2)    # independent noise even changes the marginal

    def test_lamp_read_in_x_keeps_the_marginal_but_not_the_conditional(self):
        """Light directions interfere: summing over the lamp outcome gives the same facet statistics as reading it in Z (no-signalling), but
        conditioned on an X outcome the facets are measured along a direction the lamp coin never chose."""
        z, x = STATE.cell(0, "z"), STATE.cell(0, "x")
        np.testing.assert_allclose(z.sum(axis=0), x.sum(axis=0), atol=1e-12)
        self.assertGreater(np.abs(z - x).max(), 1e-4)


class DrawsAgreeWithTheState(unittest.TestCase):
    def test_sampled_lit_fractions_converge_to_the_exact_conditional_marginals(self):
        rng = np.random.default_rng(5)
        g, w = 1, 2
        draws = [STATE.draw(64, rng, g=g, world=w) for _ in range(300)]
        by_o = {}
        for d in draws:
            o = sum((1 if p < 0 else 0) << k for k, p in enumerate(d.pol))
            by_o.setdefault(o, []).append(d.lit)
        o, rows = max(by_o.items(), key=lambda kv: len(kv[1]))
        np.testing.assert_allclose(np.mean(rows, axis=0)[:SPEC.F], STATE.marginals(g, w=w, o=o), atol=0.03)
        self.assertEqual(len(rows[0]), FACETS.n)                                  # the classical plateaus ride along after the qubits

    def test_glass_is_exactly_black_in_every_frame_from_every_observation_and_light(self):
        rng = np.random.default_rng(7)
        for g in range(4):
            for w in range(4):
                img = COMPOSER.compose(STATE.draw(8, rng, g=g, world=w).lit)
                self.assertTrue(np.all(img[SCENE.impenetrable] == 0))
                self.assertGreater(img[SCENE.frame].min(), 0)               # and nothing projectable is black either

    def test_facets_next_to_the_glass_do_not_light_it(self):
        img = COMPOSER.compose(np.ones(FACETS.n))
        self.assertTrue(np.all(img[SCENE.blocked] == 0))


class Witnesses(unittest.TestCase):
    def test_the_coherent_state_is_certified_and_the_controls_are_not(self):
        cert = rw.certificate(STATE, np.random.default_rng(0), 20000, chsh=False)
        self.assertTrue(all(r["entangled"] for r in cert["panels"]))
        for c, e in cert["controls"].items():
            self.assertFalse(any(r["entangled"] for r in e["panels"]), c)
        s = cert["sampled"]
        self.assertTrue(np.allclose(s["coherent"]["stabilizer"], 1.0))
        self.assertLess(np.max(np.abs(s["dephased"]["stabilizer"])), 0.05)          # dephased: ~0 within 5 sigma
        self.assertLess(np.max(np.abs(s["dephased"]["visibility"])), 0.05)
        self.assertLess(np.max(np.abs(s["noise"]["stabilizer"])), 0.05)
        np.testing.assert_allclose(s["coherent"]["visibility"], SPEC.visibility(), atol=0.03)

    def test_a_state_with_no_signature_is_reported_as_such(self):
        """Tilts of zero: nothing is superposed in any visible way, and the witness says so instead of certifying."""
        sp = ReliefSpec(np.zeros(4), np.zeros(4), [0, 0, 1, 1], [0.0, 0.0], [], FixedRig(), [(0.0, 0.0)])
        st = ReliefState(sp)
        # B is |+> but unentangled: V = 1, lambda_max^2 = 1, fidelity 1 is not above the bound
        cert = rw.certificate(st, np.random.default_rng(0), 2000, chsh=False)
        self.assertFalse(any(r["entangled"] for r in cert["panels"]))

    def test_a_scrambled_state_scores_low_and_a_mismatched_one_is_not_certified(self):
        rng = np.random.default_rng(3)
        bad = rng.normal(size=2 ** SPEC.F) + 1j * rng.normal(size=2 ** SPEC.F)
        st = ReliefState(SPEC, bad, label="random facet state")
        cert = rw.certificate(st, rng, 1000, chsh=False)
        self.assertFalse(any(r["entangled"] for r in cert["panels"]))
        self.assertLess(cert["panels"][0]["fidelity"], 0.01)


class NoTogglesInPerformanceMode(unittest.TestCase):
    RELIEF_PATH = ["src/texture/relief_compose.py", "src/quantum/relief_state.py", "src/quantum/relief_witness.py", "src/geometry/facets.py"]
    REMOVED_ACTIONS = ("hold_world", "hold_lamp", "release_lamp", "toggle_hold_lamp", "hold_pol", "hold_pol_all", "toggle_hold_pol", "pol_basis_toggle")

    def test_the_relief_render_path_never_imports_the_classical_bevel(self):
        banned = {"directional_bevel", "relief_profile", "polarity_map"}
        for rel in self.RELIEF_PATH + ["src/ui/session.py"]:
            with open(os.path.join(ROOT, rel)) as f:
                tree = ast.parse(f.read())
            for node in tree.body:                                              # module-level imports only: the session imports the classical code lazily
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    self.assertNotIn(mod, ("src.texture.bevel", "src.texture.compose", "src.texture.fast", "src.texture.frames"), rel)
                    self.assertFalse(banned & {a.name for a in node.names}, rel)
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(a.name.startswith("src.texture.bevel") for a in node.names), rel)
        for rel in self.RELIEF_PATH:
            with open(os.path.join(ROOT, rel)) as f:
                body = f.read()
            self.assertNotIn("directional_bevel", body.replace("`directional_bevel`", ""), rel)                 # only prose mentions are allowed
            self.assertNotIn("import relief_profile", body, rel)

    def test_session_in_performance_mode_has_no_look_choosing_actions(self):
        from src.ui.session import Session
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(bay_window_labels(), out_dir=tmp, run_dir=tmp, projector_size=(1280, 720), calib_dir="runs/calibration_5x4")
            self.assertEqual(s.family, "relief")
            for a in self.REMOVED_ACTIONS:
                with self.assertRaises(ValueError, msg=a) as cm:
                    s.dispatch(a)
                self.assertIn("unknown action", str(cm.exception))
            knobs = s.state()["knobs"]
            self.assertNotIn("lamp_hold", knobs)
            self.assertNotIn("pol_hold", knobs)

    def test_the_key_table_for_relief_has_none_of_them_and_the_server_refuses_the_routes(self):
        import http.client
        import threading
        from src.ui import keys
        from src.ui.operator_panel import make_server
        from src.ui.session import Session
        actions = {b["action"] for b in keys.bindings_json("relief")}
        self.assertFalse(actions & set(self.REMOVED_ACTIONS))
        self.assertTrue({"dephased_toggle", "witness_run", "noise_toggle", "next"} <= actions)
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(bay_window_labels(), out_dir=tmp, run_dir=tmp, projector_size=(1280, 720), calib_dir="runs/calibration_5x4")
            server, token, base = make_server(s)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                host, port = base.split("//")[1].split(":")
                for a in self.REMOVED_ACTIONS:
                    c = http.client.HTTPConnection(host, int(port), timeout=10)
                    c.request("POST", "/api/action", json.dumps(dict(name=a, args={})), {"X-Token": token, "Content-Type": "application/json", "Host": f"{host}:{port}"})
                    r = c.getresponse()
                    body = json.loads(r.read())
                    self.assertEqual(r.status, 400, a)
                    self.assertFalse(body["ok"])
                    self.assertIn("unknown action", body["error"])
            finally:
                server.shutdown()
                server.server_close()


class ProvenanceIsHonest(unittest.TestCase):
    def test_nothing_is_called_a_moth_result_without_a_moth_job(self):
        from src.ui import caption
        from src.ui.session import Session
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(bay_window_labels(), out_dir=tmp, run_dir=tmp, projector_size=(1280, 720), calib_dir="runs/calibration_5x4")
            self.assertFalse(s.prov["from_moth"])
            line = caption.provenance_line(s.prov)
            self.assertIn("not a Moth result", line)
            s.set_source("oracle")
            self.assertFalse(s.prov["quantum_backed"])
            self.assertIn("Rehearsal", caption.provenance_line(s.prov))
            s.set_source("relief")
            s.dephased_set(True)
            cap = s.live_caption()
            self.assertIn("classical mixture", cap["provenance"])
            s.dephased_set(False)
            s.noise_set(True)
            self.assertIn("random numbers", s.live_caption()["provenance"])


if __name__ == "__main__":
    unittest.main()
