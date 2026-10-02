"""The domain relief and everything around it: geometry, circuits, the exact and matrix-product backends, witnesses, ground state, Moth payloads, baselines.

    python -m unittest tests.test_domains -v
"""
import os
import tempfile
import unittest

import numpy as np

from src.capture.labels import bay_window_labels, fill_defaults, validate
from src.geometry.domains import build_domains, frustration
from src.geometry.planes import build_scene
from src.quantum import complementary as comp
from src.quantum import domain_moth as dm
from src.quantum import domain_witness as dw
from src.quantum import ground_state as gs
from src.quantum import moth_engines as me
from src.quantum.domain_state import DOMAIN_OBSERVATIONS, MPSReliefState, leaf_domains, make_state, spec_from_domains
from src.quantum.mps import MPS, permute_circuit
from src.quantum.relief_state import (LightRig, ReliefSpec, ReliefState, equal_up_to_phase, full_circuit, full_state, polarity_amplitudes,
                                      reference_circuit, statevector_of)

SCENE = DS = SPEC = None


def setUpModule():
    global SCENE, DS, SPEC
    SCENE = build_scene(validate(fill_defaults(bay_window_labels())))
    DS = build_domains(SCENE, seg_len=150, group_size=2)
    SPEC = spec_from_domains(DS, SCENE, lock=1.5708, pol_coupling=0.5)


def toy(m=2, D=3, lock=(0.9, 0.0, -0.7), J=(0.8, -0.8, 0.5), rx=None, seed=4, classical=0):
    """D domains of m facets, a ring of polarity couplings, facet couplings inside a domain, lamp lock on some domains. Small enough for the exact engine."""
    F = D * m
    rng = np.random.default_rng(seed)
    edges = [(d * m + k, d * m + k + 1, "zz", 0.7 - 0.2 * k) for d in range(D) for k in range(m - 1)]
    graph = [(0, 1), (1, 2), (0, 2)][:len(J)]
    return ReliefSpec(rng.uniform(0.35, 0.6, F), rng.uniform(-3, 3, F), np.repeat(np.arange(D), m), [35.0, 0.0, -35.0], edges, LightRig(), list(DOMAIN_OBSERVATIONS),
                      n_classical=classical, classical_panel=[1] * classical, n_groups=D, facet_panel=np.repeat(np.arange(D), m) % 3, pol_graph=graph,
                      pol_layers=[dict(zz=list(J), rx=rx)], lamp_lock=np.array(lock, float))


class Geometry(unittest.TestCase):
    def test_every_projectable_pixel_has_a_facet_glass_has_none_and_plateaus_are_classical_and_last(self):
        fs = DS.facets
        self.assertTrue(np.all(fs.label[SCENE.frame] >= 0))
        self.assertTrue(np.all(fs.label[~SCENE.frame] == -1))
        self.assertTrue(all(not f.classical for f in fs.facets[:fs.n_quantum]) and all(f.classical for f in fs.facets[fs.n_quantum:]))
        self.assertEqual(fs.n - fs.n_quantum, SCENE.n_panels)

    def test_the_tilt_budget_is_per_domain_so_the_visibility_does_not_shrink_with_the_wall(self):
        for d in range(DS.n_domains):
            t = DS.facets.qtau[DS.domain_of == d]
            self.assertAlmostEqual(float((t ** 2).sum()), len(t) * 0.5 ** 2, places=6)
            self.assertAlmostEqual(DS.visibility(d), float(np.prod(np.cos(t))), places=12)
        fine = build_domains(SCENE, seg_len=100, group_size=2)
        self.assertGreater(fine.n_quantum, DS.n_quantum)
        self.assertAlmostEqual(float(np.median(fine.visibility())), float(np.median(DS.visibility())), delta=0.1)

    def test_domains_touch_across_the_creases_between_panels(self):
        creases = [e for e in DS.edges if e[2] == "crease"]
        self.assertGreaterEqual(len(creases), 2)
        for a, b, _, n in creases:
            self.assertNotEqual(DS.domain_panel[a], DS.domain_panel[b])
            self.assertGreater(n, 50)
        self.assertTrue(any(kind == "zz" for i, j, kind, _ in SPEC.edges if SPEC.facet_panel[i] != SPEC.facet_panel[j]))     # facets too

    def test_the_domain_graph_is_balanced_alone_and_frustrated_once_the_lamp_is_joined(self):
        """Honest: with crease_sign -1 every cycle of the bay window crosses each seam an even number of times, so the signed graph is balanced. The crater
        gauge (a domain facing right and one facing left coupled to the same lamp with opposite signs) is what frustrates it."""
        self.assertEqual(frustration(DS)["frustrated_cycles"], 0)
        self.assertGreater(frustration(DS, lock=True)["frustrated_cycles"], 0)

    def test_one_leaf_domain_per_panel_with_a_nonzero_lock_sign(self):
        leaves = leaf_domains(DS)
        self.assertEqual(len(leaves), SCENE.n_panels)
        self.assertEqual(sorted(DS.domain_panel[d] for d in leaves), list(range(SCENE.n_panels)))
        self.assertTrue(all(DS.lock_sign[d] != 0 for d in leaves))
        lk = SPEC.lock_array()
        self.assertEqual(sorted(int(d) for d in np.nonzero(lk)[0]), sorted(leaves))

    def test_the_sparse_composer_matches_the_dense_one_and_keeps_glass_black(self):
        from src.texture.relief_compose import ReliefComposer, SparseReliefComposer
        lit = np.random.default_rng(0).random(DS.facets.n)
        sp, dn = SparseReliefComposer(SCENE, DS.facets), ReliefComposer(SCENE, DS.facets)
        img = sp.compose(lit)
        self.assertLess(float(np.abs(img - dn.compose(lit)).max()), 1e-3)
        self.assertTrue(np.all(img[SCENE.impenetrable] == 0))


class Circuits(unittest.TestCase):
    def test_the_qiskit_reference_circuit_equals_the_numpy_state_with_layers_field_and_couplings(self):
        sp = toy(rx=[0.3, 0.2, 0.4])
        a, b = statevector_of(reference_circuit(sp)), full_state(sp)
        self.assertTrue(equal_up_to_phase(a, b, tol=1e-10))
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-10)

    def test_the_default_domain_circuit_is_still_iqp_shaped(self):
        names = {i.operation.name for i in reference_circuit(SPEC).data}
        self.assertEqual(names, {"h", "ry", "rz", "rzz", "cz"})
        self.assertIn("rx", {i.operation.name for i in reference_circuit(toy(rx=[0.3, 0.2, 0.4])).data})

    def test_the_whole_circuit_with_the_lamp_lock_sampled_by_aer_matches_the_exact_distribution(self):
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        sp = toy(m=1, D=3, lock=(1.1, 0.0, -0.8), J=(0.7, -0.6, 0.4))
        shots = 300_000
        sim = AerSimulator(method="statevector")
        counts = sim.run(transpile(full_circuit(sp), sim), shots=shots, seed_simulator=5).result().get_counts()
        emp = {}
        for key, c in counts.items():
            co, cl, cb, cf = key.split(" ")
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

    def test_the_lamp_lock_makes_the_depth_outcome_depend_on_the_light_side_and_without_it_nothing_does(self):
        locked, free = ReliefState(toy(m=1, lock=(1.2, 1.2, 1.2), J=(0.0, 0.0, 0.0))), ReliefState(toy(m=1, lock=(0, 0, 0), J=(0.0, 0.0, 0.0)))
        g = 3                                                       # gamma 90, chi 90: the depth sphere read along Y, where the lock's rotation shows
        pl = [locked.cell(g)[w].sum(axis=1) / locked.cell(g)[w].sum() for w in (0, 1)]       # light from the right / left, polarity outcome distribution
        pf = [free.cell(g)[w].sum(axis=1) / free.cell(g)[w].sum() for w in (0, 1)]
        self.assertGreater(0.5 * np.abs(pl[0] - pl[1]).sum(), 0.2)
        self.assertLess(0.5 * np.abs(pf[0] - pf[1]).sum(), 1e-9)

    def test_classical_facets_ride_along_in_a_domain_draw(self):
        st = ReliefState(toy(classical=2))
        d = st.draw(50000, np.random.default_rng(0), g=0, world=1)
        self.assertEqual(len(d.lit), st.spec.F + 2)
        want = (1 + st.spec.rig.panel_local(1, 0.0)[2]) / 2
        self.assertAlmostEqual(d.lit[-1], want, delta=0.01)


class MatrixProductStates(unittest.TestCase):
    def setUp(self):
        from qiskit import QuantumCircuit
        rng = np.random.default_rng(3)
        n = 7
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(rng.uniform(0, 3), q), qc.rz(rng.uniform(0, 3), q)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6), (1, 5), (2, 4)]:
            qc.rzz(rng.uniform(.2, 1.5), a, b)
        for q in range(n):
            qc.ry(rng.uniform(0, 3), q), qc.rx(rng.uniform(0, 3), q)
        from qiskit.quantum_info import Statevector
        self.n, self.sv, self.mps = n, Statevector(qc).data, MPS.from_circuit(qc)

    def test_state_reduced_states_expectations_and_between_match_the_statevector(self):
        n, sv, m = self.n, self.sv, self.mps
        self.assertAlmostEqual(abs(np.vdot(sv, m.to_statevector())), 1.0, places=10)
        self.assertAlmostEqual(abs(np.vdot(sv, MPS.from_statevector(sv, n).to_statevector())), 1.0, places=10)
        np.testing.assert_allclose(m.rdm([4, 1]), comp.reduced_pair(sv, n, 4, 1), atol=1e-9)
        np.testing.assert_allclose(m.rdm([5, 0, 3]), dw.reduced_set(sv, n, [5, 0, 3]), atol=1e-9)
        self.assertAlmostEqual(m.pauli_expectation({0: "x", 3: "z"}), comp.pauli_expectation(sv, n, {0: "x", 3: "z"}), places=9)
        Z = np.diag([1, -1]).astype(complex)
        phi = np.moveaxis(np.tensordot(Z, np.moveaxis(sv.reshape((2,) * n), n - 1 - 3, 0), axes=([1], [0])), 0, n - 1 - 3).reshape(-1)
        Ma = np.moveaxis(phi.reshape((2,) * n), [n - 1 - q for q in (0, 2)], [0, 1]).reshape(4, -1)
        Mb = np.moveaxis(sv.reshape((2,) * n), [n - 1 - q for q in (0, 2)], [0, 1]).reshape(4, -1)
        np.testing.assert_allclose(m.rdm([0, 2], between={3: Z}), Ma @ Mb.conj().T, atol=1e-9)

    def test_conditional_sampling_is_exact_given_fixed_sites_and_a_traced_rest(self):
        n, sv, m = self.n, self.sv, self.mps
        rng = np.random.default_rng(1)
        smp = m.sampler([0, 2, 4], {1: 1, 5: 0})
        X = smp.draw(rng, 30000)
        idx = np.arange(2 ** n)
        bit = lambda q: (idx >> q) & 1
        mask = (bit(1) == 1) & (bit(5) == 0)
        exact = np.zeros(8)
        for i in np.nonzero(mask)[0]:
            exact[bit(0)[i] + 2 * bit(2)[i] + 4 * bit(4)[i]] += abs(sv[i]) ** 2
        exact /= exact.sum()
        emp = np.bincount(X[:, 0] + 2 * X[:, 1] + 4 * X[:, 2], minlength=8) / len(X)
        self.assertLess(0.5 * np.abs(exact - emp).sum(), 0.02)
        self.assertAlmostEqual(smp.p_fixed, float(np.abs(sv[mask]).__pow__(2).sum()), places=9)

    def test_the_bond_dimension_of_a_loop_ordered_wall_is_small(self):
        st = MPSReliefState(SPEC, max_bond=32)
        m = st.mps(0, 0)
        self.assertLessEqual(m.max_bond(), 32)
        self.assertLess(abs(1 - m.norm2()), 1e-2)


class ExactAndMpsAgree(unittest.TestCase):
    def test_the_conditioned_mps_state_has_the_exact_probabilities_for_every_observation_and_light(self):
        sp = toy(rx=[0.3, 0.3, 0.3], classical=1)
        ex, mp = ReliefState(sp), MPSReliefState(sp, truncation=1e-13, max_bond=None)
        for g in range(4):
            cell = ex.cell(g)
            for w in range(4):
                p = np.abs(mp.mps(g, w).to_statevector()) ** 2
                pe = cell[w].reshape(-1)
                np.testing.assert_allclose(p, pe / pe.sum(), atol=1e-9)

    def test_draws_agree_with_the_exact_polarity_distribution_coherent_and_dephased(self):
        sp = toy(rx=None)
        ex, mp = ReliefState(sp), MPSReliefState(sp, truncation=1e-13, max_bond=None)
        rng = np.random.default_rng(2)
        g, w, N = 2, 1, 2500
        for control in ("coherent", "dephased"):
            cnt = np.zeros(2 ** sp.P)
            for _ in range(N):
                d = mp.draw(8, rng, g=g, world=w, control=control)
                cnt[sum((1 if p < 0 else 0) << k for k, p in enumerate(d.pol))] += 1
            pe = ex.cell(g, control=control)[w].sum(axis=1)
            self.assertLess(0.5 * np.abs(cnt / N - pe / pe.sum()).sum(), 0.05, control)

    def test_conditional_facet_marginals_agree(self):
        sp = toy()
        ex, mp = ReliefState(sp), MPSReliefState(sp, truncation=1e-13, max_bond=None)
        rng = np.random.default_rng(3)
        rows = {}
        for _ in range(1500):
            d = mp.draw(64, rng, g=1, world=2)
            rows.setdefault(sum((1 if p < 0 else 0) << k for k, p in enumerate(d.pol)), []).append(d.lit[:sp.F])
        o, r = max(rows.items(), key=lambda kv: len(kv[1]))
        np.testing.assert_allclose(np.mean(r, axis=0), ex.marginals(1, w=2, o=o), atol=0.02)

    def test_lamp_read_in_x_is_exact_only(self):
        mp = MPSReliefState(toy())
        with self.assertRaises(ValueError):
            mp.draw(8, np.random.default_rng(0), lamp_mode="x")

    def test_a_scene_sized_wall_draws_frames_through_the_composer_with_glass_black(self):
        from src.texture.relief_compose import make_composer
        st = make_state(SPEC)
        self.assertEqual(st.backend, "mps")
        comp_ = make_composer(SCENE, DS.facets)
        rng = np.random.default_rng(0)
        for g, w in ((0, 0), (3, 2)):
            d = st.draw(24, rng, g=g, world=w)
            self.assertEqual(len(d.lit), DS.facets.n)
            self.assertEqual(len(d.pol), DS.n_domains)
            img = comp_.compose(d.lit)
            self.assertTrue(np.all(img[SCENE.impenetrable] == 0))
            self.assertGreater(img[SCENE.frame].min(), 0)


class Witnesses(unittest.TestCase):
    def test_exact_and_mps_certificates_agree_and_the_dephased_control_has_no_entangled_edge(self):
        sp = toy(m=1, lock=(1.2, 1.2, 1.2), J=(0.4, -0.4, 0.3))
        ce = dw.domain_certificate(ReliefState(sp))
        cm = dw.domain_certificate(MPSReliefState(sp, truncation=1e-13, max_bond=None))
        for ea, eb in zip(ce["edges"], cm["edges"]):
            self.assertAlmostEqual(ea["negativity"], eb["negativity"], places=7)
            self.assertAlmostEqual(ea["chsh"], eb["chsh"], places=6)
        self.assertAlmostEqual(ce["lamp"]["mermin"]["value"], cm["lamp"]["mermin"]["value"], places=4)
        self.assertEqual(ce["summary"]["dephased_entangled_edges"], 0)
        self.assertGreater(ce["summary"]["entangled_edges"], 0)
        self.assertLess(ce["lamp"]["mermin"]["value_dephased"], ce["lamp"]["mermin"]["local_bound"])

    def test_lamp_blocks_from_operator_insertion_equal_a_real_lamp_qubit(self):
        sp = toy(m=1, lock=(0.9, 0.0, -0.7), J=(0.8, -0.8, 0.5))
        le, lm = dw.lamp_state(ReliefState(sp)), dw.lamp_state(MPSReliefState(sp, truncation=1e-13, max_bond=None))
        n = sp.n_sim
        for qs in ([n, sp.F + 0], [sp.F + 2, n], [n, sp.F + 0, sp.F + 2], [sp.F + 0, sp.F + 1]):
            np.testing.assert_allclose(le.rdm(qs), lm.rdm(qs), atol=1e-8)

    def test_a_lamp_locked_to_one_domain_per_panel_violates_mermin_and_one_locked_to_many_does_not(self):
        """Monogamy: every extra domain locked to the lamp holds a record of it, so the lamp's coherence with any three of them dies."""
        few = dw.domain_certificate(ReliefState(toy(m=1, lock=(1.5, 1.5, 1.5), J=(0.0, 0.0, 0.0))))["lamp"]["mermin"]
        self.assertGreater(few["value"], few["local_bound"] + 0.5)
        # nine domains locked to the same lamp, the Mermin test on three of them
        D = 9
        rng = np.random.default_rng(1)
        sp = ReliefSpec(np.full(D, 0.5), rng.uniform(-3, 3, D), np.arange(D), [0.0], [], LightRig(), list(DOMAIN_OBSERVATIONS), n_groups=D, facet_panel=np.zeros(D, int),
                        pol_graph=[], pol_layers=[], lamp_lock=np.full(D, 1.5))
        # three leaves in different 'panels': give the first three distinct panels so the leaf picker takes one each
        sp.facet_panel = np.arange(D) % 3
        many = dw.domain_certificate(MPSReliefState(sp, truncation=1e-13, max_bond=None))["lamp"]["mermin"]
        self.assertLess(many["value"], many["local_bound"])

    def test_the_real_wall_shows_entangled_domain_edges_and_a_lamp_depth_mermin_violation(self):
        cert = dw.domain_certificate(make_state(SPEC))
        self.assertGreater(cert["summary"]["entangled_edges"], cert["summary"]["edges"] // 3)
        crease = [e for e in cert["edges"] if DS.edges[cert["edges"].index(e)][2] == "crease"]
        self.assertTrue(any(e["negativity"] > 1e-6 for e in crease))                 # entanglement across a geometric crease
        m = cert["lamp"]["mermin"]
        self.assertGreater(m["value"], m["local_bound"])
        self.assertEqual(cert["summary"]["dephased_entangled_edges"], 0)
        self.assertEqual(len(dw.describe_domain_certificate(cert, DS, SPEC)) >= 6, True)


class GroundState(unittest.TestCase):
    def test_classical_ground_states_are_counted_two_per_component_for_a_balanced_graph(self):
        g = gs.classical_ground(6, [(0, 1, 1.0), (1, 2, -1.0), (3, 4, 1.0)], 1.0)          # components {0,1,2}, {3,4}, {5}: 2^3 ground states
        self.assertEqual(g["degeneracy"], 8)
        self.assertAlmostEqual(g["energy"], -3.0)
        tri = gs.classical_ground(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, -1.0)], 1.0)          # a frustrated triangle: energy -1, six ground states
        self.assertAlmostEqual(tri["energy"], -1.0)
        self.assertEqual(tri["degeneracy"], 6)

    def test_the_fitted_ansatz_beats_the_product_state_and_reports_its_distance_from_the_exact_ground_state(self):
        ds = build_domains(SCENE, seg_len=300, group_size=3)
        sp = spec_from_domains(ds, SCENE, lock=0.0)
        new, rep = gs.ground_state_spec(sp, ds, J=1.0, h=0.5, p=2, restarts=2)
        self.assertLess(rep["energy"], gs.energy(sp.P, gs.signed_edges(ds), gs.layers_for(gs.signed_edges(ds), 1.0, [0.0], [0.0], sp.P), 1.0, 0.5))
        self.assertGreaterEqual(rep["energy"], rep["exact_energy"] - 1e-9)
        self.assertGreater(rep["fidelity_to_exact"], 0.3)
        self.assertEqual(rep["classical"]["degeneracy"], 2 ** 4)
        a, b = statevector_of(reference_circuit(new)) if new.n_sim <= 14 else (None, None)
        self.assertEqual(len(new.pol_layers), 2)

    def test_exact_transverse_ising_ground_energy_is_below_the_classical_one(self):
        e0, psi = gs.transverse_ising_ground(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, -1.0)], 1.0, 0.5)
        self.assertLess(e0, -3.0)
        self.assertAlmostEqual(float(np.vdot(psi, psi).real), 1.0, places=9)


class MothPayloads(unittest.TestCase):
    def test_a_patch_is_a_self_contained_smaller_design_with_a_valid_tomography_request_and_exact_moments(self):
        dom = dm.connected_patch(DS, leaf_domains(DS)[0], 5)
        payload, patch = dm.patch_tomography_payload(SPEC, dom)
        self.assertEqual(patch.P, 5)
        self.assertLessEqual(patch.n_sim, 20)
        from src.quantum.moth_client import MothClient
        self.assertEqual(MothClient().prepare("tomography-api-v2", payload["params"]).problems(), [])
        mom = me.pauli_moments(patch)
        bloch = {q: (mom[f"X{q}"], mom[f"Y{q}"], mom[f"Z{q}"]) for q in range(patch.n_sim)}
        sc = me.score_tomography((bloch, {}, None), patch)
        self.assertAlmostEqual(sc["rms_single"], 0.0, places=9)
        self.assertLess(sc["visibility"][0]["exact"], sc["visibility"][0]["isolated"])                 # neighbours lower the visibility
        self.assertTrue(any(len(k) == 4 for k in mom))

    def test_polarity_qdrive_jobs_are_small_valid_and_in_the_mapping_form(self):
        from src.quantum.moth_client import MothClient
        jobs = dm.qdrive_polarity_payloads(SPEC)
        self.assertGreaterEqual(len(jobs), 2)
        for j in jobs:
            self.assertLessEqual(len(j["params"]["targets"]), 26)
            self.assertEqual(MothClient().prepare("qdrive-api-v1", j["params"]).problems(), [])
            for t in j["params"]["targets"]:
                if t:
                    self.assertIsInstance(t["expvals"], dict)
        self.assertIsNone(jobs[0]["needs_initial_circuit_from"])

    def test_a_circuit_returned_for_the_polarity_register_becomes_its_preparation(self):
        from qiskit import QuantumCircuit, qasm3
        sp = toy(m=1, lock=(0, 0, 0))
        ref = polarity_amplitudes(sp)
        # a stand-in for a QDrive circuit: the exact same state prepared by the spec's own layers
        qc = QuantumCircuit(sp.P)
        for d in range(sp.P):
            qc.h(d)
        for (a, b), th in zip(sp.pol_graph, sp.pol_layers[0]["zz"]):
            qc.rzz(th, a, b)
        new, rep = dm.lift_polarity_circuit(sp, qasm3.dumps(qc))
        self.assertAlmostEqual(rep["fidelity"], 1.0, places=9)
        np.testing.assert_allclose(np.abs(ReliefState(new).Psi), np.abs(ReliefState(sp).Psi), atol=1e-9)
        wrong = QuantumCircuit(sp.P)
        for d in range(sp.P):
            wrong.h(d)
        _, rep2 = dm.lift_polarity_circuit(sp, qasm3.dumps(wrong))
        self.assertLess(rep2["fidelity"], 0.999)

    def test_echo_taps_modulate_the_facets_frame_by_frame_and_keep_the_geometry_azimuths(self):
        taps = me.local_echo(width=4, height=4, depth=3)
        sites = dm.facet_sites(DS, SCENE, 4, 4)
        specs = dm.dynamic_specs(SPEC, sites, taps, 3, 16)
        self.assertEqual(len(specs), 3)
        self.assertGreater(float(np.abs(specs[0].tau - specs[2].tau).max()), 1e-3)
        self.assertTrue(all(s.F == SPEC.F and s.P == SPEC.P for s in specs))
        self.assertTrue(np.all(specs[0].tau <= 1.1 + 1e-12))


class Baselines(unittest.TestCase):
    def test_the_two_branch_sampler_reproduces_the_exact_engine_and_scales_linearly(self):
        from src.quantum import classical_baselines as cb
        self.assertLess(cb.two_branch_check(N=4, frames=20000), 0.03)
        rows = cb.two_branch_timing((100, 2000), K=24, frames=3)
        self.assertLess(rows[1]["seconds_per_frame"], 2.0)

    def test_excitation_truncation_is_easy_for_the_scaled_relief_and_hard_at_the_equator(self):
        from src.quantum import classical_baselines as cb
        scaled = cb.excitation_stats(np.full(1000, 1 / np.sqrt(1000)))
        self.assertLessEqual(scaled["k_target"], 3)
        self.assertAlmostEqual(scaled["mean"], 0.25, places=2)
        self.assertGreater(cb.excitation_stats(np.full(100, np.pi / 2))["k_target"], 40)

    def test_the_graph_regime_needs_a_bigger_bond_dimension_than_the_relief_regime_on_the_same_lattice(self):
        from src.quantum import classical_baselines as cb
        r = cb._measure(("grid-relief", 3))
        g = cb._measure(("grid-graph", 3))
        self.assertGreater(g["max_bond"], r["max_bond"])
        self.assertEqual(cb._measure(("panel", 18))["max_bond"], 2)


class SessionDomainEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.ui.session import Session
        cls.tmp = tempfile.TemporaryDirectory()
        cls.s = Session(bay_window_labels(), out_dir=cls.tmp.name, run_dir=cls.tmp.name, projector_size=(1280, 720), calib_dir="runs/calibration_5x4", engine="domain",
                        domain=dict(seg_len=300, group_size=3, lock=1.5708))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_session_runs_the_domain_engine_with_honest_provenance_and_panel_level_captions(self):
        s = self.s
        self.assertEqual(s.family, "relief")
        self.assertFalse(s.prov["from_moth"])
        self.assertIn("not a Moth result", s.prov["label"])
        self.assertIn("matrix-product state", s.prov["label"])
        b = s.budget
        self.assertEqual(b["engine"], "domain")
        self.assertFalse(b["over_cap"])
        self.assertIn("domain polarity qubits", s._budget_text())
        s._new_look()
        st = s.state()
        self.assertEqual(len(st["look"]["words"]), SCENE.n_panels)
        self.assertEqual(len(st["science"]["panels"]), SCENE.n_panels)
        self.assertEqual(st["science"]["engine"]["name"], "domain")
        cap = s.live_caption()
        self.assertEqual(len(cap["panels"]), SCENE.n_panels)
        self.assertTrue(hasattr(s.current.draw, "domain_pol"))
        self.assertEqual(len(s.current.draw.domain_pol), s.relief()["ds"].n_domains)

    def test_glass_is_black_in_every_frame_and_the_overlay_draws_the_domain_graph(self):
        s = self.s
        for _ in range(4):
            s._new_look()
            self.assertTrue(np.all(s._canvas[SCENE.impenetrable] == 0))
        im = s.overlay_image()
        self.assertEqual(im.size, (SCENE.W, SCENE.H))

    def test_lamp_in_x_falls_back_with_a_message_and_the_witness_run_certifies_domain_edges(self):
        s = self.s
        s.light_interference_set(True)
        self.assertFalse(s.knobs.light_interference)
        self.assertTrue(any("exact backend" in m["msg"] for m in s.messages))
        out = s.witness_run()
        self.assertGreater(out["certificate"]["summary"]["entangled_edges"], 0)
        self.assertTrue(any("domain edges certified entangled" in m["msg"] for m in s.messages))
        self.assertIn("lamp - depth", "\n".join(s.certificate["lines"]))

    def test_controls_still_work_and_toggles_are_still_unknown_actions(self):
        s = self.s
        s.dephased_set(True)
        self.assertEqual(s.current.control, "dephased")
        s.dephased_set(False)
        s.noise_set(True)
        self.assertEqual(s.current.control, "noise")
        s.noise_set(False)
        for a in ("hold_world", "hold_pol", "pol_basis_toggle"):
            with self.assertRaises(ValueError):
                s.dispatch(a)

    def test_a_circuit_file_is_refused_by_the_domain_engine(self):
        from src.ui.session import Session
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            Session(bay_window_labels(), out_dir=tmp, run_dir=tmp, projector_size=(1280, 720), calib_dir="runs/calibration_5x4", engine="domain", circuit=__file__,
                    domain=dict(seg_len=500, group_size=4))


if __name__ == "__main__":
    unittest.main()
