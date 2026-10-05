# Standing Light v3: Domain Relief

> **Scene note.** This document describes the first build's **three-panel** bay window (left, centre, right). The built-in window, and the default of the code, is now **six panels**: each wing holds two windows stacked on top of each other with a shared rail (`left_top`, `left_bottom`, `centre_top`, `centre_bottom`, `right_top`, `right_bottom`; three planes). Counts and numbers below that say three panels, 15 or 19 qubits, or 3 polarity qubits are the three-panel ones; the README's *The default scene* has the current ones.

> **Superseded in part.** The defaults and results below are those of the first domain relief (69 qubits). The current defaults (one facet per depth qubit, weak boundaries, strong seams: 92 qubits), the parity game, the seam regime, the kicked-Ising dynamics and the corrections to this document (control B on the matrix-product backend, the site-order argument, the two lamp semantics) are in the repository README, section "Domain Relief". The audit in section 0 and the decisions in section 4 still stand.

Written 2 Oct 2026, after an audit of the v2 code against the project's own claim (quantum superposition and entanglement *between edges*, graph-structured, projected onto a wall). Read `standing-light-v2-superposed-relief-handoff.md` first; this file extends it and says where the code departs from it. Everything below is a **local simulation**; nothing has been run on a Moth engine and no credits were spent.

## 0. Why v3 exists: what the audit found in v2

Measured on the default bay window (kappa 1, entangle 0.8, 4 orientation facets per panel), before any change:

| Finding | Number |
|---|---|
| The three panels were exactly unentangled with each other | entropy of panel 0 against the rest 3e-12 bits |
| Three of the 15 facet qubits were constants | the plateau facets had tau ~ 5e-4: a \|0> that nothing couples to |
| The only cross-panel facet edges joined those two plateaus | cause: the slope was differenced across the seam, halving it there, so the seam pixels were "flat" |
| The ZZ facet couplings changed no single-facet brightness | max change of a lit marginal 0.0000; facet-facet mutual information fell 0.15 -> 0.10 bits |
| The look is mostly classical | 72% of the variance of per-frame expected brightness is the lamp world (a classical draw), 0% the observation angle, 28% the polarity outcome (a near-fair coin) |
| The scaling law undercuts the hardness story | tau = kappa/sqrt(N) holds V constant by keeping the state at ~0.25 excitations per panel at every N (P(vacuum) 0.78); the per-facet bevel signal sin tau falls to 0.03 at N = 1000, under the photon noise |

## 1. What changed (all of it tested, `tests/test_domains.py`, `tests/test_relief.py`)

1. **Cheap fixes** (affect the per-panel engine too). `slope_field` is per panel (no differencing across a seam), so band facets of neighbouring panels now touch and the panels are entangled with each other. Plateau facets are *classical* (`Facet.classical`, ordered last, `interior="drop"`): the per-panel bay window is 12 facet qubits + 3 polarity = 15 simulated (19 in the circuit), not 18/22. `interior="keep"` restores the old behaviour. Draws carry the classical facets' lit values after the qubits'.
2. **Domain relief** (`geometry/domains.py`, `quantum/domain_state.py`): polarity qubit per depth *domain*, coupled along the geometry graph. See README "Domain Relief (v3)". Key formulas: V_d = prod cos(tau_i) over the domain's facets (budget sum tau^2 = m tau^2 per domain), polarity layer `exp(-i theta_e ZZ/2)` with theta_e = J * (+1 coplanar | crease_sign), lock `exp(-i theta_d Z_L1 Z_Bd /2)` with theta_d = lock * (crater-gauge sign), one leaf domain per panel by default.
3. **Graph-state regime**: `tau_mix` (facets toward the equator) with `entangle` ZZ couplings on the facet adjacency graph. The one-body normal survives only as far as `transverse_shrink` leaves it, and V -> 0. It is the regime where classical simulation gets expensive, and not one the show is tuned for.
4. **Lamp as a quantum register**: `lamp_lock`. For the Z-read lamp the lock is a z-rotation of the locked polarity qubit conditioned on the drawn light side; `domain_witness.lamp_state` gives the exact lamp+polarity reduced states with *no lamp qubit* by inserting operators on the traced polarity qubits of the prepared MPS (tested equal to a real lamp qubit).
5. **Target-correlation preparation** (`ground_state.py`, `domain_moth.py`): the polarity register can be the fitted ground state of `H = -J sum s_ab Z Z - h sum X` on the domain graph (QAOA-style, p layers, a circuit), or a circuit returned by QDrive (`spec.pol_circuit`, `lift_polarity_circuit`). Payload builders for QDrive (polarity targets, <= 24 per job), tomography of a *patch* (<= 20 qubits, boundary couplings cut: a smaller design, not the reduced state of the wall) and the echo engine (`dynamic_specs`, not in the live show).
6. **Measurements**: `classical_baselines.py` (two-branch sampler, excitation truncation, MPS bond dimension vs size, each point in a subprocess with a time limit) and `relief_report.py` (`relief_contrast_vs_visibility.png`, `--domain`).

## 2. Results (bay window, default: seg 150, 2 facets per domain, tau 0.5, J 0.5, lock pi/2, entangle 0.8)

- 46 facet qubits + 23 domain polarity qubits = 69 simulated (73 with the lamp and observation registers), 25 domain edges (5 creases), MPS bond dimension capped at 32 (norm deficit 1.5e-3; at caps 8/16/64 it is 1.3e-2 / 6e-3 / 7.6e-4, so an exact state needs a bond dimension above 128).
- 15 of 25 domain edges entangled (largest negativity 0.083); dephased control 0. Visibility <X_B> 0.50 to 0.74 (isolated 0.77).
- Lamp + three leaves, Mermin: 4.34 against a local bound of 4 (dephased 1.0). Best lamp-depth CHSH 1.35 (no pair violates: monogamy). J 0 / 0.3 / 0.5 give 6.2 / 5.5 / 4.3: polarity coupling and lamp-depth Bell violation trade off. Locking the lamp to every domain kills the violation.
- Signed domain graph balanced alone; 3 frustrated cycles with the leaf lock at 69 qubits, 0 at 116, 10/13 with the lock on every domain.
- Classical baselines: per-panel relief, bond dimension 2 up to 401 qubits; two-branch sampler 0.04 s per frame at N = 5000 (TV to the exact engine 0.010); excitation truncation k99 = 2 for kappa/sqrt(N) at N = 1000, 12 at fixed tau 0.5 (N = 100), 62 at the equator. Domain lattice in the relief regime: bond dimension 4, 8, 16, 32 for L = 2..5 and past 128 at L = 6 (108 qubits). Graph-state regime: 8, 16, 32 for L = 2..4 and past 128 at L = 5 (75 qubits, 7% norm deficit at the cap).

## 3. What this does and does not establish

Does: the entanglement now runs across geometric edges and creases; the lamp and the depth are entangled and a Mermin test certifies it at 69 qubits; there is a measured, per-design answer to "how big can this get before the classical methods stop being cheap"; the preparation can be specified by correlations and handed to a Moth engine.

Does not: any advantage. The show's own state is approximately classically simulable (bond dimension 32 leaves a 1.5e-3 norm deficit; an exact state needs more than 128). The regime where an MPS fails is the graph-state regime, which the show does not run (V -> 0, no visible depth superposition). Nothing was run on Moth. The domain relief has not been judged by eye on a real wall. The echo-driven dynamic relief is a module, not a mode.

## 4. Decisions for you

1. Which regime is the performance? Relief regime (visible depth, MPS-approximable) or graph-state regime (hard for an MPS, no visible depth superposition)? The code supports both (`--tau-mix`); the show defaults to relief.
2. Is the leaf lock (one domain per panel) the right crater gauge, or should the lamp be locked more widely and the Bell test be dropped? Locking widely gives frustration (10-13 cycles) but loses the Mermin violation.
3. First Moth spend, in order: `tomography-api-v2` on a 14-qubit patch (1 credit); `qdrive-api-v1` polarity job 0 (1 credit; targets <= 24); everything else waits on what those return. Build with `python -m src.quantum.domain_moth build patch-tomography` / `polarity-qdrive`; both print the exact request and send nothing.

## 5. How to run

```bash
python -m src.show --engine domain                       # ~15 s to build the 16 conditioned states; ~0.06 s per frame afterwards
python -m src.studio --engine domain                     # step 3 opens it
python -m src.quantum.relief_report --domain             # frames, certificate, contrast-vs-visibility figure
python -m src.quantum.classical_baselines                # a few minutes
python -m src.quantum.domain_moth build polarity-qdrive --ground     # fits the ground state, writes the payload, sends nothing
python -m unittest tests.test_domains                    # 38 tests, ~40 s
```
