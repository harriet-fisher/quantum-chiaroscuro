# Standing Light v2: Superposed Relief

> **Scene note.** This document describes the first build's **three-panel** bay window (left, centre, right). The built-in window, and the default of the code, is now **six panels**: each wing holds two windows stacked on top of each other with a shared rail (`left_top`, `left_bottom`, `centre_top`, `centre_bottom`, `right_top`, `right_bottom`; three planes). Counts and numbers below that say three panels, 15 or 19 qubits, or 3 polarity qubits are the three-panel ones; the README's *The default scene* has the current ones.

Handoff for Claude Code. Written 1 Oct 2026. Read this after `standing-light-handoff.md` (where they differ, **this file wins** for anything about the quantum design, the bevel, and the controls). The reference simulation `superposed_relief_refsim.py` (same folder) is runnable and checks every formula below; run it first.

## 0. What changes, in five lines

1. **The bevel is no longer computed by classical code and no longer toggled.** `directional_bevel`, `relief_profile`, the lamp/polarity "hold" keys and the `X` polarity-basis key leave the performance path. Shading at the bevel is the *measurement outcome of facet qubits whose Bloch vectors are the surface normals*.
2. **Depth is in superposition.** One polarity qubit per panel puts "raised" and "sunk" depth maps in superposition. A frame shows a definite bump, a definite hollow, or something in between, according to *where on that qubit's Bloch sphere the observation lands*, and that landing point is drawn by the circuit, not set by the operator.
3. **Light direction is also drawn inside the circuit** (lamp register controlling the facets' measurement axis). One run, one bitstring: light, depth decision and photon outcomes all come out together.
4. **Quantumness is certified, not asserted.** A witness suite (CHSH / Mermin, plus a parity-visibility measurement) runs on the *actual* state and the UI reports its result with provenance. If a state shows no non-classical signature, the UI says so.
5. **There is a scaling law we can demonstrate at ~16 qubits and extrapolate to 1000+**: the visibility of interference between depth maps decays like the product of cos(tilt) over all facets, so tilt per facet must scale like 1/sqrt(N) to keep the effect constant. Section 5.

## 1. The idea, in the author's terms (use for the pitch)

A real wall has one shape. Light can make it look like many: swollen, hollowed, creased, flat. The depth is not in the wall; it is in the *relationship between the light and the viewer's guess*. Quantum mechanics has the same structure: a qubit has no definite spin until you choose an axis to ask along, and the answer depends on the question.

Standing Light v2 makes that literal. Every small patch of the wall is a qubit whose state *is* its surface normal. The wall as a whole is in a superposition of depth maps (bump and hollow, per panel). The projector is the measurement: each photon either lands or it does not, with probability given by Lambert's cosine law, which is exactly the Born rule. What you see is not a render of a shape. It is the wall answering a question, and sometimes the honest answer is "both", which looks flat, with a structure you cannot see in any single frame.

Three principles the project communicates, each tied to something a viewer can watch:

| Principle | Where it lives in the piece | What a viewer sees |
|---|---|---|
| **State = normal; measurement axis = light; Lambert = Born** | facet qubits, lamp register | bright where the surface faces the light, as a probability; hard grain with few photons, soft with many |
| **Superposition of depth maps** | polarity qubits (one per panel) | a panel that swells, sinks, and between the two goes flat |
| **Complementarity: knowing which depth costs you the interference between depths** | the observation axis on the polarity qubits | bevel contrast slides smoothly from +strong (decided raised) through 0 (undecided) to -strong (decided sunk) |

Honest framing to keep in every slide: at ~16 to 22 qubits all of this is classically simulable and the code simulates it classically. The claim is *structure and scaling*, not advantage.

## 2. Concepts, precisely

### 2.1 Facet qubit = surface normal (Gauss map)
A facet is a small region of the UV-flattened surface (an interior patch, or a segment of a *bevel band* along a panel border). In the facet's **panel-local frame** (Z = panel normal), its relief tilt is the Bloch vector

    b_i = ( sin(tau_i) cos(phi_i),  sin(tau_i) sin(phi_i),  cos(tau_i) )

`tau_i` is how steeply the relief tilts the normal at that facet; `phi_i` is the direction it tilts (the slope azimuth, e.g. toward the panel centre for a border band). Both come from a classical height field `h_raised(x, y)` built from geometry (the existing `scene.relief` distance-to-border field, scaled): average `-grad h` over the facet's pixels gives `(nx, ny)`; `tau = arctan(|grad h|)`, `phi = atan2(...)`. Panel orientation (the 35 degree side panels) is **not** in the Bloch vector: it enters through the per-facet light axis (2.3), so the relief flip (2.2) does not flip the physical angle of the panel.

State of the relief register in the raised branch: `|psi> = tensor_i Ry(tau_i) Rz(phi_i) |0>` as a *reference* (product) state; the real one is prepared with correlations (section 4.2).

### 2.2 Raised vs sunk is the parity operator Z^(x)N
`Z` maps Bloch `(x, y, z) -> (-x, -y, z)`: it reverses the in-plane slope and keeps the tilt magnitude, which is exactly "turn every bump into a hollow" (`h -> -h`). So the sunk relief of a panel is `P_p |psi>` with `P_p = product of Z over that panel's facets`. Verified in the refsim (check 2).

### 2.3 Light = measurement axis; shading = Born rule
For facet `i` and light direction `l` (unit vector expressed in facet `i`'s panel-local frame), measuring `sigma . l` gives "lit" with probability

    P(lit) = (1 + b_i . l) / 2          (= half-Lambert; refsim check 1 matches to 6 digits)

Implementation: rotate facet `i` by `U_i(l)` that takes `l` to Z, then measure Z; bit 0 = lit. One photon = one shot; `K` shots averaged = exposure. **Hardness is exposure**, not a scene property.

### 2.4 Superposition of depth maps
For each panel `p` there is a polarity qubit `B_p` in `|+>`. The relief register is

    |Psi> = (1/sqrt(2^m)) * sum over s in {0,1}^m  |s>_B (tensor_p P_p^{s_p}) |psi>        # m = number of panels

built by `H` on every `B_p`, `U_psi` on the facets, then `CZ(B_p, f)` for every facet `f` in panel `p`. No controlled version of `U_psi` is needed (that is what makes this cheap and lets us drop in a Moth-prepared `U_psi`): `Z^(x)N` is just a layer of CZ gates.

**Relief amplitude = branch distinguishability = entanglement between B and the facets.** Large tilt: the bump and hollow are nearly orthogonal and the state is GHZ-like. Tiny tilt: they are nearly the same state and nothing is superposed in any visible way. The overlap per facet is `cos(tau_i)`.

### 2.5 The observation axis: the *depth sphere*
`B_p`'s own Bloch sphere is a sphere of depth maps: north pole = raised, south pole = sunk, equator = equal superpositions (the angle around the equator picks *which ghost*, an interference phase). Reading `B_p` along `(sin(gamma) cos(chi), sin(gamma) sin(chi), cos(gamma))` leaves the facets in `cos(gamma/2)|psi> + e^(i chi) sin(gamma/2) P|psi>` (outcome 0) or its orthogonal partner (outcome 1).

- `gamma = 0`: decided. The frame shows a definite bump or hollow (the Born rule picks which, 50/50).
- `gamma = 90 deg`: fully undecided. **Bevel contrast goes to zero**: first-order shading is symmetric, the panel looks flat/ambiguous, and the evidence that both depths are present lives in *joint* statistics (2.7).
- In between: a signed bevel of intermediate strength. Refsim check 4 (8 facets, outcome 0): contrast +0.332, +0.238, +0.122, 0.000 at gamma = 0, 30, 60, 90 deg, mirrored to -0.332 at 180. Monotone, odd about 90 deg.

This is the which-path/visibility trade-off in visual form: **the more decided the depth, the less the two depth maps can interfere.** It is the cleanest quantum principle the piece can carry, and it needs no explanation of qubits to feel.

`gamma` (and optionally `chi`) are drawn by a small observation register `G` inside the circuit (2 qubits -> gamma in {0, 30, 60, 90 deg}), so nothing is toggled. Poetic reading: the room does not know how hard it is looking.

### 2.6 What is and is not quantum about the control registers
`L` (light) and `G` (observation) are uniform control registers measured in Z, so by themselves they are equivalent to classical coin flips (on a QPU they are physical randomness; in emulation they are pseudo-random). Say so. The quantum content is the facet register, the polarity qubits and the controlled operations between them. **Implementation consequence:** because `L` and `G` are unentangled uniform controls, you can sample their bits first and simulate only facets + polarity per (L, G) value (cache at most 16 states). That keeps the statevector at `N_facets + n_panels` qubits (target <= 22) while the whole thing is *mathematically one circuit*. The Moth circuit and the hardware version contain `L` and `G` as real qubits.

### 2.7 The perceptual gauge (concept to use, cheap to implement)
By the crater illusion, a bump lit from the right and a hollow lit from the left produce the same picture. The image depends only on `polarity XOR light_side`. Treat that as a gauge symmetry: the picture is a measurement of the *parity* of (depth sign, light side); neither is recoverable alone. This is the narrative bridge between perception and the Z-parity structure of 2.2, and it justifies why `L` is drawn together with `B` in the same run.

### 2.8 Invisible structure made visible: the witnesses
A fully undecided panel looks flat. Show what is not flat about it:

- **Parity visibility** `V = <psi| P |psi>` (for the whole relief, or per panel). This equals `<X_B>`. Product relief: `V = product of cos(tau_i)` (refsim check 3). It is the amplitude with which bump and hollow can interfere.
- **CHSH / Mermin witnesses** on facet/polarity subsets, using the exact machinery already in `src/quantum/complementary.py` (exact best CHSH over settings for every pair; Mermin for a hub plus its partners), now with the **light directions as the settings**.
- A *science panel* readout (never on the audience screen unless chosen): "certificate: violated / not violated, value, bound, provenance (Moth circuit / local reference / oracle)".

## 3. Concept summary: quantum vs classical (for the claims slide)

| Element | Classical or quantum | Honest status at ~20 qubits |
|---|---|---|
| Geometry capture, UV flattening, windows, masks, blur, projector warp | classical | as before |
| Height field -> tilts `tau_i, phi_i` | classical | design input |
| Facet qubits, polarity superposition, controlled flips, light-controlled measurement | quantum circuit | simulable classically at this size |
| `L`, `G` control bits | uniform randomness | classical-equivalent unless hardware QRNG |
| Interference between depth maps; which-depth/visibility trade-off | genuinely quantum (cross terms) | detectable via the coherent vs dephased control (section 6) |
| "Advantage" | none claimed | scaling story only (section 5) |

## 4. Implementation spec

### 4.1 Register layout (replaces `graph/allocate_qubits.py`)
`facets` (interior patches then bevel-band segments), `polarity` (one per panel, existing indices kept), `lamp` (2 qubits, 4 light directions; optionally 3 for 8), `observe` (2 qubits for `gamma`). `budget()` reports simulated qubits (`facets + polarity`) and full-circuit qubits (adds lamp + observe) separately. Suggested bay-window counts: ~6 interior patches + ~9 band segments = 15 facets, + 3 polarity = 18 simulated, +4 control = 22 in the circuit.

**Bevel bands.** For each panel, offset the border inward by a band width `w` (start ~4% of panel width; the existing `scene.relief` field gives it). Split the band into `M` perimeter segments (Voronoi along the boundary); each segment is a facet. The glass edge of a panel is a border too. Glass pixels still get no qubit and render exactly 0.

### 4.2 State preparation (`quantum/relief_state.py`, new)
1. `reference_circuit(tau, phi, panels)`: exact, hand-written, used for tests and as the show fallback: `H` on each polarity qubit; `Ry(tau_i)` then `Rz(phi_i)` on each facet (`Ry(tau)Rz(phi)|0>` has Bloch `(sin t cos p, sin t sin p, cos t)`); `CZ(B_p, f)` for each facet `f` of panel `p`. For the next step to be non-trivial add nearest-neighbour entangling on the facet graph (e.g. `CZ` or `exp(-i theta XX)` along coplanar edges with strength from geometry, the *same couplings the mock already uses*: positive within a plane, negative across a crease).
2. `moth_relief_circuit(...)`: ask QDrive for `U_psi` from a correlation spec (4.3), then apply the same `H` and `CZ` layer. Always compare it against the reference: because the ideal state is known exactly, *report fidelity of whatever Moth returns*. That is a clean, honest Moth result even if it misses.
3. `sampler_local.StateSource`: extend the layout to `{facets, polarity, lamp, observe}`; keep the `Pool` idea but with fields `light_idx`, `obs_idx`, `facet_bits`, `polarity_bits`. A frame = draw `(L, G)`; build/cached state; apply per-facet light rotation `U_i(L)` and per-polarity observation rotation `R(gamma_G, chi)`; sample `K` shots. All outcomes come from the one state.

### 4.3 Targets for QDrive / graph-v1 (`targets/payloads.py`, new builder)
Ideal-state moments (single branch qubit `B` shown; for several panels use the same pattern per panel; `O = product of cos(tau_k)`):

| Target | Value |
|---|---|
| `<Z_B>`, `<Y_B>` | 0 |
| `<X_B>` | `O` |
| `<Z_i>` | `cos tau_i` |
| `<X_i>`, `<Y_i>` (facet averaged over both depths) | 0 |
| `<Z_B X_i>` (slope sign locked to depth) | `sin(tau_i) cos(phi_i)` (and `Z_B Y_i` with `sin(phi_i)`) |
| `<X_B Z_i>` | `O / cos(tau_i)` |
| `<Z_i Z_j>` | `cos tau_i cos tau_j` |
| `<X_i X_j>` (same-branch slopes correlated) | `sin tau_i sin tau_j cos phi_i cos phi_j` |

These targets are *quantum-consistent* (they come from a real state) and use X/Y words, unlike the Z-only targets that made the whole v1 reproducible by a classical Ising sampler. Facet-only targets (ask for `U_psi` alone) are the simpler first job: Bloch `X, Y, Z` per facet plus `XX`, `YY`, `ZZ` on coplanar edges. Add the `B` targets only after that works.

Lessons already paid for in credits, apply them:
- graph-v1 applies operations **sequentially**; later targets overwrite earlier ones on shared qubits (first run: rms error 0.63, 32 of 66 values off by more than 0.1). Keep the target set **sparse and mutually compatible**; use the existing `weakest_first` order and `drop_below`. The unsent `payload_graph_v1_weakest_first.json` is the cheapest first experiment (5 credits).
- QDrive `expvals` must be a mapping (`{"Z": 0.3}`), proven by a failed job. Mixed Pauli words (`{"XZ": ...}`) and multi-qubit words are documented but unverified: test one with a minimal 2-qubit payload before building the full one.
- The repo's own record says QDrive has returned **no circuit** yet and the folder holds three job files (17:45, 22:49, 22:52) with no saved results for the last two. First task: find out what those two jobs returned (`GET /jobs/<id>/status` and `/result`) and write down the real error text.

### 4.4 Compose (`texture/compose.py`, `texture/bevel.py`)
New composition is purely *interpolation of measured facet outcomes inside geometry-set windows*:

    lit_i   = fraction of K shots with facet i bit == 0
    S       = interior-patch field + band field, interpolated (bands along the perimeter, interior across the panel,
              blended by distance-to-border weights, mask-aware)
    value   = scene.lo + (scene.hi - scene.lo) * S
    out     = masked_blur(value) * frame_mask          # glass exactly 0, unchanged

Delete from the render path: `directional_bevel`, `relief_profile`, `polarity_map` weights (`w_relief`, `w_dir`). Keep them importable only for `mock/` and the classical baseline, clearly marked CLASSICAL. Windows (centre vs side planes, near vs far) stay: the geometry still fences the values.

### 4.5 Controls and UI (`ui/keys.py`, `ui/session.py`, `ui/operator_panel.py`, `show.py`)
Performance mode, remove entirely: keys `1 2 3 4 L 0` (lighting world), `5 6 P 7` (polarity), `X` (polarity read basis), and the matching buttons, API routes and tests. What stays in performance mode: next/previous/auto/resample (`Space`, arrows, `A`, `R`: each is a *new measurement*, not a choice of look), exposure `[ ]` (shutter; not a scene property; if you want zero appearance controls, add a 2-qubit exposure register in the same circuit and keep the keys only in the science panel), blend `- =`, blackout, calibration, test patterns, overlay.

New *science panel* (operator laptop only, labelled as experiments, not looks):
- **Control A, independent noise** (existing `N`).
- **Control B, dephased depth**: replace each polarity qubit by its classical mixture (apply a full dephasing channel in the X-Y plane before readout, or equivalently average the two branches' probabilities). Same marginals, no cross terms.
- **Witness run**: executes a separate measurement set (B in X, facets in Z; CHSH settings; Mermin) and prints the certificate with provenance.
- **Depth-sphere overlay**: per panel, a dot on the polarity qubit's Bloch sphere showing the `(gamma, chi)` just drawn, so the audience can *see where the observation landed*.
- Always-visible provenance chip (existing): `Moth QDrive circuit` / `local reference circuit` / `classical rehearsal`.

### 4.6 Files to touch (verify names by reading first; some are inferred from the runbook)
`src/graph/allocate_qubits.py`, `src/graph/patches.py` (+ new `bands.py`), `src/geometry/relief.py` (height field to tilts), `src/targets/{payloads,calibrate_from_mock,consistency}.py`, `src/quantum/{sampler_local,solver,complementary,ab_compare,make_frames}.py`, new `src/quantum/relief_state.py`, `src/texture/{compose,bevel,frames,fast,interpolate}.py`, `src/ui/{keys,session,operator_panel,demo,caption,overlay}.py` and `src/ui/web/*.html`, `src/baseline/coherence.py` (new metrics), `tests/*`.

## 5. Scaling: the part to say out loud to Moth

**Visibility law.** For a product relief the interference visibility between depth maps is `V = product of cos(tau_i)`, about `exp(-sum(tau_i^2)/2)`. With a fixed tilt per facet it collapses as facets are added (refsim: tau = 0.35 gives 0.61 at N = 8, 0.32 at 18, 0.04 at 50, 3.7e-6 at 200, 7e-28 at 1000). Hold `sum(tau_i^2)` fixed instead (`tau_i = kappa / sqrt(N)`) and `V` stays at 0.61 from N = 8 to N = 1000 (refsim check 3). So **the same picture can be specified at any resolution if each facet's depth commitment shrinks like 1/sqrt(N)**: finer grain, more tentative depth. Allocate the tilt budget `kappa^2 = sum(tau_i^2)` by geometry (most to the bevel bands). Demonstrate at N = 12, 16, 18 with `kappa` fixed and show `V` and the frames are stable. Poetic reading: a macroscopic bump and a macroscopic hollow cannot interfere; superposed relief survives only if it is spread thin.

**What is hard at scale and what is not** (be exact):
- A product relief plus a parity layer is a two-term superposition of product states; classical sampling stays easy at any N (rejection sampling on the two-branch amplitude). Do not claim otherwise. If you want a 1000-facet *picture*, this is a legitimate classical stand-in and should be labelled as one.
- Hardness enters only if `|psi>` itself is a many-body state built from geometry (entangling layers along the facet graph, creases frustrating, light-controlled measurement angles). Sampling from such shallow, locally-measured, entangled states is the territory of IQP / shallow-circuit hardness results, but those need generic angles and ours are geometry-set. **Say "conjecture / direction", not "advantage".** Moth's Quantum Echo (`otoc-echo-v1`) is the relevant engine to ask about for dynamics-based extensions (an echo/OTOC measures how a local perturbation scrambles across the facet graph, which maps naturally to "a touch on one patch spreading as relief across the surface"); its parameters are not documented, so ask.
- Reading `V` at scale is a collective N-body measurement (parity); on hardware it is one more measurement setting, but its signal decays unless the tilt budget is respected. That is the point of the law above.
- **Don't** reuse the older "quantum marginal problem / frustration" argument: the mock's targets were derived from a classical distribution, so they are classically consistent by construction.

## 6. Acceptance tests (add under `tests/`)

1. `refsim` regression: `python superposed_relief_refsim.py` exits 0 and the printed numbers match this file (Born 0.963095; `<X_B> = prod cos`).
2. **Reference vs layout**: the Qiskit `reference_circuit` statevector equals refsim's `superposed()` to 1e-10 for a small case, with the repo's little-endian conventions pinned (as `tests/test_frames.py` already pins them).
3. **No toggles**: a test that every key and API route from section 4.5 listed as removed returns "unknown"/404 in performance mode; a static test that the render path does not import `directional_bevel` or `relief_profile`.
4. **Glass stays black** (existing checks) over frames drawn from the new source, including band facets next to glass.
5. **Which-depth/visibility**: with `G` forced to each of its four values *in a test only*, bevel contrast is monotone non-increasing in magnitude toward gamma = 90 deg and zero within sampling error at 90 deg.
6. **Coherent vs dephased**: at gamma = 90 deg, the *marginal* facet statistics are equal within error, the *joint* distribution differs (refsim at N = 8, tau 0.35 to 0.55, shows total-variation 0.10 to 0.24); independent noise differs in both.
7. **Visibility law**: for `tau_i = kappa/sqrt(N)`, `V` is within 5% across N = 8..18 in the simulator and matches `prod cos(tau_i)`.
8. **Witness honesty**: the certificate panel reports "not violated" for the independent-noise and dephased controls and never labels a classical rehearsal as a Moth result (extend `tests/test_complementary.py`).
9. **Moth fidelity**: whatever QDrive/graph-v1 returns is scored against the ideal-state moments (table in 4.3) and the *fidelity or rms is printed even when poor*. The show falls back to `reference_circuit` and says so in the chip.

## 7. Suggested order of work (about 4 days)

1. Day 1: read `keys.py`, `session.py`, `compose.py`, `sampler_local.py`; implement `relief_state.py` reference circuit, bands, new layout; get tests 1, 2, 4 green with the *classical rehearsal chip* still honest.
2. Day 2: new compose path, remove toggles, depth-sphere overlay, controls A and B; tests 3, 5, 6.
3. Day 3: Moth: reconstruct the two unsaved QDrive jobs; minimal Pauli-word probe; facet-only `U_psi` target job; graph-v1 weakest-first job for exact tomography; fidelity vs ideal. Witness panel; tests 7, 8, 9.
4. Day 4: rehearse the five-minute demo (below), freeze, write the claims slide.

**Demo beats (replace the existing seven):** (1) one wall, flat; (2) the wall as qubits, each a normal; (3) photons: Lambert is Born, hard grain to soft with exposure; (4) bump, hollow, and flat as the depth sphere dot moves; (5) control B (dephased) next to the coherent state: the picture of the marginals is the same, the correlations are not; (6) the visibility law and the 1/sqrt(N) rule; (7) honest scorecard, including what Moth did and did not return.

## 8. Moth engine plan

- **QDrive (primary).** Its design target is "specify correlations, not gates", which is exactly how `U_psi` is described. 1 credit per run. Use it for `U_psi`; the CZ layer, light rotations and polarity readout are composed locally onto its circuit.
- **graph-v1 (verifier + small samples).** Exact tomography verifies X/Y/Z facet targets and edge correlators (5 credits). Its bitstrings are only the top 20, so frames come from the QDrive circuit sampled locally, as in the existing architecture.
- **Tessa Image (case, not core).** Its colour sphere (luminance on the vertical axis, hue as azimuth) is the Bloch sphere, and a *normal map* is an image of unit vectors. Mapping: lightness <-> `cos tau`, hue <-> slope azimuth `phi`, saturation <-> `sin tau`. Raised and sunk then sit at **complementary hues** (hue + 180 deg), and an equal superposition of the two orthogonal-extreme depths sits at the pole, i.e. flat. Optional experiment: encode the facet normal map with Tessa on the emulator (about 64x64) to show the mapping, noting that Tessa decodes to an image and returns no joint state, so it cannot supply the polarity superposition or the correlations. A hardware version has a budget of about 448 data qubits, which is the only path in the platform to genuinely larger facet counts. Case to make to Moth: Tessa shows the encoding at scale, QDrive supplies the entangled state, graph-v1 verifies it.
- **Questions to send Moth:** (a) multi-qubit and mixed Pauli words in QDrive `expvals`; (b) the qubit caps for QDrive and graph-v1; (c) whether a returned QDrive circuit is exact for the targets in the emulator; (d) Quantum Echo's parameters; (e) whether Tessa's encoded circuits can be extracted.

## 9. Risks, stated plainly

- **Contrast of the ambiguous frame.** A fully undecided panel is flat at first order; the audience sees "less", not "more". Mitigate with the depth-sphere overlay and the witness readout, and keep the sequence moving so the swell-flatten-sink rhythm carries the effect.
- **Visual strength vs visibility.** `kappa^2` large gives strong bevel and tiny interference; small gives subtle bevel and large interference. Tune with the knob that costs nothing (the tilt budget) and report both numbers.
- **Moth may not reproduce the target state.** That is why the reference circuit and the fidelity report exist.
- **Parity witnesses in rotated frames.** The parity statement is exact in the unrotated Z frame; with tilted light axes the frame-level parity is only approximate. Compute the witness in the science panel's own measurement settings, not from performance frames.
- **Unverified field names.** Several file and function names above come from the runbook, not from reading the code; check before editing.

## 10. One-paragraph pitch for the repo README

*Standing Light projects shading, not images. Each patch of a real surface is a qubit whose state is its surface normal; the light is the measurement axis, and Lambert's cosine law is the Born rule. Depth itself is in superposition, a raised and a sunk version of each panel; how decided the depth looks in any frame depends on where the observation lands on the depth qubit's Bloch sphere, and that is drawn inside the circuit, never toggled. The more definite the depth, the less the two depths can interfere, and the visibility of that interference follows a law (the product of cos of each facet's tilt) that tells us how finely such a surface can be specified at 1000+ qubits.*
