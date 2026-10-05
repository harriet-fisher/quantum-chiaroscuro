# Standing Light — Quantum Chiaroscuro

> **Scene note.** This document describes the first build's **three-panel** bay window (left, centre, right). The built-in window, and the default of the code, is now **six panels**: each wing holds two windows stacked on top of each other with a shared rail (`left_top`, `left_bottom`, `centre_top`, `centre_bottom`, `right_top`, `right_bottom`; three planes). Counts and numbers below that say three panels, 15 or 19 qubits, or 3 polarity qubits are the three-panel ones; the README's *The default scene* has the current ones.

*Project specification v3 · a height-illusion projection instrument whose shading is drawn by quantum measurement, built on Moth Atlas.*

Author: Harriet Fisher · Updated: 1 Oct 2026 · Status: pre-build concept spec. A classical look-test mock exists (`mock/`); nothing has been run on a Moth engine yet and no credits have been spent.

**Evidence tags**

- **[VERIFIED]** read directly from a Moth Atlas or docs page, or observed by running the mock.
- **[INFERRED]** my reasoning from verified facts; needs a test run to confirm.
- **[UNKNOWN]** not found in the docs I could read; must be checked before relying on it.
- **[DESIGN]** a design choice or starting value of ours, not a fact about Moth.

**What changed from v2:** the overview and purpose are rewritten around the original intent (a quantum *height illusion*, not a renderer, not a light mover); the mock-up results are folded in; the qubit count and parameters match the working mock (23 qubits); the honest quantum analysis is tightened around what makes the project quantum rather than merely random. For API and engine reference, setup, tooling and the Claude Code first prompt, see `standing-light-handoff.md`; where the two differ, the handoff wins.

---

## 1. Overview

### 1.1 In the author's words

> I want this to be cool! I want it to be quantum! I think part of it can also be, it doesn't actually need to have rendering with fidelity, in the sense that part of what can be cool about it is lets say it shades some of the qubits in the center of one side of a pane dark and the areas closest to the edges bright, this creates an *illusion* of depth, where the center of that pane appears to almost sink in, while the brighter edges appear to have more height. It can then be blurred to make the colors blend more naturally and sell that illusion. I guess ultimately it's not a renderer, it's like a height illusion renderer where the shading uses quantum randomness.

> I basically want to project shading with randomness, using the actual 3D geometry. I still want to use some of the geometric data, like for example, with the bay windows, the flat center window should be a different overall brightness/darkness than the side ones, because it's on a different spatial plane. Similarly, the part of the side panes that are closer to the camera should be different than farther away, but they don't necessarily have to be darker farther, as they would be in the real world with real light, they just should respect the geometric rule of those areas existing in different spaces relative to light and consequently should have different projections (or different bounds from which the projection values can be randomly selected).

### 1.2 The idea in plain terms

Take a real surface with real geometry: the first target is a **bay window**, a flat centre panel flanked by two angled side panels, each with a pane of glass. Project light onto it from a projector. But instead of projecting a picture, project **shading**: the patterns of light and dark that the eye reads as *form*.

If the middle of a side panel is projected dark and its edges are projected bright, that panel appears to **sink**: a hollow, a recess, a crevice. Flip it, bright middle and dark edges, and it appears to **swell**: a cushion, a boss, a raised plate. Nothing physical changes. The wall is flat. Only the shading changes, and the shading is enough to make the viewer's brain build depth, height, bumps and crevices that do not exist.

Blur softens the transitions so the patches melt into smooth gradients and sell the illusion. And the shading is not authored by a person and not computed by a lighting formula. It is **drawn by quantum measurement**, inside limits that the real geometry sets.

### 1.3 The reference look: three pencil still-lifes

The three reference drawings show the same arrangement of a cone, a hexagonal block, a cube and a sphere. The forms are simple: a circle, a triangle, planar faces. What differs between the drawings is only the **shading**:

- In one the light comes from one side and the cast shadows fall the other way; in another the light comes from the opposite side, so every lit and dark face swaps.
- In the third the shading is **sharper**, with less gradient: facets are cut by hard edges of tone, and the sphere carries a crisp highlight band, giving a more graphic, faceted look.

Same forms, different light direction, different hardness, and each reads as a different scene. These are mild examples, but they are the point: *the geometry stays fixed; the shading changes the story*. Standing Light does this on a real wall, and the shading is **random**, chosen by a quantum system, while still respecting the real geometry.

### 1.4 What it is, and what it is not


| It is                                                                                                                               | It is not                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| A **height-illusion shader**: shading used as a depth cue                                                                           | A physically based renderer; it does not need rendering fidelity                                |
| Shading **drawn by quantum randomness inside geometry-set bounds**                                                                  | Colours chosen by the user, dragged by hand, or computed deterministically from surface normals |
| A device for creating **bump, height, depth and crevice relationships between shadows and highlights that do not physically exist** | Just moving a light around and seeing how it looks                                              |
| Geometry-aware: planes, creases, near/far, impenetrable regions                                                                     | Light that gets darker with distance as real light would (regions only have to *differ*)        |
| A demonstration that **quantum correlations make random output coherent**                                                           | A claim of quantum advantage at about 20 qubits                                                 |


## 2. Purpose

### 2.1 What the project is for

1. **To create an illusion of space.** The deliverable is a perceptual effect: surfaces that appear to have height, depth, hollows, ridges and creases that are not there. The relationships *between* shadows, highlights and crevices (this edge is bright because that one is dark, this panel is sunk because its neighbour is raised) are the substance of the work. It is about the **grammar of relief**, not about where the lamp is.
2. **To make quantum computing the author of that illusion.** The project exists to show a quantum system doing creative work that is visible and legible: it chooses the shading, and entanglement is what keeps the choice coherent.
3. **To answer Moth's brief.** Moth Quantum asked for a project built on the Atlas platform, with particular attention to **QDrive** and **Tessa Image**, accepting that if an engine does not suit the idea a case can be made instead. The project is a spatial, visual application of Moth's own principle of *specifying correlations, not gates*, and it gives a concrete case for or against each engine (§11). Ideas and thought process count for more than polish.
4. **To be cool.** It should look striking when it runs: a flat window that re-lights itself, one moment hollow and the next raised, lit from the left then the right, soft then sharp.

### 2.2 Quantum above all else

This is the governing design rule. **The quantum system is the author of the shading; everything classical is bookkeeping around it.**

- Classical code does the work that has no creative content: capturing geometry, labelling regions, building masks, computing windows and relief profiles, interpolating and blurring, driving the projector.
- Every decision about *how the surface looks* in a given frame (which way the light falls, which panels are raised or sunk, how hard or soft the light is, which patches are bright) comes from **quantum measurement outcomes**.
- **A test for the design:** swap the quantum layer for a plain random-number generator and the result should lose what made it work. With independent noise the panels do not agree and no relief reads. The structure that makes the illusion convincing must live in the *correlations* the quantum system is asked to produce.
- The geometry never *computes* the colours. It sets the fences; the quantum draw decides where inside the fences each value lands.

### 2.3 The honest side of "quantum"

Wanting it to be quantum above all else is exactly why the framing has to be honest. At 20 to 23 qubits everything here is classically simulable, and a purely Z-basis version could be faked with a classical correlated sampler (the mock does precisely that, to test the look). So the project pushes toward the parts that are *not* classical-fakeable: **complementary observables** (lit/dark and raised/hollow measured in different bases, so a room cannot have both a definite light state and a definite relief state), entanglement as the specification of structure, and, on hardware, physical randomness. §3.6 and §4 set out which claims can be made and which cannot.

## 3. The core concept

### 3.1 The illusion: how shading makes relief

The eye reads form from a handful of cues, and Standing Light drives them directly:


| Depth cue                                                                                                   | How the project produces it                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Edge-bright, centre-dark** (or the reverse) on a panel                                                    | A relief profile over the distance-to-border field, with sign set by a polarity qubit: centre dark and edges bright reads as *sunk*; centre bright and edges dark reads as *raised* |
| **Hollow/bump ambiguity** (the same shading reads as a dent or a bump depending on assumed light direction) | The directional bevel: one side of each trim lit, the opposite side shaded, with the sign flipped by polarity                                                                       |
| **Different planes catch light differently**                                                                | The flat centre panel has a different overall brightness window from the angled side panels                                                                                         |
| **Near and far are different places**                                                                       | Within a side panel the near and far ends get different windows (they differ; they do not have to be darker with distance)                                                          |
| **Light direction**                                                                                         | Lamp qubits choose which side is lit; panels follow by their orientation                                                                                                            |
| **Crevices and creases**                                                                                    | Contrast flips across a crease (one side lit, the other dark), producing a visible seam                                                                                             |
| **Hardness of light**                                                                                       | Few shots give hard, posterised shading; many shots average to soft gradients                                                                                                       |
| **Smooth gradients sell the form**                                                                          | Mask-aware blur blends the patches without letting black leak out of the glass                                                                                                      |


None of this needs physical correctness. A near end that is *lighter* than a far end can work as well as one that is darker, because the requirement is only that those regions **differ**, consistent with them being in different spatial positions relative to a light.

### 3.2 Geometry layer (classical)

From the captured mesh or hand-drawn shapes, compute:

1. **Planes** and their normals; **creases** between planes; dihedral angles.
2. **Region labels:** `panel` (projectable), `glass` (impenetrable), `frame`.
3. **Depth coordinate** along each panel (near end to far end as seen from the camera/projector).
4. **Distance-to-border field** per panel (0 at the border, 1 at the centre), where a border is a crease, a frame edge, or a glass edge.
5. **Visibility from the projector** (ray casting). Anything the projector cannot see is also impenetrable.

The geometry decides **three things**: (a) what may never receive light, (b) the allowed brightness range of each region, and (c) how regions relate to each other (same plane, crease, near, far).

### 3.3 Impenetrable regions: a hard rule

Glass panes, holes, hidden or shadowed regions, and anything the user marks are **impenetrable**. For these:

- Output pixels are **pure black**, always, in every frame.
- They are **not calculated on**: no qubits, no `coupling_map` edges, no targets.
- They are **excluded from blending**: blur is mask-aware (blurred(value × mask) ÷ blurred(mask)) so black never bleeds into neighbouring panels and bright values never bleed into glass.
- Their edges count as **borders** in the distance-to-border field, so panel areas next to glass behave like frame edges.
- A final multiply by the mask guarantees black even if an earlier stage misbehaves.
- Excluded cells free up qubit budget for the panels that matter.
- Practical note: a projector cannot project true black; "black" means no light, so the wall shows ambient level plus the projector's black level. Dim the room.

### 3.4 Brightness bounds from geometry

Every projectable region gets an allowed range [lo, hi] before any quantum step. The quantum draw chooses the value *inside* that range. [DESIGN]

- **Per plane:** the flat centre panel has a different overall window from the angled side panels, because it is on a different plane.
- **Per depth:** within a side panel, the near and far ends get different windows.
- **Relief profile:** a function of the distance-to-border field giving "edge raised, centre sunk" or its inverse.

Starting values (the ones that worked in the mock) are in §9.

### 3.5 What the quantum system decides: three knobs

**Knob 1: lighting world (light direction).** Two "lamp" qubits join the graph and connect to every patch qubit. L1 selects left/right light; L2 selects frontal versus grazing. Measuring the lamps picks the lighting world for that frame and the patches follow with geometry-set probabilities.

- Panels on the same plane share the same dependence, so they light up together. Across a crease between left- and right-facing panels the dependence flips sign, so one side is lit and the other dark.
- A flat front-facing centre panel depends little on L1 and mostly on L2, so it differs from the angled sides, as the geometry requires.
- Correlation magnitudes below 1 give "random within bounds" rather than a fixed pattern.
- If panel-to-panel targets come from one consistent sign pattern, the target set is mutually compatible (GHZ-like), so the solver can reach it. With magnitudes below 1 compatibility is approximate, so always log requested versus achieved. [INFERRED]

**Knob 2: relief polarity (hollow or raised).** One qubit per panel decides whether its relief reads as sunk or raised. Geometry-set correlations between polarity qubits decide whether neighbouring panels agree. This is the core of the *illusion of bump and crevice*: the same light, the same flat wall, and the panel flips between a dent and a boss. [DESIGN; the perceptual effect is well established, the implementation is ours.]

**Knob 3: hardness.** One shot per frame gives hard, posterised shading; averaging K shots converges to a smooth gradient. K is the sharp-to-soft control, matching the difference between the soft and hard-edged pencil renderings. [DESIGN]

### 3.6 Which knob is most quantum-specific

Ranked from the mock and the quantum analysis:

1. **Polarity as a complementary observable**: the most quantum-specific. Lit/dark is measured in Z, raised/hollow in X. A patch with a definite light state then has a completely random polarity, and vice versa; the room cannot decide both at once. With entanglement across creases in both bases, no assignment of pre-existing classical values reproduces the joint statistics. It needs basis-rotated measurement, which `graph-v1`'s documented sampling does not provide; a locally simulated QDrive circuit with added rotations could. [INFERRED]
2. **Lighting world**: coherent correlated randomness, but Z-only, so classically reproducible.
3. **Hardness**: sampling statistics; real noise on hardware, but classical statistics.

**Recommended hero:** lighting world (visible, easy to show) plus polarity complementarity (the quantum centrepiece). Present hardness as a free control rather than a quantum claim.

### 3.7 From measurement to pixels

For a panel region with bounds [lo, hi]:

1. Sample the circuit; read the lit/dark bit of every patch qubit (bit 0 = lit under ⟨Z⟩ = +1). [DESIGN]
2. Average over K shots to get a lit fraction per patch (K = 1 hard, large K soft).
3. Build the relief: height h = polarity × smoothed distance-to-border; shade the bevel with n = (−h_x, −h_y, 1) against the frame's light direction, so flipping polarity flips hollow/raised under the same light.
4. Combine light, relief and bevel (mock weights: 0.75, 0.20, 0.70) and map into [lo, hi].
5. Interpolate smoothly across patches (normalised per panel, so nothing leaks across creases), then apply mask-aware blur.
6. Multiply by the impenetrable mask.

### 3.8 Colour (optional)

Start greyscale or two-tone. If colour is added: Tessa's colour sphere (white top, black bottom, hue as angle, grey at the centre) matches a Bloch sphere [VERIFIED for the sphere; INFERRED for the mapping]. One idea: lit/dark from the Z measurement and hue quadrant from X/Y, giving a real rule that a patch with definite brightness has random hue. It needs basis-rotated sampling and is a stretch goal.

## 4. Why quantum, honestly

- The task needs **correlated randomness with structure set by geometry**: the draw must be coherent across planes and flip across creases. Entangled states specify and sample exactly that.
- At ≤ 23 qubits this is **classically simulable**; Moth's own Blur, Telablur and Shader pages say they run on classical simulators. **No quantum advantage is claimed.**
- The **mock is a classical sampler** and proves only that the *look* is achievable. It cannot imitate the complementary-polarity version faithfully.
- **Where it is genuinely quantum-specific:** complementary observables (§3.6), the correlation targets as the program (Moth's own "specify expectation values, let the machine find the circuit" idea), and physical randomness on hardware.
- **Scaling case:** one qubit per patch, a few lamp qubits, correlations from geometry. Real geometry makes frustrated loops, and sampling such correlated distributions is the kind of problem expected to get classically hard at scale. [INFERRED; verify before stating publicly.]

## 5. Use cases

1. **Gallery / installation:** a bay window or alcove that re-lights itself every few seconds, hollow one moment and raised the next, with no moving parts.
2. **Architectural illusion:** relief and depth on flat or lightly modelled surfaces without physical modelling.
3. **Creative-tool demo for Moth:** a spatial, visual application of "specify correlations, not gates".
4. **Education:** a tangible demonstration of entanglement producing coherent randomness and of measurement collapsing a superposition of lighting worlds.
5. **Performance:** light direction, polarity and hardness driven by measurement, optionally paired with sound.

## 6. Inputs


| #   | Input                | Format                                                                          | Source             | Notes                             |
| --- | -------------------- | ------------------------------------------------------------------------------- | ------------------ | --------------------------------- |
| 1   | Surface geometry     | Mesh (OBJ/PLY/USDZ) from a phone LiDAR app, or hand-drawn polygons over a photo | Classical capture  | Clean planes work best            |
| 2   | Region labels        | Per-polygon tag: `panel`, `glass`, `frame`, `off-limits`                        | Pen tool           | Glass = impenetrable              |
| 3   | Projector pose       | Position, orientation, field of view relative to the surface                    | Calibration        | Needed for visibility and warping |
| 4   | Brightness bounds    | Per-plane and per-depth [lo, hi]                                                | Defaults + sliders | §9 gives starting values          |
| 5   | Light-world settings | What the lamp qubits represent; correlation strength                            | UI                 | Controls randomness amount        |
| 6   | Hardness             | Shots averaged per frame (K)                                                    | UI                 | Sharp to soft                     |
| 7   | Blend amount         | Mask-aware blur radius                                                          | UI                 |                                   |
| 8   | Patch resolution     | Number of patches, within the qubit budget                                      | Setting            | Mock: 6×4 grid, 18 patches        |
| 9   | Moth credentials     | Bearer token in `MOTH_API_KEY`                                                  | Atlas → API keys   | Never paste in chat or commit     |
| 10  | Optional audio       | WAV                                                                             | User               | For the sound layer               |


## 7. Outputs

- `geometry_report.json` — planes, creases, labels, bounds, impenetrable mask, qubit allocation.
- `targets.json` — the Bloch/relationship targets sent to the engine, including lamp-hub edges.
- `quantum_state.json` — achieved single-qubit and pair values, with a requested-versus-achieved table.
- `frames/` — a stream of projection frames, each a fresh draw (lighting world, polarity, hardness).
- `coarse_map.png` — one cell per patch, glass cells blacked and labelled (diagnostic).
- `uv_or_projector_texture.png` — final texture in UV or projector space.
- `run_card.md` — parameters, seed, backend, shots, credits, timestamp.
- Overlay: the patch graph with lamp-hub edges drawn on the geometry.

**What "good" looks like:** glass exactly black in every frame; planes light and darken as groups; creases show contrast; centre and sides sit at visibly different overall levels; panels read as hollow or raised depending on the draw; blending hides seams without leaking into glass; and above all, a viewer reads **depth that is not there**.

## 8. System architecture

### 8.1 Pipeline

```
capture → geometry analysis → region labels + impenetrable mask
   → qubit allocation (glass excluded) → target table (windows, lamp hub, polarity)
   → quantum engine → samples → bounds + relief + bevel → masked blur → mask multiply → projection
                                                                      ↘ (optional) sound

```

### 8.2 Module and function map


| Module               | Responsibility            | Key functions                                                                                                                                    | Classical / Quantum       |
| -------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| `capture/`           | Get geometry              | `load_mesh()`, `draw_polygons()`, `import_lidar_export()`                                                                                        | Classical                 |
| `geometry/`          | Understand the surface    | `cluster_planes()`, `compute_normals()`, `find_creases()`, `depth_coordinate(panel)`, `distance_to_border(panel)`, `visibility_from_projector()` | Classical                 |
| `mask/`              | Impenetrable regions      | `build_mask()`, `exclude_from_graph()`, `masked_blur()`, `apply_black()`                                                                         | Classical                 |
| `graph/`             | Qubit graph               | `allocate_qubits()`, `add_lamp_hub()`, `build_edges()`                                                                                           | Classical                 |
| `bounds/`            | Allowed ranges            | `plane_window()`, `depth_shift()`, `relief_profile()`                                                                                            | Classical                 |
| `targets/`           | Geometry → engine targets | `lamp_correlation()`, `panel_correlation()`, `polarity_correlation()`, `to_graph_v1_payload()`, `to_qdrive_payload()`, `check_consistency()`     | Classical (feeds quantum) |
| `quantum/solver.py`  | Engine interface          | `solve(graph, targets)` with backends `graph_v1` and `qdrive`                                                                                    | **Quantum**               |
| `quantum/sampler.py` | Per-frame draws           | `sample_frames()` (local simulation of the returned circuit, with optional basis rotation for polarity)                                          | **Quantum (simulated)**   |
| `texture/`           | Samples → pixels          | `lit_fraction()`, `compose()`, `bevel()`, `interpolate_patches()`                                                                                | Classical                 |
| `projector/`         | Output                    | `set_pose()`, `warp_to_projector()`, `projection_window()`                                                                                       | Classical                 |
| `baseline/`          | Honest comparison         | `classical_correlated_sampler()`, independent-noise baseline                                                                                     | Classical                 |
| `audio/` (optional)  | Sound                     | `render_echo()`                                                                                                                                  | Quantum emulator          |
| `store/`             | Records                   | `save_run_card()`, `cache_job()`                                                                                                                 | Classical                 |


### 8.3 Credit-aware run loop

- `graph-v1` costs 5 credits per run; QDrive costs 1. [VERIFIED]
- Build targets once per scene, call the engine once, cache by a hash of the targets.
- For many frames, use the circuit QDrive returns (output slot `circuit`) and sample it locally; QASM3 loading is [INFERRED]. `graph-v1` returns only the top 20 outcomes plus exact tomography, so it cannot supply a varied frame stream; use it for the exact state and as a cross-check. [VERIFIED]

## 9. Worked example: bay window (as built in the mock)

Canvas 1200 × 700. Left panel, centre panel and right panel as quadrilaterals in a concave bay view; a 6 × 4 patch grid with a cell kept if at least 30% of it is projectable; glass inset to 0.45 of each panel, scaled about its centroid. Glass cells get no qubit.

**Qubit allocation:** 18 patch qubits + 2 lamp qubits + 3 polarity qubits = **23 qubits**. Emulator caps are [UNKNOWN] for graph-v1 and QDrive (Labyrinth's is 20), so the budget may need trimming (coarser grid or more glass).

**Starting windows** [DESIGN, tuned by eye in the mock]:


| Region                | lo   | hi   | Notes                                      |
| --------------------- | ---- | ---- | ------------------------------------------ |
| Centre panel          | 0.40 | 0.95 | A different overall level: different plane |
| Side panels, near end | 0.25 | 0.80 |                                            |
| Side panels, far end  | 0.10 | 0.65 | Needs only to differ, not to be darker     |
| Glass                 | —    | —    | Black, never calculated                    |


**Mock sampler parameters** (the classical stand-in): hub couplings kh = 1.6 (L1 with normal-x) and kf = 1.3 (L2 with normal-z minus mean); patch couplings coplanar +0.9, crease −0.6; polarity chain +0.9. Side-panel normals ±sin 35° in x and cos 35° in z.

**What the mock showed** [VERIFIED by rendering]:

- **Light direction** reads clearly: left-lit versus right-lit swaps which side panel is bright, and frontal versus grazing changes the centre and the bevel strength.
- **Hollow versus raised** only read once the directional bevel was added; the first pass without it was too weak. With it, the same trim visibly flips between recessed and protruding under one light. This confirmed that the *illusion* depends on the bevel, not just on edge/centre brightness.
- **Hardness** is the weakest knob: K = 1 is blotchier than K = 128 but the difference is modest. A contrast or posterise curve would help.
- Glass and outside pixels were exactly black in all 25 rendered frames.

## 10. Extending to complex surfaces (e.g. an abstract mound)

An asymmetric mound with many gradual levels of projection is the natural next surface, and the concept carries over:

- Patches come from the mesh (geodesic clustering) instead of plane clustering.
- Edge correlations become continuous functions of the normal change or slope, so gradual levels produce gradual coupling strengths instead of a plane/crease dichotomy.
- Brightness windows vary continuously with slope and relative height; "different spatial position relative to the light" generalises to a continuum.
- Visibility ray casting marks hidden areas impenetrable automatically.
- Single projector: work in projector space to avoid UV distortion; interpolate smoothly across patches.
- Hierarchical or tiled solving if the patch count outgrows the qubit budget. [INFERRED]
- First step: a small hand-made mound, one projector, about 20 patches.

## 11. Engine plan


| Engine                                    | Role                                                                                                          | Notes                                                                                                                                                                                                             |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graph-v1` (5 credits)                    | Primary solver for the first demo                                                                             | Documented schema: `coupling_map`, `bloch` and `relationship` operations, exact tomography, top-20 samples. `relationship` targets must lie on `coupling_map` edges, so lamp-hub edges must be listed. [VERIFIED] |
| `qdrive-api-v1` (1 credit)                | Run on the same targets for comparison; best route for many frames and for the complementary-polarity variant | Certainty values, weighted Pauli sums, `update_method`, `ansatz`, chaining via `initial_circuit`; returns the circuit. Target-entry syntax undocumented. [VERIFIED / UNKNOWN]                                     |
| **Tessa Image**                           | Optional; the case to make is whether its colour-sphere encoding and hardware-noise texture serve the project | Fixed lattice adjacency, one global distortion value, distortion only on QPU (~64×64 emulator, ~20×20 QPU). Does not accept our geometry-derived graph.                                                           |
| **Blur Core / Blur**                      | Optional                                                                                                      | Neither is mask-aware as §3.3 requires, so the main blend is our own masked blur.                                                                                                                                 |
| **Retrocausal Echo / Quantum Echo / QRC** | Optional sound                                                                                                | Mood layer only.                                                                                                                                                                                                  |


**Solver abstraction:** one `solve(graph, targets)` with two backends. Run the same bay-window targets through both and report achieved versus requested values, credits, and what certainty weights change. That comparison is the case for or against QDrive.

## 12. Risks and unknowns

1. QDrive's per-target syntax is not on the page I read. [UNKNOWN]
2. Qubit caps for `graph-v1` and QDrive on the emulator; 23 qubits may be over. [UNKNOWN]
3. Whether near-pinned correlations and many hub edges are honoured; infeasible targets may be silently relaxed. Always log requested versus achieved.
4. Local loading and simulation of QDrive's QASM3 circuit. [INFERRED]
5. Basis-rotated sampling for the complementary-polarity version is not offered by `graph-v1`. [INFERRED]
6. Illusion quality depends on viewpoint, ambient light and projector black level.
7. Hardness is visually subtle; may need a contrast curve.
8. No advantage claim; the scaling story is inferred and unverified.
9. Atlas is labelled ALPHA; expect rough edges. Keep the API token in an environment variable.

## 13. Evaluation

- **Hard rule:** glass pixels exactly black in 100% of frames; no bleed after blending.
- **Coherence:** compare against independent per-patch noise and a classical correlated sampler; the targeted draws should read as a single light, and independent noise should not.
- **Fidelity of targets:** requested versus achieved correlations (table and heatmap).
- **Variety:** distribution of lighting worlds and polarities over many frames.
- **Illusion:** photograph the projection on the real bay window; a short viewer test, "does the centre look sunk or raised?", and whether the answer changes with the draw.
- **Credits:** total per session.

## 14. Build plan (~5 days from 1 Oct)


| Day   | Goal                                                                                                      | Deliverable                              |
| ----- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 1     | Port the mock to modules; pen tool, labels, patch and qubit-budget printout                               | `labels.json`, `mask.png`, patch preview |
| 2     | Moth client; first `graph-v1` emulator run with lamp hub (with approval); calibrate targets from the mock | `targets.json`, `quantum_state.json`     |
| 3     | QDrive run on the same targets; local circuit sampling; frame stream                                      | A/B comparison                           |
| 4     | Projector calibration; real bay window; masked black glass verified                                       | Projection test photo                    |
| 5     | Complementary-polarity experiment (if time); polish; run card                                             | Demo material                            |
| Spare | Sound; mound; Tessa case slide                                                                            |                                          |


## 15. Open questions

1. Which knob is the hero of the demo? (Recommendation: lighting world plus polarity complementarity.)
2. Greyscale first, or colour from the start?
3. First real surface and projector availability?
4. Sound layer in or out?
5. Questions for Moth: can Tessa's encodings or circuits be extracted or reused; does QDrive accept pure-state pins and chained circuits; what are the qubit caps?

## 16. References

- Atlas: https://platform.mothquantum.com/engines (showcase pages for Labyrinth, Tessa Image, Blur, Telablur, Entanglement Shader, Retrocausal Echo)
- Docs: https://docs.mothquantum.com/docs/engines/graph-v1, `.../qdrive-api-v1`, `.../qpixl-v1`, `.../otoc-echo-v1`, `.../blur-core-v1`, `.../qrc-audio-v1`, `.../qrc-midi-v1`, `.../qrc-gen-v2`
- Project docs: "A new way to think about quantum programming" (Motte model); Wootton, *Procedural generation using quantum computation* (FDG '20); Moth's quantum games paper (arXiv 2505.13287)
- Engine reference notes saved in project memory: `topics/moth-engine-docs.md`
- Companion files: `standing-light-handoff.md`, `mock/standing_light_mock.py`, `mock/requirements.txt`

