# Standing Light

**Quantum chiaroscuro, projected onto a real surface.**

Standing Light projects *shading*, not images, onto a real object (the reference scene is a bay window with glass panes). A physical wall has one shape; light can make it look swollen, hollowed, creased or flat. The project makes that literal with quantum mechanics:

- Every patch of the wall is a **qubit whose state is its surface normal**. The light is the measurement axis, and Lambert's cosine law is the Born rule, so a frame is a measurement, not a render.
- Each panel's **depth is in superposition** (a raised and a sunk version). How decided the depth looks in any one frame depends on where the observation lands on that panel's Bloch sphere, and that is drawn *inside the circuit*, never toggled: a pole gives a definite bump or hollow, the equator gives a flat, undecided panel.
- Geometry only supplies the **barriers and limits**: the glass panes are never lit (they are exactly black, checked on every frame), shared edges set where the shading may change, and the plane windows set how bright each surface may get.

There are two relief engines. **Superposed Relief (v2, `--engine panel`, the default)** has one polarity qubit per panel: 12 facet qubits + 3 polarity qubits = 15 simulated, exactly, on your laptop. **Domain Relief (v3, `--engine domain`)** cuts the wall into local bevel facets, gives every small stretch of bevel its *own* depth qubit and couples those qubits along the geometry's edge graph, including across the creases between panels: 69 simulated qubits in the default configuration, as a matrix-product state. It is the version where the entanglement is between *edges* of the wall, and where a lamp–depth Bell-type test (Mermin) is violated by a state of that size ([Domain Relief](#domain-relief-v3)).

Honest status: at these sizes everything is classically simulable, and a benchmark (`src.quantum.classical_baselines`) measures how far each design can grow before the classical methods stop being cheap. The claim is *structure and scaling*, not quantum advantage, and nothing in the quantum design has yet been run on a Moth engine (see [Status](#status-and-limits)).

---

## Contents

1. [Quick start (the studio)](#quick-start)
2. [Installation](#installation)
3. [The ways to run it](#the-ways-to-run-it)
4. [How it works](#how-it-works)
5. [Codebase map](#codebase-map)
6. [The performance UI](#the-performance-ui)
7. [Moth engines and spending credits](#moth-engines-and-spending-credits)
8. [Tests](#tests)
9. [Safety rules the code enforces](#safety-rules-the-code-enforces)
10. [Status and limits](#status-and-limits)
11. [Further reading](#further-reading)

---

## Quick start

Run everything from the **studio**. It is one local page that walks the whole pipeline and starts each tool for you.

```bash
cd Moth_Application
source .venv/bin/activate          # Python 3.10+; see Installation if you have no .venv
python -m src.studio               # opens the studio in your browser
```

On the page, press **Next step** each time (it always offers the first step that isn't done), or click a step's own button:

| Step | Button does | Needs |
|---|---|---|
| **1 Draw the shapes** | opens the pen tool: draw the panels and glass on the projector's own screen, press **Save** in it. Or press **Use the built-in demo scene** to try everything with no projector and no drawing | nothing |
| **2 Calibrate targets** | estimates the classical correlation targets for your drawing and writes the Moth payloads. Sends nothing, takes up to a minute | step 1 |
| **3 Rehearse the show** | **opens the show on Superposed Relief**, the quantum circuit simulated exactly on your laptop. No Moth call, no credits | step 2 |
| **4 Solve on Moth** | shows the exact payload, its hash and cost. Sends only if you started the studio with `--allow-spend` *and* press the send button | step 2 |
| **5 Perform with the result** | opens the show on the circuit step 4 returned (classical QDrive flow) | a QDrive result |

Step 3 prints a URL and opens the show. That is where you see and use Superposed Relief: the operator panel (`/`), the projector window (`/output`, drag it to the projector display and press `F`) and the audience screen (`/audience`). Press `Space` for a new frame, `A` for auto-cycle, `W` for the entanglement witness, `?` for every key. Stop ends any tool the studio started; Ctrl+C stops everything.

To keep several scenes side by side, or to allow spending:

```bash
python -m src.studio --project runs/myroom     # a separate project folder per scene (default: runs/studio)
python -m src.studio --allow-spend             # lets step 4 submit to Moth (it still shows the cost and asks first)
python -m src.studio --engine domain           # step 3 opens the domain relief (about 15 s to build) instead of the per-panel one
```

The studio writes to its project folder: `labels.json` (the drawing), `calib/` (targets and payloads), `solve/` (Moth results), `studio.json` (which drawing the targets came from, so a changed drawing marks later steps stale).

**What to know about the current studio** (accurate as of the Superposed Relief build):
- Only step 3 opens the new relief engine. Steps 2, 4 and 5 are the earlier, classical QDrive flow, and step 5 stays blocked until QDrive returns a circuit (it never has yet).
- Step 3 is gated behind step 2 even though the relief engine does not use those targets: it builds its facets straight from your drawing. So to reach relief you must run step 2 once, even with the demo scene.
- Nothing from Moth is involved in what step 3 shows. Its header chip says `local reference circuit`.

Want the figures and numbers for a talk?

```bash
python -m src.quantum.relief_report     # writes runs/relief/*.png and report.json
```

---

## Installation

Python **3.10 or newer** (the code uses `tuple | None` annotations; tested on 3.10.14). `node` is only needed to run one test (`tests/test_pen_geom.py`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `Pillow`, `matplotlib`, `requests`, and `qiskit` + `qiskit-aer` + `qiskit-qasm3-import` (for the circuit cross-checks and for loading QDrive circuits). There is no OpenCV dependency: calibration is plain numpy, and all windows are web pages served by the standard library.

**Moth API key.** Only needed to submit jobs. Put `MOTH_API_KEY=...` in the environment or in `<project>/.env` (git-ignored). The code reads that one variable lazily, never prints it, and previews never need it. If it warns that `.env` is readable by others, run `chmod 600 .env`.

---

## The ways to run it

The studio ([Quick start](#quick-start)) runs these same modules for you. You can also run any of them directly, from the project root (`python -m ...`) with the same flags. Nothing sends anything to Moth unless you explicitly say so (see [Moth engines](#moth-engines-and-spending-credits)).

### 1. The show on its own (what studio step 3 opens)

Use this to skip the studio, or to pass relief options the studio does not expose (`--kappa`, `--n-dirs`, `--entangle`, a facet-register circuit).

```bash
python -m src.show                                          # built-in bay window, Superposed Relief
python -m src.show --labels runs/scene/labels.json          # shapes you drew with the pen tool
python -m src.show --kappa 1.0 --n-dirs 4 --entangle 0.8    # relief knobs (below)
python -m src.show --engine domain                          # Domain Relief: dozens of qubits, polarity per depth domain (flags below)
python -m src.show --engine domain --seg-len 100 --lock 0   # finer facets (116 qubits), lamp left as a plain coin
python -m src.show --source relief --circuit FACETS.qasm    # a Moth QDrive circuit for the 15 facet qubits
python -m src.show --source oracle                          # CLASSICAL rehearsal (also: mock, circuit, complementary)
python -m src.show --lan                                    # also serve /output and /audience to other machines
```

| Flag | Meaning |
|---|---|
| `--source` | `relief` (default) or one of the classical sources |
| `--kappa` | tilt budget per panel, `sum(tau²) = kappa²`. Interference visibility is about `exp(-kappa²/2)`: higher is a stronger bevel and weaker interference |
| `--n-dirs` | slope-orientation facets per panel (4 gives 4 band facets, a qubit each, plus a classical plateau facet: 12 qubits for the bay window) |
| `--entangle` | strength of the diagonal couplings between facets (0 = product relief) |
| `--contrast` | projection tone: stretch of the lit field about mid-grey (an exposure, not a scene property) |
| `--calib`, `--run` | directories with calibration targets and engine results (for the demo slides and re-solve) |
| `--engine` | `panel` (default) or `domain`; the flags below apply to `domain` only |
| `--seg-len` | facet length along a polygon edge in px (150: 46 facets; 100: 78; smaller: more qubits) |
| `--group-size` | facets that share one polarity qubit (2). Tilt budget per domain: V = cos(tau)^group-size, whatever the size of the wall |
| `--tau` | tilt of each facet in radians (0.5) |
| `--pol-coupling` | Ising angle between neighbouring domains' polarity qubits (0.5; sign + coplanar, `--crease-sign` across a crease) |
| `--lock` | coupling of lamp L1 to one polarity qubit per panel with the crater-gauge signs (pi/2: maximal; 0: the lamp is a plain coin) |
| `--tau-mix`, `--pol-field` | 0..1 moves the facets toward the equator (graph-state regime, V falls); transverse field on the polarity qubits |
| `--backend` | `auto` (exact up to 20 qubits, else matrix-product state), `exact`, `mps` |
| `--projector-size WxH` | initial projector size; the output window reports its real size once open |
| `--allow-spend` | lets the re-solve dialog submit to Moth (it still asks, showing hash and cost) |
| `--port`, `--no-browser` | server port (default: any free) and do not open a browser |

### 2. The studio's steps, individually

| Studio step | Module it runs |
|---|---|
| 1 Draw | `python -m src.capture.pen_tool draw --out <project>` |
| 2 Targets | `python -m src.targets.calibrate_from_mock --labels <project>/labels.json --out <project>/calib` |
| 3 Rehearse | `python -m src.show --labels <project>/labels.json --calib <project>/calib --run <project>/solve --out <project> --source relief` |
| 4 Solve | `python -m src.quantum.solver qdrive\|graph-v1 --calib <project>/calib --out <project>/solve` (add `--approve-credits N` to actually send) |
| 5 Perform | `python -m src.show ... --source circuit --circuit <project>/solve/qdrive/circuit.qasm` |

### 3. Drawing the real scene

```bash
python -m src.capture.pen_tool draw --out runs/scene          # no photo: draw directly on the projector's screen
python -m src.capture.pen_tool draw PHOTO.jpg --out runs/scene  # or trace a photo taken from the projector's position
python -m src.capture.pen_tool build LABELS.json --out DIR    # re-export without the GUI
```

The default is **no photo**: connect the projector as a display, draw panel and glass polygons while watching where the lines land on the real object, and press Save. The labels are then in projector pixels (`source: "projector"`) and need no warp. New vertices snap to existing vertices and edges so neighbouring panels share an edge exactly.

### 4. Figures and checks

```bash
python -m src.quantum.relief_report                # relief frames + both controls, complementarity curve, scaling law, witness table, contrast vs visibility
python -m src.quantum.relief_report --domain       # + the domain relief: frames, certificate (entangled edges, lamp-depth Mermin), frustration, budget
python -m src.quantum.classical_baselines          # where classical simulation stops being cheap (a few minutes; per-point time limit)
python -m src.quantum.domain_moth plan             # Moth payloads for the domain relief (build sends nothing)
python -m src.run_mock                             # the original classical look test (reproduces the 5 mock figures)
python -m src.baseline.compare                     # correlated draws vs independent noise, same compose path
python -m src.quantum.complementary --sweep        # the older polarity-complementary experiment (classical-family stand-in)
python -m src.quantum.make_frames --source oracle  # a stream of frames with a provenance manifest
```

### 5. Calibration and the glass check on the real wall

Done from the operator panel (Calibrate and Photo test tabs): outline pattern, four-corner re-alignment after the projector moves, a safety margin that shaves projector pixels off every lit region, and a photographed check that the real glass stays dark. See [`standing-light-runbook.md`](standing-light-runbook.md) §3 for the procedure and `python -m src.projector.verify --selftest` for a synthetic run.

---

## How it works

### The quantum model (Superposed Relief)

**Qubits** (little-endian: qubit *q* is bit *q* of the state index; spin +1 is bit 0):

| Register | Size (bay window) | Role |
|---|---|---|
| facets | 12 (+3 classical) | one per band facet; state = the facet's surface normal `b = (sin τ cos φ, sin τ sin φ, cos τ)` in the panel's own frame. Each panel's flat plateau has τ≈0, so it would be a constant \|0⟩ that nothing couples to: it is a *classical* facet, lit by its deterministic Lambert value |
| polarity | 3 | one per panel, in the state "plus"; controls whether that panel's relief is raised (`h`) or sunk (`-h`) |
| lamp | 2 | four light directions; controls the axis each facet is measured along |
| observe | 2 | four observation axes (γ = 0°, 30°, 60°, 90°) on each polarity qubit's Bloch sphere |

Facets + polarity = **15 simulated qubits**; the full circuit has **19**. (An earlier version kept the three plateau qubits and counted 18/22; they were constants. An earlier slope computation also halved the slope on the shared edge between two panels, so no tilted facet of one panel touched a tilted facet of its neighbour and the three panels were exactly unentangled with each other; both are fixed, tests pin them.)

**The state.** `|Ψ> = 2^(-P/2) Σ_s |s>_B ⊗ P^s |ψ_f>`, where `|ψ_f>` is the raised relief and `P = Z` on every facet of a panel (Z maps Bloch `(x,y,z)→(-x,-y,z)`: every bump becomes a hollow). It is built as: H on each polarity qubit, `Ry(τ)Rz(φ)` on each facet, optional diagonal couplings, then `CZ(B_p, f)` for every facet `f` of panel `p`.

**A frame is one run of the circuit.** The lamp register rotates each facet so measuring Z measures σ·l (`P(lit) = (1 + b·l)/2`: Lambert is Born). The observation register rotates each polarity qubit to read along the depth-sphere axis. The whole register is measured; K photons are K shots (K is the *exposure*: hard grain with few, soft with many). The polarity outcome is that frame's depth decision. The lamp and observe registers are uniform draws (pseudo-random here, physical randomness on a QPU); the quantum content is the facet register, the polarity superposition and the controlled operations between them.

**What the geometry decides.** Only the tilts. A height field (a chamfer of width `band_frac` of each panel, distance measured to the nearest border including the glass edge) gives every pixel a slope. Pixels that slope the same way form one facet, wherever they are, because under a distant light they are lit identically. Tilts are rescaled per panel so `Σ tau² = kappa²`.

### Domain Relief (v3)

The per-panel engine is three independent-ish 5-qubit systems. Domain Relief changes what a qubit stands for so that the entanglement follows the geometry:

| Piece | What it is |
|---|---|
| **facet** | a piece of the bevel strip along one polygon edge (a panel's outer edge or a glass pane's edge), about `--seg-len` px long, one slope = one Bloch vector, as before |
| **domain** | `--group-size` consecutive facets around one loop share ONE polarity qubit (one depth decision per small stretch of bevel). The tilt budget is per domain, so V = cos(τ)^m does not shrink as the wall gets more facets |
| **graph** | domains are nodes; domains whose facets touch are joined, within a plane (coplanar) or across a shared edge between panels (crease). The polarity qubits are coupled along it: `exp(-iθ ZZ/2)` on \|+⟩\|+⟩, a graph-state-like Ising layer, sign `+` coplanar and `--crease-sign` across a crease |
| **lamp lock** | lamp L1 coupled to one polarity qubit per panel with the crater-gauge signs (a raised bevel facing right lit from the right looks like a sunk one facing left lit from the left): the light side and the depth are entangled |
| **backend** | up to 20 qubits an exact statevector; beyond, Aer's matrix-product-state simulator evolves the state and `src/quantum/mps.py` samples it *exactly* (one polarity outcome per frame, then K photons given it: no post-selection) |

What it was measured to do on the bay window (`python -m src.quantum.relief_report --domain`, 69 qubits, bond dimension ≤ 32):
- 15 of 25 domain edges carry entanglement between their two domains (negativity, certified; the dephased control has none), including across creases.
- Lamp + three leaf domains (one per panel): **Mermin value 4.34 against a local bound of 4** (dephased depth: 1.0). No *pair* violates CHSH (best 1.35): that is monogamy, and the code shows it: lock the lamp to *every* domain instead of one per panel and the Mermin violation disappears (`tests.test_domains`). More polarity coupling also lowers it (J 0 → 0.3 → 0.5 gives 6.2 → 5.5 → 4.3).
- The signed domain graph is *balanced* on its own (every cycle of the bay window crosses each seam an even number of times), so there is no frustration from creases alone. The crater-gauge lock is what frustrates it, and how much depends on which domains are locked: joining the lamp to every domain gives 10 frustrated cycles (13 at 116 qubits); the one-leaf-per-panel lock the show uses gives 3 at 69 qubits and 0 at 116. The Necker-style competition is therefore a property of the lock you choose, not of the geometry.
- A frame costs about 0.06 s end to end once the 16 conditioned states (4 observations × 4 lights) are built; building them takes about 15 s at startup.

Advanced preparations (`src/quantum/ground_state.py`, `domain_moth.py`): the polarity register's state can be the ground state of the geometry-set Ising model on the domain graph (a QAOA-style fit, 4 to 6 angles, as a circuit), or a circuit returned by a Moth engine for the target moments (`spec.pol_circuit`); a *patch* of the wall (≤ 20 qubits, boundary couplings cut) can be sent to `tomography-api-v2`; echo taps from `otoc-echo-v1` modulate each facet's tilt frame by frame (`dynamic_specs`, **not wired into the live show**).

### The properties that make it more than a render

| Property | Statement | Where verified |
|---|---|---|
| Lambert = Born | `P(lit) = (1 + n·l)/2` exactly | `tests/test_relief.py` |
| Which-depth / visibility trade-off | bevel contrast slides +strong → 0 → −strong as γ goes 0° → 90° → 180°; **exactly flat at 90° for every outcome** | refsim table reproduced; tests |
| Visibility law | interference between a panel's depth maps `V = Π cos τᵢ ≈ exp(−Στ²/2)`; holding `Στ²` fixed (`τ = κ/√N`) keeps V constant as N grows | simulated N = 8…18, closed form to N = 1000 |
| Diagonal couplings are free | `zz` phases between facets leave V, the stabilizer and the flat equator untouched; the circuit stays **IQP-shaped** (single-qubit layer, commuting diagonal gates, single-qubit measurement rotations). Azimuths are pre-compensated so every facet keeps the normal the geometry gave it | tests |
| Witnesses | `X_B ⊗ Z^F` is an exact stabilizer (= 1 coherent, 0 dephased); fidelity `F > (1+V)/2` certifies entanglement across the cut B\|rest | `relief_witness.py` |
| Controls | **A** independent noise, **B** dephased depth (a classical mixture of bump and hollow). B has the same facet statistics but no interference | witness run |

### From pixels to light

```
labels.json ──► Scene (masks, planes, windows) ──► facets ──► ReliefSpec ──► ReliefState
 (pen tool)      geometry/planes.py                geometry/    quantum/        quantum/relief_state.py
                                                   facets.py    relief_state       │  draw(K shots)
                                                                                   ▼
projector frame ◄── warp + exact-black check ◄── relief_compose ◄── lit fraction per facet
 (PNG, SSE)        projector/warp.py             texture/relief_compose.py
```

`compose` is `value = lo + (hi − lo)·S`, then a mask-aware blur, then a multiply by the projectable-pixel mask, so glass is exactly 0 by construction. There is **no bevel function in the relief render path**: the shading at a bevel is a measurement outcome. (The classical bevel code in `texture/bevel.py` is kept only for the mock and the classical baseline, and a test pins that the relief path never imports it.)

### The classical family (kept as rehearsal and baseline)

The first build (still present, selectable with `--source oracle|mock|circuit|complementary`) models the scene as a patch grid: one qubit per grid cell, two lamp qubits, one polarity qubit per panel, correlation targets estimated from a classical Ising-type sampler (`calibrate_from_mock`), a graph-v1 or QDrive payload built from those targets, and frames sampled from the returned circuit. It is the *Z-only, classically reproducible* baseline; the relief engine replaces it as the performance mode.

---

## Codebase map

### `src/geometry/`, `src/mask/`, `src/bounds/`: the scene

| File | What it does |
|---|---|
| `capture/labels.py` | `labels.json` schema, validation, defaults, the synthetic bay window |
| `capture/pen_tool.py` (+ `pen_tool.html`, `pen_draw.js`, `pen_geom.js`, `projector.html`) | the drawing tool: browser UI served locally, shared-edge snapping |
| `geometry/planes.py` | `Scene`: regions, masks, plane normals, windows, relief field; `build_scene(labels)` |
| `geometry/facets.py` | **facets**: orientation classes, tilt budget, edges between facets; classical plateau facets; per-panel slope field (no differencing across a seam) |
| `geometry/domains.py` | **domains**: spatial bevel-strip facets, depth domains, the domain graph (coplanar / crease), crater-gauge lock signs, frustration of the signed graph |
| `geometry/relief.py`, `geometry/depth.py` | distance-to-border field; depth coordinate along side panels |
| `mask/build_mask.py`, `mask/masked_blur.py`, `mask/apply_black.py` | rasterise polygons; blur that never leaks light into glass; the final exact-zero multiply |
| `bounds/windows.py` | brightness window `[lo, hi]` per plane and depth |

### `src/quantum/`: states, sampling, witnesses, Moth

| File | What it does |
|---|---|
| `relief_state.py` | **the quantum core**: `ReliefSpec`, `facet_state`, `lift_polarity`, `ReliefState.draw`, light rig, Qiskit `reference_circuit` and `full_circuit` |
| `relief_witness.py` | visibility, stabilizer, fidelity witness, CHSH scan, sampled witness run, scaling law |
| `domain_state.py` | **domain relief**: `spec_from_domains`, the exact and matrix-product backends (`make_state`), `leaf_domains` |
| `mps.py` | matrix-product states on Aer: exact conditional sampling, reduced density matrices (with operator insertions), bond dimensions and entropies |
| `domain_witness.py` | domain certificate: visibility, entangled edges (negativity, CHSH) incl. creases, lamp–depth CHSH and Mermin, frustration, controls |
| `ground_state.py` | ground state of the geometry-set transverse Ising model as a fitted circuit; classical ground-state degeneracy |
| `domain_moth.py` | payloads: polarity register via QDrive targets, patch tomography, echo-driven dynamic relief |
| `classical_baselines.py` | two-branch sampler, excitation truncation, MPS bond dimension vs size: where classical simulation stops being cheap |
| `relief_report.py` | figures and `report.json` for a talk |
| `moth_client.py` | Moth Atlas API client: gated submit, never retries a POST, saves job id and raw result first |
| `moth_engines.py` | per-engine fit, payload builders (tomography, QDrive, echo, Qpixl, shader), defensive parsers, scoring against exact moments |
| `solver.py` | submit calibrated targets to graph-v1 or QDrive and score what came back (classical flow) |
| `sampler_local.py`, `sampler_mock.py` | sample a statevector / QASM circuit locally; the classical Gibbs sampler |
| `complementary.py` | the earlier polarity-complementary experiment, CHSH and Mermin machinery |
| `ab_compare.py`, `make_frames.py` | engine comparison figures; frame streams with provenance |

### `src/texture/`: pixels

| File | What it does |
|---|---|
| `relief_compose.py` | **relief frame composition** from facet outcomes; a sparse composer for hundreds of facets |
| `compose.py`, `fast.py`, `frames.py`, `interpolate.py`, `bevel.py` | the classical family's composition and bevel (reference + fast composer) |

### `src/graph/`, `src/targets/`, `src/baseline/`, `src/store/`: the classical pipeline

`graph/` (patch grid, couplings, lamp hub, qubit budget), `targets/` (calibrate targets from the mock, payload builders, consistency checks), `baseline/` (independent-noise baseline and coherence metrics), `store/` (payload-hash result cache, compact JSON).

### `src/projector/`: getting light onto the wall safely

| File | What it does |
|---|---|
| `calibrate.py` | homography from canvas to projector (scale, four corners, or dots on a photo), plain numpy |
| `warp.py` | warp frames to projector pixels and keep protected pixels exactly 0 after the warp (`GlassLeak` refuses a bad frame) |
| `patterns.py` | black, white-in-frame, outline, dots, grid, edge, alignment markers |
| `verify.py` | photographed glass-stays-black check; verdict is NOT CHECKED without an independent trace |
| `output_window.py`, `output.html` | the frame hub and the fullscreen projector page |

### `src/ui/`: the show

| File | What it does |
|---|---|
| `session.py` | **everything the operator controls**: sources, draws, knobs, history, calibration, controls, witness, re-solve gate. No HTTP in it |
| `operator_panel.py` | the local HTTP/SSE server for `/`, `/output`, `/audience`, `/api/*` |
| `keys.py` | one key table (tagged `relief` / `classical` / `both`) used by the page and by tests |
| `caption.py`, `demo.py`, `overlay.py` | audience wording that follows provenance; the seven-beat demo; the facet-graph overlay with depth-sphere dots |
| `web/` | `operator.html`, `audience.html`, `studio.html`, `common.css` |

### Top level

| File | What it does |
|---|---|
| `src/show.py` | CLI for the performance server |
| `src/studio.py` | the five-step launcher |
| `src/run_mock.py`, `mock/standing_light_mock.py` | the original classical look test and its port (checked bit-for-bit equal) |
| `superposed_relief_refsim.py` | the handoff's numpy reference simulation; its numbers are regression-tested |
| `tests/` | see [Tests](#tests) |
| `runs/` | generated outputs (`relief/`, `moth/`, `calibration_5x4/`, `first_run/`, ...) |

**Not implemented** (one-line stubs): `capture/import_roomplan.py`, `capture/load_mesh.py`, `geometry/creases.py`, `geometry/visibility.py`, `store/run_card.py`, `targets/from_geometry.py`.

---

## The performance UI

Keys on the operator page (press `?` for the live table). In the relief family there are **no keys that choose a look**: the light, each panel's depth and the observation axis come out of the circuit.

| Key | Action |
|---|---|
| `Space` / `→`, `←`, `A`, `R` | next frame (a new measurement), previous, auto-cycle, re-sample |
| `[`  `]` | harder / softer (fewer / more photons averaged, K = 1…128) |
| `-`  `=` | less / more blend |
| `N` | **control A**: independent noise through the identical compose code |
| `M` | **control B**: dephased depth |
| `I` | experiment: read the lamp register in X (light directions interfere) |
| `W` | **witness run**: measures the certificate on this state, with provenance |
| `B`, `T`, `S`, `O`, `D`, `V` | blackout, test patterns, snapshot, overlay, demo, re-solve |

The science section also shows, per panel, a **depth-sphere dot** (where the observation landed: top = decided raised, bottom = decided sunk, middle = undecided/flat), the tilt budget and per-panel visibility, and the last witness certificate. The header chip always says what the frames come from: `local reference circuit`, `Moth QDrive circuit`, or `classical rehearsal`.

`D` starts the demo (seven beats): one flat wall → the wall as qubits → photons (Lambert is Born) → bump, hollow and flat → control B beside the quantum state → the visibility law → an honest scorecard that lists what has and has not been run on Moth.

---

## Moth engines and spending credits

```bash
python -m src.quantum.moth_engines plan                 # every engine, cost, and whether it fits
python -m src.quantum.moth_engines build tomography     # write a payload and print the exact request (sends nothing)
python -m src.quantum.moth_client preview PAYLOAD.json --engine tomography-api-v2
python -m src.quantum.moth_client send    PAYLOAD.json --engine tomography-api-v2 --approve-credits 1 --out runs/solve
```

| Engine | Cost | Role here |
|---|---|---|
| `tomography-api-v2` | 1 | **verify our actual circuit**: the polarity qubit's Bloch X *is* the visibility; scored against exact moments |
| `qdrive-api-v1` | 1 | prepare the facet register from moments. Keep jobs under ~25 targets: two 201-target jobs both died with `engine_timeout` |
| `otoc-echo-v1` | 1 | dynamic relief: echo tap `F(site, depth)` → tilt `abs(F)`, azimuth `arg F`, sign of `Re F` raised/sunk, depth as frame |
| `qpixl-v1` | 1 | scaling statement: a 64×64 Lambert map on ~13 qubits |
| `entanglement-shader-v1` | 1 | angle-phase reflectance table as a quantum-made tone response (not wired in) |
| `graph-v1` | 5 | exact tomography of ≤ 20 qubits (existing; tomography-api-v2 is 1/5 the price) |

**Spending rules, enforced in code:** `send` needs `--approve-credits N` equal to the estimated cost; a POST is never retried (a retry could bill twice); the job id is written before polling and the raw response before parsing; the show's re-solve dialog and the studio refuse to spend unless started with `--allow-spend` *and* the confirm repeats the payload hash and credits it showed; a payload with a saved result costs nothing and sends nothing; IBM credentials are never sent by this client. Local checks block unknown parameters, over-size QDrive jobs and `mode: "qpu"`.

---

## Tests

```bash
for m in test_fast test_projector test_ui test_studio test_complementary test_pen_geom \
         test_frames test_payloads test_moth_client test_relief test_moth_engines test_domains; do
  python -m unittest tests.$m
done
python -m tests.check_port_matches_mock     # the src/ port still reproduces the original mock exactly
python superposed_relief_refsim.py          # the handoff's reference numbers
```

| Module | Covers |
|---|---|
| `test_domains` (38) | domain geometry (creases touch, per-domain budget, balanced vs frustrated graph), circuits (Qiskit = numpy, Aer sampling = exact with the lamp lock), the MPS tools against the statevector, exact vs MPS frames, certificates (exact = MPS, operator insertion = a real lamp qubit, monogamy), ground state, Moth payloads, baselines, the session |
| `test_relief` (36) | refsim table, Lambert = Born, conventions, Qiskit circuit = numpy state, **Aer sampling of the full circuit = exact distribution**, facets, couplings, visibility law, which-depth/visibility, controls, witnesses, no toggles in performance mode, honest provenance |
| `test_moth_engines` (12) | payloads valid and priced, QASM2 round-trip, parsers and scoring, the 201-target payload refused, echo tap mapping |
| `test_ui`, `test_studio` | key table, captions, session actions, demo, calibration, spend gate, the HTTP servers |
| `test_projector` | homography, warp, exact-black guarantee, photographed check (synthetic) |
| `test_frames`, `test_complementary`, `test_fast`, `test_payloads`, `test_moth_client`, `test_pen_geom` | the classical family and the API client, all offline |

Offline demo of the classical solve/compare flow without Moth: `python -m tests.make_selftest_fixtures` (writes SYNTHETIC results, stamped on every figure).

---

## Safety rules the code enforces

- **Glass is exactly 0.** Glass gets no qubit and no correlation; blur is mask-aware; the final frame is multiplied by the mask; and every frame is checked to be exactly 0 on the protected pixels *in projector pixels after the warp*. A frame that fails is refused and replaced with black.
- **A software check cannot see a projector that crept.** So the photographed check reports NOT CHECKED without an independent trace of the real glass, and never passes by default.
- **Nothing is called a Moth result unless a Moth job produced it.** Captions, the header chip and the demo scorecard follow the source's provenance.
- **No credits without an explicit, matching confirmation** (above).
- **The API key is never printed and `.env` is git-ignored.** Operator actions need a per-run token, a loopback client and a loopback `Host` header; `--lan` exposes only the read-only pages.

---

## Status and limits

**Done and tested:** the relief engine, its circuits and witnesses; the show, controls, demo and studio; the classical pipeline; the Moth client and engine builders; offline.

**Not verified (needs you or credits):**
- Any real Moth result for the relief design (either engine). Tomography, echo and QDrive result shapes are mostly "not documented yet", so parsers are defensive and raw results are saved first. The domain payloads (`domain_moth`) are built, validated locally and **not sent**; the qubit cap of `tomography-api-v2` is undocumented, which is why only patches of ≤ 20 qubits are offered.
- The domain relief on a projector: the frames are simulated and rendered, the glass-black checks pass, but the look on a real wall (six or so times as many facets, each with its own bevel strip) has not been judged by eye.
- Echo-driven dynamic relief is a module and a test, not a live mode.
- QDrive has never returned a circuit. Chaining uses an `input_files` asset id, and the docs list only PNG/JPEG as accepted assets; mixed Pauli words have a 2-qubit probe built but unsent.
- `local_echo` is *our reading* of the echo engine's description, not its definition.
- A real projector, a real window, real photos. The glass-check thresholds were reasoned, not calibrated.
- The studio's Solve and Perform steps are still the classical QDrive flow.

**What is and is not quantum** (an audit of the code, with the numbers):
- The facet register, the polarity superposition and the controlled operations between them are quantum content, simulated exactly or as a matrix-product state. The lamp and observation registers are uniform draws, equivalent to coin flips, except where the lamp is locked to a polarity qubit (`--lock`) or read in X (exact backend only).
- Per-panel engine: 72% of the variance of per-frame expected brightness is the lamp world (a classical draw); the polarity outcome (a near-fair coin) carries the rest. What the wall shows, the interference included, is reproduced by a classical sampler: a *two-branch sampler* draws exact frames of the uncoupled product relief at N = 5000 in 0.04 s, and the scaling law that keeps V constant (τ = κ/√N) keeps the state at ~0.25 excitations per panel, so excitation truncation (k = 2) is also exact to 0.99.
- Domain engine: capping the matrix-product bond dimension at 32 (what the show does) leaves a norm deficit of 1.5e-3 at 69 qubits (1.3e-2 at 8, 6e-3 at 16, 7.6e-4 at 64: the Schmidt tail is long, so an *exact* state needs more than 128), so the show's state is *approximately* classically simulable to a fraction of a percent and not cheaply to machine precision. The MPS benchmark shows where even the approximation stops: a lattice of domains in the relief regime passes bond dimension 128 at ~100 qubits, and in the graph-state regime (facets at the equator, CZ-like couplings) at ~75, with a 7% norm deficit. That is the regime where hardness is plausible, and it is **not** the regime the show runs in (it needs V → 0). No quantum advantage is claimed anywhere.

---

## Further reading

| File | Contents |
|---|---|
| [`standing-light-v3-domain-relief-handoff.md`](standing-light-v3-domain-relief-handoff.md) | the audit of v2 (what was and was not quantum, with numbers), the domain relief, the classical baselines, results, decisions for you (wins over v2 for polarity and entanglement design) |
| [`standing-light-v2-superposed-relief-handoff.md`](standing-light-v2-superposed-relief-handoff.md) | the v2 design: concepts, formulas, acceptance tests, scaling law (wins over the first handoff for the quantum design) |
| [`standing-light-runbook.md`](standing-light-runbook.md) | projection, calibration, performance mode, the glass check; §7 Superposed Relief with the findings that corrected the handoff; §8 Moth engines |
| [`standing-light-handoff.md`](standing-light-handoff.md), [`standing-light-project-spec.md`](standing-light-project-spec.md) | the original handoff and spec (classical family, targets, A/B protocol) |
| [`superposed_relief_refsim.py`](superposed_relief_refsim.py) | runnable numpy reference simulation |
| [`docs.mothquantum.com/docs/engines`](https://docs.mothquantum.com/docs/engines) | Moth engine documentation |
