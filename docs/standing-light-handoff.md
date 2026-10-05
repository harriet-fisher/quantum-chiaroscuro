# Standing Light — Project Handoff and Reference

> **Scene note.** This document describes the first build's **three-panel** bay window (left, centre, right). The built-in window, and the default of the code, is now **six panels**: each wing holds two windows stacked on top of each other with a shared rail (`left_top`, `left_bottom`, `centre_top`, `centre_bottom`, `right_top`, `right_bottom`; three planes). Counts and numbers below that say three panels, 15 or 19 qubits, or 3 polarity qubits are the three-panel ones; the README's *The default scene* has the current ones.

*Quantum chiaroscuro: a height-illusion projection instrument whose shading is drawn by quantum measurement, built on Moth Atlas.*

Author: Harriet Fisher · Written: 1 Oct 2026 · Brief context: Moth Quantum asked for something built on the Atlas platform (they specifically pointed to **QDrive** and **Tessa Image**; if a case can be made that an engine does not suit the project, that is acceptable). Ideas and thought process matter more than a polished prototype. About 5 days remained on 1 Oct.

**How to use this file:** put it in the project folder (or name it `CLAUDE.md`) and open Claude Code there. Section 1 is the quick card, §17 is the suggested first prompt, and §14 is the build order. Where this file conflicts with `standing-light-project-spec.md` (v2), **this file wins**: it adds the mock-up findings, the engine documentation read afterwards, and the narrowed quantum framing.

**Evidence tags**

- **[VERIFIED]** read directly from a Moth Atlas or docs page, or observed by running the mock.
- **[INFERRED]** my reasoning from verified facts; needs a test.
- **[UNKNOWN]** not found in the docs; check before relying on it.
- **[DESIGN]** a design choice or starting value of ours.
- **[SUGGESTION]** a tool or library recommendation from general knowledge, not checked in this project.

**Nothing has been run against Moth yet.** No credits spent, no API calls made. All Moth facts below come from reading pages.

---

## 1. Quick card

| Item | Value |
|---|---|
| Project | Standing Light, "quantum chiaroscuro" |
| One line | Geometry sets the allowed brightness ranges and the no-light zones; quantum measurement draws the shading inside them, so flat panels read as hollow, raised, or lit from different sides |
| First surface | A bay window: flat centre panel plus two angled side panels, glass in each |
| Hard rule | Glass, holes and anything the projector cannot see are pure black, get no qubits, and are excluded from blending |
| Three knobs | (1) lighting world, (2) relief polarity (hollow vs raised), (3) hardness (shots averaged) |
| Primary engine | `graph-v1` (documented schema, exact tomography, 5 credits) |
| Comparison engine | `qdrive-api-v1` (1 credit; returns the circuit; certainty weights; schema undocumented) |
| Optional engines | Tessa Image, Blur Core/Blur/Telablur, Retrocausal Echo, Quantum Echo, QRC |
| Mock status | Built and working; script and figures in `mock/` |
| Size at demo scale | 18 patch qubits + 2 lamp + 3 polarity = 23 qubits (cap unknown) |
| Python | venv + `pip install -r requirements.txt` |
| Secrets | `MOTH_API_KEY` environment variable; never in files or chat |

## 2. The concept

### 2.1 What it is, and what it is not

- It **is** a height-illusion shader: shading used as a depth cue, projected onto a real surface.
- It **is not** a physically based renderer. Brightness need not fall off with distance; regions only have to *differ* because they sit at different positions relative to the light.
- Colours/brightness are **not chosen by the user and not computed deterministically from normals**. Geometry sets bounds and rules; **quantum measurement picks values inside the bounds.**
- Reference look: the three pencil still-lifes (cone, block, sphere) shown lit from different sides and with different hardness. Same forms, different light and hardness. Standing Light does this on a real wall, with randomness.

### 2.2 Why random shading needs entanglement

If every patch picked its brightness independently you would see noise, not relief. A depth illusion needs the draw to be *coherent*: coplanar patches agree, patches across a crease contrast, and everything looks lit by one light. Correlations set by geometry give coherent-but-random shading, and correlations are what the Moth engines specify and sample.

### 2.3 The geometry decides three things

1. **Impenetrable regions** (see §2.4).
2. **Allowed brightness windows** [lo, hi] per plane and per depth.
3. **Relationships** between regions (same plane, crease, near, far) which become correlation targets.

### 2.4 Impenetrable regions (hard rule)

Glass panes, holes, hidden regions (not visible from the projector), and anything the user marks:

- Output is **pure black in every frame**.
- **No qubits**, no edges in any `coupling_map`, no targets: they are not calculated on.
- **Excluded from blending**: blur is mask-aware (normalised masked blur) so black does not bleed into panels and brightness does not bleed into glass.
- Their edges count as borders in the distance-to-border relief field.
- A final multiply by the mask guarantees black.
- Excluded cells free up qubit budget.
- Practical note: a projector cannot project true black; "black" means no light, so the wall shows at ambient level plus the projector's black level. Dim the room.

### 2.5 The three knobs

1. **Lighting world.** Two lamp qubits connected to every patch. L1 = left/right light, L2 = frontal vs grazing. Measuring the lamps selects the light direction; patches follow with geometry-set probabilities. Panels on the same plane share the same dependence, so they light up together; across a crease the dependence flips sign, so one side lights and the other darkens.
2. **Relief polarity.** One qubit per panel decides whether its trim reads sunk (centre dark, edges bright) or raised. Geometry correlations between polarity qubits decide whether neighbouring panels agree. Uses the hollow/bump ambiguity of shading: the same light reads as hollow or raised depending on the sign of the height field.
3. **Hardness.** One shot per frame gives a hard, blotchy shading; averaging K shots gives a smooth gradient.

### 2.6 What the mock found (all [VERIFIED] by rendering)

- **Light direction** reads clearly: left vs right lighting flips which side panel is lit; frontal vs grazing changes the centre panel and the bevel strength.
- **Hollow vs raised** only read once a **directional bevel** was added (height field h = polarity × smoothed distance-to-border; shade = n·l with n = (−hx, −hy, 1)). The first pass without bevel was too weak. With the bevel, the trim visibly flips between recessed and protruding under the same light.
- **Hardness** is the weakest knob visually. K=1 is blotchier than K=128, but the difference is modest. A contrast or posterise curve would help.
- Glass/outside pixels were exactly black in all 25 rendered frames.
- The classical stand-in sampler is an Ising-type Gibbs sampler over patch spins with lamp spins clamped per frame (lamps sampled uniformly), plus a tiny exact sampler for the three polarity spins.

Mock parameters that produced the working look [DESIGN]:

| Parameter | Value |
|---|---|
| Canvas | 1200 × 700 |
| Patch grid | 6 × 4 cells, included if ≥ 30% of the cell is projectable; gives 18 patches |
| Glass inset | 0.45 of each panel, scaled about its centroid |
| Windows [lo, hi] | centre (0.40, 0.95); side near (0.25, 0.80); side far (0.10, 0.65) |
| Side panel normals | ±sin(35°) in x, cos(35°) in z (concave view from inside the bay) |
| Hub couplings | kh = 1.6 (L1 with normal-x), kf = 1.3 (L2 with normal-z minus mean) |
| Patch couplings | coplanar +0.9, crease −0.6; polarity chain +0.9 |
| Weights | light 0.75, relief 0.20, bevel 0.70; bevel gain 40; grazing doubles bevel strength |
| Interpolation | Gaussian σ = 0.18 × cell width per panel (normalised, no leaking across creases) |
| Final blend | mask-aware Gaussian σ = 5 |

## 3. How this is quantum, honestly

Read this before pitching. It decides what you can and cannot claim.

**3.1 What is not quantum-specific.** Any pattern that depends only on Z-basis bit correlations can be reproduced by a classical correlated sampler (the mock proves the look is achievable classically). So, as currently designed:

- **Lighting world** (lamp qubits measured in Z) is a classical random choice that patches follow. Least quantum-specific.
- **Hardness** is sampling statistics. On hardware the noise is physically real, but the statistics are classical.
- At ≈ 20–23 qubits everything here is classically simulable. Moth's own pages say Blur, Telablur and the Shader run on classical simulators, and `graph-v1`/QDrive emulation runs on Aer. **Do not claim quantum advantage.**

**3.2 Where it genuinely can be quantum-specific.**

1. **Complementary observables (the strongest case).** Make *polarity* an observable that does not commute with *lighting*: lit/dark measured in one basis (Z), raised/hollow in another (X). A patch with a definite light state then has a completely random relief polarity, and vice versa; the room cannot decide both at once. This mirrors the Bloch-sphere colour logic (a pure black/white pole has no hue). With correlations across creases in both bases, no classical assignment of predetermined values reproduces the joint statistics (the structure Bell-type tests probe). **Requires basis-rotated measurement**, which `graph-v1`'s documented sampling does not offer (it returns Z-basis bitstrings); locally simulating a QDrive-returned circuit with added rotations could. [INFERRED]
2. **Entanglement as the specification of structure.** The correlation targets (coplanar agree, crease disagree, lamp hub) are the program. This is the Motte-model idea in Moth's own write-up: specify expectation values, let the machine find the circuit.
3. **Hardware randomness.** On a QPU the draw is physical rather than pseudo-random. Not scoped or priced.
4. **Scaling story.** One qubit per patch plus a few lamp qubits, with local targets from geometry. Real geometry creates frustrated loops (no single lighting is consistent everywhere); sampling such correlated distributions is the kind of problem expected to become classically hard at scale. [INFERRED; verify before stating publicly. I recall the quantum marginal problem being QMA-complete but have not verified it.]
5. **Moth's framing.** Labyrinth's own FAQ treats walls as an emergent readout of a many-body state and noise as part of the artefact. Standing Light can use the same stance.

**3.3 Recommended narrowing.** Hero = lighting world (visible, easy to show) + polarity as the complementarity centrepiece (the quantum-specific part). Present hardness as a free control, not a quantum claim.

**3.4 What the mock is and is not.** The mock's sampler is classical. It tests the *look*. It cannot imitate the complementary-observable version faithfully.

## 4. Moth Atlas: platform and API

### 4.1 Platform

- Atlas: `https://platform.mothquantum.com`. Labelled **ALPHA**, "optimized for larger screens". Sidebar: Dashboard, Engines, Jobs, Assets, API keys, API docs, Community. [VERIFIED]
- Engine showcase pages: `/engines/showcase/<slug>` for labyrinth, tessa-image, blur, entanglement-shader, retrocausal-echo, telablur, coin-toss. QDrive has **no showcase page** (the link returned "not found"). [VERIFIED]
- Docs: `https://docs.mothquantum.com/docs/engines/<engine-id>`; intro at `/docs/intro`; "Submitting jobs" at `/docs/submitting-jobs` (not read); interactive API reference at `https://api.mothquantum.com/docs`. [VERIFIED]

### 4.2 API flow [VERIFIED unless noted]

- Base URL: `https://api.mothquantum.com/api/v1`
- Auth: `Authorization: Bearer <key>` from Atlas > API keys. Docs use `MOTH_API_KEY` (showcase curl samples use `$MOTH_TOKEN`); use `MOTH_API_KEY`.
- Submit: `POST /engines/<engine-id>/process` with JSON `{"params": {...}}` → returns `job_id`.
- Poll: `GET /jobs/<job_id>/status` every ~2 s until `completed`, `failed` or `cancelled` (error text under `error`).
- Result: `GET /jobs/<job_id>/result` → `outputs` list, each with `slot` and `url`; download each URL.
- File inputs: upload as **Assets** first (presigned URLs), pass the asset id under the engine's slot name. [VERIFIED from intro/QDrive pages]
- Chaining: later jobs can reference an earlier job's named output as `job:<id>/<name>` (documented on the QRC engines, e.g. `job:<id>/model`). Whether QDrive accepts `initial_circuit: job:<id>/circuit` is **[UNKNOWN]**.
- Rate limits: referenced but not specified. [UNKNOWN]

Skeleton (adapted from the QDrive example page):

```python
import os, time, requests
API = "https://api.mothquantum.com/api/v1"
H = {"Authorization": f"Bearer {os.environ['MOTH_API_KEY']}"}

def run(engine_id, params):
    job = requests.post(f"{API}/engines/{engine_id}/process", headers=H, json={"params": params}).json()
    while True:
        st = requests.get(f"{API}/jobs/{job['job_id']}/status", headers=H).json()
        if st["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(2)
    if st["status"] != "completed":
        raise RuntimeError(f"job {st['status']}: {st.get('error')}")
    return requests.get(f"{API}/jobs/{job['job_id']}/result", headers=H).json()
```

(`graph-v1` is documented as JSON in; the request-body shape for it is shown in §5.1. Confirm the exact envelope with the dashboard's "Try the API call" before relying on it.)

## 5. Engines

### 5.1 `graph-v1` — Quantum Graph Engine (PRIMARY)

5 credits/run, JSON→JSON, updated Sep 9 2026. [VERIFIED] Prepares a quantum graph state from per-qubit Bloch targets and per-edge Pauli correlations, returns exact tomography, and samples it.

Request (documented example):

```json
{
  "num_qubits": 4,
  "coupling_map": [[0,1],[1,2],[2,3]],
  "operations": [
    {"type": "bloch", "qubit": 0, "paulis": {"X": 1.0}},
    {"type": "relationship", "qubits": [0,1], "paulis": {"ZZ": 1.0}}
  ],
  "shots": 1024,
  "mode": "emu"
}
```

- `coupling_map`: optional; default fully connected. **Every `relationship` operation's pair must be an edge.** So lamp-hub edges must be listed.
- `operations`: applied in order; each has `update` (default true) that refreshes the tracked state so later ops account for earlier ones; false = faster, blind.
- Omit `operations` for a random state (random Bloch vectors, random ±1 ZZ per edge); `seed` makes it repeatable.
- `mode`: `"emu"` (local noiseless simulator, seconds) or `"qpu"` (IBM hardware, queue minutes–hours, needs `qpu_token` and `qpu_instance`). Stay on `emu` for now.
- Other params: `num_qubits`, `shots`, `backend_name`.

Returns: `output.tomography` (exact prepared state: `bloch` per-qubit X/Y/Z and `relationships` per-edge values keyed `"a,b"`, computed classically so noise-free even on QPU), `output.measurements` (**top 20 bitstrings** with count/probability, `dominant_bitstring`, `edge_agreement_score`), and the resolved `coupling_map`. Bitstrings are **qubit-0-leftmost**.

Use here: primary solver for the first demo; exact readout of per-patch Bloch vectors and correlators in one call. **Limitation:** only the top 20 outcomes, so it cannot supply a varied frame stream; use the exact state to drive a local sampler, or use QDrive for the circuit. Qubit cap: [UNKNOWN].

### 5.2 `qdrive-api-v1` — QDrive (COMPARISON, and best route for many frames)

1 credit/run, Text→Text, updated Sep 1 2026, "Build a quantum circuit by specifying target expectation values instead of gates." [VERIFIED]

Params: `machine` (default `aer`), `n_qubits` (required unless `initial_circuit` given), `coupling_map`, `targets` (ordered list of `QDrive.target()` calls; an empty/null entry calls `update()`; an expvals value can be a number, `[value, certainty]`, or a `measured[...]`-style expression `[[word, qubits, coefficient], ...]`, optionally `[terms, certainty]`), `update_method` (`spectral` default / `direct` / `constrained`), `ansatz` (QASM3 circuits keyed by qubit count), `sample` (also run a sampler), `shots` (default 1024), `tomography` (0/1/2), `seed`. One optional file slot: `initial_circuit` (text/plain). **Output slot: `circuit`** (QASM). "No further details provided by the engine author."

[UNKNOWN]: the exact JSON shape of each `targets` entry (use the dashboard's "Try the API call"), qubit cap, behaviour of the three `update_method`s, output format of the tomography/sampler, whether job-reference chaining works.

Why it matters: certainty values map to hard vs soft constraints (pane pins vs lighting hints); weighted Pauli sums could express a Lambert-like term; the returned circuit can be sampled locally for unlimited frames (loading QASM3 via Qiskit is [INFERRED], untested) and rotated into other bases for the complementary-polarity design. Moth specifically pointed to QDrive, so keep it in the story.

### 5.3 Tessa Image — `tessa-image-v1` (OPTIONAL)

[VERIFIED] Encodes an image onto a quantum device colour by colour (pipeline encode → transform → decode). Colour sphere: 50% grey at the centre, white pole top, black pole bottom, vivid colours on the equator, hue as angle; a pixel is a point on a sphere, same shape as a qubit state. Params: `machine` (aer, or emulated IBM fez/marrakesh/torino/brisbane/kyiv/sherbrooke/kyoto/osaka/quebec/cusco/strasbourg/brussels), `fixed_palette`, `separate_rgb`, `range_correction`, `shots` (4096), `distortion` + `distortion gate` (XY/YX/XX/YY/ZZ/XZ/ZX/YZ/ZY), `downscale`. The **distortion gate rotates adjacent cells and is hardware-only** (greyed out in emulation). Emulator lattice about 64×64; QPU budget about 448 data qubits (≈ 20×20 image). 10 MB image max. Showcase says emulation (noiseless Aer or emulated IBM chip) is free with no credits.

Fit: the colour sphere = Bloch sphere gives the colour convention; optional seed-image mode (image → initial Bloch targets, then geometry imposed); optional hardware-noise texture layer. Limits: fixed lattice adjacency (ignores seams/depth jumps), one global distortion value, distortion only on a QPU, small resolution. **Cut first if time is short; make the case in a slide** (why it did not fit) if not used. Whether its encodings/circuits can be extracted or reused in another job: [UNKNOWN] (ask Moth).

### 5.4 Other engines

| Engine | ID | Credits | Summary |
|---|---|---|---|
| Labyrinth | `labyrinth-v1` | n/a | One qubit per room; `coupling_map` = open room pairs (grid); `level_data` {grid_size, num_qubits, coupling_map, initial_states per room {X,Y,Z,radiating}}, `shots`, `mode` emu/qpu, `steps`. Emulator capped at 20 qubits (4×5); QPU for larger. Outputs per-room Bloch vectors, sampled maze, metrics `sz_tomo` vs `sz_samp`. Walls = ⟨ZZ⟩ = −1. Good baseline/inspiration. |
| Quantum Blur Core | `blur-core-v1` | 1 | Interference blur on any N-D rectangular grid of non-negative numbers (not image files). Params `values`, `strength` (number or per-axis list), `style` ("x"), `reach`, `axes`, `shots`, `max_qubits` (default 20; 1024×1024 = 20). Output: nested list, rescaled to input max. Not mask-aware. |
| Blur / Telablur | `blur-v1`, `telablur-v1` | n/a | Image blur / two-image morph with optional mask (white = effect applies). Simulator only. |
| Entanglement Shader | (page: entanglement-shader) | n/a | Baked reflectance/transmittance tables indexed by incidence angle; ZIP of OSL, Marmoset .frag, HLSL, MaterialX, R_lut/T_lut.hdr. Simulator only. Weakest quantum claim (baked classical lookup at render time). |
| Qpixl | `qpixl-v1` | 1 | Encodes an array of floats onto a quantum device with Interwoven QPIXL; machines aer / fake_<chip> / qpu (`backend_name` e.g. `ibm_fez`); shots default 4096. |
| Quantum Echo | `otoc-echo-v1` | 1 | Core echo engine (v0.1.2) the audio echo engines are built on; no parameter details on the page. |
| Retrocausal Echo | `retrocausal-echo-v1` | n/a | Audio: qubit chain/square lattice, forward scramble + impulse + reverse; taps with level/pan/time/polarity; emulator or QPU; can re-render from a saved tap map. |
| QRC Audio / MIDI / Generate | `qrc-audio-v1`, `qrc-midi-v1`, `qrc-gen-v2` | 5 / 5 / 1 | Quantum-reservoir sequencing of audio chunks / MIDI notes / generated sequences. Optional sound layer only. |
| Coin Toss | `coin-toss-v1` | n/a | One measured qubit; the docs' getting-started example. |

Also listed in the docs sidebar, not read: Quantum Labyrinth Engine page, Blur Jazz (`blur-midi-v1`), QRC Train (`qrc-train-v1`). Credit costs marked n/a were not shown on the pages read.

## 6. Recommended setup

### 6.1 Environment

```
cd Moth_Application/mock          # or a new project folder outside iCloud
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy, scipy, Pillow, matplotlib, requests
export MOTH_API_KEY="..."         # never commit
```

Notes: avoid creating the venv inside iCloud Drive if it syncs slowly; keep it e.g. under `~/venvs/standing-light`. In VS Code select that interpreter (Cmd+Shift+P → Python: Select Interpreter) or Pylance will report missing imports. The mock opens **no window**; it writes PNGs and takes roughly 30–60 s.

### 6.2 Solver strategy

- Build `solve(graph, targets) -> state` with two backends: `graph_v1` and `qdrive`.
- **Calibrate targets from the mock** so the quantum run reproduces the working look: estimate ⟨Z_p⟩, ⟨Z_pZ_q⟩, ⟨Z_L Z_p⟩ from the mock's Gibbs sampler, pass those as `bloch` / `relationship` operations, then compare achieved vs requested. Z-diagonal statistics are realisable as a pure state (√p amplitudes) but the engine builds states from pairwise targets, so expect a gap. Log it. [INFERRED]
- Run both engines on identical targets once (about 6 credits) and show: achieved vs requested, credits, what QDrive's certainty weights change. That comparison is the case to Moth for or against QDrive.
- Frame stream: feed the exact state (or QDrive's circuit) into a **local sampler** for many frames; cache results by a hash of the targets.

### 6.3 Credits discipline

graph-v1 = 5, QDrive = 1. Check the balance first. One engine call per scene state; cache; sample locally for animation; never one call per frame.

### 6.4 Suggested repo layout

```
standing-light/
  CLAUDE.md                 # this file
  requirements.txt
  mock/standing_light_mock.py   # working classical look test (keep as reference)
  src/
    capture/      load_mesh.py, pen_tool.py, import_roomplan.py
    geometry/     planes.py, creases.py, depth.py, relief.py, visibility.py
    mask/         build_mask.py, masked_blur.py, apply_black.py
    graph/        patches.py, edges.py, lamp_hub.py, allocate_qubits.py
    bounds/       windows.py
    targets/      from_geometry.py, calibrate_from_mock.py, payloads.py, consistency.py
    quantum/      moth_client.py, solver.py, sampler_local.py, sampler_mock.py
    texture/      compose.py, interpolate.py, bevel.py
    projector/    calibrate.py, warp.py, output_window.py
    ui/           operator_panel.py, keys.py
    store/        run_card.py, cache.py
  runs/           # run cards, cached states, frames
```

## 7. Tools for uploading geometry and identifying patches

All items in this section are **[SUGGESTION]** unless stated; none were tested here.

### 7.1 Two capture routes

**Route A: projector-space prototype (recommended for the bay window and for speed).** Because a single projector only ever shows a 2D image, work in the projector's/camera's view. Take a photo from the projector's position (or a camera beside it), draw polygons over it, and enter each panel's angle by hand. No 3D scan is needed for this demo. The mock already works this way.

- Pen tool (draw polygons on the photo, assign labels `panel` / `glass` / `frame`, enter a plane id and angle): Matplotlib `PolygonSelector`, OpenCV mouse callbacks, or a small local web canvas. The Matplotlib route fits the existing dependencies.
- Normals per panel from user-entered angle, or estimated from vanishing lines; centre panel facing the viewer.

**Route B: LiDAR/scan mesh (for the mound and real 3D).**

- Phone capture apps for iPhone Pro: Polycam, Scaniverse, 3d Scanner App; Apple RoomPlan-based apps give parametric rooms with labelled surfaces such as windows. Export OBJ / PLY / GLB / USDZ.
- Glass is often a **hole or missing surface in a scan**, which conveniently marks it impenetrable; windows are labelled by RoomPlan-style scanners. [INFERRED]

### 7.2 Processing libraries

| Task | Libraries |
|---|---|
| Load and clean meshes | `trimesh`, `open3d`, `pymeshlab` |
| Plane detection | `open3d` `segment_plane` (RANSAC) and region growing; or k-means on face normals with `scikit-learn` |
| Polygons / masks | `shapely`, Pillow `ImageDraw` (used in the mock), OpenCV |
| Graph building | `networkx` |
| UV unwrap (only if you need UVs) | `xatlas`, Blender |
| Geodesic clustering for the mound | `potpourri3d` or `gdist` |
| Ray casting for visibility from the projector | `trimesh.ray` (embree if available), or render a depth map from the projector pose with `pyrender`/`moderngl` |
| Projector calibration | OpenCV `findHomography` (4 corners / checkerboard / ArUco markers); structured light later if needed |

### 7.3 Patch identification (the process)

1. Label regions: `panel`, `glass`, `frame`, `off-limits`; plane ids per panel.
2. Impenetrable mask = glass ∪ off-limits ∪ outside ∪ not visible from the projector.
3. Grid the projector image into cells (prototype: 6×4); a cell becomes a patch only if ≥ 30% projectable; its plane = the majority plane of its projectable pixels.
4. Edges: 4-neighbour cells, +coupling if same plane, −coupling if different planes (crease).
5. Hub: add 2 lamp qubits connected to every patch; 1 polarity qubit per panel, chained to neighbour panels.
6. Mound extension: patches from mesh clustering (geodesic Voronoi / k-means on face centroids + normals), edge coupling as a continuous function of normal change or slope, hidden areas auto-impenetrable by ray casting.
7. Print and plot the qubit budget; refuse to run if it exceeds the engine cap.

## 8. Pipeline

```
photo / scan
  → region labels + plane ids + angles
  → impenetrable mask (glass ∪ hidden ∪ outside)
  → bounds per plane/depth + relief field (distance to border)
  → patch grid → qubit allocation (impenetrable cells get none) + lamp hub + polarity qubits
  → target table (Bloch + relationships, calibrated from the mock sampler)
  → solver (graph-v1 / QDrive) → exact state (+ circuit)
  → local sampler: draw a lighting world, polarity, K shots per frame
  → per-patch lit fraction → per-panel normalised interpolation
  → relief + directional bevel from polarity and light
  → map into [lo, hi] → mask-aware blur → multiply by impenetrable mask
  → projector warp → output window on the projector display
```

Stage-by-stage contracts:

| Stage | Input | Output | Key check |
|---|---|---|---|
| Label | photo/mesh | `labels.json` | glass polygons present |
| Mask | labels | `mask.png` | impenetrable = black everywhere downstream |
| Patches | mask, labels | `graph.json` | qubit count ≤ cap |
| Targets | graph, geometry | `targets.json` | consistency check, Bloch-length ≤ 1 |
| Solve | targets | `state.json` (+ `circuit.qasm`) | requested vs achieved table |
| Sample | state/circuit | frame configs | seed recorded |
| Compose | frame configs | frames | glass pixels exactly 0 |
| Project | frames | projector window | warp aligned |

## 9. The processes in detail

### 9.1 Lamp-hub correlation design [DESIGN, untested]

- Lamp L1 (horizontal): coupling to patch p ∝ horizontal normal component n_x. Lamp L2 (frontal vs grazing): ∝ (n_z − mean n_z). A flat front-facing centre panel depends little on L1 and mostly on L2, so it differs from the sides.
- Panel-to-panel: coplanar positive, crease negative (the same sign pattern keeps the target set mutually consistent, GHZ-like; with magnitudes < 1 consistency is approximate).
- Keep correlation magnitudes below 1 to leave randomness within bounds.

### 9.2 Frame generation

- Per frame: sample the lamps (uniform) → sample K patch configurations with lamps conditioned (post-select) → lit fraction per patch → polarity bits (once per frame) → compose.
- Hardness = K. Hold K per frame fixed or let it drift as a performance control.

### 9.3 Brightness composition (as in the mock)

For each pixel in a panel: `u = (w_light·S + w_relief·profile + w_dir·bevel) / (w_light + w_relief + w_dir)`, `value = lo(t) + (hi(t) − lo(t))·u`, where S is the interpolated lit fraction, `profile` is (1 − R) for sunk or R for raised (R = normalised distance-to-border), `bevel = clip(0.5 + gain·strength·(−hx·lx − hy·ly))` with h = polarity × smoothed R. Then the mask-aware blur and the impenetrable multiply.

### 9.4 Complementary-polarity variant (the quantum-specific one) [DESIGN, untested]

Take the circuit from QDrive, append basis-rotation gates (e.g. H) to the polarity qubits before measurement, and simulate locally; measure patches in Z. Keep per-panel polarity qubits entangled with the lamp/patch structure so that raised/hollow and lit/dark cannot both be definite. Compare frames against the Z-only version.

### 9.5 Run record

Each run writes a run card: parameters, seed, engine, backend, shots, credits spent, timestamp, requested-vs-achieved table, qubit allocation, git hash.

## 10. User experience

### 10.1 Setup flow (operator, laptop screen)

1. Photograph the surface from the projector's position.
2. Draw panel polygons; mark glass; set each plane's angle.
3. Preview patch graph overlaid on the photo; adjust grid; read the qubit count.
4. Set ranges (centre/side windows), knob defaults.
5. Calibrate the projector (four corners or markers) and check the glass is black.
6. Press run: engine call(s) (one at a time, with credit estimate shown first).

### 10.2 Performance mode

- Projector shows the full-screen output on the projector display; the laptop shows an operator panel.
- Auto-cycle frames every few seconds, or step manually.
- Controls (keyboard or sliders): lighting world (auto / hold), polarity (auto / hold), hardness K, blend amount, "re-sample", "re-solve" (costs credits; asks first).
- Overlay toggle (laptop only): patch graph with correlation-coloured edges, current lamp bits, achieved-vs-requested gaps.
- Optional sound: Retrocausal Echo or QRC audio keyed to light changes.

### 10.3 What the audience sees

A bay window that re-lights itself every few seconds: left-lit, right-lit, frontal; panels sinking into the wall or standing out; sometimes crisp and posterised, sometimes soft. Glass stays black. A caption or companion screen explains that each look is a measurement, and that geometry made the draw coherent.

### 10.4 Five-minute demo script for Moth

1. The problem: random shading that still reads as one light.
2. Show the geometry labels and the black glass rule.
3. Show the qubit graph (patches + lamps + polarity) and the targets.
4. Run: show the achieved-vs-requested table.
5. Live frames: three knobs in turn.
6. QDrive vs graph-v1 comparison slide.
7. Honest slide: what is quantum-specific and what is not; scaling story.

## 11. Limitations and risks

1. **Qubit caps** for `graph-v1` and QDrive on the emulator are unknown. The demo needs 23; Labyrinth's emulator is capped at 20. If capped at 20, reduce the patch grid or drop to 2 polarity qubits (merge centre with a side).
2. **QDrive target syntax** not documented on the page; use the dashboard's "Try the API call".
3. **`graph-v1` returns only the top 20 outcomes**; use QDrive's circuit or the exact state for many frames.
4. **Pinned or extreme targets** may be relaxed silently; always log requested vs achieved.
5. **Local QASM3 loading** with Qiskit is untested.
6. **Basis-rotated sampling** (needed for the complementary variant) is not exposed by `graph-v1`.
7. **No quantum advantage** at this scale; present structure and scaling instead.
8. **Hardness knob** is visually weakest; add a contrast curve if kept.
9. **Illusion is viewpoint- and room-dependent**; ambient light and the projector's black level limit contrast; "black" glass still reflects ambient light.
10. **Projecting onto glass** (if panels contain real glass) behaves badly; the rule that glass is black avoids it, and the projector's light should be masked off those pixels.
11. **Calibration drift** if the projector moves; bake in a quick re-calibration.
12. **Atlas is ALPHA**: expect rough edges and possible API changes.
13. **Mock look ≠ final look**: the Gibbs sampler is classical; the quantum-calibrated draws may differ.
14. **iCloud folders** can slow venv and image I/O.
15. **Time**: about 5 days; prioritise (see §14).

## 12. Evaluation

- Glass/outside pixels exactly black in 100% of frames; no bleed after blending.
- Coherence: compare against independent per-patch noise and the classical correlated sampler; targeted draws should read as one light.
- Requested vs achieved correlation table and heatmap.
- Variety: distribution of lighting worlds and polarities over many frames.
- Illusion: photographed projection on the real bay window; quick viewer test ("does the centre look sunk or raised?").
- Credits per session.

## 13. Extensions

- **Mound / complex surface**: mesh-based patches; continuous edge couplings from normals/slope; auto-impenetrable hidden regions via ray casting; single projector in projector space; hierarchical or tiled solving if patches exceed the qubit budget.
- **Colour**: Tessa's sphere convention; Z measurement gives lit/dark, X/Y measurements give hue quadrant; needs basis-rotated sampling.
- **Sound**: Retrocausal Echo, QRC audio/MIDI.
- **Hardware**: a QPU run via `graph-v1` `mode: "qpu"` (needs `qpu_token` and `qpu_instance`) or Tessa's distortion gate; unscoped and unpriced.

## 14. Build order (suggested, ~5 days)

| Day | Goal | Done when |
|---|---|---|
| 1 | Reuse `mock/standing_light_mock.py` as the reference; build the pen tool and label format; qubit-budget printout | A photo with polygons produces `labels.json`, `mask.png`, and a patch preview |
| 2 | `moth_client.py`; first `graph-v1` emulator run with lamp hub (ask first; 5 credits); calibrate targets from the mock | `state.json` and requested-vs-achieved table |
| 3 | QDrive run on the same targets (1 credit); local circuit sampling; frame stream through the existing compose code | A/B figures and a frame stream |
| 4 | Projector calibration and output window; real bay window; masked black glass verified | Projection test photo |
| 5 | Polarity-complementary experiment (if time); polish; run card; demo script | Demo ready |
| Spare | Sound; mound; Tessa case slide | |

## 15. Decisions made and open

**Made:** concept = quantum chiaroscuro / height illusion; geometry sets bounds and impenetrable rules; `graph-v1` first, QDrive compared; Tessa optional; patches in projector space for the prototype; qubits only for projectable patches.

**Open:** which knob is the hero (recommendation: lighting world + polarity complementarity); greyscale first or colour; first real surface and projector availability; whether to include sound; whether to ask Moth (a) if Tessa's encodings/circuits can be extracted or reused, (b) whether QDrive accepts pure-state pins and chained circuits, and (c) the qubit caps.

## 16. File inventory

In the Moth_Application folder:

- `standing-light-project-spec.md` — v2 spec (earlier; this handoff supersedes where they differ)
- `standing-light-handoff.md` — this file
- `mock/standing_light_mock.py` — working classical look test (run `python3 standing_light_mock.py 6 4`; no window; writes 5 PNGs in ~30–60 s)
- `mock/fig0_geometry.png` … `fig4_random_sheet.png` — mock outputs
- `mock/requirements.txt` — Python dependencies

Also saved in the Claude Project memory: `topics/moth-engine-docs.md` (engine reference notes).

## 17. First prompt for Claude Code

> Read `CLAUDE.md` (this file) and `mock/standing_light_mock.py`. Do not call any Moth API yet. Step 1: create the repo layout in §6.4 and port the mock into `src/` modules without changing its output (compare images). Step 2: build a pen-tool script that takes a photo, lets me draw panel and glass polygons, assigns plane ids and angles, and saves `labels.json` and `mask.png`; print the patch count and the qubit budget (patches + 2 lamps + one per panel) and warn if it exceeds 20. Step 3: write `moth_client.py` from §4.2 with the key read from `MOTH_API_KEY`, and a `calibrate_from_mock.py` that estimates ⟨Z⟩ and ⟨ZZ⟩ targets from the mock sampler. Stop and show me the payload and the estimated credit cost before sending anything to Moth.
