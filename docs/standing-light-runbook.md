# Standing Light: runbook for projection, performance and the complementary experiment

Written 1 Oct 2026. Companion to `standing-light-project-spec.md` (v3) and `standing-light-handoff.md`; where they differ, the handoff wins. This file covers handoff §14 days 4 to 5 and §10 (performance mode). **Nothing in it has been run on a real projector or a real bay window yet**: every number below that comes from a camera or a wall is a *synthetic* test of the tools, and each section says what is and is not verified.

> **Scene note (the default window).** This runbook was written for the first build's three-panel bay window. The built-in window, and the default of every command below, is now **six panels**: three wall sections (left wing, centre, right wing) each holding two windows stacked on top of each other with a shared rail (`left_top`, `left_bottom`, `centre_top`, `centre_bottom`, `right_top`, `right_bottom`; three planes; a pane of glass in every window). Where a number below says "three panels", "15" or "19 qubits" it describes the first layout; the README's *The default scene* has the current ones (classical grid 4x4 = 20 qubits; per-panel relief 12 facet + 6 polarity = 18 simulated, 22 in the circuit; domain relief 64 + 64 = 128 qubits).

## 1. What was built

| Area | Files | Verified how |
|---|---|---|
| Projector output and calibration | `src/projector/{calibrate,warp,patterns,output_window,verify}.py`, `src/projector/output.html` | unit tests with synthetic cameras; the output page was run in a browser and reported its real pixel size |
| Performance mode and UX | `src/show.py`, `src/ui/{session,operator_panel,keys,caption,overlay,demo}.py`, `src/ui/web/*.html`, `src/texture/fast.py` | 27 UI tests, an HTTP smoke test, and the three pages driven in a browser |
| Polarity-complementary experiment | `src/quantum/complementary.py`, `qdrive_complementary_payload` in `src/targets/payloads.py`, `StateSource` Bloch-axis bases in `sampler_local.py` | witness maths checked against textbook states; the experiment itself is **UNTESTED against any Moth circuit** |

No new Python dependency: calibration is a plain-numpy homography (no OpenCV), the windows are web pages served by the standard library.

## 2a. One interface for the whole pipeline: the studio

```
python -m src.studio                           # project folder runs/studio (use --project runs/myroom for another scene)
python -m src.studio --allow-spend             # also lets step 4 submit to Moth, after asking
```
One local page walks the steps you used to run by hand, each one running the same module with the paths filled in:

| Step | Runs | Writes (under the project folder) |
|---|---|---|
| 1 Draw | `src.capture.pen_tool draw` (opens in your browser; press Save in it) | `labels.json`, `mask.png` |
| 2 Targets | `src.targets.calibrate_from_mock` (optional grid and sweeps) | `calib/` (targets and both payloads) |
| 3 Rehearse | `src.show --source relief` (Superposed Relief, the quantum circuit simulated exactly here; no Moth call, no credits) | `calibration.json` if you align |
| 4 Solve | an engine picked in the dropdown: **Relief on Aer** (default; `src.quantum.aer_relief`: the whole relief circuit executed on Aer, 1M shots, checked against the reference state; free), **Verify on Moth** (`src.quantum.verify_moth`: tomography-api-v2 on the reference circuit, 1 credit), **QDrive facet register** (`src.quantum.facet_qdrive`: chained jobs, one confirm and one credit each; the Aer run can then execute the relief on its circuit), **Echo dynamic relief** (`src.quantum.echo_relief`: one Aer circuit per echo depth, taps from the local echo (free) or Moth's otoc-echo-v1 (1 credit)), or the older `src.quantum.solver qdrive` / `graph-v1` | `solve/aer/`, `solve/tomography/`, `solve/qdrive_facets/`, `solve/echo/`, `solve/` |
| 5 Perform | `src.show --source relief --aer-run solve/aer` (every look is one measured shot), `--echo-run solve/echo` (each look steps to the next echo depth), or `src.show --source circuit` on the circuit QDrive returned (the older classical-family flow) | |

The **Next** button does the right thing for the first step that is not done. A step is blocked until the one it needs is current: change the drawing and steps 2 to 5 say so (the studio remembers which drawing the targets came from, in `studio.json`). Stop ends a tool; Ctrl+C or closing the terminal stops everything the studio started. "Use the built-in demo scene" fills step 1 with the six-pane bay window so the rest can be tried before anything is drawn.

**Spending is unchanged and stricter, not looser.** Step 4 shows the payload hash and cost (a payload with a saved result is free and sends nothing) and the send button stays disabled unless the studio was started with `--allow-spend`; the confirm must repeat the hash and credits it showed. The show it opens is never given `--allow-spend`. Jobs that were submitted but have no saved result are listed in the plan, so a failed paid run is not forgotten. Tested with fake child processes (`tests/test_studio.py`) and once for real on the demo scene up to the refused spend; **no credits were sent through the studio**, and QDrive's behaviour on a real send is unchanged and still unverified.

The modules still run on their own with the same flags; the studio only supplies them.

## 2. Quick start (rehearsal, no projector needed)

```
source .venv/bin/activate             # or: .venv/bin/python -m ...   (run from the project folder)
python -m src.show                    # the built-in six-pane bay window, Superposed Relief (section 7)
python -m src.show --source oracle    # the classical rehearsal source instead
```
It prints three URLs: the **operator panel** (`/`, this laptop only), the **projector window** (`/output`) and the **audience screen** (`/audience`). The header chip says `local reference circuit` for the default relief source and `classical rehearsal (no quantum)` for the classical ones; the audience screen says so too, in words.

Sources (`--source`): `relief` (default: Superposed Relief, section 7), `oracle` (the mock's distribution as a state, classical), `mock` (Gibbs sampler), `complementary` (the experiment's stand-in state, UNTESTED), `circuit` (a QASM3 file, e.g. a QDrive circuit, simulated locally). Nothing is called a Moth result unless the circuit came from a Moth job.

## 3. The real bay window

1. **Connect the projector as a display.** Draw the shapes live with the pen tool in no-photo mode (`python -m src.capture.pen_tool draw --out runs/scene`) while watching where the lines land; Save writes `labels.json` with `source: "projector"`, so no warp is needed. Neighbouring panels can share an edge: new vertices snap to existing vertices (ring) and edges (diamond), Alt places freely, closing a shape welds matching vertices into the neighbour, and dragging a vertex or edge moves everything that shares it (shared vertices are filled yellow); Esc cancels a drag, U undoes.
2. `python -m src.show --labels runs/scene/labels.json`. Open `/output` in a browser window, drag it onto the projector display and press **F** (Chrome: the page's "Fullscreen on the projector screen" button picks the screen). The page reports its real pixel size and the server re-renders at it; the operator chip turns green (`fullscreen`).
3. **Check alignment.** Calibrate tab, pattern `outline`: panel outlines solid, glass dashed. They should sit on the real edges.
4. **If the projector moved**, or for photo-traced labels, re-align: Calibrate tab, *Start corner alignment*. Four numbered markers are projected at real corners of the drawn shapes. Drag each onto the physical corner it was drawn on (or select with 1 to 4 and nudge with the arrow keys, Shift = 10 px); the outline stretches live. *Apply and save* writes `calibration.json` next to the labels.
   - Labels traced on a **photo** need a camera calibration first: photograph the `dots` pattern and the view with the projector black, then `python -m src.projector.calibrate dots --on dots.jpg --off black.jpg --labels labels.json --projector 1920x1080 --out calibration.json`. Dots on glass are swallowed; if fewer than nine are seen it says how many and why.
5. **Safety margin**: 2 projector pixels by default are shaved off every lit region (Calibrate tab). Every frame is also checked to be exactly 0 on the glass **in projector pixels, after the warp**, and a frame that fails is refused and replaced with black (the glass chip counts frames checked and refused).

### The photographed glass-stays-black check

The software can prove the glass pixels it *sends* are 0. Whether the real glass stays dark depends on the real wall, so photograph it. Fixed camera, manual exposure and focus, room dimmed, nothing moves between shots. Photo test tab: show **black**, **white-in-frame** (and optionally **dots**), save as `black.jpg`, `white.jpg`, `dots.jpg` in one folder. Then trace the real glass and panels on `black.jpg` with the pen tool (`python -m src.capture.pen_tool draw black.jpg`): that trace says where the glass physically is, independent of the projector.

```
python -m src.projector.verify --photos PHOTOS_DIR --truth TRUTH_labels.json --labels runs/scene/labels.json --calib runs/scene/calibration.json
```
- **PASS**: light seen in the glass is at most 2% of the panels' on average and 6% in the brightest 1% (thresholds in `verify.py`).
- **FAIL**: with the reason (calibration off or projector crept, glass reflecting, or bloom: raise the margin).
- **INCONCLUSIVE**: overexposed, too dark for the noise, or too little glass in view; it says which. It never passes by default.
- **NOT CHECKED**: no `--truth` trace was given. This is deliberate: without an independent trace the tool can only measure pixels the software sent as black, which cannot see a projector that crept (the self-test demonstrates exactly this). The optical result is still printed, labelled as the weaker check.

`python -m src.projector.verify --selftest` runs three synthetic cases (aligned, projector crept 14 px, glass reflecting 8%) and writes annotated pictures. The **snapshot** button (key `S`) writes `frame_projector.png` for a photo of a real frame (`frame.jpg`), which adds a correlation between what was sent and what the camera saw.

**Not yet done, needs you:** a real projector, a real window, real photos. The thresholds were chosen by reasoning, not calibrated on a real surface, and the dot registration assumes the wall is flat enough for one homography.

## 4. Performance mode

> **Superposed Relief (section 7) is now the default source, and in it the "Lighting world", "Relief polarity" keys (`1-7`, `L`, `0`, `P`, `X`) do not exist**: the light, the depth and the observation are drawn inside the circuit. The table below describes the CLASSICAL sources (`--source oracle|mock|circuit|complementary`). Relief adds `M` (control B), `I` (lamp read in X) and `W` (witness run).

The operator panel is the laptop's view: a live preview of exactly what the projector shows, the knobs, the patch-graph overlay, calibration, the photo test and a log. The audience screen is a companion display (fullscreen with F). Auto-cycle draws a new frame every few seconds with a crossfade; a frame costs about 70 ms end to end (compose, warp to 1920x1080, exact-black check).

| Key | Action |
|---|---|
| **Look** | |
| `Space / →` | next look (a new draw) |
| `←` | previous look |
| `A` | auto-cycle on / off |
| `R` | re-sample (new draw, same knobs) |
| **Demo** | |
| `Space / →` | next demo beat |
| `←` | previous demo beat |
| **Lighting world** | |
| `1` | hold: light from right, frontal |
| `2` | hold: light from left, frontal |
| `3` | hold: light from right, grazing |
| `4` | hold: light from left, grazing |
| `L` | hold / release the current lighting world |
| `0` | lighting world: auto |
| **Relief polarity** | |
| `5` | hold: all panels sunk |
| `6` | hold: all panels raised |
| `P` | hold / release the current polarity |
| `7` | polarity: auto |
| `X` | read polarity in Z / X (complementary experiment) |
| **Hardness and blend** | |
| `[` | harder (fewer shots averaged) |
| `]` | softer (more shots averaged) |
| `-` | less blend |
| `= / +` | more blend |
| **Projector** | |
| `B` | blackout (projector black) |
| `N` | swap the quantum draw for independent noise (the §2.2 test) |
| `T` | test patterns: black, white, outline, dots, grid, edge, off |
| `S` | snapshot the projector frame for the photographed test |
| **Panel** | |
| `O` | patch-graph overlay (laptop only) |
| `D` | demo mode (the five-minute script) |
| `V` | re-solve (shows the cost first; never spends without a click) |
| `? / H` | this help |

- **Independent noise (N)** swaps the draw for plain random numbers through the identical compose code: the spec §2.2 test, live.
- **Overlay (O)**: patches coloured by this look's lit fraction, edges coloured by the requested correlation, lamp and polarity bits, and red rings where an engine's *returned* correlation is more than 0.1 from the request (choose graph-v1 or QDrive under "overlay gap rings compare against"; on the first graph-v1 run that is 13 rings, which is the honest picture).
- **Re-solve (V)** opens a dialog that shows the payload, its hash and its cost. It cannot submit unless the server was started with `--allow-spend` **and** you press the spend button, and the server refuses if the payload or cost differs from what was shown. A payload that already has a saved result is free and sends nothing. Default: preview only.
- `--lan` serves only the read-only pages (`/output`, `/audience`) to other machines; the operator page and every action stay on this machine.

### The five-minute demo (D)

Seven steps, built from what is on disk and the current provenance: (1) the problem, independent noise then the correlated draw; (2) geometry and the black-glass rule (outline, then white-in-frame); (3) the qubit graph; (4) requested vs achieved; (5) the three knobs in turn, plus the complementary beat; (6) QDrive vs graph-v1 (it says plainly when QDrive has not returned a circuit for this scene); (7) an honest scorecard of what is and is not quantum-specific. Space or the arrows step through beats; each beat sets the projector, the knobs and the audience screen together.

## 5. The polarity-complementary experiment (UNTESTED)

```
python -m src.quantum.complementary --out runs/complementary --sweep      # ~25 s; add --circuit QDRIVE.qasm for a real circuit
python -m src.show --source complementary                                  # see it on the wall
```
**What it is.** Lamps uniform, patches Boltzmann given the lamps (the mock's distribution as a pure state), polarity qubits in |+> with a controlled phase to the lamp that governs their panel. Controlled phases are diagonal, so every lit/dark and lamp statistic is exactly the Z-only baseline's. **Read in Z, polarity is a fair coin; read in X, it is locked to its lamp** (correlation 1.00). This is a *stand-in*: no QDrive circuit for this state exists (at the time of writing QDrive's one job had failed on the target syntax; since then small jobs have run, see `qdrive-field-notes.md`), so no Moth result is involved.

**What it found** (on the first build's three-panel window, 19 qubits; the six-pane default is described below the list; the outputs are `runs/complementary/report.json`, `complementary_frames.png`, `complementary_witness.png`):
- The witness is a Bell-type test needing no assumption about the circuit: the exact best CHSH value over all settings for every qubit pair (Horodecki), and a Mermin inequality for the lamp plus its polarity qubits. Checked against textbook states and against sampling.
- Coupling all polarity qubits to L2: the 4-party Mermin value is **7.46 against a classical bound of 4** (maximum 8), 479 sigma with 20,000 shots per term. No *pair* shows it (every pairwise CHSH is exactly 2.00), because the other polarity qubits hold a copy of the lamp's phase.
- Coupling by geometry (sides to L1, centre to L2): only the centre pair violates (CHSH 2.74). **The side panels' lamp L1 gives exactly 2.00 because the patches follow L1 strongly and so already hold a record of it.** This is a real trade-off: the more the lighting is coherent, the less a polarity qubit tied to that lamp can be non-classical; `--sweep` draws the curve.
- The Z-only baseline's *statistics* are classical but its sqrt(P) *state* is not classical in other bases (one patch pair scores 2.35), so its S_max is reported for contrast, not as a "classical" label.
- **On the six-pane default** every lamp is followed by two polarity qubits per wing (sides to L1: four panes; centre to L2: two), so each lamp is recorded twice over: *no* pane-lamp pair violates CHSH (every one is exactly 2.00), and only the many-party Mermin test over the lamp and its polarity qubits does (value 46.7 of a possible 64 at the default phase, `lamp2` coupling, seven parties). The centre pane's 2.74 above is a three-panel result; `tests/test_complementary.py` pins both.

**What it does not show.** A violation says the *state* is non-classical; it is not a quantum-advantage claim (this whole script is classically simulable). The audience cannot see it in any single frame: it lives in statistics across measurement settings. It says nothing about what QDrive will return for these targets. `payload_qdrive_complementary.json` is built and validated locally (1 credit if sent) but **NOT SENT**, and the mixed-Pauli-word target syntax is unverified.

## 6. Tests

```
python -m pytest tests -q                     # everything (about 7 minutes), or one module at a time with: python -m unittest tests.test_relief
python -m unittest tests.test_moth_pipeline   # the whole Moth path against a fake server on localhost
python -m tests.check_port_matches_mock       # the src port still reproduces the original mock exactly
```
UI tests install a stub for the Moth solver and fail if anything reaches the real one.

## 7. Superposed Relief (v2): the default performance mode

Built 2 Oct 2026 from `standing-light-v2-superposed-relief-handoff.md`. **Everything below is a local exact simulation; no Moth engine has been run on it.** Where this section differs from the v2 handoff, this section describes what the code does and why.

```
python -m src.show                      # relief is now the default source; the classical sources are --source oracle|mock|circuit|complementary
python -m src.quantum.relief_report     # figures + report.json in runs/relief (frames with both controls, complementarity curve, scaling law, witness table)
python -m unittest tests.test_relief tests.test_moth_engines
```

**What the wall is.** `src/geometry/facets.py` cuts every panel into facets, and **a facet is an orientation class, not a patch**: all pixels whose relief slopes the same way share one qubit (the outer-left bevel and the glass's right-hand bevel are one facet), because under a distant light they are lit identically. Per panel: `--n-dirs` slope facets + 1 flat one (the default is the most that keeps the exact circuit within 20 qubits: 4 for three panels, **2 for the six panes of the default window**, 1 for eight), so 18 facets for the default window, 12 of them quantum. A facet's state is its surface normal, `b = (sin t cos p, sin t sin p, cos t)` in the panel's own frame. `src/quantum/relief_state.py` builds `|Psi> = 2^(-P/2) sum_s |s>_B (x) P^s |psi_f>` (H on a polarity qubit per panel, Ry/Rz per facet, optional couplings, CZ(B, facet)); 12 facets + 6 polarity = 18 simulated qubits, 22 with the lamp (2) and observation (2) registers (the plateau facets are classical and carry no qubit).

**A frame is one run of the circuit.** The lamp register picks one of four lights and rotates every facet so measuring Z measures sigma.l (Lambert is the Born rule), the observation register picks gamma on each polarity qubit's depth sphere, everything is measured once, K photons deep. The polarity outcome is the depth decision; where gamma landed decides how definite it looks (pole: a bump or hollow; equator: **flat**; between: partial). Nothing in performance mode chooses a look: the lighting-world, relief-polarity and polarity-basis keys, buttons and actions exist only for the classical sources (`tests/test_relief.py::NoTogglesInPerformanceMode`).

**Controls and experiments (operator "science" section, never looks).** `N` control A, independent noise. `M` control B, dephased depth (each polarity qubit replaced by its classical mixture). `I` lamp read in X: the light directions interfere. `W` witness run. The depth-sphere dots show where each observation landed.

### What was found while building it (these correct or extend the handoff)

| Finding | Consequence |
|---|---|
| **Diagonal couplings only.** `zz` phases between facets keep the visibility law `V = prod cos(tau)`, the stabilizer `X_B Z^F = 1`, and an *exactly* flat bevel at the equator for every polarity outcome. The parity-preserving `xy` exchange keeps the first two but leaves a residual bevel at 90 deg (0.05 to 0.15 in lit fraction). | Default `coupling="zz"`; `xy` is an opt-in experiment. With only diagonal gates the whole circuit is **IQP-shaped** (single-qubit layer, commuting diagonal gates, single-qubit measurement rotations): the family whose classical sampling hardness is conjectured. A direction, not an advantage claim at 18 qubits. |
| A diagonal coupling rotates each facet's azimuth about z (by `sum atan2(sin th cos tau_j, cos th)`), exactly, because it leaves Z populations alone. | The preparation azimuths are pre-rotated by the opposite angle: every facet keeps the normal the geometry gave it; only its transverse length shrinks (closed form, tested). |
| At gamma = 90 deg the facets' **unconditioned** statistics are identical for the coherent state and for control B (no-signalling). What differs is the joint with the polarity **outcome**: `P(o = 0) = (1 + V)/2` against 1/2, and the stabilizer 1 against 0. | Handoff test 6 is stated that way in `tests/test_relief.py`. The visibility is also a frame-level frequency: undecided frames show outcome 0 about 70% of the time instead of 50%. |
| `X_B (x) Z^{F_p}` is an exact stabilizer, and `F > (1 + V_p)/2` for the fidelity to the ideal state certifies entanglement across the cut B_p | rest (Bourennane witness; `F` is the probability of all zeros after un-computing the reference circuit, so a device can measure it). Control B has `F = 2^-P`, noise `2^-n`. | The certificate in `relief_witness.py`. Pairwise CHSH is reported too and is **not violated** at these tilts (1.64 for polarity-facet); the report says so instead of hiding it. |
| Qiskit `prepare_state` of an exact vector, transpiled to u/cx, did not reproduce the state (fidelity 0.78); gate-by-gate circuits do. | Circuits are built gate by gate everywhere. |

### Verified how
Reference numbers: the engine reproduces the handoff's check-4 table (contrast +.332 +.238 +.122 0 ..., P(outcome 0) .500 .603 .678 .706 ...). The Qiskit reference circuit equals the numpy state to 1e-10 with both coupling kinds; **the full circuit (lamp and observation registers inside) sampled by Aer matches the exact distribution** (total variation 0.013, sampling floor 0.020). the tests are in `tests/test_relief.py` and `tests/test_moth_engines.py` (their counts are in the README's Tests table).

### Not done
No real projector, no Moth engine run on this design, QDrive has returned circuits only for small jobs (never for a whole scene), the studio's Solve/Perform steps run the relief circuit on Aer (a simulation here, per-panel engine only), not on a Moth engine or a QPU, and the lamp/observation registers are uniform draws (physical randomness only on a QPU).

## 8. Moth engines for this project

`python -m src.quantum.moth_engines plan` lists them; `build <what>` writes a payload and prints the exact request (nothing is sent). Costs and parameter whitelists are registered in `moth_client.py`, so a typo or an IBM credential is blocked locally.

| Engine | Cost | Fit | Use |
|---|---|---|---|
| tomography-api-v2 | 1 | high | **Best first spend.** Our reference circuit as QASM2: polarity Bloch X = visibility, (polarity, facet) pair = slope locked to depth, mutual information. Scored against exact moments by `score_tomography`. Replaces graph-v1 (5) for verification. |
| qdrive-api-v1 | 1 | high, blocked | Facet-register preparation. Both jobs of 201 targets died with `engine_timeout` at target 56 and 61 (a mapping `{"Z": v}` / `{"ZZ": v}` is accepted). Jobs of <= 25 targets, chained through `initial_circuit` (an `input_files` asset id; accepted asset types are unverified), and a 2-qubit probe for mixed words first. The client now refuses > 100 targets. |
| otoc-echo-v1 | 1 | medium-high | Dynamic relief: echo tap `F(site, depth)` -> tilt `|F|`, azimuth `arg F`, sign(Re F) raised/sunk, depth = frame. Aer <= 24 qubits; IBM backends up to 156 sites need your own IBM token (never sent by this client). `local_echo` is *our reading* of the description, not the engine's definition. |
| qpixl-v1 | 1 | medium | The scaling statement: a 64x64 Lambert map on ~13 qubits. |
| entanglement-shader-v1 | 1 | low-medium | Angle-phase reflectance LUT as a quantum-made tone response; not wired in. |
| blur-core-v1, coin-toss-v1 (2), labyrinth-v1 (5), tessa-image-v1 | | no | Frame blur must be live and mask-aware; coin-toss returns counts, not bits; labyrinth has no honest mapping; Tessa returns an image with no joint state. |

### Pitch paragraph for the README
*Standing Light projects shading, not images. Each patch of a real surface is a qubit whose state is its surface normal; the light is the measurement axis, and Lambert's cosine law is the Born rule. Depth itself is in superposition, a raised and a sunk version of each panel; how decided the depth looks in any frame depends on where the observation lands on the depth qubit's Bloch sphere, and that is drawn inside the circuit, never toggled. The more definite the depth, the less the two depths can interfere, and the visibility of that interference follows a law (the product of cos of each facet's tilt) that tells us how finely such a surface can be specified at 1000+ qubits.*

## 9. Domain Relief (v3)

Built 2 Oct 2026; design, audit findings and results in `standing-light-v3-domain-relief-handoff.md`. **Local simulation only; nothing sent to Moth.**

```
python -m src.show --engine domain                # 128 qubits (six-pane window) as a matrix-product state; about 5 s to build, 0.07 s per frame
python -m src.studio --engine domain              # step 3 opens it
python -m src.quantum.relief_report --domain      # frames, certificate, contrast vs visibility
python -m src.quantum.classical_baselines         # where classical simulation stops being cheap
python -m src.quantum.domain_moth plan            # payloads for the polarity register (QDrive), a patch (tomography), dynamic relief (echo)
```

In the operator panel everything of section 7 still applies (controls A and B, witness run `W`); differences:
- the **witness run** prints the domain certificate: entangled edges (negativity), lamp-depth CHSH and Mermin, frustration, bond dimension, with the dephased control beside each;
- the **lamp in X** (`I`) is exact-only: the domain engine falls back to the Z read and says so in the log;
- the overlay draws the domain graph (blue coplanar, orange crease), a ring per polarity qubit (yellow raised, red sunk this frame, +/- the lamp-lock sign);
- captions and depth spheres speak of panels: each panel shows the mean over its domains.

Added since: `G` toggles the **parity game** (some looks are rounds of a Mermin game between the lamp and three depth qubits, one leaf in each wing; the tally against the classical 75% is in the science panel), `E` evolves the relief one kicked-Ising step (entanglement grows; on the six-pane window the game's predicted win rate falls from 93% to 74% after two steps, see the README), and `--seam-mix`, `--crease-coupling`, `--leaf-tau` set the seam regime (see the README). The lamp in X (`I`) now runs on this backend too.

The glass checks are unchanged (exact 0 on the glass after the warp; a sparse composer is used beyond 24 facets). The projector procedure of section 3 applies as is. **Not yet done: judging the domain relief by eye on the real wall.**
