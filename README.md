# Quantum Chiaroscuro

**Quantum chiaroscuro, projected onto a real surface.**

Quantum Chiaroscuro projects *shading*, not images, onto a real object. The reference scene, and the default of every tool here, is a **bay window of six panels**: three wall sections (left wing, centre, right wing), each holding **two windows stacked on top of each other with one shared rail between them**, and a pane of glass in every window ([The default scene](#the-default-scene)). A physical wall has one shape; light can make it look swollen, hollowed, creased or flat. The project makes that literal with quantum mechanics:

- Every patch of the wall is a **qubit whose state is its surface normal**. The light is the measurement axis, and Lambert's cosine law is the Born rule, so a frame is a measurement, not a render.
- Each panel's **depth is in superposition** (a raised and a sunk version). How decided the depth looks in any one frame depends on where the observation lands on that panel's Bloch sphere, and that is drawn *inside the circuit*, never toggled: a pole gives a definite bump or hollow, the equator gives a flat, undecided panel. The azimuth χ round the equator chooses *which* raised-versus-sunk coherence the frame samples (`P(depth) = (1 + V sin γ cos χ)/2`, and a different bevel signature in the facets): χ = 90° reads the Y component, which is a fair coin on the depth but still shapes the bevel. The equator is exactly flat at χ = 0° and 180° only; at χ = 90° a small outcome-dependent residual bevel (about ±0.06 on the six-pane default, ±0.03 on the original three-panel window) remains.
- Geometry only supplies the **barriers and limits**: the glass panes are never lit (they are exactly black, checked on every frame), shared edges set where the shading may change, and the plane windows set how bright each surface may get.

There are two relief engines. **Superposed Relief (v2, `--engine panel`, the default)** has one polarity qubit per panel: on the six-panel bay window 12 facet qubits (two slope facets per pane) + 6 polarity qubits = 18 simulated, exactly, on your laptop, 22 in the full circuit. **Domain Relief (v3/v4, `--engine domain`)** cuts the wall into local bevel facets, gives every facet its *own* depth qubit and couples those qubits along the geometry's edge graph, weakly along a boundary and strongly across the seams between wings: 128 simulated qubits in the default configuration (64 facets + 64 depth qubits), as a matrix-product state. Every crease between wings is entangled (4 of 4 crease edges, largest negativity 0.28), a lamp-plus-three-depth-qubits **Mermin game** (one leaf in each wing) is predicted to be won in 93% of rounds, 92.5% when sampled, against a classical maximum of 75% (`G` key, `--game`), and a kicked-Ising evolution (`E` key, `--evolve-steps`) makes the entanglement grow with every step ([Domain Relief](#domain-relief-v3)).

Honest status: at these sizes everything is classically simulable, and a benchmark (`src.quantum.classical_baselines`) measures how far each design can grow before the classical methods stop being cheap. The claim is *structure and scaling*, not quantum advantage, and nothing in the quantum design has yet been run on a Moth engine (see [Status](#status-and-limits)).

---

## Contents

1. [The default scene](#the-default-scene)
2. [Quick start (the studio)](#quick-start)
3. [Installation](#installation)
4. [The ways to run it](#the-ways-to-run-it)
5. [How it works](#how-it-works)
6. [Codebase map](#codebase-map)
7. [The performance UI](#the-performance-ui)
8. [Moth engines and spending credits](#moth-engines-and-spending-credits)
9. [Tests](#tests)
10. [Safety rules the code enforces](#safety-rules-the-code-enforces)
11. [Status and limits](#status-and-limits)
12. [Further reading](#further-reading)

---

## The default scene

Everything that needs a window and has not been given a drawing (`python -m src.show`, the studio's *built-in demo scene*, every `--labels`-less tool, the tests) uses `bay_window_labels()` in `src/capture/labels.py`: **six panels in three planes**.

```
   left wing (+35°)          centre (0°)              right wing (-35°)
 ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
 │ 0  left_top        │  │ 2  centre_top      │  │ 4  right_top       │
 ├──────── rail ──────┤  ├──────── rail ──────┤  ├──────── rail ──────┤
 │ 1  left_bottom     │  │ 3  centre_bottom   │  │ 5  right_bottom    │
 └────────────────────┘  └────────────────────┘  └────────────────────┘
        plane 0                  plane 1                  plane 2
```

| | Default scene |
|---|---|
| Panels | 6, ids 0-5 in the order above: `left_top`, `left_bottom`, `centre_top`, `centre_bottom`, `right_top`, `right_bottom` |
| Planes | 3 (`plane_id` 0, 1, 2; yaw +35°, 0°, -35°). The two windows of one wing are **coplanar**: the rail between them is a *coplanar* edge, not a crease. The only creases are the two vertical edges between the wings |
| Glass | 6 panes, one in every window; exactly black on every frame |
| Classical patch grid (studio steps 2, 4, 5) | 4 x 4: 12 patches + 2 lamps + 6 polarity qubits = **20 qubits**, exactly the graph-v1 cap |
| Per-panel relief (`--engine panel`) | 2 slope facets per pane (the most that keeps the circuit exact, see `--n-dirs`): **12 facet qubits + 6 polarity = 18 simulated, 22 in the circuit**, plus 6 classical plateau facets |
| Domain relief (`--engine domain`) | 64 facets + 64 depth qubits = **128 simulated** (132 in the circuit), 74 domain edges of which 4 are creases |
| Parity game | the lamp and three leaves, **one leaf per wing** (left, centre, right), not one per pane |

The first build modelled each wing as ONE undivided panel (three panels, `bay_window_labels(stacked=False)`). That layout survives only so that `src.run_mock` and `tests/check_port_matches_mock.py` can still reproduce the original classical look test bit for bit (`mock/standing_light_mock.py`); `python -m src.run_mock --stacked` renders the six-pane default. Older numbers in the `docs/` handoffs that say "three panels", "15 qubits" or "19 qubits" describe that first layout.

Your own window is drawn in the pen tool with the same convention: give the top and the bottom window of one wing the **same plane id**, so that they share an angle and the rail between them is not a crease.

---

## Quick start

Run everything from the **studio**. It is one local page that walks the whole pipeline and starts each tool for you.

```bash
cd chiaroscuro                     # the project folder (the one holding src/, tests/ and this README)
source .venv/bin/activate          # Python 3.10+; see Installation if you have no .venv
python -m src.studio               # opens the studio in your browser
```

### Common startup modes

Use these when you want to start the system in a particular mode instead of the default local studio run:

| Command | Purpose |
|---|---|
| `python -m src.studio` | default studio launcher; opens the guided workflow in the browser |
| `python -m src.studio --project runs/myroom` | keep a separate project folder for a different scene or run |
| `python -m src.studio --allow-spend` | allow the studio to submit or spend on Moth during the solve/send flow; still asks for confirmation before sending |
| `python -m src.studio --engine domain` | launch the domain-relief workflow instead of the default per-panel relief |
| `python -m src.show --engine domain --game` | start the live show directly in domain mode with the parity-game witness enabled |
| `python -m src.show --engine domain --evolve-steps 4` | start the live show directly with Floquet entanglement growth enabled |
| `python -m src.show --lan` | serve the output and audience pages on the local network as well as the operator page |

> The spend gate is the one you want when you are intentionally testing the Moth-backed flow: `--allow-spend` is the actual flag in the current codebase, and it still asks for confirmation before any submission occurs.

On the page, press **Next step** each time (it always offers the first step that isn't done), or click a step's own button:

| Step | Button does | Needs |
|---|---|---|
| **1 Draw the shapes** | opens the pen tool: draw the panels and glass on the projector's own screen, press **Save** in it. Or press **Use the built-in demo scene** (the six-pane bay window above) to try everything with no projector and no drawing | nothing |
| **2 Calibrate targets** | estimates the classical correlation targets for your drawing and writes the Moth payloads. Sends nothing, takes up to a minute | step 1 |
| **3 Rehearse the show** | **opens the show on Superposed Relief**, the quantum circuit simulated exactly on your laptop. No Moth call, no credits | step 2 |
| **4 Solve on Moth** | shows the exact payload, its hash and cost. Sends only if you started the studio with `--allow-spend` *and* press the send button | step 2 |
| **5 Perform with the result** | opens the show on the circuit step 4 returned (classical QDrive flow) | a QDrive result |

Step 3 prints a URL and opens the show. That is where you see and use Superposed Relief: the operator panel (`/`), the projector window (`/output`, drag it to the projector display and press `F`) and the audience screen (`/audience`). Press `Space` for a new frame, `A` for auto-cycle, `W` for the entanglement witness, `?` for every key. Stop ends any tool the studio started; Ctrl+C stops everything.

To keep several scenes side by side, or to allow spending:

```bash
python -m src.studio --project runs/myroom     # a separate project folder per scene (default: runs/studio)
python -m src.studio --allow-spend             # lets step 4 submit to Moth (it still shows the cost and asks first)
python -m src.studio --engine domain           # step 3 opens the domain relief (a few seconds to build) instead of the per-panel one
```

The studio writes to its project folder: `labels.json` (the drawing), `calib/` (targets and payloads), `solve/` (Moth results), `studio.json` (which drawing the targets came from, so a changed drawing marks later steps stale).

**What to know about the current studio** (accurate as of the Superposed Relief build):
- Only step 3 opens the new relief engine. Steps 2, 4 and 5 are the earlier, classical QDrive flow, and step 5 stays blocked until there is a QDrive circuit for the project (no whole-scene job has finished yet; `python -m src.quantum.qdrive_rounds` builds one in small chained jobs and the studio recognises its finished result).
- Step 3 is gated behind step 2 even though the relief engine does not use those targets: it builds its facets straight from your drawing. So to reach relief you must run step 2 once, even with the demo scene.
- Step 2 for the demo scene writes the same calibration as `runs/calibration_bay_4x4/` (12 patches, 20 qubits), which is what the pipeline's CLIs (`solver`, `qdrive_rounds`, `show`...) use as their default `--calib`.
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

**Moth API key.** Only needed to submit jobs. Put `MOTH_API_KEY=...` in the environment or in the `.env` file in the project folder, next to this README (`chiaroscuro/.env`, git-ignored; it is the same file whichever `--project` the studio uses). The code reads that one variable lazily, never prints it, and previews never need it. If it warns that `.env` is readable by others, run `chmod 600 .env`.

**If `source .venv/bin/activate` seems to do nothing** (it silently keeps using the system Python and `pip`/`pytest` fail with "bad interpreter"): the folder was moved after the venv was made, and the venv still points at the old path. Recreate it (`rm -rf .venv && python3 -m venv .venv && pip install -r requirements.txt`) or run `.venv/bin/python -m ...` directly.

---

## The ways to run it

The studio ([Quick start](#quick-start)) runs these same modules for you. You can also run any of them directly, from the project root (`python -m ...`) with the same flags. Nothing sends anything to Moth unless you explicitly say so (see [Moth engines](#moth-engines-and-spending-credits)).

### 1. The show on its own (what studio step 3 opens)

Use this to skip the studio. The studio's step 3 and step 5 each have a "Show options" section that sets these same flags from the page (greyed out when the source or engine you picked ignores them); blank leaves the show's own default.

```bash
python -m src.show                                          # built-in six-pane bay window, Superposed Relief
python -m src.show --labels runs/scene/labels.json          # shapes you drew with the pen tool
python -m src.show --kappa 1.0 --n-dirs 2 --entangle 0.8    # relief knobs (below; --n-dirs 2 is what six panes get by default)
python -m src.show --engine domain                          # Domain Relief: dozens of qubits, polarity per depth domain (flags below)
python -m src.show --engine domain --seg-len 100 --lock 0   # finer facets, lamp left as a plain coin
python -m src.show --engine domain --game                   # some looks are rounds of the Mermin parity game (G toggles)
python -m src.show --engine domain --evolve-steps 4         # entanglement grown by four kicked-Ising steps (E advances one more)
python -m src.show --engine domain --seam-mix 1             # the seams in the graph-state regime (what it costs the game: see below)
python -m src.show --source relief --circuit FACETS.qasm    # a Moth QDrive circuit for the 12 facet qubits (one per band facet; polarity is lifted locally)
python -m src.show --source oracle                          # CLASSICAL rehearsal (also: mock, circuit, complementary)
python -m src.show --lan                                    # also serve /output and /audience to other machines
```

| Flag | Meaning |
|---|---|
| `--source` | `relief` (default) or one of the classical sources |
| `--kappa` | tilt budget per panel, `sum(tau²) = kappa²`. Interference visibility is about `exp(-kappa²/2)`: higher is a stronger bevel and weaker interference |
| `--n-dirs` | slope-orientation facets per panel, a qubit each, plus one classical plateau facet. Default: the most that keeps facets + panels within 20 exact qubits, at most 4. That is **2 for the six panes of the default bay window (12 facet + 6 polarity = 18 qubits)**, 4 for three panels (15), 1 for eight. Asking for more is refused above 22 qubits, with the number to use |
| `--entangle` | strength of the diagonal couplings between facets (0 = product relief) |
| `--contrast` | projection tone: stretch of the lit field about mid-grey (an exposure, not a scene property) |
| `--lamp-mode` | `z` (default: a light world is drawn per look) or `x` (the lamp register is read in X, so the light directions interfere; the `x` experiment key, set from the start) |
| `--calib`, `--run` | directories with calibration targets and engine results (for the demo slides and re-solve). Defaults: `runs/calibration_bay_4x4` and `runs/bay_run`, the default scene's. The re-solve dialog refuses a calibration made for a different scene |
| `--engine` | `panel` (default) or `domain`; the flags below apply to `domain` only |
| `--seg-len` | facet length along a polygon edge in px (150: 64 facets and 128 qubits on the default bay window; 200: 60 facets; 100: 112; 60: 176; smaller: more qubits) |
| `--group-size` | facets that share one polarity qubit (1: one facet per depth qubit, the most coherent). Tilt budget per domain: V = cos(tau)^group-size, whatever the size of the wall |
| `--tau` | tilt of each facet in radians (0.5) |
| `--pol-coupling` | Ising angle between neighbouring domains' polarity qubits along a boundary (0.3, weak; sign + coplanar) |
| `--crease-coupling`, `--crease-sign` | the same angle across a crease between panels (1.0, strong) and its sign (-1: the depths want to differ across a shared edge) |
| `--seam-coupling` | ZZ angle between facets that touch across a crease and among seam facets (1.0; pi/2 is a CZ up to local phases) |
| `--seam-mix` | 0..1 moves ONLY the seam facets toward the equator: the graph-state regime on the seams, visible relief elsewhere (0) |
| `--leaf-tau`, `--leaf-prefer` | tilt of the facets of the game's leaf domains (0.3: more visibility, more rounds won) and which domains are the leaves (`visible`, or `seam`) |
| `--game`, `--game-fraction` | start with the parity game on; the fraction of looks that are rounds (0.5) |
| `--evolve-steps`, `--evolve-zz`, `--evolve-x` | kicked-Ising steps on the facet graph before the polarity attaches (0), and their angles (0.7, 0.5) |
| `--lock` | coupling of lamp L1 to one leaf polarity qubit per plane (the left, centre and right wing of the bay window: three in all) with the crater-gauge signs (pi/2: maximal; 0: the lamp is a plain coin) |
| `--tau-mix`, `--pol-field` | 0..1 moves the facets toward the equator (graph-state regime, V falls); transverse field on the polarity qubits |
| `--backend` | `auto` (exact up to 20 qubits, else matrix-product state; the default 128-qubit domain relief is always a matrix-product state), `exact`, `mps` |
| `--projector-size WxH` | initial projector size; the output window reports its real size once open |
| `--allow-spend` | lets the re-solve dialog submit to Moth (it still asks, showing hash and cost) |
| `--port`, `--no-browser` | server port (default: any free) and do not open a browser |

#### Relief flags: what each knob changes, conceptually and in practice

These are not cosmetic controls. In this project, a “relief knob” changes the quantum state that is sampled, the geometry it is allowed to express, and the trade-off between “visible relief,” “entangled wall,” and “certified parity-game behavior.”

The simplest rule of thumb is:

- `--kappa` sets how much tilt the wall is allowed to carry.
- `--entangle` sets how strongly the facets are coupled together.
- `--seam-mix` and `--tau-mix` move the state toward a graph-state / seam-heavy regime.
- `--game` and `--game-fraction` switch the scene into the Mermin certifier.
- `--evolve-steps` decides whether the state stays static or grows entanglement in time.
- `--lock` decides whether the lamp acts like a coherent control or a plain coin.

##### Geometry and visibility

###### `--kappa`
- Conceptually: this is the per-panel tilt budget. The relief state is built from facet normals, and the total allowed angular spread is constrained by `sum(tau²) = kappa²`.
- Practically: larger `kappa` means stronger relief and stronger bevel, but also weaker interference because visibility falls roughly like `exp(-kappa²/2)`. In other words, it increases the sculptural effect while reducing the “quantum coherence” of the state.

###### `--n-dirs`
- Conceptually: how many slope directions each panel is decomposed into. More directions means more facet classes, more qubits, more structure.
- Practically: a panel with few directions is simpler and cleaner; a panel with many directions has more nuanced relief but more complexity and more measurement overhead. This is the knob that chooses the granularity of the panel geometry.

###### `--tau`
- Conceptually: the actual facet tilt angle.
- Practically: it is the “shape” parameter for each facet normal. If `kappa` sets the total available budget, `tau` is the per-facet tilt that spends it. In the wall, this controls how much each local patch leans relative to the panel plane.

###### `--contrast`
- Conceptually: not a geometry or Hamiltonian parameter; it is an output/exposure control.
- Practically: it stretches the lit field around mid-grey. It changes how the projector looks, but it does not change the underlying quantum structure. This is the “make the image easier to read” knob, not the “change the physics” knob.

##### Domain graph and seam structure

###### `--seg-len`
- Conceptually: the length of a facet segment along an edge.
- Practically: smaller segment lengths produce more facets, more qubits, and a finer-grained relief. Larger values simplify the wall and reduce the Hilbert space. This is the coarse/fine control for how much detail the geometry exposes.

###### `--group-size`
- Conceptually: how many facets share a single polarity qubit.
- Practically: `group-size = 1` gives the most coherent result because each facet has its own depth qubit. Larger groups smear the polarity across multiple facets, which reduces coherence but can simplify the model. It is the main way to trade off “finer physical detail” against “cleaner, more stable domain behavior.”

###### `--pol-coupling`
- Conceptually: the Ising coupling between neighbouring domain polarity qubits along a boundary.
- Practically: this is the strength of the signed domain interaction that makes local relief decisions correlate across a panel. Weak values keep nearby domains mostly independent; stronger values add structural continuity and make the wall act like one correlated object.

###### `--crease-coupling`, `--crease-sign`
- Conceptually: the same idea as above, but specifically across a crease between panels.
- Practically: these decide how strongly the relief wants to differ across the shared edge and whether the sign of that difference is positive or negative. This is where the wall’s “crease” behavior is encoded. The sign matters because a crease is not just a boundary; it is a geometric frustration line that can prefer “bump on one side, hollow on the other.”

###### `--seam-coupling`
- Conceptually: the `ZZ` coupling among facets that touch across a seam or crease.
- Practically: this is the direct graph-state coupling that makes the seam behave like an entangled edge rather than a simple boundary. High values create strong seam entanglement and can dramatically change the parity-game behavior.

###### `--seam-mix`
- Conceptually: a “move only the seam facets toward the equator” control.
- Practically: this is the knob that shifts the wall from ordinary relief into the seam-heavy regime. At `seam-mix = 0`, the visible relief stays ordinary and the seams remain weakly perturbed. At `seam-mix = 1`, the seam facets are pushed toward the equatorial graph-state regime, which gives stronger seam entanglement but destroys the ordinary relief behavior on those geometry lines. In short: it makes the wall look like a graph-state skeleton rather than a clean relief surface.

###### `--tau-mix`
- Conceptually: a general “mix the facet state toward the equator” knob.
- Practically: it makes the facets less visible as ordinary relief and more like an entangled graph-state arrangement. It is a broader, less targeted version of `--seam-mix`, and it reduces visibility while pushing the state toward the seam regime.

###### `--pol-field`
- Conceptually: a transverse field on the polarity qubits.
- Practically: it acts like a local drive that competes with the Ising order. Large fields push the polarity state away from clean ordered relief and into a more fluctuating, graph-like regime. This is the “how energetic is the local polarity field?” knob.

##### Game and certification

###### `--game`
- Conceptually: enable the Mermin parity-game witness.
- Practically: the system starts using a four-party parity-game frame for some fraction of the looks, and the operator page shows a running tally versus the classical bound of 75%. This is the knob that turns the installation into a nonclassical certification mode rather than a purely visual render.

###### `--game-fraction`
- Conceptually: how often a look is treated as a parity-game round instead of an ordinary relief frame.
- Practically: the project will still show a normal scene, but a fraction of the frames become game rounds. This is a way to trade between “presentation” and “proof”: more game fraction means more certification, but less ordinary visual output.

###### `--leaf-tau`, `--leaf-prefer`
- Conceptually: the leaf domains used in the parity game and the tilt assigned to those leaves.
- Practically: more visible / more coherent leaves produce a stronger game signal; choosing `seam` or `visible` leaves changes which facets participate directly in the certification. This is the part of the design that makes the game “about” the same geometry that produces the wall.

###### `--lock`
- Conceptually: how strongly the lamp couples to the polarity register.
- Practically: `--lock 0` leaves the lamp effectively as a plain coin; `--lock π/2` gives it maximal coherent control over the polarity. This is the prime knob for turning the lamp from a passive observer into an active driver of the relief state.

##### Time evolution and entanglement growth

###### `--evolve-steps`
- Conceptually: the number of kicked-Ising steps applied to the facet graph before the polarity is attached.
- Practically: this is the “dynamic entanglement” knob. Each step entangles the graph more deeply, makes the relief evolve in time, and changes the normal field from one frame to the next. In the code, it runs a Floquet-style evolution on the facet graph; it is not a cosmetic animation, it is a real change in the state’s entanglement structure.

###### `--evolve-zz`, `--evolve-x`
- Conceptually: the `ZZ` and `X` angles of the kicked-Ising step.
- Practically: they set how strongly the graph is entangling and how much local transverse field is applied. Larger values push the system toward a more deeply entangled, more time-dependent regime, and they also hurt the parity-game win rate as the entanglement grows.

##### Backend and runtime

###### `--backend`
- Conceptually: whether the state is built as an exact simulator or an MPS-backed approximation.
- Practically: use `auto` unless you need to debug a specific compression regime. `exact` is the “full” physically faithful path for smaller systems; `mps` is the scalable approximation that keeps large domains tractable but changes the numerical behavior and the classical cost.

###### `--projector-size WxH`
- Conceptually: the initial output resolution of the projector page.
- Practically: this is a runtime/output sizing choice, not a physics knob.

###### `--allow-spend`
- Conceptually: allow the system to spend credits when submitting to Moth or re-solving against a remote engine.
- Practically: it is a safety gate around Moth integration. The show still asks for confirmation and shows the hash/cost before sending anything.

###### `--port`, `--no-browser`
- Conceptually: local hosting options.
- Practically: choose where the show serves and whether it opens a browser automatically.

##### In practice: what changes when you “turn a knob up”?

The most important pattern in this project is that the same state can be tuned in three different directions:

1. Visibility / relief mode
   - increase `--kappa`
   - adjust `--tau`
   - keep `--entangle` modest
   - this gives a strong, readable relief wall

2. Seam / graph-state mode
   - increase `--seam-mix`
   - increase `--seam-coupling`
   - raise `--tau-mix` or `--pol-field`
   - this makes the wall more entangled along seams and less like an ordinary optical relief

3. Game / witness mode
   - turn on `--game`
   - set `--game-fraction`
   - adjust `--leaf-tau` and `--lock`
   - this emphasizes the parity-game certification and measures the wall against the classical 75% bound

4. Dynamical entanglement mode
   - use `--evolve-steps`
   - increase `--evolve-zz` and `--evolve-x`
   - this creates entanglement growth over time, but it costs the game and often destroys the clean relief regime

This is why the project is not just a renderer: each knob changes the state, not merely the appearance. The live image is the measured outcome of that state, and the operator is really choosing a physics regime for the wall.

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
python -m src.quantum.seam_benchmark               # seams, the parity game and Floquet time against the classical cost (several minutes)
python -m src.quantum.domain_moth plan             # Moth payloads for the domain relief (build sends nothing)
python -m src.run_mock                             # the original classical look test on the first build's three-panel window (reproduces the 5 mock figures; --stacked: the six-pane default)
python -m src.baseline.compare                     # correlated draws vs independent noise, same compose path
python -m src.quantum.complementary --sweep        # the older polarity-complementary experiment (classical-family stand-in; on six panes no single pane-lamp pair violates CHSH, only the many-party Mermin test does)
python -m src.quantum.make_frames --source oracle  # a stream of frames with a provenance manifest
```

### 5. Calibration and the glass check on the real wall

Done from the operator panel (Calibrate and Photo test tabs): outline pattern, four-corner re-alignment after the projector moves, a safety margin that shaves projector pixels off every lit region, and a photographed check that the real glass stays dark. See [`docs/standing-light-runbook.md`](docs/standing-light-runbook.md) §3 for the procedure and `python -m src.projector.verify --selftest` for a synthetic run.

---

## How it works

### The quantum model (Superposed Relief)

**Qubits** (little-endian: qubit *q* is bit *q* of the state index; spin +1 is bit 0):

| Register | Size (default bay window) | Role |
|---|---|---|
| facets | 12 (+6 classical) | one per band facet; state = the facet's surface normal `b = (sin τ cos φ, sin τ sin φ, cos τ)` in the panel's own frame. Each panel's flat plateau has τ≈0, so it would be a constant \|0⟩ that nothing couples to: it is a *classical* facet, lit by its deterministic Lambert value |
| polarity | 6 | one per panel (pane), in the state "plus"; controls whether that panel's relief is raised (`h`) or sunk (`-h`) |
| lamp | 2 | four light directions; controls the axis each facet is measured along |
| observe | 2 | four observation axes (γ, χ) on each polarity qubit's Bloch sphere: by default the ring (0°, 0°), (60°, 0°), (60°, 90°), (90°, 180°); `--observations line` restores γ = 0°, 30°, 60°, 90° at χ = 0°, or give four `γ,χ` pairs in degrees |

Facets + polarity = **18 simulated qubits** (12 facets + 6 polarity); the full circuit has **22** (+ 2 lamp + 2 observe). Two band facets per pane is `--n-dirs`' default for six panes: it is the most that keeps the exact statevector within 20 qubits, and the three-panel window the first build used got four per panel (15 simulated, 19 in the circuit). Historical note, so that the numbers are not confused: the first build kept the constant plateau qubits too and counted 18 and 22 *for three panels*; the plateaus are now classical facets, and the six-pane default happens to total 18 and 22 again with a different make-up. An earlier slope computation also halved the slope on the shared edge between two panels, so no tilted facet of one panel touched a tilted facet of its neighbour and the panels were exactly unentangled with each other; fixed, tests pin it.

With two facets a pane, the two slopes of a pane are *opposite diagonals* (about ±55° from the horizontal), not four axis-aligned sides: they still carry the raised-versus-sunk coherence and still react to a light from the left or right, but the picture is coarser than with four. The facets of one pane share a plane with the pane above or below it, so the rail between two stacked windows is a coplanar edge of the facet graph (a weak `+` coupling), and only the two vertical wing edges are creases (`-`).

**The state.** `|Ψ> = 2^(-P/2) Σ_s |s>_B ⊗ P^s |ψ_f>`, where `|ψ_f>` is the raised relief and `P = Z` on every facet of a panel (Z maps Bloch `(x,y,z)→(-x,-y,z)`: every bump becomes a hollow). It is built as: H on each polarity qubit, `Ry(τ)Rz(φ)` on each facet, optional diagonal couplings, then `CZ(B_p, f)` for every facet `f` of panel `p`.

**A frame is one run of the circuit.** The lamp register rotates each facet so measuring Z measures σ·l (`P(lit) = (1 + b·l)/2`: Lambert is Born). The observation register rotates each polarity qubit to read along the depth-sphere axis. The whole register is measured; K photons are K shots (K is the *exposure*: hard grain with few, soft with many). The polarity outcome is that frame's depth decision. The lamp and observe registers are uniform draws (pseudo-random here, physical randomness on a QPU); the quantum content is the facet register, the polarity superposition and the controlled operations between them.

**What the geometry decides.** Only the tilts. A height field (a chamfer of width `band_frac` of each panel, distance measured to the nearest border including the glass edge) gives every pixel a slope. Pixels that slope the same way form one facet, wherever they are, because under a distant light they are lit identically. Tilts are rescaled per panel so `Σ tau² = kappa²`. A panel is one *window* (pane), not one wing: the budget, the polarity qubit and the visibility `V` are per pane, so each of the six panes has the same `V` = ∏cos τ ≈ 0.58 at `kappa` 1.

### Domain Relief (v3)

The per-panel engine is six independent-ish 3-facet systems (two band facets and a classical plateau per pane, each with its own polarity qubit; they are coupled only through the few facet couplings across shared edges). Domain Relief changes what a qubit stands for so that the entanglement follows the geometry:

| Piece | What it is |
|---|---|
| **facet** | a piece of the bevel strip along one polygon edge (a panel's outer edge or a glass pane's edge), about `--seg-len` px long, one slope = one Bloch vector, as before |
| **domain** | `--group-size` consecutive facets around one loop share ONE polarity qubit (one depth decision per small stretch of bevel). The tilt budget is per domain, so V = cos(τ)^m does not shrink as the wall gets more facets |
| **graph** | domains are nodes; domains whose facets touch are joined, within a plane (coplanar) or across a shared edge between panels (crease). The polarity qubits are coupled along it: `exp(-iθ ZZ/2)` on \|+⟩\|+⟩, a graph-state-like Ising layer, sign `+` coplanar and `--crease-sign` across a crease |
| **lamp lock** | lamp L1 coupled to ONE leaf polarity qubit per *plane* (one in each wing: three, whatever the number of panes) with the crater-gauge signs (a raised bevel facing right lit from the right looks like a sunk one facing left lit from the left): the light side and the depth are entangled |
| **backend** | up to 20 qubits an exact statevector; beyond, Aer's matrix-product-state simulator evolves the state and `src/quantum/mps.py` samples it *exactly* (one polarity outcome per frame, then K photons given it: no post-selection) |

**Measured at the show's defaults** on the six-pane bay window (`python -m src.quantum.relief_report --domain`: 64 facets + 64 domain qubits = 128 simulated, 132 in the circuit, 74 domain edges of which 4 are creases, bond dimension capped at 32 with a norm deficit of 1.8e-2; weak coupling 0.3 along a boundary, strong 1.0 across a crease, leaf tilt 0.3; the figures are in `runs/relief/`, the three-panel ones in `runs/relief_three_panel/`):
- 55 of 74 domain edges carry entanglement between their two domains, and **all 4 creases do** (largest negativity 0.28; at the first domain-relief defaults none did on this window). The dephased control has none.
- Lamp + three leaf domains (one per wing, domains 8, 29 and 50): **Mermin value 6.94 against a local bound of 4** (dephased depth: 1.0). No lamp-depth *pair* violates CHSH (best 1.85) and no depth-depth pair is certified to (best 2.008, which is inside the matrix-product truncation error of 1.8e-2): monogamy. Lock the lamp to *every* domain with a crater-gauge sign (37 of them) and the Mermin violation disappears (0.75; `tests.test_domains`).
- The signed domain graph is *balanced* on its own, so there is no frustration from creases alone, and the three-leaf lock does not frustrate it (0 frustrated cycles with or without the lamp). The Necker-style competition is a property of the lock you choose, not of the geometry.
- Frame-level interference signature: measured on the three-panel window (not repeated on the six-pane one), the K-photon frame statistics of the coherent state and of control B were identical at the decided observation and differed at the equatorial ones by up to 0.12 against a photon noise of 0.06. Single frames do not show it; an ensemble of a few hundred does.
- Speed: Aer builds ONE state (about 2-3 s, about 5 s for the whole domain session); the lock, the light and the observation are single-qubit gates on its tensors. A look costs about 0.07 s (0.04 s for the per-panel engine).

### The parity game: the frames certify the state

Every look can be one round of a four-party Mermin (GHZ-style) game between the lamp and the depth qubits of three *leaf* domains, one in each wing of the bay window (the left, centre and right: not one per pane, which would lock the lamp to six qubits and lose the violation) (`src/quantum/parity_game.py`, key `G`, `--game`). If a wall has more than three planes the leaves are the left-most, the middle and the right-most. A round draws an input S, a uniformly random *even-size* subset of the four parties. A party in S reads its qubit along its setting A', the others along A (the axes are the numerically optimised Mermin frames of the state). Everyone gets a ±1 outcome, and the round is won when `(product of the four outcomes) × (−1)^(|S|/2) = +1`.
- The eight inputs are the eight terms of the Mermin operator, so a round is won with probability `(1 + M/8)/2` for a state with Mermin value M. **No classical strategy wins more than 75%** (`classical_bound` proves it by brute force over all 4⁴ strategies); a perfect GHZ state wins every round. At the defaults M = 6.94, so the prediction is 93.4%, and 400 rounds gave **370 won (92.5%)**, 8.1 σ above the classical bound. Dephased depth cannot exceed 75% (tested).
- The lamp is read along A or A′, a rotated axis: this is the lamp read in X. Its outcome then *chooses the light side by feed-forward*, so the facets keep no record of the lamp and its coherence with the depth survives; each leaf's depth-sphere dot sits on its own setting; every other domain keeps the frame's ordinary observation. The look is the picture those outcomes make; the operator page and the caption show the round (inputs, outcomes, won or lost) and the running tally against 75%. One look is one bit; the win *rate* over many looks is what certifies.
- **What the game cannot do:** the rate is `(1 + M/8)/2` for THIS state after every dephasing the facets and neighbours cause, and exceeds 75% only if M > 4.
- **It does not survive the hard seam.** Put the leaves on a seam whose facets sit at the equator and visibility is 0: the predicted win rate falls from 76% (seam_mix 0) to 56% (seam_mix 1), below the classical bound, while leaves away from the seam stay at 93-94% (`seam_benchmark`).

### Seams: where the entanglement between edges lives

- **One facet per depth qubit** (`--group-size 1`) raises each qubit's own visibility V from about 0.77 to 0.88; with **weak coupling along a boundary and strong across a crease** (`--pol-coupling 0.3`, `--crease-coupling 1.0`, `--seam-coupling 1.0`) every crease is entangled. Measured on the six-pane window, the lamp–depth Mermin value is 2.5 at the first domain-relief defaults (group size 2, uniform coupling 0.8: no crease entangled), 4.6 with one facet per qubit and uniform coupling 0.8 (2 of 4 creases), 6.3 with weak boundaries and strong seams (4 of 4) and 6.9 once the leaves' tilt is lowered to 0.3 (strong coupling everywhere lowers it: monogamy again).
- **Seam regime** (`--seam-mix`): only the facets on a crease move toward the equator. At seam_mix 1 each seam facet is maximally entangled with the rest of the wall (1.00 bits against 0.29 in the relief regime, and 0.27 for other facets; a pair across a crease 2.0 bits against 0.57), the seam domains have visibility exactly 0 yet keep a GHZ stabilizer of 0.51–0.56, and the classical cost at bond dimension 32 rises, by much less than on the first three-panel window: norm deficit 1.8e-2 → 2.5e-2 (about 1.4×; the three-panel window went from 1.3e-3 to 1.6e-2, 12×, but its relief state started far easier for a bond-32 stand-in). The price is that the creases' *polarity* entanglement (negativity 0.28) is destroyed, and the game cannot use those qubits (above).
- **Honest limit:** the seams of a wall are a thin skeleton of lines, and an MPS handles a thin skeleton. On tiled walls (n × n panels, every shared edge a seam: up to 368 qubits) the norm deficit at bond dimension 32 stays between 8e-3 and 4e-2 for the relief seam and between 4e-2 and 1e-1 for the equatorial one, and does not grow from 200 to 368 qubits; the equatorial seam costs about 2-3× the relief seam there (in these tiled walls every panel is its own plane, so the game's three leaves are the left-most, middle and right-most). The lattice benchmark in which *every* facet is equatorial and CZ-coupled in 2D (which an MPS cannot do beyond ~75 qubits) is a different object: that is a wall with no visible relief.

### Dynamics: entanglement that grows in time

`--evolve-steps T` (key `E` adds one) runs T kicked-Ising steps on the facet graph after the facet register is prepared and before the polarity attaches: `exp(-iθ_zz ZZ/2)` on every facet coupling and `exp(-iθ_x X/2)` on every facet, per step (`ReliefSpec.floquet`; the exact numpy state, the Qiskit circuit and the MPS agree, tested). Each step entangles further and moves the facets' normals, so the relief evolves. Measured on the six-pane bay window at bond dimension 32 (`seam_benchmark`): the mean bond entropy rises from 1.21 bits to 2.3–2.4 within 6–8 steps (θ_zz 0.7, θ_x 0.5) or 2.7–2.85 (1.0, 0.9), and the norm deficit of the best bond-32 classical stand-in grows from 1.8e-2 to 7.6e-2 (4×) or 1.4e-1 (8×), then saturates (finite size and the cap). The relative growth is smaller than the 70–100× the three-panel window showed, because the six-pane state is already 1.8e-2 off at bond 32 before any step. A static state never gets harder; this one does. **It costs the game:** the same steps spread the leaf qubits' coherence over the facets, so the Mermin value and the predicted parity-game win rate fall with them (static 6.94 → 93%; 1 step 4.94 → 81%; **2 steps 3.79 → 74%, below the classical 75%**; 3 steps 4.43 → 78%; 4 steps 4.44 → 78%; 6 steps 4.33 → 77%; 8 steps 5.13 → 82%; the Mermin value is a numerical maximum over measurement frames, so the series is not exactly monotone), and the crease entanglement (negativity 0.28) is gone by step 3 (4 of 4 creases entangled static, 0 of 4 at steps 3-4, partly back at 6-8). Entanglement that grows in time and the lamp–depth Bell test pull against each other in this design: a performance can have the game early or the dynamics late, not both at once, unless the leaves are decoupled from the evolved facets (not built). The echo engine (`otoc-echo-v1`) is the same kicked-Ising family; its payload and the echo-tap-to-tilt mapping remain in `domain_moth.py`, **not run on Moth and not wired into the show**.

### Things that were wrong and are fixed (read before trusting older numbers)
- Control B on the matrix-product backend kept one hidden depth for a whole frame, where the exact engine redraws it per photon (frame-to-frame spread 0.14 against 0.06). Fixed, with a frame-level regression test.
- `spec_from_domains(order="loop")` was silently ignored (a local variable shadowed it), so every earlier number used the x-sweep order. Measured now: sweep 1.5e-3 against loop 5.2e-3 norm deficit at bond dimension 32, so the sweep is the better order and the default; the docstring that promised loop order would win was wrong.
- The lamp read in X (`I`, the exact engine's experiment) makes the lamp *control the light coherently*, so the facets record the light side and dephase the lamp–depth coherence. The Mermin numbers above are for the *feed-forward* lamp (read first, then it picks the light), which is what the parity game and every Z-read frame do. The two semantics are separate in the code (`lamp_mode` against `game_axis`). The lamp read in X now also runs on the matrix-product backend (tested against the exact engine).

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
| `capture/labels.py` | `labels.json` schema, validation, defaults, the built-in six-pane bay window (`bay_window_labels()`; `stacked=False` is the three-panel layout of the mock), the default calibration and run folders |
| `capture/pen_tool.py` (+ `pen_tool.html`, `pen_draw.js`, `pen_geom.js`, `projector.html`) | the drawing tool: browser UI served locally, shared-edge snapping |
| `geometry/planes.py` | `Scene`: regions, masks, plane normals, windows, relief field, panel adjacency; `build_scene(labels)` |
| `geometry/facets.py` | **facets**: orientation classes (`auto_n_dirs`: how many per panel keep the circuit exact), tilt budget, edges between facets; classical plateau facets; per-panel slope field (no differencing across a seam) |
| `geometry/domains.py` | **domains**: spatial bevel-strip facets, depth domains, the domain graph (coplanar / crease), crater-gauge lock signs, frustration of the signed graph |
| `geometry/relief.py`, `geometry/depth.py` | distance-to-border field; depth coordinate along side panels |
| `mask/build_mask.py`, `mask/masked_blur.py`, `mask/apply_black.py` | rasterise polygons; blur that never leaks light into glass; the final exact-zero multiply |
| `bounds/windows.py` | brightness window `[lo, hi]` per plane and depth |

### `src/quantum/`: states, sampling, witnesses, Moth

| File | What it does |
|---|---|
| `relief_state.py` | **the quantum core**: `ReliefSpec`, `facet_state`, `lift_polarity`, `ReliefState.draw`, light rig, Qiskit `reference_circuit` and `full_circuit` |
| `relief_witness.py` | visibility, stabilizer, fidelity witness, CHSH scan, sampled witness run, scaling law |
| `domain_state.py` | **domain relief**: `spec_from_domains`, the exact and matrix-product backends (`make_state`), `leaf_domains` (one leaf per plane, at most three) |
| `mps.py` | matrix-product states on Aer: exact conditional sampling, reduced density matrices (with operator insertions, and between two different states), superposition of states (bond dimensions add), single-site gates, bond dimensions and entropies |
| `parity_game.py` | the Mermin parity game: frames of the state, rounds as frames, win rate against the classical 75%, brute-force classical bound |
| `seam_benchmark.py` | seams, the game and Floquet time against the classical cost; tiled walls; `runs/seams/seams.png` |
| `domain_witness.py` | domain certificate: visibility, entangled edges (negativity, CHSH) incl. creases, lamp–depth CHSH and Mermin, frustration, controls |
| `ground_state.py` | ground state of the geometry-set transverse Ising model as a fitted circuit; classical ground-state degeneracy |
| `domain_moth.py` | payloads: polarity register via QDrive targets, patch tomography, echo-driven dynamic relief |
| `classical_baselines.py` | two-branch sampler, excitation truncation, MPS bond dimension vs size: where classical simulation stops being cheap |
| `relief_report.py` | figures and `report.json` for a talk |
| `moth_client.py` | Moth Atlas API client: gated submit, never retries a POST, saves job id and raw result first |
| `moth_engines.py` | per-engine fit, payload builders (tomography, QDrive, echo, Qpixl, shader), defensive parsers, scoring against exact moments |
| `solver.py` | submit calibrated targets to graph-v1 or QDrive and score what came back (classical flow) |
| `qdrive_plan.py`, `qdrive_rounds.py`, `motte_mock.py` | QDrive in small chained, scored rounds (plan, resume, assemble; `--engine local` rehearses it with no credits), and the paper-based local mock of the engine they are tested against |
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
| `qdrive_lab/` | the QDrive field lab: trial scripts, offline checks (`python qdrive_lab/test_offline.py`) and every raw result, see `docs/qdrive-field-notes.md` |
| `runs/` | generated outputs: `calibration_bay_4x4/` (the default scene's targets and payloads), `bay_run/` (where its Moth results go), `relief/`, `seams/`, `moth/`, and the first build's three-panel record (`calibration_5x4/`, `first_run/`, `relief_three_panel/`, `seams_three_panel/`) |

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
| `G` | **parity game** (domain engine): some looks become rounds of a Mermin game; the tally must beat 75% |
| `E` | **evolve** (domain engine): one more kicked-Ising step on the facet graph; the entanglement grows |
| `B`, `T`, `S`, `O`, `D`, `V` | blackout, test patterns, snapshot, overlay, demo, re-solve |

The science section also shows, per panel, a **depth-sphere dot** (where the observation landed: top = decided raised, bottom = decided sunk, middle = undecided/flat), the tilt budget and per-panel visibility, and the last witness certificate. The header chip always says what the frames come from: `local reference circuit`, `Moth QDrive circuit`, or `classical rehearsal`.

`D` starts the demo (seven beats): one flat wall → the wall as qubits → photons (Lambert is Born) → bump, hollow and flat → control B beside the quantum state → the visibility law → an honest scorecard that lists what has and has not been run on Moth.

---

## Moth engines and spending credits

**QDrive in rounds.** One QDrive job per component can stall; `python -m src.quantum.qdrive_rounds --help` sends the same plan as small chained jobs (a spend decision per job, each scored before the next), keeps every piece on disk, and assembles the show's `qdrive/circuit.qasm` from them. `--engine local` rehearses it with no credits. See `docs/qdrive-solution-design.md` section 8.


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

**What the default scene sends** (all of it previewable with no key and no credits):

| Path | Payload for the six-pane bay window | Credits |
|---|---|---|
| `python -m src.quantum.solver graph-v1` | 20 qubits (the cap), 45 edges, 20 `<Z>` + 45 `<ZZ>` operations (from `runs/calibration_bay_4x4`) | 5 |
| `python -m src.quantum.solver qdrive` | one job, 20 qubits, 66 targets: sendable (the client refuses above 100 targets and warns above 30), **but every 19-qubit QDrive job in the lab timed out**, so use the rounds below | 1 |
| `python -m src.quantum.qdrive_rounds plan` | two independent components (patches + lamps: 14 qubits; polarity: 6), 10 chained jobs of 1-2 layers | 10 |
| `python -m src.quantum.moth_engines build tomography` | the 18-qubit per-panel circuit as OpenQASM 2, with the qubit pairs that carry the physics | 1 |
| `python -m src.quantum.moth_engines build qdrive-facets` | the 12-qubit facet register, 62 targets cut into jobs of at most 24 | 1 per job |
| `python -m src.quantum.domain_moth build ...` | polarity register of the domain relief (builds with `--group-size 2`, 34 domain qubits, *not* the show's 64: a 64-qubit QDrive job is far beyond anything that has run), a patch of at most 20 qubits for tomography, the echo | 1 each |

**The first day the servers are back** (a suggested order, cheapest and most informative first; every command previews unless you add `--approve-credits N`): (1) `python -m src.quantum.qdrive_rounds canary --out runs/bay_run --approve-credits 1` tells a service problem from a payload problem for one credit; (2) `moth_engines build tomography` and send it: it verifies the circuit the show actually runs, 1 credit; (3) `qdrive_rounds run --max-steps 2` for the first two rounds, then `qdrive_rounds assemble`; (4) only then `solver graph-v1` (5 credits). `python -m unittest tests.test_moth_pipeline` rehearses every one of these paths against a fake Moth server on localhost (payloads accepted, jobs recorded before polling, circuits downloaded without the key, results parsed and scored, the show switching to the returned circuit, nothing paid twice); if it passes, the first real call differs only in who answers.

---

## Tests

```bash
python -m pytest tests -q                   # everything: 316 tests, about 7 minutes (pytest is in the venv; unittest runs the same files)
for m in test_fast test_projector test_ui test_studio test_complementary test_pen_geom test_frames test_payloads test_moth_client \
         test_relief test_moth_engines test_domains test_qdrive_plan test_qdrive_rounds test_moth_pipeline; do
  python -m unittest tests.$m
done
python -m tests.check_port_matches_mock     # the src/ port still reproduces the original three-panel mock exactly, and the six-pane default is the same wall
python superposed_relief_refsim.py          # the handoff's reference numbers
python -m pytest qdrive_lab/test_offline.py -q   # the QDrive lab's scorer against known states (8 more tests)
```

| Module | Covers |
|---|---|
| `test_domains` (60) | domain geometry on the six-pane default (creases touch, rails coplanar, per-domain budget, one leaf per plane and at most three, balanced vs frustrated graph), circuits (Qiskit = numpy, Aer sampling = exact with the lamp lock), the MPS tools against the statevector, exact vs MPS frames (both controls, lamp in X), the parity game (classical bound, exact joint distribution, win rate), the lamp lock's monogamy, seams and Floquet dynamics, certificates (exact = MPS, operator insertion = a real lamp qubit), ground state, Moth payloads, baselines, the session |
| `test_relief` (46) | refsim table, Lambert = Born, conventions, Qiskit circuit = numpy state, **Aer sampling of the full circuit = exact distribution**, the default scene (six panes, three planes, 12 + 6 = 18 qubits) and the three-panel layout it replaced, facets, couplings, visibility law, which-depth/visibility, controls, witnesses, no toggles in performance mode, honest provenance |
| `test_moth_pipeline` (16) | **the whole Moth path against a fake Moth server on localhost**: the default calibration and its payloads, the solver for graph-v1 and QDrive end to end (job recorded, circuit downloaded without the key, parsed, scored, loaded by the show), the show's re-solve gate over real HTTP, every engine builder's payload accepted by the fake API's schema check, tomography scored against itself, the domain payloads, a calibration for another scene refused, a new payload never inheriting an old result, and every README command (previews and the local rounds rehearsal) run as a subprocess |
| `test_qdrive_plan` (15), `test_qdrive_rounds` (22) | QDrive growth plans, cutting into chained rounds, resuming, failing, assembling, provenance (against an ideal fake engine) |
| `test_moth_engines` (12), `test_moth_client` (19) | payloads valid and priced, QASM2 round-trip, parsers and scoring, the 201-target payload refused, echo tap mapping; the client's safety properties (never retries a POST, key never printed, gate before any HTTP) |
| `test_ui` (27), `test_studio` (37) | key table, captions, session actions, demo, calibration, spend gate, the HTTP servers, the studio's five steps with fake child processes |
| `test_projector` (20) | homography, warp, exact-black guarantee, photographed check (synthetic) |
| `test_frames` (14), `test_complementary` (15), `test_fast` (4), `test_payloads` (5), `test_pen_geom` (4) | the classical family: sampler, oracle, frames, composers (six panes and the original three), the polarity stand-in |

Tests never read your real `.env` and never reach Moth: the Moth tests patch the API address to a localhost server and set a fake key in the environment.

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

**Done and tested:** the relief engines (per-panel and domain) on the six-pane default window, their circuits and witnesses; the parity game, the seam regime and the kicked-Ising dynamics; the show, controls, demo and studio; the classical pipeline; the Moth client and engine builders and the whole Moth path against a fake server; offline. The whole suite (`tests/`, 15 modules, 316 tests) and the mock-port check pass.

**Not verified (needs you or credits):**
- Any real Moth result for the relief design (either engine). Tomography, echo and QDrive result shapes are mostly "not documented yet", so parsers are defensive and raw results are saved first. The domain payloads (`domain_moth`) are built, validated locally and **not sent**; the qubit cap of `tomography-api-v2` is undocumented, which is why only patches of ≤ 20 qubits are offered.
- The domain relief on a projector: the frames are simulated and rendered, the glass-black checks pass, but the look on a real wall (six or so times as many facets, each with its own bevel strip) has not been judged by eye.
- The parity game's win rate is the *simulated* state's prediction, confirmed by sampling the simulation (370 of 400 rounds against a predicted 93.4%). It has not been measured on a device.
- The kicked-Ising evolution is wired into the show (`E`, `--evolve-steps`), but the echo engine's own taps (`otoc-echo-v1`, `domain_moth.dynamic_specs`) are a module and a test, not a live mode, and not run on Moth. Evolution and the parity game conflict: the game falls below 75% by step 4.
- QDrive has returned circuits for small jobs only (a 2-qubit Bell state, a 3-qubit GHZ state, chained jobs: 21 of 28 jobs completed on 2 and 4 Oct 2026) and never for a whole scene: every 19-qubit job and every 8-body target timed out. Chaining through an output asset UUID works, the docs' PNG/JPEG-only asset limit does not hold, and mixed Pauli words work as a mapping (all recorded in `docs/qdrive-field-notes.md`); what is untested is anything near 20 qubits, which is why the default scene is sent in rounds of two components (14 + 6 qubits).
- `local_echo` is *our reading* of the echo engine's description, not its definition.
- A real projector, a real window, real photos. The glass-check thresholds were reasoned, not calibrated.
- The studio's Solve and Perform steps are still the classical QDrive flow.

**What is and is not quantum** (an audit of the code, with the numbers):
- The facet register, the polarity superposition and the controlled operations between them are quantum content, simulated exactly or as a matrix-product state. The lamp and observation registers are uniform draws, equivalent to coin flips, except where the lamp is locked to a polarity qubit (`--lock`) or read in X (both backends).
- Per-panel engine (the following two figures were measured on the first three-panel window and not repeated on the six-pane one): 72% of the variance of per-frame expected brightness is the lamp world (a classical draw); the polarity outcome (a near-fair coin) carries the rest. What the wall shows, the interference included, is reproduced by a classical sampler: a *two-branch sampler* draws exact frames of the uncoupled product relief at N = 5000 in 0.04 s, and the scaling law that keeps V constant (τ = κ/√N) keeps the state at ~0.25 excitations per panel, so excitation truncation (k = 2) is also exact to 0.99.
- Domain engine: capping the matrix-product bond dimension at 32 (what the show does) leaves a norm deficit of 1.8e-2 at the default 128 qubits of the six-pane window (1.3e-3 at the 92 qubits of the three-panel window; at the earlier 69-qubit configuration: 1.3e-2 at bond 8, 6e-3 at 16, 7.6e-4 at 64: the Schmidt tail is long, so an *exact* state needs more than 128), so the show's state is *approximately* classically simulable to about 2% and not cheaply to machine precision. Making the seams equatorial raises the classical cost about 1.4× (12× on the three-panel window, which started far easier) and destroys what the game needs; the kicked-Ising evolution raises it 4–8× over 6–8 steps (70–100× there), also at the game's expense; tiled walls of up to 368 qubits show no growth in the seam-only regime. The MPS benchmark shows where even the approximation stops: a lattice of domains in the relief regime passes bond dimension 128 at ~100 qubits, and in the graph-state regime (facets at the equator, CZ-like couplings) at ~75, with a 7% norm deficit. That is the regime where hardness is plausible, and it is **not** the regime the show runs in (it needs V → 0). No quantum advantage is claimed anywhere.

---

## Further reading

| File | Contents |
|---|---|
| [`docs/standing-light-v3-domain-relief-handoff.md`](docs/standing-light-v3-domain-relief-handoff.md) | the audit of v2 (what was and was not quantum, with numbers) and the first domain relief. **Its defaults and numbers are superseded by the README's Domain Relief section** (parity game, seams, dynamics, corrections) |
| [`docs/standing-light-v2-superposed-relief-handoff.md`](docs/standing-light-v2-superposed-relief-handoff.md) | the v2 design: concepts, formulas, acceptance tests, scaling law (wins over the first handoff for the quantum design) |
| [`docs/standing-light-runbook.md`](docs/standing-light-runbook.md) | projection, calibration, performance mode, the glass check; §7 Superposed Relief with the findings that corrected the handoff; §8 Moth engines |
| [`docs/standing-light-handoff.md`](docs/standing-light-handoff.md), [`docs/standing-light-project-spec.md`](docs/standing-light-project-spec.md) | the original handoff and spec (classical family, targets, A/B protocol) |
| [`superposed_relief_refsim.py`](superposed_relief_refsim.py) | runnable numpy reference simulation |
| `mock/` | the original classical look test (`standing_light_mock.py`, its figures); `tests/check_port_matches_mock.py` checks the `src/` port against it |
| [`docs.mothquantum.com/docs/engines`](https://docs.mothquantum.com/docs/engines) | Moth engine documentation |
