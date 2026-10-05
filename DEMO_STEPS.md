# Demo steps: setup, Moth tomography, three runs

All commands run from the project folder. `runs/myroom` is the folder name used for your scene; use any name, but use the same one everywhere.

```bash
cd /Users/harrietfisher/Moth_Application/chiaroscuro
source .venv/bin/activate
```

The plan in one line: **draw the scene, send one 1-credit Moth tomography job, score it, then record three runs.**

| | What | Moth? | Credits |
|---|---|---|---|
| Part 1 | Set up: key check, draw the scene, calibrate | no | 0 |
| Part 2 | Prepare the Moth job (build and preview, nothing sent) | no | 0 |
| Part 3 | Send the job, then score it | **yes** | **1** |
| Run 1 | The wall is a measurement (panel engine) | shows the Moth result | 0 |
| Run 2 | Proof it is quantum: the parity game (domain engine) | no, local | 0 |
| Run 3 | Entanglement that grows, and what it costs (domain engine) | no, local | 0 |
| Part 5 | Optional: QDrive canary | yes | 1 |

---

## Part 1: Set up (free)

1. **Check the key exists** (this only checks the file, it does not print it):
   ```bash
   test -f .env && echo "key file present" || echo "MISSING: create .env with MOTH_API_KEY=..."
   chmod 600 .env
   ```
2. **Start the studio** (no `--allow-spend`; nothing in steps 1-3 sends anything):
   ```bash
   python -m src.studio --project runs/myroom
   ```
3. **Step 1, Draw the shapes.** Press the button, draw on the projector's own screen: the 6 panels with a glass pane in each, then press **Save**. Give the top and bottom window of one wing the **same plane id**, so the rail between them is not a crease.
   - No projector today? Press **Use the built-in demo scene** instead. It is the same six-pane bay window.
4. **Step 2, Calibrate targets.** Press it and wait up to a minute. It is free. It is required even though relief does not use its targets, because step 3 is blocked until it is done.
5. **Stop here.** Do not press step 4 (Solve on Moth) or step 5. They only send the old classical flow (`graph-v1` 5 credits, QDrive 1 credit), and tomography is not part of the studio.

## Part 2: Prepare the Moth job (free, nothing sent)

6. Build the tomography payload **from your own drawing**. The payload must match the circuit the show runs, so use the show's default relief settings (kappa 1.0, entangle 0.8) in Run 1:
   ```bash
   .venv/bin/python -m src.quantum.moth_engines build tomography --labels runs/myroom/labels.json
   ```
   Built-in scene instead: leave off `--labels`.
7. Preview the exact request, price and checks. This sends nothing:
   ```bash
   .venv/bin/python -m src.quantum.moth_client preview runs/moth/payload_tomography-api-v2_tomography.json --engine tomography-api-v2
   ```
   Expect: engine `tomography-api-v2`, **1 credit**, an 18-qubit OpenQASM 2 circuit, provider `aer`.
8. **Check your credit balance on the Moth dashboard** (the code cannot read it). The lab's earlier 15 credits are spent.

## Part 3: Use Moth (1 credit; send only when you are ready)

9. **Send it.** `--approve-credits 1` must equal the price shown, or the command refuses. The job id is saved before polling and the raw result before parsing, so a paid result is never lost:
   ```bash
   .venv/bin/python -m src.quantum.moth_client send runs/moth/payload_tomography-api-v2_tomography.json --engine tomography-api-v2 --approve-credits 1 --out runs/myroom/solve/tomography
   ```
10. **Score it** against the exact numbers from the local circuit. This writes `score.json`, which the demo's final slide reads:
    ```bash
    .venv/bin/python -m src.quantum.moth_engines score runs/myroom/solve/tomography --labels runs/myroom/labels.json
    ```
    It prints, per panel, Moth's measured depth-qubit ⟨X⟩ (the visibility) next to the exact value (about 0.58 at kappa 1), plus rms errors. Score before you start Run 1, or restart the show, because the scorecard is built when you press `D`.

**What this is and is not.** Moth's tomography measured the exact circuit the show simulates. It runs on `aer`, Moth's simulator, not a quantum computer. This engine has never been run before, so its result shape is a guess: if scoring says it cannot find Bloch vectors, the raw file is safe in `runs/myroom/solve/tomography/raw_result_*.json`. Send it to me and I will fix the parser; do not resend.

**If it fails.** `422` or a size error: build a smaller patch instead (`python -m src.quantum.domain_moth build patch-tomography`). A timeout: do not resend; check the job with `MothClient().status(job_id)` first.

---

## Part 4: The three runs (studio step 3, then **Stop** between runs)

For every run: in the studio, open the **Show options** under step 3, set what the run says, press step 3. The show opens in the browser. Drag the **projector window** (`/output`) to the projector display and press `F` for fullscreen. Press `T` for the test patterns and check the glass is black before recording.

### Run 1: The wall is a measurement (panel engine)

**Settings:** engine `panel` (default). Leave every other option at its default. Hardness (K) is set during the run.

1. Press `D` to start the demo. It walks through seven beats and sets the controls for you.
2. Beats 1-2: flat wall with black glass, then the overlay, showing the wall as qubits.
3. Beat 3 (photons): hardness sweeps from K=1 up. Say: Lambert's cosine law is the Born rule.
4. Beat 4 (bump, hollow, flat): K=48, auto-cycle. Watch the depth-sphere dots on the operator panel: pole = decided bump or hollow, middle = flat.
5. Beat 5: press `W` (**Witness run**). It always measures the quantum state and prints the controls' numbers in the same list, so it does **not** change when you toggle A or B. Read it as a table: every panel's fidelity is above its bound (certified entangled), while "control dephased" and "control noise" are below it. Then the demo switches on **B: dephased depth** and **A: independent noise** on the wall:
   - **A** is visible: the wall turns to structureless speckle.
   - **B** is not visible in a single frame. Every patch keeps the same statistics. Say that this is the point: what B removes is interference, which the witness certifies and single frames cannot show. The header chip changes to `CONTROL B`.
6. Beat 6: the visibility law slide.
7. Beat 7: the scorecard slide. **This is where the Moth tomography line appears.** It says what has and has not run on Moth.

### Run 2: Proof it is quantum, the parity game (domain engine)

**Settings:** `engine = domain`, `game = on`, `game_fraction = 1`. Leave the rest at default. The show takes about 5 s to build its 128-qubit state.

1. Press `W`. Show: 55 of 74 domain edges entangled, all 4 creases, lamp-plus-leaves Mermin value **6.94 against a local bound of 4**.
2. Press **reset tally**. Set auto-cycle to about 1 s and let about 40 rounds run. Expect a win rate around **92%** against the classical ceiling of **75%**.
3. Press `M` (dephased depth), reset the tally, run again. Expect the rate to fall to **75% or below**. Press `M` again to turn it off.
4. Optional: `N` (independent noise). The wall turns to speckle. (The line under the sliders shows the circuit's size and visibility budget in the relief engine, not a coherence score, so it will not change.)

### Run 3: Entanglement that grows, and what it costs (domain engine)

**Settings:** `engine = domain`, `game = on`, `game_fraction = 1`, `evolve_steps = 0`, leave `evolve_zz 0.7` and `evolve_x 0.5`.

1. Start static. Reset the tally, run about 40 rounds: about 93%.
2. Press `E` once (one kicked-Ising step), **reset tally**, run 40 rounds: about 81%.
3. Press `E` again (2 steps), reset tally, run: about **74%, below the classical 75%**.
4. Press `E` to step 4 and run again: about 78%.
5. Press **static** to return to step 0.
6. Say: this is a real trade-off in this design. The state gets harder to simulate classically (the matrix-product stand-in's error rises 4-8x over 6-8 steps), but it stops winning the Bell game. You cannot have the proof and the growth at the same time, unless the game leaves are decoupled from the evolved facets. That is not built.

Run 3 is the same engine as Run 2 with `E` pressed, so you can also do both in one take.

---

## Part 5 (optional, 1 credit): QDrive canary

QDrive has failed every job since 4 Oct 22:13 (a known-good job also failed), so it may be down. One credit tells you whether it is back:

```bash
.venv/bin/python -m src.quantum.qdrive_rounds canary --out runs/myroom/solve --approve-credits 1
```

If it passes, the only claim you can make is "QDrive is working again". It does not drive the wall.

---

## What to say honestly

- The wall's state is simulated locally, and everything at these sizes is classically simulable. No quantum advantage is claimed.
- Moth's part: independent tomography of the exact circuit (`aer` simulator), plus the QDrive lab work and its documented findings.
- No Moth circuit has driven the wall yet. QDrive built a Bell state and a GHZ state; every 19-qubit job timed out.
- The parity game rate is a prediction of the simulated state, confirmed by sampling the simulation. It has not been measured on a device.

## If something goes wrong

- **Studio step 3 is greyed out:** step 2 is not done, or the drawing changed after it. Run step 2 again.
- **Show will not start, or the studio says "refused":** read the line it prints; most are scene and calibration mismatches.
- **The Parity game, Evolve and reset-tally buttons are greyed out:** you launched the panel engine. Stop and relaunch with `engine = domain`.
- **Glass not black on the wall:** blackout (`B`), recalibrate the projector corners on the Calibrate tab, and do not record.
