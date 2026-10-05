# Demo steps (updated 5 Oct 2026 for the Aer path, the Moth engine cards and the echo)

All commands run from the project folder. `runs/myroom` is the project for your scene (six panes, 22 qubits; drawing and targets already done). Use any name, but use the same one everywhere.

```bash
cd /Users/harrietfisher/Moth_Application/chiaroscuro
source .venv/bin/activate
```

**The story in one line:** *draw the wall, execute its quantum circuit, perform from the measured shots, show the controls that remove the quantum part, then show the entanglement growing and what it costs.* Moth's engines are the state (QDrive), the verdict (tomography) and the dynamics (echo); Aer executes and measures. **Nothing needs Moth to work on camera**, and the Moth service has failed every job since 4 Oct about 21:13Z, so the script below is built so that the Moth parts are cards you can show without sending.

| | What | Moth? | Credits | Takes |
|---|---|---|---|---|
| Part 0 | Pre-flight (off camera) | no | 0 | 10 min |
| Part 1 | The studio, steps 1-3 | no | 0 | 1 min |
| Part 2 | **Solve: execute the circuit on Aer** | no | 0 | 1 min (the run is 7 s) |
| Part 3 | Moth cards: tomography, QDrive facets (show the request; do not send) | cards only | 0 | 1 min |
| Part 4 | Echo: dynamic relief on Aer (optional) | no | 0 | 1 min (the run is 13 s) |
| Part 5 | **Perform from the Aer run** and the scripted demo (key `D`, nine beats) | no | 0 | 5 min |
| Part 6 | Echo performance (optional) | no | 0 | 1 min |
| Part 7 | The parity game and the growing entanglement (domain engine) | no | 0 | 4 min |
| Part 8 | Optional: really send tomography to Moth | **yes** | **1** | only if the service is back |

---

## Part 0: Pre-flight (off camera, about 10 minutes)

1. **Free the machine.** Quit browsers tabs you do not need, Dropbox/VS Code syncs, anything heavy. The Aer run and the domain show use several cores; with other heavy jobs running, steps that take 7 s can take minutes.
2. **Start the studio** (no `--allow-spend`; the chip at the top should read *Moth submit: preview only*):
   ```bash
   python -m src.studio --project runs/myroom
   ```
   Steps 1-3 should already be green and **Next: Solve** offered. If step 2 or 3 says *stale*, press step 2 (it takes up to a minute) so it is green before you film.
3. **Reset the Show options** in step 3 (open **Show options**, press **Reset options**). The Aer and echo engines refuse a *domain* engine setting (they say so); a leftover `engine = domain` from an earlier take is the most likely thing to break the first minute.
4. **Rehearse the whole script once.** Do Parts 2 to 7 quickly. Then **delete the saved runs so the take starts clean** (or keep them: a saved result shows as *done*, and "Review circuit" says *a run for exactly this is already on disk*, which is fine but less dramatic):
   ```bash
   rm -rf runs/myroom/solve/aer runs/myroom/solve/echo
   ```
   Do **not** delete `runs/myroom/solve/tomography` (it holds the records of the two failed Moth jobs, which the Moth card will list honestly as *earlier jobs with no saved result*).
5. **Windows for filming.** The studio opens one browser tab; each show opens the **operator page** (`/`) by itself. You also want:
   - **`/audience`** (the clean picture and the captions): open it in a second window and make it the one you film full-size.
   - **`/output`** (the projector frame): drag to the projector display and press `F`. No projector today: skip it; `/audience` shows the same wall in a small preview.
   - Press `T` on the operator page to cycle test patterns (black, white, outline). Check the **glass is black** before recording.
6. **Moth go/no-go.** Sending anything to Moth is Part 8 and is optional. Only do it if you have checked your credit balance on the Moth dashboard *and* a 1-credit canary passed (`python -m src.quantum.qdrive_rounds canary --out runs/myroom/solve --approve-credits 1`). If you do not, nothing on camera changes: Part 3 shows the cards.
7. **Zoom the browser** so the log text is readable on video (Cmd +).

---

## Part 1: The studio (on camera, 1 minute)

Show the page top to bottom and say one sentence per step:

1. **Draw the shapes** (done): the six panes and a glass pane in each, drawn on the projector's own screen.
2. **Calibrate targets** (done): the older classical correlation targets (needed because step 3 is gated behind it, not used by the relief).
3. **Rehearse the show** (done): the relief drawn from the exact numpy state. Do not open it now.
4. **Solve: execute the circuit**: this is where we start.

---

## Part 2: Solve, execute the circuit on Aer (on camera)

1. In **step 4** the dropdown reads **Relief on Aer (quantum circuit, local, 0 credits)**. The select beside it should read **facet register: hand-built circuit**. Press **Review circuit**.
2. The card shows (read it aloud): *relief-aer (this laptop)*, **circuit sha256 …**, **0 credits: nothing will be sent**. Open **Show what will run**: *12 facet qubits + 6 polarity + 2 lamp + 2 observe = 22 qubits; 6 flat facets stay classical; 1,000,000 shots on Aer's statevector simulator, once with the lamp register read in Z and once in X; then a check against the reference state and a witness run.*
3. Press **Run on Aer here (0 credits)**. The **Log** opens by itself. It takes about 7 seconds. Point at:
   - `lamp read in Z: ~300,000 distinct outcomes` (hundreds of thousands of measured bitstrings, not probabilities),
   - `check against the reference state: TV 0.0xx (shot-noise floor 0.0xx) … consistent`: **the shots of the executed circuit match the exact state to within shot noise**,
   - `witness run on Aer: … stabilizer 1.000`: the polarity qubit read in X and the facets read in Z are perfectly correlated, while each alone is only the visibility (about 0.58). That is the entanglement signature.
4. Step 4 turns **done**: *Superposed Relief was executed on Aer: 22 qubits, 1,000,000 shots per lamp reading; the shots agree with the reference state within shot noise.* Step 5 turns **ready**.

What to say: *this is a real circuit executed gate by gate, then measured; it is a simulator on this laptop, so it is quantum in what is executed and measured, not in being hard to simulate.*

---

## Part 3: The Moth cards (on camera, do not send)

Keep these as "this is how Moth plugs in". Each card shows the request, its hash and its price; **Cancel** closes it.

1. **Verify on Moth.** Choose **Verify on Moth: tomography-api-v2 (1 credit)** and press **Review payload**. Say: *the same circuit, 18 qubits without the lamp and observe registers, would go to Moth's tomography engine for 1 credit; the result is scored against the exact moments and the polarity visibilities are compared with the Aer witness we just saw.* The card will show **1 credit will be spent** and a note *1 earlier job has no saved result: cece6571*: that is the real failed job from 5 Oct; say so. The send button is greyed out (preview-only studio). Press **Cancel**.
2. **QDrive facet register.** Choose **QDrive facet register (1 credit per job)**, **Review payload**. Say: *Moth's QDrive would prepare the 12 facet qubits from correlation targets, one chained job at a time, each scored against the ideal state before the next is offered; when the chain is finished the Aer run can execute the whole relief on that Moth-prepared state.* Show **job 1 of 5**. **Cancel.**
3. Do **not** press anything that says *Send to Moth*.

---

## Part 4: Echo, dynamic relief on Aer (optional, on camera)

1. Choose **Echo dynamic relief on Aer (local echo taps, 0 credits)**. Three small boxes appear (depth, width, height); leave them blank (4, 4, 4). **Review payload**.
2. Read the card: *4 echo depths on a 4x4 lattice (kick at site 8); the taps scale each facet's tilt and turn its slope direction; one relief circuit per depth (22 qubits) is executed on Aer with 200,000 shots each.* **Run on Aer here.** It takes about 13 seconds; the log shows four lines *depth 1 … consistent; mean tilt 0.xxx* (each depth is its own circuit, checked against its own reference).
3. Say: *a touch on one patch spreads across the wall from one depth to the next; with Moth's `otoc-echo-v1` the taps would come from the engine instead, and the circuits are still executed here.* (The Moth option is in the dropdown for 1 credit; do not run it.)

---

## Part 5: Perform from the Aer run, and the scripted demo (on camera)

1. Press **Open performance (Aer run)** (the first button in step 5). The show builds in about 1 second; the operator page opens by itself. Open **`/audience`** in the filming window.
   - The caption at the bottom of the audience page says *executed gate by gate on the Aer simulator on this laptop (not a Moth result)*, and the operator page's source chip reads **Aer circuit run**. Point at both.
2. On the operator page press **`D`** (demo mode). Then **Space** moves to the next beat. There are **nine beats**; the demo sets the controls for you (never the light, depth or observation: those come out of the circuit):

   | Beat | What is on screen | What to say |
   |---|---|---|
   | 1 | slide *A wall has one shape*; white test pattern | the real window, flat; point out the black panes (the glass is exactly zero) |
   | 2 | overlay of the qubit graph | every patch is a qubit whose state is its surface normal; facet, polarity, lamp and observe registers |
   | 3 | live; hardness sweeps K = 1 upward | Lambert's cosine law is the Born rule; hard grain with few photons, soft with many |
   | 4 | live, K = 48 | watch the depth-sphere dots: pole = decided bump or hollow, equator = undecided and flat |
   | 5a | slide *Same pictures, different correlations* | **press `W` now** (witness run). It measures on Aer (the certificate says *the circuit executed on Aer*): every panel's fidelity above its bound = certified entangled; the dephased and noise controls below it |
   | 5b | live, **control B (dephased depth)** on | not visible in one frame: every patch keeps the same statistics; what B removes is interference (header chip: CONTROL B) |
   | 5c | live, **control A (independent noise)** on | structureless speckle: the panels no longer agree on a light |
   | 6 | slide *How finely can relief be specified?* | the visibility law; no claim of advantage |
   | 7 | slide *What is and is not quantum here* | the honest scorecard: the provenance line (Aer on this laptop), *Nothing has been run on a Moth engine for this scene yet*, and the not-an-advantage lines |

3. Beat 7 shows the **Moth tomography line only if a real Moth run saved `solve/tomography/score.json`** (Part 8). Otherwise it says nothing has been run on Moth; that is the honest state, leave it.
4. Press **`D`** again to leave demo mode. Optional extras while live: **`I`** (read the lamp register in X: the light directions interfere; works because the Aer run saved an X reading), **`[` / `]`** (harder / softer), **`N`** and **`M`** (controls A and B).

---

## Part 6: Echo performance (optional, on camera)

1. Back on the studio page, in step 5 press **Open performance (echo: dynamic relief)** (the second button; it replaces the running show).
2. On the operator page press **`A`** (auto-cycle). Every look steps to the next echo depth. On `/audience` the headline reads *Light from the right, frontal · echo depth 2 of 4* and the caption says *one per echo depth … taps of a local echo (our own reading, not Moth's engine)*; the operator chip reads **Aer circuits, one per echo depth (local echo taps)**.
3. Do **not** press `I` here: the echo run only saved the Z reading and says so.

---

## Part 7: The parity game and the growing entanglement (domain engine, on camera)

These use the **domain engine** (128 qubits as a matrix-product state), opened from step 3, not step 4: **Solve on Aer does not cover it** and will refuse if you leave the engine set to domain, so do this part **last**.

Two things differ from Parts 1 to 6 here. The demo slides for beats 6 and 7 are written for the per-panel engine (their scorecard says everything is simulated exactly at 18 qubits), so do not use them for Runs 2 and 3. And the domain engine uses its own four observation axes, (0,0), (45,90), (90,0), (90,90), not the ring the per-panel engine uses.

For each run: in step 3 open **Show options**, set what the run says, press **Open rehearsal**. The show takes about 5 s to build. Stop the previous show first.

### Run 2: proof it is quantum, the parity game

**Settings:** `engine = domain`, `game = on` (tick), `game_fraction = 1`. Everything else default.

1. Press `W`: about 55 of 74 domain edges entangled, all 4 creases, lamp-plus-leaves **Mermin value about 6.9 against a local bound of 4**.
2. Press **reset tally**. Set auto-cycle to about 1 s and let about 40 rounds run (a parity-game round took about 2.6 s in a cloud test, so time a few rounds on your Mac first and lengthen the interval if rounds overlap). Expect a win rate around **92%** against the classical ceiling of **75%**.
3. Press `M` (dephased depth), **reset tally**, run again: the rate falls to **75% or below**. Press `M` again.

### Run 3: entanglement that grows, and what it costs

**Settings:** same as Run 2 plus `evolve_steps = 0`.

1. Static: reset tally, about 40 rounds: about 93%.
2. Press `E` once, reset tally, run 40 rounds: about 81%.
3. Press `E` again (2 steps), reset tally, run: about **74%, below the classical 75%**.
4. Press **static** to return to step 0.
5. Say: the state gets harder to approximate classically, but it stops winning the Bell game. You cannot have the proof and the growth together unless the game leaves are decoupled from the evolved facets; that is not built.

(These rates are the simulated state's predictions confirmed by sampling the simulation. They have not been measured on a device.)

---

## Part 8 (optional, 1 credit): really send the tomography job

Only if the service is back (canary passed) and you have checked your credit balance.

1. Stop the studio (Ctrl+C) and restart it **with spending enabled**:
   ```bash
   python -m src.studio --project runs/myroom --allow-spend
   ```
2. Step 4 → **Verify on Moth** → **Review payload**. The button now reads **Send to Moth: spend 1 credit(s)**. The card shows the hash; the confirm repeats it, so if anything changed the studio refuses.
3. Press it. The job id is written before polling and the raw result before parsing. Whatever Moth returns, the raw file is safe in `runs/myroom/solve/tomography/raw_result_*.json`. If the result shape is not recognised, the log stops with the keys it saw: **do not resend**; keep the file.
4. On success the page says *Moth tomography-api-v2 verified the reference circuit … polarity visibilities within 0.0xx of the Aer witness run*, and `solve/tomography/score.json` is written. **Restart the show** (or press `D` again after a beat) so beat 7's scorecard picks the line up.
5. Never press a Moth send during filming unless you want that to be the take.

---

## What to say honestly

- The wall's state is a simulation on this laptop (Aer, a statevector simulator). Everything at these sizes is classically simulable. **No quantum advantage is claimed.**
- Moth's part today: the engine requests are built, priced and gated (cards); QDrive built a Bell state and a GHZ state in the lab; **no Moth result exists for this wall**, and the service has been failing since 4 Oct. When it answers, tomography gives a verdict on the circuit, QDrive can prepare the facet register, and the echo supplies the dynamics.
- The echo mapping from taps to tilt and azimuth is ours, not the engine's definition; the local echo is our reading of the engine's description.
- The parity-game rate is a prediction of the simulated state, confirmed by sampling the simulation.

## If something goes wrong on camera

- **"Review circuit" says the engine must be panel / the domain engine is not part of it:** a leftover `engine = domain` in Show options. Press **Reset options** (step 3) and review again.
- **Step 5 buttons are missing or greyed:** step 4 has not finished (the log is still running) or the drawing changed after the run (step 4 then says *made from an earlier drawing*). Run step 4 again.
- **A step takes minutes instead of seconds:** something else is using the CPU. Quit it; stop and restart the step.
- **"Review payload" for Verify on Moth shows an earlier job with no result:** expected (the 5 Oct job `cece6571`); it is a true statement, not an error.
- **The send button is greyed:** the studio was started without `--allow-spend`; that is what you want while filming.
- **Glass not black on the wall:** blackout (`B`), recalibrate the projector corners on the Calibrate tab, and do not record.
- **The show will not start:** read the line it prints; most are scene and calibration mismatches.
- **The Parity game, Evolve and reset-tally buttons are greyed out:** you opened the panel engine. Stop and relaunch step 3 with `engine = domain`.
