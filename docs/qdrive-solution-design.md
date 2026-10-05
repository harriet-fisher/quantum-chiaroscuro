# Running QDrive in Standing Light: derived solution

> **Scene note.** The worked example here is the user's living room (8 panels in 5 planes, 19 qubits). The built-in default scene is the six-pane bay window: its calibration (`runs/calibration_bay_4x4`) has 20 qubits in two components, patches + lamps (14 qubits) and polarity (6), which `python -m src.quantum.qdrive_rounds plan` cuts into 10 chained jobs; the `--engine local` rehearsal of that plan is in `tests/test_moth_pipeline.py`.

Inputs: the Motte-model paper (arXiv 2605.22744 v3), `docs/qdrive-field-notes.md` (28 QDrive jobs, 15 of them lab trials), and the
living-room run (`runs/harrietlivingroom`: `mask.png`, `patches_preview.png`, `labels.json`, `calib/targets.json`).
Tags: **[exact]** proven offline from the project's own oracle state, **[lab]** verified by a QDrive job, **[paper]** stated by the paper,
**[hyp]** hypothesis, not yet tested live. Code: `src/quantum/qdrive_plan.py`, tests `tests/test_qdrive_plan.py`, ladder
`qdrive_lab/ladder.py`. **Nothing in this document has spent credits.**

---------------------------------------------------------------------------------------------------------------------------------

## 1. Short answer

1. **Split the 19 qubits into the two independent components they already are** and run each as its own small job (patches + lamps: 11
   qubits; polarity: 8 qubits). Merge the circuits locally. [exact: zero covariance between them]
2. **Build each component by sequential growth**, one new qubit per step, one target per step, a `null` between steps. A step's target is the
   qubit's group = the new qubit plus its already-built neighbours. For your scene that group is at most 3 qubits (polarity) and 4 qubits
   (patches + lamps, with one virtual lamp-lamp edge). The growth structure reproduces your target distribution **exactly (total variation
   0.0)** from group marginals alone. [exact]
3. **Give each step the full reduced density matrix of the partly built state** (X, Y and Z words), not Z words only. Z-only group targets
   describe a fully dephased state that a gate on an entangled group cannot reach. [paper + exact; live status hypothesis]
4. **Yes, targets are approached gradually, but not everywhere.** A single consistent group is reached in one layer (lab: ZZ 0.9976 in one
   layer, `T01`). Gradual repetition helps only where groups compete or the engine under-delivers; repeating the whole target list is
   costly and sometimes harmful (`T03`). So: measure after every job (we can: the returned circuit simulates locally) and **repeat only the
   groups that are still off**, aimed slightly past the target. [lab + paper §4.3; the loop is built, convergence on the real engine untested]
5. **Validate in four small live rungs (4 credits)**; rungs R3 and R4 are the production runs themselves.

## 2. What your scene actually is

`mask.png` / `patches_preview.png` / `labels.json`: a bay window photographed head-on, 8 panels in 5 planes (angles -36, -15, 0, +15, +36 deg),
white = surfaces the projector may light, black = glass (no light, no qubits). A 5 x 4 grid lays 9 patch cells on the trim and mullions that
survive the mask (`min_cover` 0.3): cells (1..3, 0) along the top, and columns 1 and 3 beside the centre panes below that. The calibration then
allocates 19 qubits (`meta.qubit_map`):

| Role | Qubits | Targets (all Z-type) |
|---|---|---|
| patches | 0-8 | `<Z>` ~ 0.001-0.005; 8 patch-patch `<ZZ>`: six at +0.72 (same plane), two at -0.54 (across the centre/right crease, patch 2 with patches 1 and 4) |
| lamps | 9, 10 | `<Z>` = 0 (fair coins); 18 lamp-patch `<ZZ>`: L1 from -0.38 to +0.71 (strong on patch 2), L2 weak throughout (magnitudes 0.09 to 0.14) |
| polarity | 11-18 | `<Z>` ~ -0.008; 11 `<ZZ>` between 0.90 and 0.97: the eight polarity qubits behave as one spin |

Structure that drives everything:
- The patches form a **single chain** 8-6-4-2-1-0-3-5-7. Each lamp touches **every** patch (a hub of degree 9). The polarity qubits form a
  **chain of four triangles** (11,12,13)(13,14,15)(14,15,16)(16,17,18). No coupling joins the patch/lamp part to the polarity part, and the
  oracle state has exactly zero covariance between them (max 0.0 over all 88 cross pairs). [exact]
- Everything is Z-type: `calib/targets.json` says "z-only baseline ... a classical correlated sampler reproduces it exactly". The oracle
  state `sampler_local.oracle_state` (positive amplitudes sqrt(P)) reproduces the 37 stored pair targets to within 0.0047 (their Monte Carlo
  error is about 0.003). It is the state the plan aims at. [exact]
- All 11 polarity `<ZZ>` are at least 0.899, so the polarity part is almost a single pinned spin; the calibration already warns about it ("near-pinned": 9 of 37 pair targets above 0.9).

## 3. Why the payload we sent could not work

| What we sent | What the lab and paper say |
|---|---|
| 19 singles + 37 overlapping pairs, one layer | one target = one gate; overlapping targets in a layer fight (`T02`/`T03`); the polarity-chain jobs showed exactly that |
| One 19-qubit register | every 19-qubit job failed (4 of 4); every 8-qubit low-order job passed. Width limit unlocated |
| Z-only words per pair | a Z-only RDM is dephased: a gate on a group that is entangled with the rest cannot change the group's spectrum, so it cannot reach it |
| Hub qubits in many targets | a hub in every group serialises the whole job into the longest chain of overlapping groups |

## 4. Your question: gradual pushing and repeated sampling

The paper's loop (Fig. 1, §2.1) is: measure the <=k-body Paulis of the *current* state, name new targets, the interface derives and appends a
gate, repeat; an optional fractional power `U^f` makes each step partial (§4.3: smaller `f` = more faithful trajectory, more rounds). So a
step is gradual only if you ask for it. Our own evidence:

| Observation | Reading |
|---|---|
| One `ZZ` target, one layer: ZZ = 0.9976 (`T01`) | a consistent group is one-shot |
| `{ZZ, XX}` target: Bell state in 2 rounds (`T04`) and chained repair in 1 layer (`T09`) | same |
| 8 singles at `Z` = 0, rounds 1/2/4: error 0.41 / 0.21 / 0.086 | gradual convergence **does** occur where the engine under-delivers |
| overlapping singles + pairs: 1/2/4 rounds, pairs stay at ~1.0 / 0.88 / 0.90 | more rounds do not fix a conflict, they just repeat it |

Design consequence: repetition is a **per-group** decision driven by measured residuals, not a global `rounds` knob. `qdrive_plan.closed_loop`
does this: job 1 is the full sweep, each later job repeats only the groups still more than `tol` off and aims `gain` (default 0.5) of the
remaining error past the target (clipped at 0.98). The paper's `f` is not a QDrive parameter (the schema has none; the per-word certainty had
no visible effect, `T14`), so the overshoot-and-remeasure loop is how we emulate a controllable step. The loop needs the returned circuit
simulated locally between jobs: cheap at 8-11 qubits.

## 5. The derived solution

### 5.1 Decompose
`qdrive_plan.components` splits the target graph into independent components: one job each, circuits merged on their global qubits with
`qdrive_plan.merge_circuits` (a product of independent states needs no engine). This also keeps every register at 8-11 qubits.

### 5.2 Grow, one qubit per step (the paper's GHZ construction, §3.2.5)
`qdrive_plan.elimination` does min-degree elimination of the target graph; the reverse order is the growth order; a qubit's **parents** are its
neighbours built earlier; its group is `parents + [qubit]`. A fresh qubit starts in |0>, so a gate on its group only has to set that qubit's
conditional distribution: exactly reachable, and the rest of the register is not disturbed in the Z basis.

| Component | Group sizes | Fill edges (virtual coupling) | Steps | Markov-growth TV vs oracle |
|---|---|---|---|---|
| polarity (11-18) | <= 3 | none | 8 | **0.0** |
| patches + lamps (0-10) | <= 4 | (9,10) lamp-lamp | 11 | **0.0** |

(`tests.test_qdrive_plan.LivingRoom` asserts both to 9 decimals.) The lamp-lamp fill edge is added to `coupling_map` only: the engine runs on
a simulator, the projector does not care.

### 5.3 Targets are full reduced density matrices of the partly built state
Step k's target is every Pauli word of the reduced density matrix, on the step's group, of the state after step k
(`qdrive_plan.prefix_states`, `rdm_words`, `build_state_prep_job`). Z-type words are always included, even at 0, because the one lab target
that set `<Z>` = 0 explicitly (`T07`) worked. This matches the paper's statement that the interface works with the tomography of the circuit
so far and a target RDM, and that an m-qubit gate is only unique when it is determined from (m+1)-qubit information (§2.2.3). Payload size is
modest: 2-75 words per target (`ladder.py build`).

### 5.4 One layer per step, never overlapping targets in one layer
`pack_layers` puts a `null` between any two overlapping groups. Under both hypotheses about when the engine measures (once per layer vs per
gate) the result is the same, so the plan does not depend on that unresolved point.

### 5.5 Cost and time
2 jobs (one per component), 1 credit each, 8 and 11 layers. Lab timing: 5-17 s for 1-3-body targets at 2-3 qubits; ~30-65 s for 80
low-order targets at 8 qubits. The 4-body steps and 11-qubit width are unmeasured (rungs R2 and R4).

### 5.6 What this does and does not give you
The Z-only baseline is a classical distribution. The plan reproduces it exactly, but a circuit for it can also be written locally with
multiplexed rotations at zero credits: `prefix_states` already builds that state. QDrive's value for the project is the *quantum-specific*
variant (polarity read in X, entangled with the lamps, `docs/standing-light-handoff.md` section 9.4): the same machinery applies, with the
targets taken from that designed state instead of sqrt(P). Do not present the Z-only run as a quantum result.

## 6. Evidence ledger and the live ladder


**Live result, R1 (4 Oct 2026): FAILED, job `c53afb2a-f67e-47fc-9d8b-d3b47c2abf9f`.** `engine_timeout` after 62 s on a 3-qubit, 3-target job, with
an EMPTY `progress` (the engine never reported a target). R2-R4 were not sent (stop-on-error). 1 credit spent (charge for failed jobs
unknown). What the payload had that no successful lab job had: (a) a 1-qubit target containing an X word (`{"X": 1.0, "Z": 0.0}`),
(b) a fully specified pure 2-qubit RDM (`ZZ = XX = 1`, `YY = -1`, plus the zero Z words), i.e. a rank-1 target with a triple-degenerate
zero eigenspace, (c) values of exactly +/-1.0 (the planner clips Z-only words to 0.98 but `rdm_words` did not), (d) an explicit
`coupling_map`. Lab jobs that completed had either partial word sets (T04: `ZZ`, `XX` only) or no `coupling_map` (T01-T14). The 62-63 s
failure with empty progress also matches T16 (also 63 s, empty progress) and is closer to 2 missed 30 s heartbeats than to the 120 s limit,
i.e. the engine appears to stall in a computation, not run slowly. [hyp: culprit is (b) or (c); not isolated]

**Live result, R1 revised (4 Oct 2026): FAILED again, job `30f5419b-5582-4f59-9a15-5873f4dcb932`.** Same `engine_timeout` at 64 s, empty `progress`.
Payload: `n_qubits 3`, `coupling_map [[0,1],[1,2]]`, target 1 = `{qubits [1,2], IZ 0, ZI 0, XX 0.98, ZZ 0.98}` (a Bell state, no Y word, values clipped,
no one-qubit target), `null`, target 2 = `{qubits [0,1], IZ 0, ZI 0, ZZ 0.98}`, `null`. So the one-qubit X target, the fully specified rank-1 RDM
and the +/-1.0 values are NOT the cause (all removed). Two failures, ~63 s each, empty progress both times (T16 matches). What remains
different from every job that completed: (i) an explicit `coupling_map` together with a 3-qubit register, (ii) first target on qubits `[1,2]`
(an untouched qubit 0), and (iii) the SECOND target acts on a qubit already entangled with another one (a mixed reduced state, the thing R1 was
built to test). Empty progress suggests the stall is at or before the first target, which would point at (i)/(ii) rather than (iii), but that is
not established. Cleanest bisect: send ONLY target 1 (+ `null`); if it completes, the second target (entangled group) is the problem; if it stalls,
the setup (coupling_map / qubit placement) is. R2-R4 remain unsent.

**Offline-proven [exact]:** component split; widths and fill edges; Markov-growth exactness; that the full-RDM targets of the prefix states are
reachable by construction; planner structure and layering (15 unit tests).

**Lab-verified [lab]:** group targets, identity padding, mixed-word mappings, 3-body GHZ, chaining by output asset UUID, `null` semantics,
update methods, timing at <= 8 qubits (`docs/qdrive-field-notes.md`).

**Not verified, and why the mock does not settle them:** `src/quantum/motte_mock.py` is a paper-based ideal-gate approximation. Calibrated
against the lab it reproduces the grouped successes and, with gates derived from the layer-start state, the overlapping-pair failure; it
does *not* reproduce `T02` and it fits the pairs-only run better with the opposite assumption. Run on the growth plan it reached total
variation 0.37-0.79, because its gate search picks *some* unitary matching the group's words and that need not be the controlled rotation
the plan relies on. That is a limit of the mock, not evidence about QDrive. It is kept for plumbing tests and as an ideal-gate reference
only. The questions below need the real engine.

| Rung | Question (what it decides) | Payload | Pass if | Credits |
|---|---|---|---|---|
| **R1** | Does growth work when the group is already entangled with the rest? (GHZ-3 in two steps.) If not, the plan must use (m+1)-body targets | `runs/harrietlivingroom/qdrive_plan/R1` | TV < 0.03 | 1 |
| **R2** | Do 4-qubit targets run, in time, and is a 4-body step honoured? (synthetic 4-qubit distribution) | `.../R2` | TV < 0.05, < 60 s | 1 |
| **R3** | The real polarity component, full words | `.../R3` | TV < 0.05, rms < 0.03 | 1 (= production run for that component) |
| **R4** | The real patches + lamps component | `.../R4` | TV < 0.05 and completes | 1 (= production run) |
| R3b | Same as R3 with Z words only: do X/Y words matter? Only if R3 passes narrowly or to diagnose a failure | `.../R3b` | informational | 1 |

Run order and decisions: R1 first. If R1 fails, stop and use (m+1)-body targets (one extra neighbour per group) or the local state prep. If
R1 passes, R2 and R3 are independent. If R4 times out, fall back to 2-body groups: treat the two lamps as classical coin flips and run one
9-qubit patch chain per lamp world (a tree, group size 2; 4 jobs, fewer by the left/right symmetry), picking the world's circuit at show
time. If a rung under-delivers by a small amount, switch on `closed_loop` for that component (1 credit per repeat).

```bash
python qdrive_lab/ladder.py build                        # regenerates the payloads, sends nothing
python qdrive_lab/ladder.py score R1 <returned circuit>  # local scoring against the exact oracle
python qdrive_lab/ladder.py send R1 --approve-credits 1  # the only command that spends (1 credit)
```

## 7. Open questions

- Is the fraction `f` reachable through any QDrive parameter? (Not in the schema; certainty looked inert.)
- Width cliff between 8 and 19 qubits; body-size cliff between 3 and 8.
- Whether failed jobs are charged (no balance endpoint; check the dashboard around one failure).
- `update_method` choice for growth steps: `spectral` is the default, `direct` was fastest in the lab.
- For the quantum-specific variant: the designed state's prefix RDMs include X-type words on the polarity qubits; same machinery, new targets.

## 8. Sending the plan in rounds (implemented 4 Oct 2026, never run live)

`src/quantum/qdrive_rounds.py` cuts section 5's plan into small chained jobs instead of one job per component:

- Round r is job r of every component (living room: patches+lamps 11 qubits = 5 jobs, polarity 8 qubits = 4 jobs; 9 credits with the default
  `--layers-per-round 2`). Job r continues job r-1's returned circuit as `initial_circuit` (its output ASSET uuid; `n_qubits` is then omitted).
- After each job returns it is simulated locally and compared with the state the plan wants at that point (TV and fidelity). The run stops before
  the next credit when a job fails or lands further than `--max-tv` (default 0.10) from its target. A failed or unrecorded job is never resent
  without `--retry-failed`.
- State is kept per step under `<solve>/qdrive/rounds/` (`rounds_state.json` records every job id), so a run can be stopped and resumed.
  `assemble` merges what is done: all components finished -> `circuit.qasm` + `state.json` + `requested_vs_achieved.*` (what the show and studio read);
  otherwise `circuit_partial.qasm`, never `circuit.qasm`. `--local-fill` prepares an unfinished component exactly, locally, and `provenance.json`
  then says so (the show's label drops "Moth").
- `--engine local` is an ideal stand-in engine (no network, no credits, kept apart in `<solve>/qdrive_local/`, always labelled a local rehearsal): it
  exercises the whole pipeline and merges to the calibrated targets at rms 0.004 on the living room. It says nothing about the real engine.
- `canary` sends the lab's known-good 2-qubit job (T07, 1 credit): run it first when a round fails with `engine_timeout`; on 4 Oct 2026 it also timed out,
  which is how the service (not the payloads) was found to be down.
- Spend gate: `run` previews unless `--approve-credits N` equals the credits of the jobs queued now (`--max-steps` limits that).
- Found while building it: qiskit's `prepare_state` drops amplitudes about 1e-4 of the largest (42% fidelity on the 11-qubit component); `local_circuit` uses
  an exact multiplexed-Ry tree instead.

Unvalidated live: 11-qubit register and 4-body groups (first jobs of the patches+lamps chain), chaining through `output_asset_id` beyond T09's 2-qubit case,
and `coupling_map` on chained jobs (`--no-coupling-map` leaves it out).
