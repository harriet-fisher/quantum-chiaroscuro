# QDrive field notes (qdrive-api-v1)

A working supplement to the skeletal docs at <https://docs.mothquantum.com/docs/engines/qdrive-api-v1>, built from the engine's own
machine-readable record (`GET /api/v1/engines/qdrive-api-v1`, saved in `qdrive_lab/results/engine_spec.json`), the Motte-model paper
(arXiv 2605.22744), and the 28 QDrive jobs run on 2 and 4 Oct 2026 (21 completed, 7 failed). Each claim is tagged:

- **[V]** verified by a job or request we ran (job id given, or the lab file that holds it)
- **[S]** stated by the engine record / docs / paper, not independently tested
- **[H]** hypothesis, consistent with data but not proven

Raw evidence: `qdrive_lab/results/` (every trial JSON, every returned circuit, the ledger, the free-probe log).
Re-run offline checks with `python qdrive_lab/test_offline.py`.

---------------------------------------------------------------------------------------------------------------------------------

> **Correction (4 Oct 2026, later):** in the living-room scene the qubit roles are patches 0-8, **lamps 9-10**, **polarity 11-18**
> (`runs/harrietlivingroom/calib/targets.json`, `meta.qubit_map`). Everything below that says "lamp group" about qubits 11-18, including
> the labels of trials T15 and T16, means the **polarity chain**. The conclusions do not change. The follow-up analysis and the derived plan
> are in `docs/qdrive-solution-design.md`.


## 1. The mental model in six lines

1. A QDrive job **builds a circuit by simulation, not by sampling hardware**: you list `targets` (Pauli expectation values on named
   qubits), the engine derives gates that move the state toward them, and returns the circuit as QASM3. [S, V]
2. **One target = one gate on its own qubit group.** The group is the target's `qubits` list; the target's `expvals` mapping holds
   *every* Pauli word you want true on that group. A 2-qubit target yields a full 2-qubit unitary (4 single-qubit blocks + `rxx`,
   `ryy`, `rzz`); a 1-qubit target yields one 1-qubit unitary; a 3-qubit target yields a 3-qubit gate. [V: gate counts in T01, T02, T04, T08]
3. **`null` in `targets` = `update()` = "end this layer and apply it".** Targets before a `null` form one layer. Repeating
   `[targets..., null]` N times builds N layers, each re-measured against the current state. [S: paper, V: gate counts double per round]
   A job with targets and no `null` still applies one layer (official sample, T01). [V]
4. **Overlapping targets inside one layer fight each other.** Putting `Z` on q0, `Z` on q1 and `ZZ` on (q0,q1) as three targets did
   NOT build the intended state (T02, T03); putting every word for the pair in ONE target did (T04, T07). [V]
5. **Cost is driven by target body size and register width, not by the number of targets.** 1–3-body targets finish in 5–17 s;
   one 8-body target hung until the engine timed out (T15); 19-qubit registers hung with plain 2-body targets (4 jobs). [V]
6. A job that hangs is killed by the engine and reported as `engine_timeout` (retryable: true); there is no partial result. [V]

## 2. Calling the API (what actually works)

| Thing | Verified fact |
|---|---|
| Submit | `POST https://api.mothquantum.com/api/v1/engines/qdrive-api-v1/process`, JSON `{"params": {...}, "input_files": {...}}`. Returns **HTTP 202** `{job_id, ...}`. [V all trials] |
| Wrong route in the engine's own code samples | The samples use `https://api.mothquantum.com/v1/generation/qdrive-api-v1/process` with multipart `_params`; that URL returns **404**. Use the `/api/v1/engines/...` route above. [V: `qdrive_lab/results/probe_free.json`] |
| Poll | `GET /api/v1/jobs/{id}/status` -> `status` queued/running/completed/failed, `progress{detail,step}`, `error{type,message,retryable}`, `submitted_at`, `updated_at`. Progress detail is only a hint: the engine can sit on one target for minutes after the last progress tick. [V] |
| Result | `GET /api/v1/jobs/{id}/result` -> `{"outputs":[{slot:"circuit", url (presigned, no auth), output_asset_id, expires_at, content_type:"text/plain", size_bytes}]}`. `result` (inline) is always null. Presigned URL expires about 15 min after the job. [V] |
| List jobs | `GET /api/v1/jobs` lists all of your jobs with status (no params, no cost). Useful for an audit of failures. [V] |
| Credit balance | **No endpoint found** (`/me`, `/me/credits`, `/me/usage`, `/credits`, `/balance`, `/account`, `/usage`, `/billing` all 404 except `/me` which has no credits field). Whether failed jobs are charged is still unknown; read the dashboard. [V] |
| Price | 1 credit per run (`credits_per_run`). Accepted jobs: the lab ledger counts 15; with the earlier runs of 4 Oct that is 21 completed + 7 failed QDrive jobs on the account. [V] |
| Run policy | `timeout: 120`, `max_retries: 3`, `heartbeat: 30` (engine record). [S] The observed failure time of 4 m 13 s - 4 m 16 s for the big jobs is consistent with two or more 120 s attempts + overhead [H]. T15 failed at 186 s and T16 at 63 s, so the clock is not a fixed 255 s. [V] |
| Sync/async | `is_async: false`, `execution_mode: handler`. [S] |

### Chaining (feeding one job's circuit into the next)

- Input slot is `initial_circuit` (text/plain QASM3, optional). With it, `n_qubits` is not needed. [S]
- **Pass an asset UUID.** `"job:<id>/circuit"` (the form shown in the engine's samples) is rejected on the JSON route with
  `422 "must be an asset UUID"`. [V: T06]
- Every completed job's output is itself an asset: its UUID is `outputs[0].output_asset_id` in the result, or
  `GET /assets?kind=output` (rows carry `job_id` and `slot`). Using that UUID as `initial_circuit` works. [V: T09 continued from T02's
  Bell-ish circuit, added a Bell target, got rms 0.008]
- The returned circuit after chaining is *the old circuit plus new gates* (T09 ops: `u 8, cx 3` from the input + `rz/ry/rxx/ryy/rzz` new). [V]
- The engine re-serialises an input circuit even when asked to do nothing: our `h q[0]; cx` came back as `U(pi/2,0,pi) q[0]; cx` (T05). [V]
- Chaining is the intended way to build deep circuits within the 120 s limit: one job per few layers. [H, supported by T09]

### Assets (uploading your own circuit or image)

- Flow works exactly as documented: `POST /assets {filename, content_type, size_bytes}` -> `{asset_id, upload{url,headers}}` ->
  `PUT` bytes to the presigned URL with the returned headers -> `POST /assets/{id}/complete` -> `status: uploaded`. [V]
- **The docs say only `image/png` and `image/jpeg` are accepted. That is wrong in practice:** `text/plain`,
  `application/octet-stream` and `application/json` all registered AND completed. [V: `probe_free.json`, ad-hoc check]
- A `text/plain` QASM3 asset is accepted by QDrive as `initial_circuit` and the job runs (T05, job 9e36fe64). [V]
- `GET /assets/{id}/download` -> `{download_url, expires_at}`; `DELETE /assets/{id}` -> 204. `GET /me/storage` shows usage and
  quotas (1 GiB uploads, 10 GiB total, 2 GiB notebooks). [V]
- Nothing in the asset responses mentions a price; whether uploads consume credits is not documented and we could not measure it (no balance endpoint). [V that no price is shown]
- PNG uploads get automatic image labelling in `metadata` (irrelevant to QDrive). [V]

## 3. Parameters (from the engine's `params_schema`, schema behaviour verified with free 422 probes)

The schema has `additionalProperties: false`: an unknown key is a 422 (`additional properties 'bogus_key' not allowed`). [V]

| Param | Type / default | Notes |
|---|---|---|
| `n_qubits` | int > 0 or null | Required unless `initial_circuit` given (checked at run time, not by the schema). **No upper bound in the schema** (500 passed validation); the limit is the 120 s clock. Float 2.5 -> 422. [V] |
| `targets` | array of object/null, array of string, or a **plain string** | Schema accepts a bare string and a list of strings, so a text/DSL form of targets exists but is **undocumented** [V schema, H meaning]. Lists of numbers -> 422. |
| `update_method` | str, default `spectral` | `spectral`, `direct`, `constrained`. A bogus value is NOT a schema error (checked at run time as `invalid_update_method`). All three reached a Bell state (see section 5). |
| `machine` | str, default `aer` | Bogus value also run-time (`invalid_machine`). Only `aer` is used by this project. |
| `shots` | int > 0, default 1024 | Shared by the "estimator and sampler" (engine text). 0 -> 422. |
| `sample` | bool, default false | "Also run a shot-based sampler on the final circuit". **No sample output was observed** (T11: still only `circuit`, `result` null). [V] |
| `tomography` | int 0-2, default 0 | 0 none / 1 single-qubit / 2 two-qubit. 3 -> 422. **Returned nothing visible** (T05, T11: no extra output, no inline result, no warning). [V] Its effect on the engine's internal fitting is unknown. |
| `seed` | int >= 0 or null | Same payload + same seed does **not** give a byte-identical circuit (T04 vs T10 differ in rotation angles) although both are equally good. The seed does not make runs reproducible. [V] |
| `coupling_map` | list of edges, string, or null | `[[0,1],[1,2]]`. Out-of-range edges are a run-time error (`coupling_map_out_of_range`). Effect on results untested in the lab (the project uses it for the real data). |
| `ansatz` | object (qubit-count -> QASM3 string), string, or null | **Never tested** (the format of parameterised QASM3 is undocumented). Per-target `ansatz` keys are rejected (`unsupported_target_ansatz`). |

Run-time error types (engine record): `invalid_params, invalid_initial_circuit, n_qubits_mismatch, invalid_coupling_map,
coupling_map_out_of_range, invalid_ansatz, invalid_ansatz_qasm, ansatz_width_mismatch, unsupported_target_ansatz, invalid_target,
invalid_update_method, invalid_machine`. `invalid_target` means: missing `qubits`, duplicate or out-of-range qubits, or non-mapping
`expvals` (a bare number gave exactly this on 2 Oct, job f54945ab). Run-time errors surface as a **failed job**, so they are
potentially billed; schema errors are 422 at submit and create no job. [S, V]

## 4. Targets in detail

### 4.1 Shape

```json
{"qubits": [0, 1], "expvals": {"ZZ": 1.0, "XX": 1.0}}
```

- `expvals` **must be a mapping**; a bare number fails with `invalid_target`. [V]
- Keys are **Pauli words with one letter per entry of `qubits`**, letter *i* acting on `qubits[i]`. Use `I` to pad. [V: T07 `ZI`,`IZ`,`ZZ`;
  T08 `ZZI`,`IZZ`,`XXX`; T04 `ZZ`,`XX`]
- Values may be a number or `[value, certainty]` per word: `{"ZZ": [1.0, 0.9]}` was accepted and gave a good Bell state (T14); **no
  measurable effect of the certainty was seen**, and a low certainty (0.1) on single-qubit targets did not stop them fighting the pair targets
  (polarity-chain test 1f48f12f). [V]
- The schema text also mentions an expression form `[[word, qubits, coefficient], ...]` (optionally `[terms, certainty]`). **Not tested.**
- A `null` entry = `update()`.

### 4.2 Qubit-order convention (verified)

Little-endian like Qiskit: qubit 0 is the rightmost character of a Qiskit Pauli string and the least-significant bit. The lab scorer
(`qdrive_lab/lab.py::pauli_expval`) passes known-answer tests (X on q0, Bell, GHZ, letter order, QASM3 round trip) and agrees with the
project loader `src/quantum/sampler_local.py`. Letter order inside a word follows the `qubits` list. [V: `qdrive_lab/test_offline.py`]
Reversing the qubit order or switching measurement basis does not explain any bad result we had. [V]

### 4.3 Grouping rule (the central finding)

| Trial | Targets | Result | Reading |
|---|---|---|---|
| T02 job 9d96016d | `Z`(q0)=0, `Z`(q1)=0, `ZZ`(0,1)=1, `null` | `<Z>` = -0.48, -0.34; ZZ = 0.80 | overlapping targets conflict |
| T03 job 8dbe4f12 | same x 3 rounds | `<Z>` = -0.36, -0.34; ZZ = 0.83 | more layers do not repair a conflict |
| T07 job 27dd8027 | ONE target `{ZI:0, IZ:0, ZZ:1}` | `<Z>` = 0.03, 0.03; ZZ = 0.9994 | one group, one gate: works |
| T04 job 4e77a23b | ONE target `{ZZ:1, XX:1}` x 2 rounds | Bell state, rms 0.0085 | works |
| T08 job 6d5c10c5 | ONE 3-qubit target `{ZZI:1, IZZ:1, XXX:1}` x 2 | GHZ-3, rms 0.0014 | multi-qubit groups work |
| T09 job d4d7c270 | chain from T02's circuit + Bell target | Bell, rms 0.008 | a later layer can repair an earlier one |
| T01 job 408249a8 | the engine's own sample (one `ZZ` target, no `null`) | ZZ = 0.9976 | one layer applied without `null` |

Gate-count evidence for "one target = one gate": a lone 2-qubit target gives `rz 8, ry 4, rxx 1, ryy 1, rzz 1` per layer (T01, T07);
adding single-qubit targets adds `rz 4, ry 2` per layer (T02: `rz 12, ry 6`); rounds multiply everything (T04: x2, T03: x3). [V]

Consequence for the old 8-qubit polarity-chain jobs (qubits 11-18 of the living-room scene; earlier notes mislabelled them "lamp"): asking for 8 single-qubit `Z`~0 targets and 11 overlapping `ZZ`~0.9 pair targets in the same layer
set up exactly the conflict of T02/T03. The results (rms 0.81 with 1 round, 0.69 with 2, pairs ~0 after 4 rounds, jobs
9bb6eb06 / 4a789894 / 26a59376, plus the weighted / reordered / direct / constrained variants) are what that conflict looks like
at scale. Pairs-only (job b24a61e0) honoured `ZZ` because no single-qubit targets competed, and landed on the trivial all-zeros state. [V results, H cause]

### 4.4 What does NOT work (or is untested)

- A bare number for `expvals`. [V]
- Many overlapping low-order targets in one layer asking for incompatible things (above). [V]
- **Large group targets**: one 8-qubit target with 19 words timed out after 186 s (T15, job 678811ca); a 19-qubit job with an 11-body
  and an 8-body target failed after 63 s (T16, job ddb6c2b2). The paper's tomography cost is `3^k` settings for k-body targets [S],
  which fits. Work with groups of **2-3 qubits**. [V that 1-3 body is fast, V that 8-body fails, H on the exact threshold between 3 and 8]
- Overlapping groups (a triangle sharing an edge with the next triangle) were **not** tested: the real scene's chain-of-triangles polarity graph needs this.

## 5. Update methods (T12, T13 vs T04/T10, same Bell payload x 2 rounds)

| method | job | rms to Bell | seconds |
|---|---|---|---|
| spectral | 4e77a23b / b3f07578 | 0.0085 / 0.0018 | 12 |
| direct | ce29be03 | 0.0089 | 5 |
| constrained | 4e0e446d | 0.0342 | 8 |

On an easy single-group target all three work, `direct` was fastest and `constrained` slightly less accurate. Differences between
methods only showed in the old conflicting-target polarity-chain runs: `direct` drove singles to -0.99, the others flattened the pairs. [V; H that differences matter on harder problems]

## 6. Timing and failure data

| Job | Register | Targets | Result | Time |
|---|---|---|---|---|
| T01 | 2q | 1 pair | ok | 10 s |
| T04 | 2q | 1 group x 2 | ok | 12 s |
| T08 | 3q | 1 3-body x 2 | ok | 12.5 s |
| polarity chain 1/2/4 rounds | 8q | 20/40/80 low-order | ok | 37 / 54 / 65 s |
| T15 | 8q (polarity chain) | one 8-body target (19 words) x 2 | **timeout** | 186 s |
| T16 | 19q | 11-body + 8-body x 2 | **timeout** | 63 s |
| 19q payloads | 19q | 46-201 low-order targets | **timeout** (4 jobs, all stalled at a pair touching the hub qubit after 38-61 targets) | 255 s |

- Fixed startup is 4-7 s; each additional round costs roughly 5-10 s at 2-8 qubits. [V]
- Register width appears to matter: all 8-qubit and smaller low-order jobs finished, every 19-qubit job failed regardless of target
  count (46, 57, 201). [V; H that width itself is the cause: not isolated, no intermediate width such as 12-14 was tried]
- The dashboard's "Processing target 43/57" is a stale progress tick: the real failure came minutes later. [V: 6fee4d14]

## 7. Outputs

- Exactly one file output: `circuit`, OpenQASM 3, text/plain, size ~1-12 kB. It uses `rz, ry, rxx, ryy, rzz` (and `u, cx` for carried-in
  gates) and no measurement. Local Aer or Qiskit `Statevector` reads it directly; `src/quantum/sampler_local.py` does the same. [V]
- `result` (inline) is null, there is no estimator report, no sampler counts (even with `sample: true`), no tomography output. [V: T11]
- Achieved expectations must be computed locally by simulating the circuit; that is how every number in these notes was obtained. [V]

## 8. Docs vs reality

| The docs / samples say | What we found |
|---|---|
| Use `/v1/generation/qdrive-api-v1/process` with form `_params` | 404. Use `/api/v1/engines/.../process` with JSON. |
| Chain with `initial_circuit: "job:<id>/circuit"` | 422 "must be an asset UUID". Use the output asset UUID. |
| Assets: PNG and JPEG only | `text/plain`, `application/octet-stream`, `application/json` all upload and complete. |
| "120 s timeout" | Jobs can run for 186-255 s before failing (retries). |
| `expvals` "may be a number" | A bare number is `invalid_target`; it must be a mapping of words. |
| `tomography` 1/2, `sample` | No visible output or effect. |
| Example payload shows no `n_qubits`/`targets` | Fine for the chaining form; targets can also be omitted when only re-serialising. |

## 9. A recipe that is supported by the data, and what is still guesswork

Supported by evidence:
1. Express everything you want about a qubit group as **one** target; pad with `I`; use groups of 2-3 qubits.
2. End each layer with `null`; use 2 rounds for an easy target, chain jobs for more depth.
3. Check every returned circuit locally with `qdrive_lab/lab.py::score` (known-answer-tested).
4. Keep a single job to a register that finishes: 8 qubits was reliable, 19 never was.

Guesswork / next experiments (none run, the 15-credit budget is spent):
- Cover the polarity chain (8 qubits, 11 edges, Z~0 + ZZ~0.9) with **overlapping 3-body targets** (triangles) or per-edge 2-body targets that carry
  the identity-padded single-qubit words, and check whether overlapping groups in one layer conflict like T02, or whether separate
  layers (a `null` between groups) fix it.
- Locate the register-width cliff (10, 12, 14, 16 qubits) with a single easy 2-body target.
- Locate the group-size cliff (4, 5, 6 qubits) with a GHZ-k target.
- Chain two 8-qubit jobs (polarity, patches+lamps) and merge locally rather than using one 19-qubit register.
- Test `ansatz`, the expression form of `expvals`, and whether the string form of `targets` is a program.
- Find whether failed jobs are charged (compare the dashboard balance around one deliberately failing job).

## 10. Job index

Earlier runs (2 Oct): f54945ab (bare-number expvals, failed), 6957c6c1 and 32564c08 (19q, 201 targets, timeout).
4 Oct, project data: 6fee4d14 (19q 57 targets, timeout), ebebc343 (19q 46 positive-only, timeout), 9bb6eb06 / 4a789894 / 26a59376
(polarity chain 1/2/4 rounds), b24a61e0 (polarity pairs-only), 1f48f12f (weighted), 7daf7a49 (pairs first), b7eeb8b6 (direct), 6c7d7419 (constrained).
4 Oct, lab trials (`qdrive_lab/results/trial_*.json`): T01 408249a8, T02 9d96016d, T03 8dbe4f12, T04 4e77a23b, T05 9e36fe64,
T06 rejected 422 (no job), T07 27dd8027, T08 6d5c10c5, T09 d4d7c270, T10 b3f07578, T11 9b2ab6e2, T12 ce29be03, T13 4e0e446d,
T14 bc4b13b1, T15 678811ca (timeout), T16 ddb6c2b2 (timeout).
