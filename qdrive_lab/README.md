# qdrive_lab (temporary)

Throw-away investigation of QDrive. Delete this whole folder when finished; nothing else imports it.
The write-up is `../docs/qdrive-field-notes.md`.

- `lab.py` shared helpers: raw API calls, ledger + 15-job budget cap, known-answer scorer
- `test_offline.py` no-network tests (scorer vs known circuits; client pre-checks). `python qdrive_lab/test_offline.py`
- `probe_free.py` free probes: schema 422s (always carry `shots: 0`, so no job can be created), alternate route, asset flow
- `trials.py` paid trials T01-T16 (`python qdrive_lab/trials.py stage1|stage2|stage3`); skips trials that already have a result
- `results/` every trial JSON, returned circuits, `ledger.jsonl`, `engine_spec.json` (the engine's own params schema), `probe_free.json`

The budget is spent (15/15). To run more: `QDRIVE_LAB_BUDGET=20 python qdrive_lab/trials.py ...` after adding trials.
