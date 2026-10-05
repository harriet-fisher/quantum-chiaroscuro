#!/usr/bin/env python3
"""Standing Light studio: the whole pipeline from one page, step by step.

    python -m src.studio                          # project folder runs/studio
    python -m src.studio --project runs/myroom    # keep several scenes side by side
    python -m src.studio --allow-spend            # let step 4 submit to Moth (it still shows the cost and asks first)

One local page (printed URL, opened for you) walks the five steps in the order you ran them by hand:

    1 Draw     the pen tool           -> <project>/labels.json
    2 Targets  calibrate_from_mock    -> <project>/calib/        (targets + both engine payloads)
    3 Rehearse show on Superposed Relief (local reference circuit; the classical sources remain via `src.show --source oracle`)
    4 Solve    the relief circuit executed on Aer (free, local) -> <project>/solve/aer/, or QDrive / graph-v1 on Moth -> <project>/solve/ (the only way to spend credits)
    5 Perform  show, driven by the Aer run's measured shots (quantum), or by the circuit QDrive returned (the older classical-family flow)

Each step runs the existing module as a child process, so every module still works on its own with the same flags. The studio only
decides what runs next, with which paths, and whether the step before it is still current (change the drawing and everything
downstream says so). Long-running tools (pen tool, show) are started on a free port and opened in your browser; Stop ends them.

The Aer path is the default for steps 4 and 5: step 4 transpiles the whole relief circuit (facet, polarity, lamp and observe registers) for Aer, runs it for a
million shots, saves the measured bitstrings and checks them against the reference state; step 5 draws every look from one of those shots. It needs no key and
no credits. The per-panel engine only (the domain engine's 69 qubits are beyond a statevector run).

Spending. The Moth engines of step 4 cannot submit unless the studio was started with --allow-spend AND you press the send button on the plan it shows
(payload hash and credits); the request must repeat that hash and cost, so a payload that changed after you read it is refused. A
payload that already has a result on disk is free and sends nothing. The show started here is given --allow-spend only if the studio has it too and you tick it in the show options (the show's own
re-solve dialog still asks, with hash and cost), and the Moth key is read by the solver itself, never by this file.
"""
import argparse
import collections
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "src", "ui", "web", "studio.html")
CSS = os.path.join(ROOT, "src", "ui", "web", "common.css")
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")
MAX_BODY = 20_000
LOG_LINES = 300

STAGES = ("draw", "targets", "preview", "solve", "final")
TITLES = dict(draw="Draw the shapes", targets="Calibrate targets", preview="Rehearse the show", solve="Solve: execute the circuit", final="Perform with the result")
ECHO = "echo"                                                                 # the Perform choice that plays the echo run
ECHO_LOCAL, ECHO_MOTH = "echo-local", "echo-moth"                                 # step 4: dynamic relief, taps from our local echo (free) or Moth's otoc-echo-v1 (1 credit), circuits executed on Aer
QFAC = "qdrive-facets"                                                        # step 4: the facet register prepared by QDrive, one chained job per confirm
TOMO = "tomography"                                                          # step 4: verify the reference circuit on Moth's tomography-api-v2 (1 credit)
AER = "aer"                                                                  # step 4's local engine: the relief circuit executed on Aer (no Moth call)
ENGINES = {"qdrive": ("payload_qdrive.json", "qdrive-api-v1", "qdrive"), "graph-v1": ("payload_graph_v1.json", "graph-v1", "graph_v1")}


CLASSICAL = ("oracle", "mock", "circuit", "complementary")
PI = 3.1416

# One row per `src.show` flag the studio can set before step 3 (and the runtime ones before step 5). (key, kind, default, limits, stages, needs, group)
#   kind    float | int | bool | choice | path | size        needs   None | relief | domain | classical (checked against the source/engine chosen)
# The key is the flag with underscores ("seg_len" -> --seg-len). A blank value means "leave the show's own default". The page is built from this table.
KNOBS = [
    ("source", "choice", "relief", ("relief",) + CLASSICAL, ("preview",), None, "Show"),
    ("engine", "choice", "panel", ("panel", "domain"), ("preview",), "relief", "Show"),
    ("backend", "choice", "auto", ("auto", "exact", "mps"), ("preview",), "domain", "Show"),
    ("lamp_mode", "choice", "z", ("z", "x"), ("preview",), "relief", "Show"),
    ("kappa", "float", 1.0, (0.01, 8.0), ("preview",), "relief", "Relief"),
    ("n_dirs", "int", None, (1, 16), ("preview",), "relief", "Relief"),
    ("entangle", "float", 0.8, (0.0, PI), ("preview",), "relief", "Relief"),
    ("contrast", "float", 1.15, (0.1, 5.0), ("preview",), "relief", "Relief"),
    ("observations", "choice", "", ("ring", "line"), ("preview",), "relief", "Relief"),
    ("seg_len", "float", 150.0, (10.0, 2000.0), ("preview",), "domain", "Domain"),
    ("group_size", "int", 1, (1, 64), ("preview",), "domain", "Domain"),
    ("tau", "float", 0.5, (0.0, PI), ("preview",), "domain", "Domain"),
    ("pol_coupling", "float", 0.3, (-PI, PI), ("preview",), "domain", "Domain"),
    ("crease_coupling", "float", 1.0, (-PI, PI), ("preview",), "domain", "Domain"),
    ("crease_sign", "float", -1.0, (-1.0, 1.0), ("preview",), "domain", "Domain"),
    ("seam_coupling", "float", 1.0, (-PI, PI), ("preview",), "domain", "Domain"),
    ("seam_mix", "float", 0.0, (0.0, 1.0), ("preview",), "domain", "Domain"),
    ("leaf_tau", "float", 0.3, (0.0, PI), ("preview",), "domain", "Domain"),
    ("leaf_prefer", "choice", "visible", ("visible", "seam"), ("preview",), "domain", "Domain"),
    ("lock", "float", 1.5708, (-PI, PI), ("preview",), "domain", "Domain"),
    ("tau_mix", "float", 0.0, (0.0, 1.0), ("preview",), "domain", "Domain"),
    ("pol_field", "float", 0.0, (-PI, PI), ("preview",), "domain", "Domain"),
    ("game", "bool", False, None, ("preview",), "domain", "Game and evolution"),
    ("game_fraction", "float", 0.5, (0.0, 1.0), ("preview",), "domain", "Game and evolution"),
    ("evolve_steps", "int", 0, (0, 40), ("preview",), "domain", "Game and evolution"),
    ("evolve_zz", "float", 0.7, (-PI, PI), ("preview",), "domain", "Game and evolution"),
    ("evolve_x", "float", 0.5, (-PI, PI), ("preview",), "domain", "Game and evolution"),
    ("polarity_basis", "choice", "", ("z", "x"), ("preview", "final"), None, "Classical stand-ins"),
    ("coupling", "choice", "lamp2", ("hub", "lamp1", "lamp2"), ("preview", "final"), None, "Classical stand-ins"),
    ("pool", "int", 60000, (100, 10_000_000), ("preview", "final"), None, "Classical stand-ins"),
    ("seed", "int", 2026, (0, 2**31 - 1), ("preview", "final"), None, "Classical stand-ins"),
    ("calib", "path", "", "dir", ("preview", "final"), None, "Folders and files"),
    ("run", "path", "", "dir", ("preview", "final"), None, "Folders and files"),
    ("calibration", "path", "", "file", ("preview", "final"), None, "Folders and files"),
    ("circuit", "path", "", "file", ("preview",), "relief", "Folders and files"),
    ("projector_size", "size", "", None, ("preview", "final"), None, "Server"),
    ("allow_spend", "bool", False, None, ("preview", "final"), None, "Server"),
    ("lan", "bool", False, None, ("preview", "final"), None, "Server"),
    ("port", "int", 0, (1, 65535), ("preview", "final"), None, "Server"),
    ("no_browser", "bool", False, None, ("preview", "final"), None, "Server"),
]
KNOB = {k[0]: k for k in KNOBS}


def knob_schema():
    return [dict(key=k, flag="--" + k.replace("_", "-"), kind=kind, default=d, limits=(list(lim) if isinstance(lim, tuple) else lim), stages=list(st), needs=needs, group=g)
            for k, kind, d, lim, st, needs, g in KNOBS]


class StageError(ValueError):
    """The request cannot be done right now; the message is shown to the operator."""


def sha_file(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


class Job:
    """One child process with a rolling log. `url` is set for the tools that serve a page."""

    def __init__(self, name, argv, spawn, url=None, **meta):
        self.name, self.argv, self.url, self.meta = name, argv, url, meta
        self.log = collections.deque(maxlen=LOG_LINES)
        self.started = time.time()
        self.returncode = None
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.proc = spawn(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.log.append("$ " + " ".join(argv[2:] if argv[1:2] == ["-m"] else argv))
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self.log.append(line.rstrip("\n"))
        except (ValueError, OSError):
            pass
        self.returncode = self.proc.wait()
        self.log.append(f"[exited with code {self.returncode}]")

    @property
    def running(self):
        return self.returncode is None

    def stop(self):
        if self.running:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def view(self):
        return dict(name=self.name, running=self.running, returncode=self.returncode, url=self.url, log=list(self.log)[-80:], **self.meta)


class Studio:
    def __init__(self, project, allow_spend=False, spawn=subprocess.Popen, open_browser=True, python=None, log=print, engine="panel"):
        self.project = os.path.abspath(project)
        self.labels_path = os.path.join(self.project, "labels.json")
        self.calib = os.path.join(self.project, "calib")
        self.solve = os.path.join(self.project, "solve")
        self.record_path = os.path.join(self.project, "studio.json")
        self.allow_spend, self.spawn, self.open_browser, self.say = allow_spend, spawn, open_browser, log
        self.engine = engine                                           # relief engine step 3 opens: 'panel' or 'domain' (dozens of qubits, a matrix-product state)
        self.python = python or sys.executable
        self.lock = threading.RLock()
        self.jobs = {}                       # "pen" | "show" | "targets" | "solve" -> Job (the latest of each)
        self._budget = (None, None)          # (labels sha, summary)
        os.makedirs(self.project, exist_ok=True)

    # ------------------------------------------------------------------ what is on disk
    def _record(self):
        try:
            with open(self.record_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _remember(self, **kv):
        rec = self._record()
        rec.update(kv)
        tmp = self.record_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, self.record_path)

    def labels_info(self):
        """(sha, summary or None, error or None) for the drawing; the budget is cached per sha."""
        sha = sha_file(self.labels_path)
        if sha is None:
            return None, None, None
        if self._budget[0] == sha:
            return sha, self._budget[1], None
        try:
            from src.capture.labels import load_labels
            from src.capture.pen_tool import summarise
            labels = load_labels(self.labels_path)
            b, _, _ = summarise(labels)
            summary = dict(patches=b["patches"], total=b["total"], over_cap=b["over_cap"], cap=b["cap"], text=b["text"],
                           source=labels.get("source", "photo"), grid=[labels["grid"]["nx"], labels["grid"]["ny"]])
        except Exception as e:                                          # noqa: BLE001 - a bad file is shown, not fatal
            return sha, None, f"{type(e).__name__}: {e}"
        self._budget = (sha, summary)
        return sha, summary, None

    def targets_ready(self):
        return all(os.path.exists(os.path.join(self.calib, n)) for n in ("targets.json", "payload_graph_v1.json", "payload_qdrive.json"))

    def rounds_state(self):
        """provenance.json of a QDrive built in rounds (python -m src.quantum.qdrive_rounds), or None. The studio does not send rounds itself
        (each is its own spend decision); it recognises the finished result so that step 5 can perform with it."""
        try:
            with open(os.path.join(self.solve, "qdrive", "provenance.json")) as f:
                d = json.load(f)
            return d if d.get("kind") == "qdrive-rounds" else None
        except (OSError, ValueError):
            return None

    @property
    def aer_dir(self):
        return os.path.join(self.solve, "aer")

    def aer_info(self):
        """state.json of the Aer run in <project>/solve/aer when it was made from the current drawing, else None."""
        from src.quantum.aer_relief import read_state
        st = read_state(self.aer_dir)
        sha = sha_file(self.labels_path)
        return st if st and sha and st.get("labels_sha") == sha else None

    @property
    def echo_dir(self):
        return os.path.join(self.solve, "echo")

    def echo_info(self):
        """state.json of the echo run in <project>/solve/echo when it was made from the current drawing, else None."""
        from src.quantum.echo_relief import read_state
        st = read_state(self.echo_dir)
        sha = sha_file(self.labels_path)
        return st if st and sha and st.get("labels_sha") == sha else None

    @property
    def qfacets_dir(self):
        return os.path.join(self.solve, "qdrive_facets")

    def qfacets_info(self):
        """state.json of the QDrive facet chain in <project>/solve/qdrive_facets when it is of the current drawing, else None."""
        from src.quantum.facet_qdrive import read_state
        st = read_state(self.qfacets_dir)
        sha = sha_file(self.labels_path)
        return st if st and sha and st.get("labels_sha") == sha else None

    def tomography_info(self):
        """state.json of the Moth tomography check in <project>/solve/tomography when it is of the current drawing and came from Moth (a local rehearsal does not count)."""
        from src.quantum.verify_moth import read_state
        st = read_state(os.path.join(self.solve, "tomography"))
        sha = sha_file(self.labels_path)
        return st if st and not st.get("local") and sha and st.get("labels_sha") == sha else None

    def circuit_path(self):
        p = os.path.join(self.solve, "qdrive", "circuit.qasm")
        return p if os.path.exists(p) else None

    def _running(self, name):
        j = self.jobs.get(name)
        return j if j and j.running else None

    # ------------------------------------------------------------------ the five stages
    def stage_views(self):
        sha, summary, err = self.labels_info()
        rec = self._record()
        pen, tgt, show, solve = (self.jobs.get(n) for n in ("pen", "targets", "show", "solve"))
        views = {}

        # 1 draw
        v = dict(status="ready", note="Opens the pen tool. Draw the panels live on the projector, then press Save in the tool.")
        if sha:
            v.update(status="done", note=("Saved drawing: " + summary["text"] + f" ({summary['source']} space, grid {summary['grid'][0]}x{summary['grid'][1]})") if summary
                     else f"labels.json exists but cannot be read: {err}")
            if summary and summary["over_cap"]:
                v["warn"] = f"{summary['total']} qubits is over the {summary['cap']}-qubit graph-v1 cap; use a coarser grid in step 2 or draw fewer shapes."
        if pen and pen.running:
            v.update(status="running", note="Pen tool is open. Press Save in it; this step turns to done when labels.json is written.")
        views["draw"] = v

        # 2 targets
        used = rec.get("targets_for")
        v = dict(status="blocked", note="Needs a saved drawing (step 1).")
        if sha:
            v.update(status="ready", note="Estimates the target <Z> and <ZZ> values for this drawing and writes both engine payloads. Sends nothing.")
            if self.targets_ready():
                if used == sha:
                    v.update(status="done", note="Targets match the current drawing.")
                else:
                    v.update(status="stale", note="Targets exist but were not made from the current drawing (or the studio cannot tell). Re-run." if used else
                             "Targets exist in the project folder but the studio did not make them, so it cannot tell they match this drawing. Re-run to be sure.")
        if tgt and tgt.running:
            v.update(status="running", note="Estimating targets (this can take a minute).")
        elif tgt and tgt.returncode not in (None, 0):
            v.update(status="failed", note=f"The last run failed (exit {tgt.returncode}); see the log.")
        views["targets"] = v

        ok_targets = views["targets"]["status"] == "done"
        blocked_targets = "Needs up-to-date targets (step 2)."

        # 3 preview
        v = dict(status="ready" if ok_targets else "blocked", note="Opens the show on Superposed Relief: a quantum circuit simulated exactly on this laptop (no Moth call, no credits)." if ok_targets else blocked_targets)
        if rec.get("previewed") == sha and ok_targets:
            v.update(status="done", note="Rehearsal opened for this drawing. Open it again any time.")
        if show and show.running and show.meta.get("mode") == "oracle":
            v.update(status="running", note="The show is open (Superposed Relief, local reference circuit). Stop it before you continue if the projector is needed elsewhere.")
        views["preview"] = v

        # 4 solve (a result is "current" when the payload on disk now has a saved result)
        v = dict(status="ready" if ok_targets else "blocked", note="Shows exactly what will run, its hash and its cost first. Nothing runs or is sent until you confirm: the default runs the relief circuit on Aer here (free); QDrive and graph-v1 are Moth engines that cost credits." if ok_targets else blocked_targets)
        results = {}
        if ok_targets:
            for eng in ENGINES:
                try:
                    p = self.plan(eng)
                except (StageError, OSError, ValueError):
                    continue
                results[eng] = dict(cached=p["cached"], orphans=p["orphans"])
            q = results.get("qdrive")
            rs = self.rounds_state()
            if rs and rs.get("complete") and self.circuit_path():
                jobs = sum(len(p.get("jobs", [])) for p in rs["pieces"])
                v.update(status="done", note=f"QDrive was built in rounds: {jobs} chained job(s)" + (f" plus {len(rs['local_pieces'])} locally prepared component(s)" if rs.get("local_pieces") else "") + ".")
                results["qdrive"] = dict(results.get("qdrive") or dict(orphans=[]), cached=True, rounds=True)
            elif rs and not rs.get("complete") and not (q and q["cached"]):
                v["warn"] = "QDrive rounds are unfinished: python -m src.quantum.qdrive_rounds run (see --help). Only circuit_partial.qasm exists, and the show does not use it."
            elif q and q["cached"]:
                v.update(status="done", note="QDrive has a saved result for exactly this payload.")
            elif q and q["orphans"]:
                v["warn"] = (f"{len(q['orphans'])} QDrive job(s) were submitted for this project but no result was saved ({', '.join(o[:8] for o in q['orphans'])}). "
                             "Credits may have been spent; check the job in Moth before sending again.")
        aer = self.aer_info() if ok_targets else None
        if aer:
            sc = aer["score"]
            text = (f"Superposed Relief was executed on Aer: {aer['n_qubits']} qubits, {aer['shots']:,} shots per lamp reading"
                    + (f", facet register: {aer['facets']['label']}" if aer.get("facets") else "") + "; "
                    + ("the shots agree with the reference state within shot noise." if sc["consistent"] else "the shots do NOT agree with the reference state (see state.json)."))
            v.update(status="done", note=(v["note"] + " " if v["status"] == "done" else "") + text)
            results[AER] = dict(cached=True, orphans=[])
        echo = self.echo_info() if ok_targets else None
        if echo:
            src = "Moth's otoc-echo-v1" if echo["taps_source"] == "moth" else "the local echo"
            text = (f"Dynamic relief was executed on Aer: {echo['depth']} circuits of {echo['n_qubits']} qubits, {echo['shots']:,} shots each, taps from {src}; "
                    + ("every depth agrees with its reference state within shot noise." if echo["consistent"] else "a depth does NOT agree with its reference state (see state.json)."))
            v.update(status="done", note=(v["note"] + " " if v["status"] == "done" else "") + text)
            results[ECHO] = dict(cached=True, orphans=[])
        elif ok_targets and os.path.exists(os.path.join(self.echo_dir, "state.json")):
            v["warn"] = (v.get("warn", "") + " " if v.get("warn") else "") + "The saved echo run was made from an earlier drawing; run it again."
        if not aer and ok_targets and read_aer_exists(self.aer_dir):
            v["warn"] = (v.get("warn", "") + " " if v.get("warn") else "") + "The saved Aer run was made from an earlier drawing; run it again."
        v["extras"] = []
        chain = self.qfacets_info() if ok_targets else None
        if chain:
            results[QFAC] = dict(cached=bool(chain["complete"]), orphans=[])
            v["chain"] = (f"QDrive facet register: {chain['jobs_done']} of {chain['jobs_total']} job(s) done, fidelity to the ideal facet state {chain['score']['fidelity']:.3f}"
                          + (" (LOCAL rehearsal, not a Moth result)" if chain.get("local") else "") + (": complete, ready for the Aer run." if chain["complete"] else "."))
            v["extras"].append(v["chain"])
        tomo = self.tomography_info() if ok_targets else None
        if tomo:
            sc, cmp_aer = tomo["score"], tomo.get("aer_comparison") or {}
            results[TOMO] = dict(cached=True, orphans=[])
            v["verified"] = (f"Moth tomography-api-v2 verified the reference circuit ({tomo['n_qubits']} qubits): rms error of the single-qubit moments {sc['rms_single']:.3g}"
                             + (f", of {sc['n_pair_values']} pair correlators {sc['rms_pairs']:.3g}" if sc.get("rms_pairs") is not None else "")
                             + (f"; polarity visibilities within {cmp_aer['max_abs_difference']:.3f} of the Aer witness run" if cmp_aer.get("max_abs_difference") is not None else "") + ".")
            v["extras"].append(v["verified"])
        if solve and solve.running:
            v.update(status="running", note="Running the solver.")
        elif solve and solve.returncode not in (None, 0):
            v.update(status="failed", note=f"The last run failed (exit {solve.returncode}); see the log.")
        v["results"] = results
        views["solve"] = v

        # 5 final
        circuit = self.circuit_path()
        choices = ([AER] if aer else []) + ([ECHO] if echo else []) + (["qdrive"] if ok_targets and circuit and views["solve"].get("results", {}).get("qdrive", {}).get("cached") else [])
        used = None
        if choices:
            used = {AER: lambda: sha_file(os.path.join(self.aer_dir, "shots.npz")), ECHO: lambda: sha_file(os.path.join(self.echo_dir, "shots.npz")),
                    "qdrive": lambda: sha_file(circuit)}[choices[0]]()
        both = " More results are available (more buttons)." if len(choices) > 1 else ""
        v = dict(status="ready" if choices else "blocked", choices=choices,
                 note=({AER: "Opens the show on the Aer run: every look is one measured shot of the relief circuit executed on Aer (a statevector simulation here, not a Moth result).",
                        ECHO: "Opens the show on the echo run: dynamic relief, every look steps to the next echo depth and is one measured shot of that depth's circuit executed on Aer.",
                        "qdrive": "Opens the show on the circuit QDrive returned (simulated locally; the older classical-family flow)."}[choices[0]] + both if choices else
                       "Needs a result from step 4: the Aer run or the echo run (free, local) or a QDrive result for the current payload; graph-v1 returns no circuit to draw from."))
        if choices and rec.get("final") == used:
            v.update(status="done", note="Performed with this result. Open it again any time." + both)
        if show and show.running and show.meta.get("mode") in ("circuit", "aer", "echo"):
            v.update(status="running", note={"aer": "The show is open on the Aer run.", "echo": "The show is open on the echo run.", "circuit": "The show is open on the QDrive circuit."}[show.meta["mode"]])
        views["final"] = v

        for name, job_name in dict(draw="pen", targets="targets", solve="solve").items():
            j = self.jobs.get(job_name)
            views[name]["job"] = j.view() if j else None
        for name, modes in dict(preview=("oracle",), final=("circuit", "aer", "echo")).items():
            views[name]["job"] = show.view() if show and show.meta.get("mode") in modes else None
        order = [s for s in STAGES if views[s]["status"] not in ("done", "blocked")]
        return [dict(id=s, title=TITLES[s], **views[s]) for s in STAGES], (order[0] if order else None)

    def state(self):
        stages, nxt = self.stage_views()
        return dict(project=os.path.relpath(self.project, ROOT) if self.project.startswith(ROOT) else self.project,
                    allow_spend=self.allow_spend, engine=self.engine, knobs=knob_schema(), stages=stages, next=nxt, busy=bool(self._running("targets") or self._running("solve")))

    # ------------------------------------------------------------------ starting things
    def _py(self, *args):
        return [self.python, "-m", *args]

    def _server_job(self, name, argv, port, open_browser=None, **meta):
        old = self.jobs.get(name)
        if old:
            old.stop()
        url = f"http://127.0.0.1:{port}/"
        job = self.jobs[name] = Job(name, argv, self.spawn, url=url, **meta)
        if self.open_browser if open_browser is None else open_browser:
            threading.Thread(target=self._open_when_up, args=(job, port), daemon=True).start()
        return job

    def _open_when_up(self, job, port):
        for _ in range(120):
            if not job.running:
                return
            if port_open(port):
                webbrowser.open(job.url)
                return
            time.sleep(0.5)

    def _require_idle(self):
        for n in ("targets", "solve"):
            if self._running(n):
                raise StageError(f"the {n} step is still running; wait for it to finish")

    def start(self, stage, opts=None):
        opts = opts or {}
        with self.lock:
            if stage not in STAGES:
                raise StageError(f"unknown step {stage!r}")
            views = {s["id"]: s for s in self.stage_views()[0]}
            if views[stage]["status"] == "blocked":
                raise StageError(views[stage]["note"])
            return getattr(self, "_start_" + stage)(opts)

    def _start_draw(self, opts):
        port = free_port()
        argv = self._py("src.capture.pen_tool", "draw")
        photo = (opts.get("photo") or "").strip()
        if photo:
            if not os.path.isfile(photo):
                raise StageError(f"no photo at {photo!r}")
            argv.append(photo)
        if os.path.exists(self.labels_path):
            from src.capture.labels import load_labels
            if load_labels(self.labels_path).get("source", "photo") == "photo" and not photo:
                raise StageError("this drawing was traced on a photo; give the photo's path to resume it (Save would overwrite it otherwise)")
            argv += ["--labels", self.labels_path]
        argv += ["--out", self.project, "--port", str(port), "--no-browser"]
        self._server_job("pen", argv, port)
        return dict(url=self.jobs["pen"].url)

    def _start_targets(self, opts):
        self._require_idle()
        sha = self.labels_info()[0]
        argv = self._py("src.targets.calibrate_from_mock", "--labels", self.labels_path, "--out", self.calib)
        grid = opts.get("grid")
        if grid:
            nx, ny = (int(v) for v in grid)
            if not (1 <= nx <= 40 and 1 <= ny <= 40):
                raise StageError("grid must be between 1x1 and 40x40")
            argv += ["--grid", str(nx), str(ny)]
        if opts.get("sweeps"):
            sweeps = int(opts["sweeps"])
            if not 1000 <= sweeps <= 2_000_000:
                raise StageError("sweeps must be between 1000 and 2,000,000")
            argv += ["--sweeps", str(sweeps)]
        job = self.jobs["targets"] = Job("targets", argv, self.spawn, labels_sha=sha)
        threading.Thread(target=self._after_targets, args=(job, sha), daemon=True).start()
        return {}

    def _after_targets(self, job, sha):
        while job.running:
            time.sleep(0.05)
        if job.returncode == 0:
            self._remember(targets_for=sha, previewed=None, final=None)

    def _knob_values(self, stage, raw, engine_default=None):
        """Check the show knobs the page sent for this stage; returns {key: typed value} for the ones that differ from the show's own default."""
        raw = dict(raw or {})
        if engine_default and not raw.get("engine") and (raw.get("source") or "relief") == "relief":
            raw["engine"] = engine_default
        for key in raw:
            if key not in KNOB:
                raise StageError(f"unknown option {key!r}")
        given = {}
        for key, kind, default, lim, stages, needs, _ in KNOBS:
            v = raw.get(key)
            if v is None or (isinstance(v, str) and not v.strip()) or (kind == "bool" and not v):
                continue
            flag = "--" + key.replace("_", "-")
            if stage not in stages:
                raise StageError(f"{flag} does not apply to the {TITLES[stage].lower()} step")
            try:
                if kind == "float":
                    v = float(v)
                    if not (v == v and lim[0] <= v <= lim[1]):
                        raise ValueError
                elif kind == "int":
                    f = float(v)
                    if f != int(f) or not lim[0] <= int(f) <= lim[1]:
                        raise ValueError
                    v = int(f)
                elif kind == "bool":
                    if v is not True:
                        raise ValueError
                elif kind == "choice":
                    v = str(v).strip()
                    if v not in lim:
                        raise ValueError
                elif kind == "size":
                    w, h = (int(x) for x in str(v).lower().split("x"))
                    if not (1 <= w <= 16384 and 1 <= h <= 16384):
                        raise ValueError
                    v = f"{w}x{h}"
                else:                                                   # path: relative paths are relative to the project root, where the show runs
                    v = os.path.abspath(os.path.join(ROOT, os.path.expanduser(str(v).strip())))
                    if not (os.path.isdir(v) if lim == "dir" else os.path.isfile(v)):
                        raise StageError(f"{flag}: no {'folder' if lim == 'dir' else 'file'} at {v!r}")
            except StageError:
                raise
            except (ValueError, TypeError, IndexError, OverflowError):
                shown = f"between {lim[0]} and {lim[1]}" if kind in ("float", "int") else f"one of {', '.join(lim)}" if kind == "choice" else "like 1920x1080" if kind == "size" else "on or off"
                raise StageError(f"{flag} must be {shown}")
            if v != default:
                given[key] = v
        source = given.get("source", "relief") if stage == "preview" else "circuit"
        engine = given.get("engine", "panel")
        for key in given:
            needs = KNOB[key][5]
            flag = "--" + key.replace("_", "-")
            if needs in ("relief", "domain") and source != "relief":
                raise StageError(f"{flag} applies to the relief source only")
            if needs == "domain" and engine != "domain":
                raise StageError(f"{flag} applies to the domain engine only")
        if given.get("lamp_mode") == "x" and given.get("game"):
            raise StageError("--game needs --lamp-mode z: with the lamp read in X no round is played")
        if given.get("allow_spend") and not self.allow_spend:
            raise StageError("the studio was started without --allow-spend, so the show cannot be given it")
        return given

    def _show_argv(self, port, source, knobs=None, extra=()):
        knobs = dict(knobs or {})
        calib, run = knobs.pop("calib", self.calib), knobs.pop("run", self.solve)
        no_browser, user_port = knobs.pop("no_browser", False), knobs.pop("port", None)
        argv = self._py("src.show", "--labels", self.labels_path, "--calib", calib, "--run", run, "--out", self.project, "--source", knobs.pop("source", source), *extra)
        for key, v in knobs.items():
            flag = "--" + key.replace("_", "-")
            argv += [flag] if v is True else [flag, str(v)]
        return argv + ["--port", str(port), "--no-browser"]

    def _show_port(self, knobs):
        port = knobs.get("port") or free_port()
        if knobs.get("port") and port_open(port):
            raise StageError(f"port {port} is already in use")
        return port

    def _start_preview(self, opts):
        knobs = self._knob_values("preview", (opts or {}).get("show"), engine_default=(opts or {}).get("engine", self.engine))
        port = self._show_port(knobs)
        self._server_job("show", self._show_argv(port, "relief", knobs), port, open_browser=False if knobs.get("no_browser") else None, mode="oracle")
        self._remember(previewed=self.labels_info()[0])
        return dict(url=self.jobs["show"].url)

    def _start_final(self, opts):
        opts = opts or {}
        knobs = self._knob_values("final", opts.get("show"))
        choices = next(s for s in self.stage_views()[0] if s["id"] == "final")["choices"]
        want = opts.get("from") or choices[0]
        if want not in choices:
            raise StageError(f"there is no current {'Aer run' if want == AER else 'echo run' if want == ECHO else 'QDrive result' if want == 'qdrive' else repr(want)} to perform with; step 4 makes one")
        port = self._show_port(knobs)
        no_browser = False if knobs.get("no_browser") else None
        if want == AER:
            flags = [x for k, v in (self.aer_info().get("flags") or {}).items() for x in ("--" + k.replace("_", "-"), str(v))]
            self._server_job("show", self._show_argv(port, "relief", knobs, ("--aer-run", self.aer_dir, *flags)), port, open_browser=no_browser, mode="aer")
            self._remember(final=sha_file(os.path.join(self.aer_dir, "shots.npz")))
        elif want == ECHO:
            flags = self._flag_args(self.echo_info().get("flags") or {})
            self._server_job("show", self._show_argv(port, "relief", knobs, ("--echo-run", self.echo_dir, *flags)), port, open_browser=no_browser, mode="echo")
            self._remember(final=sha_file(os.path.join(self.echo_dir, "shots.npz")))
        else:
            circuit = self.circuit_path()
            self._server_job("show", self._show_argv(port, "circuit", knobs, ("--circuit", circuit)), port, open_browser=no_browser, mode="circuit")
            self._remember(final=sha_file(circuit))
        return dict(url=self.jobs["show"].url)

    def _start_solve(self, opts):
        raise StageError("step 4 is started from its plan: use 'Review payload' and then the send button")

    def stop(self, name):
        with self.lock:
            job = self.jobs.get(name)
            if not job:
                raise StageError(f"nothing named {name!r} is running")
            job.stop()

    def stop_all(self):
        for j in list(self.jobs.values()):
            j.stop()

    def use_demo_scene(self):
        """Start from the built-in bay window (three wall sections, two stacked windows each: six panels; 4x4 grid, 20 qubits) so every step can be
        tried before anything is drawn."""
        with self.lock:
            if os.path.exists(self.labels_path):
                raise StageError("this project already has a drawing; it will not be overwritten")
            from src.capture.labels import bay_window_labels, fill_defaults, save_labels
            save_labels(fill_defaults(bay_window_labels()), self.labels_path)

    # ------------------------------------------------------------------ solve: plan, then an explicit matching confirm
    # the relief engines (their plans are built from the drawing and the show options, not from calib/): engine id -> plan method
    RELIEF_ENGINES = {"aer": "plan_aer", "tomography": "plan_tomography", "qdrive-facets": "plan_qfacets", "echo-local": "plan_echo_local", "echo-moth": "plan_echo_moth"}

    def _relief_flags(self, show, what):
        """The relief flags among the show options (kappa, n_dirs, entangle, contrast, observations), validated. Refuses what a per-panel circuit cannot be."""
        from src.quantum import aer_relief as ar
        if not os.path.exists(self.labels_path):
            raise StageError("needs a saved drawing (step 1)")
        raw = {k: v for k, v in (show or {}).items() if k in ar.RELIEF_FLAGS or k in ("engine", "source")}
        if (raw.get("engine") or self.engine) != "panel" or (raw.get("source") or "relief") != "relief":
            raise StageError(f"{what} covers the per-panel relief circuit; the domain engine's 69 qubits and the classical sources are not part of it (set show option engine to panel)")
        given = self._knob_values("preview", raw)
        return {k: given[k] for k in ar.RELIEF_FLAGS if k in given}

    @staticmethod
    def _opt_int(opts, key, default, lo, hi, what):
        v = (opts or {}).get(key)
        try:
            v = default if v in (None, "") else int(v)
        except (TypeError, ValueError):
            raise StageError(f"{what} must be a whole number")
        if not lo <= v <= hi:
            raise StageError(f"{what} must be between {lo:,} and {hi:,}")
        return v

    @staticmethod
    def _flag_args(flags):
        return [x for k, v in flags.items() for x in ("--" + k.replace("_", "-"), str(v))]

    @staticmethod
    def _orphans(run_dir):
        """Job ids submitted into run_dir that never produced a raw result (credits may have been spent)."""
        done = {os.path.basename(p)[len("raw_result_"):-5] for p in _glob(run_dir, "raw_result_*.json")}
        return [os.path.basename(p)[4:-5] for p in _glob(run_dir, "job_*.json") if os.path.basename(p)[4:-5] not in done and not os.path.basename(p)[4:].startswith("local-")]

    def plan_aer(self, show=None, opts=None):
        """Step 4's local engine: what the Aer run would execute, its hash, and whether a run for exactly this is already on disk. Costs nothing and sends nothing."""
        from src.capture.labels import load_labels
        from src.quantum import aer_relief as ar
        from src.quantum.relief_state import full_circuit
        flags = self._relief_flags(show, "the Aer run")
        shots = self._opt_int(opts, "shots", ar.DEFAULT_SHOTS, 10_000, 20_000_000, "shots")
        _, _, spec = ar.build_spec(load_labels(self.labels_path), flags.get("kappa", 1.0), flags.get("n_dirs"), flags.get("entangle", 0.8), flags.get("observations"))
        facets = (opts or {}).get("facets") or "ideal"
        if facets not in ("ideal", "qdrive"):
            raise StageError("facets must be 'ideal' or 'qdrive'")
        fqc = fpath = fsha = fnote = None
        if facets == "qdrive":
            chain = self.qfacets_info()
            fpath = os.path.join(self.qfacets_dir, "circuit.qasm")
            if not chain or not chain.get("complete") or not os.path.exists(fpath) or chain.get("spec") != ar.spec_fingerprint(spec):
                raise StageError("there is no finished QDrive facet chain for these settings and this drawing: run the 'QDrive facet register' engine until its jobs are done "
                                 "(or choose the hand-built facet circuit)")
            fqc, _, fsha = ar.load_facet_circuit(fpath, spec)
            fnote = (f"facet register: {ar.facet_provenance(fpath)['label']}; fidelity to the ideal facet state {chain['score']['fidelity']:.4f}")
        qc = full_circuit(spec, facet_circuit=fqc)
        sha = ar.plan_sha(spec, shots, ar.DEFAULT_SEED, fsha)
        have = ar.read_state(self.aer_dir)
        cached = bool(have and have["sha256"] == sha)
        ops = ", ".join(f"{n} x{c}" for n, c in sorted(qc.count_ops().items()))
        describe = (f"Superposed Relief, per-panel engine: {spec.F} facet qubits + {spec.P} polarity + 2 lamp + 2 observe = {spec.n_full} qubits"
                    + (f"; {spec.n_classical} flat facets stay classical (no qubit)" if spec.n_classical else "") + "\n"
                    f"circuit: {qc.size()} operations ({ops})\n"
                    f"{shots:,} shots on Aer's statevector simulator with seed {ar.DEFAULT_SEED}, once with the lamp register read in Z and once in X; then a check of the shots against the "
                    "reference state and a witness run (polarity in X, facets in Z).\n" + (fnote + "\n" if fnote else "") + "Executed here by qiskit-aer: no network, no key, no credits.")
        command = self._py("src.quantum.aer_relief", "--labels", self.labels_path, "--out", self.aer_dir, "--shots", str(shots), "--seed", str(ar.DEFAULT_SEED), *self._flag_args(flags),
                           *(["--facet-circuit", fpath] if fpath else []))
        return dict(engine=AER, engine_id="relief-aer (this laptop)", sha256=sha, credits=0, cached=cached, problems=ar.check_size(spec), describe=describe,
                    allow_spend=self.allow_spend, orphans=[], flags=flags, shots=shots, opts=dict(shots=shots, facets=facets), command=command, action="Run on Aer here (0 credits)", hash_label="circuit", local=True,
                    note=("A run for exactly this circuit, shot count and seed is already on disk: nothing is executed." if cached else
                          "Executes this circuit on Aer on this machine (a few seconds) and saves the measured shots. Nothing is sent anywhere and no credits are spent."))

    def plan_echo_local(self, show=None, opts=None):
        return self._plan_echo("local", show, opts)

    def plan_echo_moth(self, show=None, opts=None):
        return self._plan_echo("moth", show, opts)

    def _plan_echo(self, source, show, opts):
        """Dynamic relief: the echo's taps (local, or Moth's otoc-echo-v1) modulate the relief and one circuit per echo depth is executed on Aer."""
        from src.capture.labels import load_labels
        from src.quantum import aer_relief as ar
        from src.quantum import echo_relief as er
        from src.store.cache import cached_result
        flags = self._relief_flags(show, "the echo run")
        depth = self._opt_int(opts, "depth", er.DEFAULT_DEPTH, 1, er.MAX_DEPTH, "echo depth")
        width = self._opt_int(opts, "width", 4, 1, 24, "lattice width")
        height = self._opt_int(opts, "height", 4, 1, 24, "lattice height")
        shots = self._opt_int(opts, "shots", er.DEFAULT_SHOTS, 10_000, 5_000_000, "shots per depth")
        spec = ar.build_spec(load_labels(self.labels_path), flags.get("kappa", 1.0), flags.get("n_dirs"), flags.get("entangle", 0.8), flags.get("observations"))[2]
        problems = ar.check_size(spec) + er.check_lattice(width, height, depth, source)
        kick = width * height // 2
        moth_sha = job = None
        moth_cached, orphans, credits = True, [], 0
        if source == "moth":
            _, moth_sha, job = er.moth_job(width, height, depth, kick)
            problems += job.problems()
            moth_cached = bool(cached_result(os.path.join(self.echo_dir, "moth"), moth_sha))
            orphans = self._orphans(os.path.join(self.echo_dir, "moth"))
            credits = 0 if moth_cached else job.credits
        sha = er.plan_sha(spec, source, width, height, depth, kick, shots, ar.DEFAULT_SEED, moth_sha)
        have = er.read_state(self.echo_dir)
        cached = bool(have and have["sha256"] == sha)
        flat = f"; {spec.n_classical} flat facets stay classical" if spec.n_classical else ""
        describe = (f"Dynamic relief: {depth} echo depth(s) on a {width}x{height} lattice (kick at site {kick}); the taps scale each facet's tilt and turn its slope direction, "
                    f"and one relief circuit per depth ({spec.n_full} qubits{flat}) is executed on Aer with {shots:,} shots each (lamp read in Z), checked against its reference state.\n"
                    + ("Taps: the LOCAL echo, our reading of the engine's description (free)." if source == "local" else
                       "Taps: Moth's otoc-echo-v1 (machine aer).\n" + job.describe(max_ops=3)[:2500]))
        command = self._py("src.quantum.echo_relief", "--labels", self.labels_path, "--out", self.echo_dir, "--source", source, "--depth", str(depth), "--width", str(width),
                           "--height", str(height), "--shots", str(shots), "--seed", str(ar.DEFAULT_SEED), *self._flag_args(flags))
        local_run = source == "local" or moth_cached
        return dict(engine=ECHO_LOCAL if source == "local" else ECHO_MOTH, engine_id="relief-echo (taps: " + ("local echo" if source == "local" else "otoc-echo-v1") + ")",
                    sha256=sha, credits=credits, cached=cached, problems=problems, describe=describe, allow_spend=self.allow_spend, orphans=orphans, flags=flags,
                    opts=dict(depth=depth, width=width, height=height, shots=shots), command=command, hash_label="echo run", local=local_run,
                    action="Run on Aer here (0 credits)" if local_run else None,
                    note=("A run for exactly this echo, circuit, shot count and seed is already on disk: nothing is executed." if cached else
                          (f"Executes {depth} circuits on Aer on this machine (about {depth * 4} s). Nothing is sent anywhere and no credits are spent." if source == "local" else
                           "The Moth result for this echo request is already on disk (0 credits); the circuits are executed on Aer here." if moth_cached else
                           f"Sends the echo request to Moth and spends {job.credits} credit(s), then executes the {depth} circuits on Aer here."
                           + ("" if self.allow_spend else " Disabled: the studio was started without --allow-spend."))))

    def plan_qfacets(self, show=None, opts=None):
        """Prepare the facet register with QDrive: the NEXT job of the chain (each job is its own spend decision), or a note that the chain is finished."""
        import hashlib
        from src.capture.labels import load_labels
        from src.quantum import aer_relief as ar
        from src.quantum import facet_qdrive as fq
        flags = self._relief_flags(show, "the QDrive facet register")
        spec = ar.build_spec(load_labels(self.labels_path), flags.get("kappa", 1.0), flags.get("n_dirs"), flags.get("entangle", 0.8), flags.get("observations"))[2]
        chain = fq.plan_chain(spec)
        st = fq.chain_state(self.qfacets_dir, chain, spec)
        k, blocked = fq.next_index(chain, st)
        command = self._py("src.quantum.facet_qdrive", "send", "--labels", self.labels_path, "--out", self.qfacets_dir, "--jobs", "1", *self._flag_args({x: v for x, v in flags.items() if x != "contrast"}))
        base = dict(engine=QFAC, engine_id=fq.ENGINE + " (facet register)", allow_spend=self.allow_spend, flags=flags, opts={}, command=command, action=None, hash_label="job",
                    orphans=[blocked] if blocked else [])
        if k is None:
            return dict(base, sha256=hashlib.sha256(f"{chain['sha']}:done".encode()).hexdigest(), credits=0, cached=True, problems=[], describe=f"all {len(chain['jobs'])} job(s) are done",
                        note="The QDrive facet chain is finished: nothing is sent. Choose 'facet register: QDrive' in the Aer run to execute the relief on it.")
        job = fq.preview_job(chain, st, k)
        sha = hashlib.sha256(f"{chain['sha']}:{k}".encode()).hexdigest()
        n_t = sum(1 for t in job.params["targets"] if t)
        return dict(base, sha256=sha, credits=job.credits, cached=False, problems=job.problems() + ([blocked] if blocked else []),
                    describe=f"job {k + 1} of {len(chain['jobs'])} of the facet-register chain: {n_t} target(s) on {spec.F} qubits" + (f", continuing job {k} (its output asset)" if k else "") + "\n" + job.describe(max_ops=3)[:3000],
                    note=f"Sends job {k + 1} of {len(chain['jobs'])} to Moth and spends {job.credits} credit(s); the returned circuit is scored against the ideal facet state before the next job is offered."
                         + ("" if self.allow_spend else " Disabled: the studio was started without --allow-spend."))

    def plan_tomography(self, show=None, opts=None):
        """Verify on Moth: the reference relief circuit sent to tomography-api-v2 (1 credit), scored against the exact moments and the Aer witness run."""
        from src.capture.labels import load_labels
        from src.quantum import verify_moth as vm
        from src.store.cache import cached_result
        flags = self._relief_flags(show, "the tomography check")
        shots = self._opt_int(opts, "shots", 4096, 256, 100_000, "tomography shots")
        spec, params, sha, job = vm.plan(load_labels(self.labels_path), flags, shots)
        run_dir = os.path.join(self.solve, "tomography")
        cached = bool(cached_result(run_dir, sha))
        command = self._py("src.quantum.verify_moth", "--labels", self.labels_path, "--out", run_dir, "--aer-run", self.aer_dir, "--shots", str(shots), *self._flag_args(flags))
        return dict(engine=TOMO, engine_id=vm.ENGINE, sha256=sha, credits=0 if cached else job.credits, cached=cached, problems=job.problems(), describe=job.describe(max_ops=3)[:4000],
                    allow_spend=self.allow_spend, orphans=self._orphans(run_dir), flags=flags, opts=dict(shots=shots), command=command,
                    action=None, hash_label="payload",
                    note=("A result for exactly this payload is already on disk: nothing is sent and it costs nothing." if cached else
                          f"Sends the reference circuit ({spec.n_sim} qubits, no lamp or observe registers) to Moth and spends {job.credits} credit(s); the result is scored against the exact "
                          "moments (and the Aer witness run if there is one)." + ("" if self.allow_spend else " Disabled: the studio was started without --allow-spend.")))

    def plan(self, engine, show=None, opts=None):
        if engine in self.RELIEF_ENGINES:
            return getattr(self, self.RELIEF_ENGINES[engine])(show, opts)
        from src.quantum.moth_client import MothClient
        from src.store.cache import cached_result
        from src.targets.payloads import payload_sha256
        if engine not in ENGINES:
            raise StageError(f"unknown engine {engine!r}")
        fname, eid, sub = ENGINES[engine]
        path = os.path.join(self.calib, fname)
        if not os.path.exists(path):
            raise StageError("no payload yet: run step 2")
        with open(path) as f:
            params = json.load(f)["params"]
        sha = payload_sha256({"engine": eid, "params": params})
        run_dir = os.path.join(self.solve, sub)
        cached = cached_result(run_dir, sha)
        job = MothClient().prepare(eid, params)                   # builds the request only: no key, no network
        done = {os.path.basename(p)[len("raw_result_"):-5] for p in _glob(run_dir, "raw_result_*.json")}
        orphans = [os.path.basename(p)[4:-5] for p in _glob(run_dir, "job_*.json") if os.path.basename(p)[4:-5] not in done]
        return dict(engine=engine, engine_id=eid, sha256=sha, credits=0 if cached else job.credits, cached=bool(cached), problems=job.problems(),
                    describe=job.describe(max_ops=6), allow_spend=self.allow_spend, orphans=orphans,
                    note=("A result for exactly this payload is already on disk: nothing is sent and it costs nothing." if cached else
                          f"Sends this payload to Moth and spends {job.credits} credit(s)." + ("" if self.allow_spend else
                          " Disabled: the studio was started without --allow-spend (preview only). Restart it with --allow-spend to send.")))

    def confirm(self, engine, sha256, credits, show=None, opts=None):
        with self.lock:
            self._require_idle()
            if engine in self.RELIEF_ENGINES:
                return self._confirm_relief(engine, sha256, credits, show, opts)
            plan = self.plan(engine)
            if plan["problems"]:
                raise StageError("this payload cannot be sent: " + "; ".join(plan["problems"]))
            if sha256 != plan["sha256"] or credits != plan["credits"]:
                raise StageError("the payload or its cost changed since it was shown: review it again and read it before sending")
            if plan["credits"] is None:
                raise StageError("the price of this engine is not known, so it will not be sent")
            if plan["credits"] and not self.allow_spend:
                raise PermissionError("the studio was started without --allow-spend, so it will not submit to Moth")
            argv = self._py("src.quantum.solver", engine, "--calib", self.calib, "--out", self.solve)
            if plan["credits"]:
                argv += ["--approve-credits", str(plan["credits"])]
            self.jobs["solve"] = Job("solve", argv, self.spawn, engine=engine, credits=plan["credits"], sha256=plan["sha256"])
            return dict(started=True, plan=plan)


    def _confirm_relief(self, engine, sha256, credits, show, opts):
        """Run a relief engine's plan: the request must repeat the hash and the cost it was shown with; Moth engines also need --allow-spend."""
        plan = self.plan(engine, show, opts)
        if plan["problems"]:
            raise StageError("this cannot be run: " + "; ".join(plan["problems"]))
        if sha256 != plan["sha256"] or credits != plan["credits"]:
            raise StageError("the request or its cost changed since it was shown: review it again before running")
        if plan["credits"] and not self.allow_spend:
            raise PermissionError("the studio was started without --allow-spend, so it will not submit to Moth")
        argv = list(plan["command"])
        if plan["credits"]:
            argv += ["--approve-credits", str(plan["credits"])]
        self.jobs["solve"] = Job("solve", argv, self.spawn, engine=engine, credits=plan["credits"], sha256=plan["sha256"])
        return dict(started=True, plan=plan)


def read_aer_exists(folder):
    return os.path.exists(os.path.join(folder, "state.json"))


def _glob(d, pattern):
    import glob
    return glob.glob(os.path.join(d, pattern))


# ---------------------------------------------------------------------- the page's server
def make_handler(studio, token, hosts):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            body = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj), "application/json")

        def _ok(self):
            """Loopback client, loopback Host header (DNS-rebinding guard): the studio is for this machine only."""
            if self.client_address[0] not in LOOPBACK or self.headers.get("Host") not in hosts:
                self._send(403, "forbidden: the studio is available on this machine only", "text/plain")
                return False
            return True

        def do_GET(self):
            if not self._ok():
                return
            path = urlparse(self.path).path
            if path == "/":
                with open(PAGE, encoding="utf-8") as f:
                    return self._send(200, f.read().replace("__TOKEN__", token), "text/html; charset=utf-8")
            if path == "/web/common.css":
                with open(CSS, encoding="utf-8") as f:
                    return self._send(200, f.read(), "text/css; charset=utf-8")
            if path == "/api/state":
                return self._json(200, studio.state())
            self._send(404, "not found", "text/plain")

        def do_POST(self):
            if not self._ok():
                return
            if self.headers.get("X-Token") != token:
                return self._json(403, dict(ok=False, error="bad token"))
            path = urlparse(self.path).path
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > MAX_BODY:
                    return self._json(413, dict(ok=False, error="body too large"))
                body = json.loads(self.rfile.read(n) or b"{}")
                if path == "/api/start":
                    return self._json(200, dict(ok=True, **studio.start(body["stage"], body.get("opts"))))
                if path == "/api/stop":
                    studio.stop(body["job"])
                    return self._json(200, dict(ok=True))
                if path == "/api/demo-scene":
                    studio.use_demo_scene()
                    return self._json(200, dict(ok=True))
                if path == "/api/solve/plan":
                    return self._json(200, dict(ok=True, plan=studio.plan(body.get("engine", "qdrive"), body.get("show"), body.get("opts"))))
                if path == "/api/solve/confirm":
                    return self._json(200, dict(ok=True, **studio.confirm(body["engine"], body["sha256"], body["credits"], body.get("show"), body.get("opts"))))
                self._json(404, dict(ok=False, error="not found"))
            except PermissionError as e:
                self._json(403, dict(ok=False, error=str(e)))
            except (StageError, ValueError, KeyError, TypeError, OSError) as e:
                self._json(400, dict(ok=False, error=str(e) if isinstance(e, StageError) else f"{type(e).__name__}: {e}"))

    return Handler


def make_server(studio, port=0):
    """Create (not start) the server. Returns (server, token, url)."""
    token = secrets.token_urlsafe(16)
    server = ThreadingHTTPServer(("127.0.0.1", port), lambda *a, **k: None)
    server.daemon_threads = True
    real = server.server_address[1]
    server.RequestHandlerClass = make_handler(studio, token, {f"127.0.0.1:{real}", f"localhost:{real}"})
    return server, token, f"http://127.0.0.1:{real}/"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="runs/studio", help="folder for this scene's drawing, targets and results (default runs/studio)")
    ap.add_argument("--allow-spend", action="store_true", help="let step 4 submit to Moth (still shows the cost and asks first)")
    ap.add_argument("--port", type=int, default=0, help="default: any free port")
    ap.add_argument("--engine", choices=["panel", "domain"], default="panel", help="relief engine step 3 opens: panel (exact: facets + one polarity qubit per panel, 18 qubits for the six-pane bay window) or domain (one polarity qubit per depth domain)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    studio = Studio(a.project, allow_spend=a.allow_spend, open_browser=not a.no_browser, log=lambda m: print("  " + m), engine=a.engine)
    server, _, url = make_server(studio, a.port)
    print(f"Standing Light studio: {url}\n  project: {studio.project}\n  Moth submit from step 4: "
          + ("ENABLED (--allow-spend), still asks first" if a.allow_spend else "off (preview only; restart with --allow-spend to send)") + "\nCtrl+C to stop (this also stops the tools it started).")
    if not a.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    def _stop(*_):                                       # closing the terminal or `kill` must not leave the show or pen tool running
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGHUP, _stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        studio.stop_all()
        server.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
