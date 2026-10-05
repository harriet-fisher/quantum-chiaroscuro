"""Performance session (handoff §10.2): everything the operator controls, independent of any HTTP or browser code.

A Session owns the scene, a pool of measurement outcomes (from a quantum state, the classical mock, or independent noise), the
knobs, the history of looks, the calibration and the output size. It turns a draw into a projector frame (compose -> warp ->
exact-black check -> PNG) and hands finished frames and state to a FrameHub. All operator actions go through dispatch(), so the
keyboard, the sliders and the demo script share one code path and tests can drive it without a browser.

Honesty plumbing: every source carries a provenance (quantum_backed, from_moth, untested), and captions and slides follow it.
Spending: nothing here can submit to Moth unless the session was created with allow_spend=True AND a plan with the exact payload
hash and credits is confirmed (resolve_plan / resolve_confirm); the default is preview only.
"""
import io
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.baseline.coherence import coherence_metrics
from src.capture.labels import DEFAULT_CALIB_DIR, DEFAULT_RUN_DIR, bay_window_labels, fill_defaults, validate
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import allocate, budget, format_budget
from src.projector import calibrate as cal_mod
from src.projector import patterns
from src.projector.output_window import FrameHub
from src.projector.warp import GlassLeak, Warper
from src.quantum import qdrive_rounds
from src.quantum import sampler_local as sl
from src.quantum.sampler_mock import Sampler
from src.ui import caption as caption_mod
from src.ui import demo as demo_mod
from src.ui import overlay as overlay_mod

def _classical():
    """The classical (pool-based, hand-composed-bevel) pipeline, imported only when a CLASSICAL source is in use: the relief render path
    never imports it (tests pin that)."""
    from src.texture.fast import FastComposer
    from src.texture.frames import PoolIndex, draw_frame
    return FastComposer, PoolIndex, draw_frame


LADDER = [1, 2, 4, 8, 16, 32, 64, 128]
HISTORY = 40
PATTERN_CYCLE = ["black", "white", "outline", "dots", "grid", "edge", None]
SOURCES = ("relief", "oracle", "mock", "circuit", "complementary")
CLASSICAL_SOURCES = ("oracle", "mock", "circuit", "complementary")
# Actions that choose a look. They exist only for the CLASSICAL sources: in the relief performance family the light, the depth decision and the
# observation axis are all drawn inside the circuit, and nothing here lets the operator pick them (v2 handoff section 4.5).
LEGACY_ONLY = ("hold_world", "hold_lamp", "release_lamp", "toggle_hold_lamp", "hold_pol", "hold_pol_all", "toggle_hold_pol", "pol_basis_toggle")


@dataclass
class Knobs:
    lamp_hold: list = field(default_factory=lambda: [None, None])       # +1 / -1 / None per lamp bit
    pol_hold: list = field(default_factory=list)                        # +1 / -1 / None per panel
    K: int = 8
    blend: float = 5.0
    gain: float = 1.0
    cycle_s: float = 4.0
    fade_ms: int = 700
    auto: bool = False
    blackout: bool = False
    noise: bool = False                  # control A: independent noise, in the relief family and the classical one
    hardness_sweep: bool = False
    dephased: bool = False               # control B (relief only): every polarity qubit replaced by its classical mixture
    light_interference: bool = False     # relief experiment: the lamp register read in X, so light directions interfere
    contrast: float = 1.0                # projection tone (relief): stretch of the lit field about mid-grey, not a scene property
    game: bool = False                   # relief (domain engine): a fraction of the looks are rounds of the parity game
    game_fraction: float = 0.5           # how many of them

    def snapshot(self, relief=False):
        out = dict(K=self.K, blend=self.blend, gain=self.gain, cycle_s=self.cycle_s, fade_ms=self.fade_ms, auto=self.auto, blackout=self.blackout,
                   noise=self.noise, hardness_sweep=self.hardness_sweep)
        if relief:
            out.update(dephased=self.dephased, light_interference=self.light_interference, contrast=self.contrast, game=self.game, game_fraction=self.game_fraction)
        else:
            out.update(lamp_hold=list(self.lamp_hold), pol_hold=list(self.pol_hold))
        return out


@dataclass
class Look:
    id: int
    draw: object
    blend: float
    gain: float
    noise: bool
    source: str
    t: float
    control: str = "coherent"


def _png(u8):
    buf = io.BytesIO()
    Image.fromarray(u8, "L").save(buf, "PNG", compress_level=1)
    return buf.getvalue()


# One facet per depth qubit; weak coupling along a boundary, strong across the seams; the lamp locked to one leaf per panel at the maximal angle, the leaves' tilt
# lowered so their visibility is high. Measured on the bay window (README): crease negativity 0.26, Mermin 6.6, predicted game win 91%.
DOMAIN_DEFAULTS = dict(seg_len=150.0, group_size=1, tau=0.5, pol_coupling=0.3, crease_coupling=1.0, seam_coupling=1.0, seam_mix=0.0, leaf_tau=0.3, lock=1.5708,
                       tau_mix=0.0, pol_field=0.0, crease_sign=-1.0, backend="auto", floquet_steps=0, floquet_zz=0.7, floquet_x=0.5, leaf_prefer="visible")


def _clean(o):
    """Make numpy scalars and tuples JSON-friendly."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


class Session:
    def __init__(self, labels=None, calibration=None, *, source="relief", pol_basis=None, circuit=None, coupling="lamp2",
                 calib_dir=DEFAULT_CALIB_DIR, run_dir=DEFAULT_RUN_DIR, out_dir="runs/show", projector_size=None, seed=2026,
                 pool_size=60000, hub=None, allow_spend=False, solve_fn=None, log=None, complementary_report="runs/complementary/report.json",
                 kappa=1.0, n_dirs=None, entangle=0.8, contrast=1.15, engine="panel", domain=None, game=False, game_fraction=0.5, lamp_mode="z", observations=None, aer_run=None):
        self.labels = validate(fill_defaults(labels if labels is not None else bay_window_labels()))
        self.scene = build_scene(self.labels)
        g = self.labels["grid"]
        self.NX, self.NY = g["nx"], g["ny"]
        self.sampler = Sampler(self.scene, self.NX, self.NY, min_cover=g["min_cover"])
        self.layout = allocate(self.sampler.n, self.sampler.n_panels)
        self.budget = budget(self.sampler.n, self.sampler.n_panels)
        self.panel_names = [p.name for p in self.scene.panels]
        self.hub = hub or FrameHub()
        self.calib_dir, self.run_dir, self.out_dir = calib_dir, run_dir, out_dir
        self.allow_spend, self._solve_fn, self.complementary_report = allow_spend, solve_fn, complementary_report
        self.coupling, self.pool_size = coupling, pool_size
        self.rng = np.random.default_rng(seed)
        self.lock = threading.RLock()
        self._log = log or (lambda m: None)
        self.messages = deque(maxlen=40)
        self.knobs = Knobs(pol_hold=[None] * self.scene.n_panels)
        self.history, self.cursor, self._next_id = [], -1, 1
        self.error = None
        self.pattern = None
        self.align = None
        self.glass = dict(frames_checked=0, violations=0)
        self.output = dict(connected=False, w=None, h=None, fullscreen=False, dpr=1.0, last_seen=None)
        self.demo = dict(active=False, index=0)
        self.ui = dict(overlay=False, help=False, resolve=False)
        self._resolve_plan = None
        self._composer = None
        self._pools = {}
        self._cycler = None
        self._last_published = None
        self._batch_depth = 0
        self._dirty = False
        self.circuit = circuit
        self.aer_run = aer_run                                          # folder of a saved Aer run: the relief source then draws its looks from those shots
        self.pol_basis = pol_basis
        self.source_kind = None
        if engine not in ("panel", "domain"):
            raise ValueError("engine must be 'panel' (one polarity qubit per panel) or 'domain' (one per depth domain, coupled along the geometry graph)")
        from src.quantum.relief_state import parse_observations
        self.observations = parse_observations(observations) if isinstance(observations, str) or observations is None else [tuple(o) for o in observations]
        self.relief_params = dict(kappa=kappa, n_dirs=n_dirs, entangle=entangle, engine=engine, domain=dict(DOMAIN_DEFAULTS, **(domain or {})))
        self._rel = None                                                # facets, spec, composer: built on first use of the relief source
        self.rstate = None
        self.knobs.contrast = float(contrast)
        if game and engine != "domain":
            raise ValueError("the parity game needs --engine domain")
        self.knobs.game, self.knobs.game_fraction = bool(game), float(game_fraction)
        if lamp_mode not in ("z", "x"):
            raise ValueError("lamp_mode must be 'z' or 'x'")
        self.knobs.light_interference = lamp_mode == "x"
        self.certificate = None
        self._game, self._game_for, self.evolution = None, None, None

        lw = self.labels["image"]
        self.size = tuple(projector_size) if projector_size else ((lw["width"], lw["height"]) if self.labels.get("source") == "projector" else (1920, 1080))
        self.calib_path = os.path.join(out_dir, "calibration.json")
        self.cal_base = self._initial_calibration(calibration)
        self._rebuild_warper()
        os.makedirs(out_dir, exist_ok=True)
        self.photos_dir = os.path.join(out_dir, "photos")
        self.set_source(source, pol_basis, circuit, announce=False)
        if self.budget["over_cap"]:
            self.say(f"WARNING: {format_budget(self.budget)}")
        mismatch = self.calibration_mismatch()
        if mismatch:
            self.say(f"WARNING: {mismatch}")
        self._new_look()
        self._notify()

    # ------------------------------------------------------------------ plumbing
    def say(self, msg):
        self.messages.append(dict(t=time.strftime("%H:%M:%S"), msg=msg))
        self._log(msg)

    def _notify(self):
        if self._batch_depth:
            self._dirty = True
        else:
            self.hub.publish_state(self.state())

    class _Batch:
        def __init__(self, s):
            self.s = s

        def __enter__(self):
            self.s._batch_depth += 1

        def __exit__(self, *a):
            self.s._batch_depth -= 1
            if not self.s._batch_depth and self.s._dirty:
                self.s._dirty = False
                self.s.hub.publish_state(self.s.state())

    def batch(self):
        return Session._Batch(self)

    @property
    def family(self):
        """'relief' (Superposed Relief: everything is drawn inside the circuit) or 'classical' (the pool-based sources, kept as the rehearsal
        and as the baseline the relief state is compared with)."""
        return "relief" if self.source_kind == "relief" else "classical"

    @property
    def composer(self):
        if self._composer is None:
            FastComposer = _classical()[0]
            self._composer = FastComposer(self.scene, self.sampler.cells, self.NX)
        return self._composer

    def relief(self):
        """Facets, circuit spec and composer for this scene, built once (about a second; a few more for the domain engine)."""
        if self._rel is None:
            from src.texture.relief_compose import make_composer
            rp = self.relief_params
            if rp["engine"] == "domain":
                from src.geometry.domains import build_domains
                from src.quantum.domain_state import spec_from_domains
                dp = rp["domain"]
                ds = build_domains(self.scene, seg_len=dp["seg_len"], group_size=dp["group_size"], tau=dp["tau"], crease_sign=dp["crease_sign"])
                spec = spec_from_domains(ds, self.scene, entangle=rp["entangle"], pol_coupling=dp["pol_coupling"], lock=dp["lock"], tau_mix=dp["tau_mix"],
                                         pol_field=dp["pol_field"], crease_coupling=dp["crease_coupling"], seam_coupling=dp["seam_coupling"], seam_mix=dp["seam_mix"],
                                         leaf_tau=dp["leaf_tau"], leaf_prefer=dp["leaf_prefer"], observations=self.observations, floquet=(dp["floquet_steps"], dp["floquet_zz"], dp["floquet_x"]))
                self._rel = dict(facets=ds.facets, spec=spec, composer=make_composer(self.scene, ds.facets), ds=ds)
            else:
                from src.geometry.facets import EXACT_QUBITS_MAX, auto_n_dirs, build_facets
                from src.quantum.relief_state import spec_from_facets
                fs = build_facets(self.scene, n_dirs=rp["n_dirs"], kappa=rp["kappa"])
                spec = spec_from_facets(fs, self.scene, entangle=rp["entangle"], observations=self.observations)
                if spec.n_sim > EXACT_QUBITS_MAX:
                    raise ValueError(f"the per-panel engine would simulate {spec.n_sim} qubits exactly ({spec.F} facets + {spec.P} panels), more than the {EXACT_QUBITS_MAX} it "
                                     f"attempts; use a smaller --n-dirs (the default for {self.scene.n_panels} panels is {auto_n_dirs(self.scene.n_panels)}) or --engine domain")
                self._rel = dict(facets=fs, spec=spec, composer=make_composer(self.scene, fs), ds=None)
        return self._rel

    def _legacy_only(self, name):
        if self.family == "relief":
            raise ValueError(f"unknown action {name!r}")

    @property
    def budget(self):
        if self.family == "relief":
            sp = self.relief()["spec"]
            domain = self.relief_params["engine"] == "domain"
            return dict(patches=sp.F, lamps=2, panels=sp.P, total=sp.n_full, simulated=sp.n_sim, observe=2, cap=None if domain else 22,
                        over_cap=False if domain else sp.n_full > 22, family="relief", engine=self.relief_params["engine"])
        return self._classical_budget

    @budget.setter
    def budget(self, v):
        self._classical_budget = v

    # ------------------------------------------------------------------ calibration and warp
    def _initial_calibration(self, given):
        if given is not None:
            return given
        if os.path.exists(self.calib_path):
            try:
                c = cal_mod.load_calibration(self.calib_path)
                if tuple(c.canvas) == (self.scene.W, self.scene.H):
                    self.say(f"loaded calibration {self.calib_path} ({c.method}, rms {c.rms_px:.2f} px)")
                    return c
                self.say(f"ignored {self.calib_path}: it was made for a {c.canvas[0]}x{c.canvas[1]} canvas, this scene is {self.scene.W}x{self.scene.H}")
            except (ValueError, KeyError, OSError) as e:
                self.say(f"ignored {self.calib_path}: {e}")
        return cal_mod.labels_calibration(self.labels, self.size)

    @property
    def cal(self):
        return self.cal_base.for_size(self.size)

    def _rebuild_warper(self):
        c = self.cal
        self.warper = Warper(c.H, self.scene.frame, self.size, c.margin_px)

    def _scene_polys(self):
        sx = self.scene.W / self.labels["image"]["width"]
        panels = [p.polygon for p in self.scene.panels]
        glass = [[(x * sx, y * sx) for x, y in g["polygon"]] for key in ("glass", "off_limits") for g in self.labels[key]]
        return panels, glass

    def set_output(self, w, h, fullscreen=False, dpr=1.0):
        """The output page reports its real pixel size; render at it."""
        with self.lock:
            w, h = int(w), int(h)
            self.output.update(connected=True, w=w, h=h, fullscreen=bool(fullscreen), dpr=float(dpr), last_seen=time.time())
            if w >= 200 and h >= 200 and (w, h) != self.size:
                old = self.size
                self.size = (w, h)
                self._rebuild_warper()
                if self.align:                                   # keep the markers on the same physical spots
                    sx, sy = w / old[0], h / old[1]
                    self.align["quad"] = [[(x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5] for x, y in self.align["quad"]]
                self.say(f"output window is {w}x{h}: rendering at that size")
                if self.align:
                    self._render_align()
                else:
                    self._republish()
            self._notify()

    def output_gone(self):
        with self.lock:
            self.output["connected"] = False
            self._notify()

    # ------------------------------------------------------------------ sources
    def _build_relief(self, circuit):
        """(ReliefState, provenance) for the relief source: the reference circuit, or the facet register replaced by a QASM3 circuit file
        (e.g. a QDrive result) with the polarity superposition lifted on top locally. The domain engine builds its own state (exact up to 20
        qubits, else a matrix-product state) and takes no circuit file."""
        from src.quantum.relief_state import ReliefState
        spec = self.relief()["spec"]
        if self.aer_run:
            if circuit:
                raise ValueError("--aer-run and --circuit both replace the relief state; give one")
            if self.relief_params["engine"] != "panel":
                raise ValueError("--aer-run executes the per-panel engine's circuit on Aer; the domain engine's qubits do not fit a statevector run")
            from src.quantum.aer_relief import AerReliefState
            st = AerReliefState(spec, self.aer_run)
            return st, dict(label=st.label, quantum_backed=True, from_moth=False, untested=False, synthetic=False, executed="aer")
        if self.relief_params["engine"] == "domain":
            if circuit:
                raise ValueError("--circuit replaces the facet register of the per-panel engine; the domain engine prepares its own register")
            from src.quantum.domain_state import MPSReliefState, make_state
            st = make_state(spec, backend="mps" if self.knobs.game else self.relief_params["domain"]["backend"])
            if isinstance(st, MPSReliefState):
                secs = st.warm()
                acc = st.accuracy()
                label = (f"local reference circuit ({spec.n_sim} qubits as a matrix-product state on this laptop, bond dimension <= {acc['max_bond']}, norm deficit "
                         f"{acc['norm_deficit']:.1e}; not a Moth result)")
                self.say(f"matrix-product states built in {secs:.1f}s")
            else:
                label = f"local reference circuit ({spec.n_sim} qubits, exact statevector simulated here; not a Moth result)"
            st.label = label
            if hasattr(st, "prep") and self.relief_params["domain"]["floquet_steps"]:
                self._note_evolution(st, self.relief_params["domain"]["floquet_steps"])
            return st, dict(label=label, quantum_backed=True, from_moth=False, untested=False, synthetic=False)
        if not circuit:
            return ReliefState(spec), dict(label="local reference circuit (exact statevector simulated here; not a Moth result)", quantum_backed=True,
                                           from_moth=False, untested=False, synthetic=False)
        if not os.path.exists(circuit):
            raise ValueError(f"no circuit file at {circuit!r}")
        with open(circuit) as f:
            psi, nq = sl.statevector_from_qasm(f.read())
        if nq != spec.F:
            raise ValueError(f"{circuit} has {nq} qubits but this scene's facet register has {spec.F}")
        folder = os.path.dirname(os.path.abspath(circuit))
        synthetic = any(os.path.exists(os.path.join(d, "SYNTHETIC")) for d in (folder, os.path.dirname(folder)))
        rounds = None if synthetic else qdrive_rounds.circuit_provenance(circuit)
        from_moth = rounds["from_moth"] if rounds else any(f.startswith("job_") for f in os.listdir(folder)) and not synthetic
        label = ("SYNTHETIC self-test circuit (not a Moth result)" if synthetic else rounds["label"] + " (facet register, polarity lifted locally)" if rounds
                 else "Moth QDrive circuit for the facet register, polarity lifted locally" if from_moth
                 else "circuit file of unknown origin for the facet register, polarity lifted locally")
        return ReliefState(spec, psi, label=label), dict(label=label, quantum_backed=True, from_moth=from_moth, untested=not from_moth, synthetic=synthetic)

    def _build_pool(self, kind, basis, circuit):
        L, M, rng = self.layout, self.pool_size, self.rng
        if kind == "oracle":
            src = sl.StateSource(sl.oracle_state(self.sampler, L), L, {q: basis for q in L["polarity"]} if basis == "x" else None)
            return src.pool(M, rng), dict(label="classical stand-in (the mock's distribution as a state)", quantum_backed=False, from_moth=False, untested=False)
        if kind == "mock":
            if basis == "x":
                raise ValueError("the classical mock has no X basis to read polarity in")
            from src.quantum.ab_compare import mock_pool
            return mock_pool(self.sampler, M // 4, rng), dict(label="classical Gibbs sampler (the look-test mock)", quantum_backed=False, from_moth=False, untested=False)
        if kind == "complementary":
            from src.quantum import complementary as comp
            psi = comp.complementary_state(self.sampler, L, self.coupling)
            pool = comp.make_pool(psi, L, basis, M, rng)
            return pool, dict(label=f"complementary-polarity stand-in ({self.coupling}), simulated locally", quantum_backed=True, from_moth=False, untested=True)
        if kind == "circuit":
            if not circuit or not os.path.exists(circuit):
                raise ValueError(f"no circuit file at {circuit!r}")
            with open(circuit) as f:
                text = f.read()
            pool = sl.source_from_qasm(text, L, {q: basis for q in L["polarity"]} if basis == "x" else None).pool(M, rng)
            folder = os.path.dirname(os.path.abspath(circuit))
            synthetic = any(os.path.exists(os.path.join(d, "SYNTHETIC")) for d in (folder, os.path.dirname(folder)))
            job = [f for f in os.listdir(folder) if f.startswith("job_")]
            rounds = None if synthetic else qdrive_rounds.circuit_provenance(circuit)
            from_moth = rounds["from_moth"] if rounds else bool(job) and not synthetic
            label = ("SYNTHETIC self-test circuit (not a Moth result)" if synthetic else rounds["label"] if rounds
                     else "QDrive circuit, simulated locally" if from_moth else "circuit file of unknown origin, simulated locally")
            return pool, dict(label=label, quantum_backed=True, from_moth=from_moth, untested=not from_moth, synthetic=synthetic)
        raise ValueError(f"unknown source {kind!r}; choose from {SOURCES}")

    def set_source(self, kind, pol_basis=None, circuit=None, announce=True):
        """Switch the source of the measurement outcomes (builds its state or pool; a second or two for quantum states)."""
        with self.lock:
            if kind not in SOURCES:
                raise ValueError(f"unknown source {kind!r}; choose from {SOURCES}")
            if kind == "relief":
                circuit = circuit if circuit is not None else (self.circuit if self.source_kind in (None, "relief") else None)
                key = ("relief", circuit, self.relief_params["domain"]["floquet_steps"], bool(self.knobs.game), self.aer_run)
                if key not in self._pools:
                    t0 = time.time()
                    state, prov = self._build_relief(circuit)
                    prov.update(kind="relief", pol_basis="z", relief=True)
                    self._pools[key] = (state, None, prov)
                    self.say(f"source 'relief' ready in {time.time() - t0:.1f}s: {prov['label']}")
                self.rstate, _, self.prov = self._pools[key]
                self.index, self.pool = None, None
                self.source_kind, self.pol_basis, self.circuit = "relief", "z", circuit
                self.certificate = None
                self.knobs.lamp_hold, self.knobs.pol_hold = [None, None], [None] * self.scene.n_panels
                if announce:
                    self._new_look()
                self._notify()
                return self.prov
            basis = pol_basis or ("x" if kind == "complementary" else "z")
            circuit = circuit or self.circuit
            key = (kind, basis, circuit if kind == "circuit" else None, self.coupling if kind == "complementary" else None)
            if key not in self._pools:
                t0 = time.time()
                pool, prov = self._build_pool(kind, basis, circuit)
                prov.update(kind=kind, pol_basis=basis)
                PoolIndex = _classical()[1]
                self._pools[key] = (PoolIndex(pool), pool, prov)
                self.say(f"source '{kind}' ready in {time.time() - t0:.1f}s: {prov['label']}")
            self.index, self.pool, self.prov = self._pools[key]
            self.rstate = None
            self.source_kind, self.pol_basis, self.circuit = kind, basis, circuit
            if announce:
                self._new_look()
            self._notify()
            return self.prov

    def _noise_index(self):
        key = ("independent",)
        if key not in self._pools:
            n, P, M = self.sampler.n, self.sampler.n_panels, self.pool_size
            pool = sl.Pool(self.rng.choice([-1, 1], (M, 2)), self.rng.choice([-1, 1], (M, n)), self.rng.choice([-1, 1], (M, P)))
            PoolIndex = _classical()[1]
            self._pools[key] = (PoolIndex(pool), pool, dict(kind="independent", label="independent noise", quantum_backed=False, from_moth=False, untested=False, pol_basis="z"))
        return self._pools[key][0]

    # ------------------------------------------------------------------ frames
    def _held(self):
        lamp = tuple(self.knobs.lamp_hold)
        pol = tuple(self.knobs.pol_hold)
        return (None if all(v is None for v in lamp) else lamp), (None if all(v is None for v in pol) else pol)

    def _control(self):
        k = self.knobs
        return "noise" if k.noise else ("dephased" if k.dephased else "coherent")

    def _new_look(self):
        k = self.knobs
        if k.hardness_sweep:
            k.K = LADDER[(LADDER.index(k.K) + 1) % len(LADDER)] if k.K in LADDER else LADDER[0]
        if self.family == "relief":
            game = self._game_obj() if (k.game and self._control() == "coherent" and not k.light_interference) else None
            if game is not None and self.rng.random() < k.game_fraction:
                d, _ = game.play(self.rng, k.K)                           # one round of the parity game, and the frame its outcomes make
            else:
                try:
                    d = self.rstate.draw(k.K, self.rng, control=self._control(), lamp_mode="x" if k.light_interference else "z")
                except ValueError as e:
                    k.light_interference = False
                    self.say(str(e))
                    d = self.rstate.draw(k.K, self.rng, control=self._control())
            self._panel_view(d)
            self.error = None
        else:
            lamp, pol = self._held()
            index = self._noise_index() if k.noise else self.index
            try:
                d = _classical()[2](index, k.K, self.rng, lamp, pol)
            except ValueError as e:
                self.error = str(e)
                self.say(f"cannot draw that look: {e}")
                return None
            self.error = None
        look = Look(self._next_id, d, k.blend, k.gain, k.noise, self.source_kind, time.time(), getattr(d, "control", "coherent"))
        self._next_id += 1
        self.history = self.history[:self.cursor + 1] + [look]
        self.history = self.history[-HISTORY:]
        self.cursor = len(self.history) - 1
        self._show(look)
        return look

    def _panel_view(self, d):
        """Domain engine: one polarity outcome per depth domain. The captions, spheres and operator page speak of panels, so `pol` and `lean` become
        the panel's mean over its domains, and the per-domain values stay on the draw as `domain_pol` / `domain_lean` (the overlay draws those)."""
        ds = self.relief()["ds"]
        if ds is None or hasattr(d, "domain_pol"):
            return d
        d.domain_pol, d.domain_lean = tuple(d.pol), tuple(d.lean)
        pol, lean = [], []
        for k in range(self.scene.n_panels):
            idx = [i for i in range(ds.n_domains) if ds.domain_panel[i] == k]
            pol.append(1 if not idx or np.mean([d.domain_pol[i] for i in idx]) >= 0 else -1)
            lean.append(float(np.mean([d.domain_lean[i] for i in idx])) if idx else 0.0)
        d.pol, d.lean = tuple(pol), tuple(lean)
        return d

    def _compose(self, look):
        if look.source == "relief":
            return self.relief()["composer"].compose(look.draw.lit, blend_sigma=look.blend, contrast=self.knobs.contrast)
        return self.composer.compose(look.draw.lit, look.draw.lamp, look.draw.pol, blend_sigma=look.blend)

    def _show(self, look):
        """Render, check and publish one look (or black when blacked out)."""
        self.current = look
        if self.knobs.blackout:
            return self._publish_array(np.zeros((self.size[1], self.size[0]), np.uint8), dict(kind="black", name="blackout"))
        canvas = self._compose(look)
        self._canvas = canvas
        u8 = self.warper(canvas, look.gain)
        try:
            self.warper.check(u8)
            self.glass["frames_checked"] += 1
        except GlassLeak as e:
            self.glass["violations"] += 1
            self.say(f"REFUSED to send a frame: {e}")
            u8 = np.zeros_like(u8)
        return self._publish_array(u8, dict(kind="frame", name=f"look {look.id}", look=look.id))

    def _publish_array(self, u8, meta):
        self._last_u8 = u8
        png = _png(u8)
        meta = dict(meta, w=int(u8.shape[1]), h=int(u8.shape[0]), fade_ms=self.knobs.fade_ms)
        self._last_meta = meta
        self.hub.publish_frame(png, meta)
        return png

    def _republish(self):
        if self.pattern:
            return self._show_pattern(self.pattern)
        if getattr(self, "current", None) is not None:
            return self._show(self.current)
        return self._new_look()

    def _recompose_current(self):
        look = getattr(self, "current", None)
        if look is None:
            return
        look.blend, look.gain = self.knobs.blend, self.knobs.gain
        if not self.pattern:
            self._show(look)

    # ------------------------------------------------------------------ patterns and calibration
    def _pattern_array(self, name):
        panels, glass = self._scene_polys()
        c = self.cal
        landmarks = self.align["landmarks"] if self.align else cal_mod.default_landmarks(self.labels, (self.scene.W, self.scene.H))
        H = self.align["H"] if self.align else c.H
        return patterns.render(name, size=self.size, warper=self.warper, H=H, panel_polys=panels, glass_polys=glass,
                               landmarks_canvas=landmarks, selected=self.align["selected"] if self.align else None)

    def _show_pattern(self, name):
        u8 = self._pattern_array(name)
        if name in patterns.SCENE_PATTERNS:
            try:
                self.warper.check(u8)
            except GlassLeak as e:
                self.glass["violations"] += 1
                self.say(f"REFUSED to send pattern {name}: {e}")
                u8 = np.zeros_like(u8)
        return self._publish_array(u8, dict(kind="pattern", name=name))

    def set_pattern(self, name):
        with self.lock:
            if name is not None and name not in patterns.ALL:
                raise ValueError(f"unknown pattern {name!r}; choose from {patterns.ALL}")
            if self.align and name != "align":
                self.align = None
            self.pattern = name
            if name is None:
                self._republish()
            else:
                self._show_pattern(name)
            self._notify()

    def pattern_cycle(self):
        i = PATTERN_CYCLE.index(self.pattern) if self.pattern in PATTERN_CYCLE else -1
        self.set_pattern(PATTERN_CYCLE[(i + 1) % len(PATTERN_CYCLE)])

    def _render_align(self):
        c = self.cal
        self.align["H"] = cal_mod.fit_homography(self.align["landmarks"], self.align["quad"])
        self.pattern = "align"
        self._show_pattern("align")

    def align_start(self):
        """Corner calibration: project the four landmark markers and the scene outline; the operator drags the markers onto their
        physical features (align_move / align_nudge), then align_apply."""
        with self.lock:
            lm = cal_mod.default_landmarks(self.labels, (self.scene.W, self.scene.H))
            quad = [list(map(float, p)) for p in cal_mod.apply_homography(self.cal.H, lm)]
            self.align = dict(landmarks=lm, quad=quad, selected=0, H=self.cal.H.copy(), start_quad=[list(q) for q in quad])
            self._render_align()
            self.say("alignment: drag the four numbered markers onto their physical corners, then Apply")
            self._notify()

    def align_move(self, index, x, y):
        with self.lock:
            if not self.align:
                raise ValueError("start alignment first")
            q = [list(p) for p in self.align["quad"]]
            q[int(index)] = [float(np.clip(x, 0, self.size[0] - 1)), float(np.clip(y, 0, self.size[1] - 1))]
            if not cal_mod.quad_is_convex(q):
                return dict(ok=False, error="that would cross two markers over; the outline must stay a convex quadrilateral")
            self.align["quad"], self.align["selected"] = q, int(index)
            self._render_align()
            self._notify()
            return dict(ok=True)

    def align_nudge(self, index, dx, dy):
        with self.lock:
            x, y = self.align["quad"][int(index)]
            return self.align_move(index, x + dx, y + dy)

    def align_apply(self):
        with self.lock:
            if not self.align:
                raise ValueError("no alignment in progress")
            c = cal_mod.corner_calibration((self.scene.W, self.scene.H), self.size, self.align["landmarks"], self.align["quad"], self.cal.margin_px)
            self.cal_base = c
            self.align, self.pattern = None, None
            self._rebuild_warper()
            os.makedirs(self.out_dir, exist_ok=True)
            cal_mod.save_calibration(c, self.calib_path)
            self.say(f"calibration saved to {self.calib_path} (4 corners, rms {c.rms_px:.3f} px)")
            self._republish()
            self._notify()
            return dict(path=self.calib_path, rms_px=c.rms_px)

    def align_cancel(self):
        with self.lock:
            self.align, self.pattern = None, None
            self._republish()
            self._notify()

    def align_reset(self):
        """Back to the scale-only calibration (labels drawn in projector space need nothing else)."""
        with self.lock:
            self.cal_base = cal_mod.labels_calibration(self.labels, self.size, margin_px=self.cal_base.margin_px)
            self.align, self.pattern = None, None
            self._rebuild_warper()
            if os.path.exists(self.calib_path):
                os.replace(self.calib_path, self.calib_path + ".bak")
            self.say("calibration reset to the scale-only mapping")
            self._republish()
            self._notify()

    def set_margin(self, px):
        with self.lock:
            px = int(np.clip(px, 0, 20))
            self.cal_base.margin_px = px
            self._rebuild_warper()
            if self.cal_base.method != "scale" or os.path.exists(self.calib_path):
                cal_mod.save_calibration(self.cal_base, self.calib_path)
            self._republish()
            self._notify()

    def snapshot(self):
        """Write what is on the wall right now (projector pixels) for the photographed test: frame_projector.png and a timestamped copy."""
        with self.lock:
            os.makedirs(self.photos_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            u8 = self._last_u8
            Image.fromarray(u8, "L").save(os.path.join(self.photos_dir, "frame_projector.png"))
            Image.fromarray(u8, "L").save(os.path.join(self.photos_dir, f"frame_{stamp}_projector.png"))
            info = dict(time=stamp, meta=self._last_meta, knobs=self.knobs.snapshot(), provenance=self.prov)
            if getattr(self, "current", None):
                info["look"] = dict(id=self.current.id, lamp=list(self.current.draw.lamp), pol=list(self.current.draw.pol), K=self.current.draw.K)
            with open(os.path.join(self.photos_dir, f"frame_{stamp}.json"), "w") as f:
                json.dump(_clean(info), f, indent=1)
            self.say(f"snapshot written to {self.photos_dir}/frame_projector.png")
            return dict(dir=self.photos_dir)

    # ------------------------------------------------------------------ actions
    def next(self):
        with self.lock:
            if self.pattern:
                self.align, self.pattern = None, None
            if self.cursor < len(self.history) - 1:
                self.cursor += 1
                self._show(self.history[self.cursor])
            else:
                self._new_look()
            self._notify()

    def prev(self):
        with self.lock:
            if self.pattern:
                self.align, self.pattern = None, None
                self._republish()
            elif self.cursor > 0:
                self.cursor -= 1
                self._show(self.history[self.cursor])
            self._notify()

    def auto_set(self, on, seconds=None):
        with self.lock:
            self.knobs.auto = bool(on)
            if seconds is not None:
                self.knobs.cycle_s = float(np.clip(seconds, 0.5, 60))
            self._notify()

    def auto_toggle(self):
        self.auto_set(not self.knobs.auto)

    def cycle(self, seconds):
        self.auto_set(self.knobs.auto, seconds)

    def hold_world(self, l1, l2):
        self._legacy_only('hold_world')
        with self.lock:
            self.knobs.lamp_hold = [l1, l2]
            self._new_look()
            self._notify()

    def hold_lamp(self, index, value):
        self._legacy_only('hold_lamp')
        with self.lock:
            self.knobs.lamp_hold[int(index)] = value
            self._new_look()
            self._notify()

    def release_lamp(self):
        self._legacy_only('release_lamp')
        with self.lock:
            self.knobs.lamp_hold = [None, None]
            self._notify()

    def toggle_hold_lamp(self):
        self._legacy_only('toggle_hold_lamp')
        with self.lock:
            if all(v is None for v in self.knobs.lamp_hold) and getattr(self, "current", None):
                self.knobs.lamp_hold = list(self.current.draw.lamp)
            else:
                self.knobs.lamp_hold = [None, None]
            self._notify()

    def hold_pol(self, index, value):
        self._legacy_only('hold_pol')
        with self.lock:
            self.knobs.pol_hold[int(index)] = value
            self._new_look()
            self._notify()

    def hold_pol_all(self, value):
        self._legacy_only('hold_pol_all')
        with self.lock:
            self.knobs.pol_hold = [value] * self.scene.n_panels
            self._new_look()
            self._notify()

    def toggle_hold_pol(self):
        self._legacy_only('toggle_hold_pol')
        with self.lock:
            if all(v is None for v in self.knobs.pol_hold) and getattr(self, "current", None):
                self.knobs.pol_hold = list(self.current.draw.pol)
            else:
                self.knobs.pol_hold = [None] * self.scene.n_panels
            self._notify()

    def pol_basis_toggle(self):
        self._legacy_only("pol_basis_toggle")
        self.set_source(self.source_kind, "z" if self.pol_basis == "x" else "x")

    def K(self, value):
        with self.lock:
            self.knobs.K = int(np.clip(round(float(value)), 1, 256))
            self.knobs.hardness_sweep = False
            self._new_look()
            self._notify()

    def K_step(self, step):
        cur = min(LADDER, key=lambda v: abs(v - self.knobs.K))
        self.K(LADDER[int(np.clip(LADDER.index(cur) + int(step), 0, len(LADDER) - 1))])

    def hardness_sweep_set(self, on):
        with self.lock:
            self.knobs.hardness_sweep = bool(on)
            if on:
                self.knobs.K = LADDER[-1]                      # the first new look wraps round to K = 1
                self._new_look()
            self._notify()

    def blend(self, value):
        with self.lock:
            self.knobs.blend = float(np.clip(value, 0.5, 20))
            self._recompose_current()
            self._notify()

    def blend_step(self, step):
        self.blend(self.knobs.blend + 1.5 * int(step))

    def gain(self, value):
        with self.lock:
            self.knobs.gain = float(np.clip(value, 0.05, 1.0))
            self._recompose_current()
            self._notify()

    def fade(self, value):
        with self.lock:
            self.knobs.fade_ms = int(np.clip(value, 0, 3000))
            self._notify()

    def blackout_set(self, on):
        with self.lock:
            self.knobs.blackout = bool(on)
            self._republish()
            self._notify()

    def blackout_toggle(self):
        self.blackout_set(not self.knobs.blackout)

    def noise_set(self, on):
        with self.lock:
            self.knobs.noise = bool(on)
            self._new_look()
            self._notify()

    def noise_toggle(self):
        self.noise_set(not self.knobs.noise)

    def dephased_set(self, on):
        """Control B (relief): each polarity qubit replaced by its classical mixture. Same one-body statistics, no interference between depths."""
        if self.family != "relief":
            raise ValueError("unknown action 'dephased_set'")
        with self.lock:
            self.knobs.dephased = bool(on)
            if on:
                self.knobs.noise = False
            self._new_look()
            self._notify()

    def dephased_toggle(self):
        self.dephased_set(not self.knobs.dephased)

    # ------------------------------------------------------------------ the parity game and the Floquet dynamics (domain engine)
    def _require_domain(self, name):
        if self.family != "relief" or self.relief_params["engine"] != "domain":
            raise ValueError(f"{name} needs the domain engine (launch the show with --engine domain)")

    def _game_obj(self):
        """The ParityGame on this state (built once: it optimises the Mermin frames, about a second). It needs the lock and the matrix-product backend."""
        if getattr(self, "_game", None) is None or self._game_for is not self.rstate:
            from src.quantum.parity_game import ParityGame
            if getattr(self.rstate, "backend", "exact") != "mps":
                raise ValueError("the parity game needs the matrix-product backend: restart with --backend mps")
            spec = self.relief()["spec"]
            if spec.lock_array() is None:
                if self.relief_params["domain"]["lock"]:
                    raise ValueError("the parity game needs leaf domains that face left or right, and no domain does with these facets: use a smaller --group-size "
                                     "(1 or 2) or --seg-len, so that a domain holds one stretch of bevel rather than a whole loop")
                raise ValueError("the parity game needs the lamp locked to the depth qubits (--lock > 0)")
            if len(spec.leaves or []) < 2:
                raise ValueError(f"the parity game needs at least two leaf domains besides the lamp (one per plane, up to three); this wall has {len(spec.leaves or [])}")
            self._game = ParityGame(self.rstate, restarts=8)
            self._game_for = self.rstate
            self.say(f"parity game ready: Mermin value {self._game.M:.2f}, predicted win rate {100 * self._game.predicted:.1f}% against a classical maximum of 75%")
        return self._game

    def game_set(self, on, fraction=None):
        """Turn the parity game on or off; `fraction` of the looks are then rounds (the others are ordinary frames)."""
        self._require_domain("game_set")
        with self.lock:
            if on and self.rstate is not None and getattr(self.rstate, "backend", "exact") != "mps":
                self.knobs.game = True
                self.set_source("relief", announce=False)                  # rebuild as a matrix-product state
            if fraction is not None:
                self.knobs.game_fraction = float(np.clip(fraction, 0.0, 1.0))
            self.knobs.game = bool(on)
            if on:
                try:
                    self._game_obj()
                except ValueError:                                           # a game that cannot be built must not stay on: every look would raise
                    self.knobs.game = False
                    raise
            self._new_look()
            self._notify()

    def game_toggle(self):
        self.game_set(not self.knobs.game)

    def game_reset(self):
        self._require_domain("game_reset")
        with self.lock:
            if getattr(self, "_game", None) is not None:
                from src.quantum.parity_game import GameStats
                self._game.stats = GameStats(self._game.predicted)
            self._notify()

    def evolve_set(self, steps):
        """Run the facet register for `steps` kicked-Ising steps before the polarity attaches (0: static). Rebuilds the state (a second or two)."""
        self._require_domain("evolve_set")
        with self.lock:
            steps = int(max(0, min(steps, 40)))
            self.relief_params["domain"]["floquet_steps"] = steps
            self._rel, self._game = None, None
            self.say(f"evolving the relief: {steps} kicked-Ising step(s) on the facet graph")
            self.set_source("relief", announce=False)
            if hasattr(self.rstate, "prep"):
                self._note_evolution(self.rstate, steps)
            else:
                self.evolution = None
            self.certificate = None
            self._new_look()
            self._notify()

    def _note_evolution(self, state, steps):
        m = state.prep()
        en = m.entropies()
        self.evolution = dict(steps=steps, max_bond=m.max_bond(), mean_entropy=float(np.mean(en)), max_entropy=float(max(en)), norm_deficit=float(abs(1 - m.norm2())))
        self.say(f"step {steps}: bond dimension {m.max_bond()}, mean bond entropy {self.evolution['mean_entropy']:.2f} bits, norm deficit {self.evolution['norm_deficit']:.1e}")

    def evolve_step(self):
        self._require_domain("evolve_step")
        self.evolve_set(self.relief_params["domain"]["floquet_steps"] + 1)

    def light_interference_set(self, on):
        """Relief experiment: read the lamp register in X, so the light directions interfere instead of being a coin flip between them."""
        if self.family != "relief":
            raise ValueError("unknown action 'light_interference_set'")
        with self.lock:
            self.knobs.light_interference = bool(on)
            self._new_look()
            self._notify()

    def light_interference_toggle(self):
        self.light_interference_set(not self.knobs.light_interference)

    def contrast(self, value):
        with self.lock:
            self.knobs.contrast = float(np.clip(value, 0.5, 3.0))
            self._recompose_current()
            self._notify()

    def witness_run(self, shots=20000):
        """Measure the witnesses on the state in use (a separate measurement set from the frames) and keep the certificate for the science
        panel. Provenance is part of the result, and a state with no non-classical signature is reported as such."""
        if self.family != "relief":
            raise ValueError("unknown action 'witness_run'")
        from src.quantum import relief_witness as rw
        with self.lock:
            if self.relief_params["engine"] == "domain":
                from src.quantum import domain_witness as dw
                cert = dw.domain_certificate(self.rstate)
                cert["lines"] = dw.describe_domain_certificate(cert, self.relief()["ds"], self.relief()["spec"])
                self.certificate = cert
                s = cert["summary"]
                self.say(f"witness run: {s['entangled_edges']} of {s['edges']} domain edges certified entangled"
                         + (f"; lamp-depth Mermin {cert['lamp']['mermin']['value']:.2f} vs {cert['lamp']['mermin']['local_bound']}" if "mermin" in cert.get("lamp", {}) else ""))
                self._notify()
                return dict(certificate=_clean(cert))
            cert = rw.certificate(self.rstate, np.random.default_rng(int(self.rng.integers(1 << 31))), int(shots))
            cert["lines"] = rw.describe_certificate(cert)
            self.certificate = cert
            self.say("witness run: " + ("entanglement certified for every panel" if all(r["entangled"] for r in cert["panels"]) else "not every panel certified"))
            self._notify()
            return dict(certificate=_clean(cert))

    def overlay_toggle(self):
        with self.lock:
            self.ui["overlay"] = not self.ui["overlay"]
            self._notify()

    def help_toggle(self):
        with self.lock:
            self.ui["help"] = not self.ui["help"]
            self._notify()

    def resolve_open(self):
        with self.lock:
            self.ui["resolve"] = True
            self._notify()

    def resolve_close(self):
        with self.lock:
            self.ui["resolve"] = False
            self._notify()

    # ------------------------------------------------------------------ cycler
    def start_cycler(self):
        if self._cycler is None:
            self._stop = threading.Event()
            self._cycler = threading.Thread(target=self._cycle_loop, daemon=True)
            self._cycler.start()

    def stop_cycler(self):
        if self._cycler is not None:
            self._stop.set()
            self._cycler.join(timeout=2)
            self._cycler = None

    def _cycle_loop(self):
        last = time.time()
        while not self._stop.wait(0.1):
            k = self.knobs
            if not k.auto or k.blackout or self.pattern or self.align:
                last = time.time()
                continue
            if time.time() - last >= k.cycle_s:
                last = time.time()
                try:
                    self.next_new()
                except Exception as e:                                       # noqa: BLE001 - never kill the show
                    self.say(f"auto-cycle error: {e}")

    def next_new(self):
        """Always a fresh draw (the auto-cycle), even when the cursor is back in the history."""
        with self.lock:
            self.cursor = len(self.history) - 1
            self._new_look()
            self._notify()

    # ------------------------------------------------------------------ demo
    def _demo_steps(self):
        ab_path = os.path.join(self.run_dir, "ab", "ab_summary.json")
        ab = json.load(open(ab_path)) if os.path.exists(ab_path) else None
        comp = None
        if os.path.exists(self.complementary_report):
            with open(self.complementary_report) as f:
                comp = json.load(f)
        ctx = dict(prov=self.prov, budget=self._budget_text(), family=self.family, moth_lines=self._moth_lines(), ab=ab, comp=comp, can_complementary=not self.budget["over_cap"],
                   qdrive_circuit=os.path.exists(os.path.join(self.run_dir, "qdrive", "circuit.qasm")),
                   ab_images=all(os.path.exists(self.asset_path(k) or "") for k in ("ab_requested_vs_achieved", "ab_frames")))
        return demo_mod.build_steps(ctx)

    def _moth_lines(self):
        """What has actually been run on Moth for this scene, read from disk (the scorecard never says more)."""
        out = []
        for engine, label in (("tomography", "tomography-api-v2"), ("qdrive", "QDrive"), ("echo", "otoc-echo-v1")):
            path = os.path.join(self.run_dir, engine, "score.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        sc = json.load(f)
                    out.append(f"Moth {label}: " + ", ".join(f"{k} {v:.3g}" for k, v in sc.items() if isinstance(v, (int, float)) and not isinstance(v, bool)))
                except (OSError, ValueError):
                    pass
        if os.path.exists(os.path.join(self.run_dir, "qdrive", "circuit.qasm")):
            out.append("A QDrive circuit exists for this run (see the facet-register source).")
        return out

    def asset_path(self, key):
        names = dict(ab_requested_vs_achieved="ab/ab_requested_vs_achieved.png", ab_frames="ab/ab_frames.png", ab_coherence="ab/ab_coherence.png")
        return os.path.join(self.run_dir, names[key]) if key in names else None

    def _apply_beat(self, index):
        """Run a beat's actions, then rebuild the script so its slides are written from the provenance and files as they are NOW."""
        beat = self._flat[index][3]
        with self.batch():
            for name, args in beat.actions:
                self.dispatch(name, **args)
            self._flat = demo_mod.flatten(self._demo_steps())

    def demo_toggle(self):
        with self.lock:
            self.demo["active"] = not self.demo["active"]
            if self.demo["active"]:
                self.demo["index"] = 0
                self._flat = demo_mod.flatten(self._demo_steps())
                self._apply_beat(0)
            self._notify()

    def demo_goto(self, delta):
        with self.lock:
            if not self.demo["active"]:
                return
            i = int(np.clip(self.demo["index"] + delta, 0, len(self._flat) - 1))
            if i != self.demo["index"]:
                self.demo["index"] = i
                self._apply_beat(i)
            self._notify()

    def demo_next(self):
        self.demo_goto(1)

    def demo_prev(self):
        self.demo_goto(-1)

    def demo_view(self):
        if not self.demo["active"]:
            return None
        si, bi, step, beat = self._flat[self.demo["index"]]
        return dict(index=self.demo["index"], total=len(self._flat), step=step.title, beat=bi + 1, beats=len(step.beats), notes=beat.notes,
                    steps=[s.title for _, _, s, b in self._flat if b is s.beats[0]], step_index=si)

    # ------------------------------------------------------------------ re-solve (never spends without an explicit, matching confirm)
    def calibration_mismatch(self):
        """None when the targets in calib_dir were made for THIS scene's patch grid (same cells, same panel of each cell), else a sentence saying
        how they differ. The payloads are built from those targets, so sending them for another scene would pay for the wrong state."""
        from src.targets.consistency import scene_problem
        try:
            with open(os.path.join(self.calib_dir, "targets.json")) as f:
                meta = json.load(f)["meta"]
        except (OSError, KeyError, ValueError):
            return None                                                    # no targets: nothing to compare
        why = scene_problem(meta, self.sampler.cells, (self.NX, self.NY), self.scene.n_panels)
        if why is None:
            return None
        return f"{why}; folder {self.calib_dir}: run studio step 2 for this drawing, or point --calib at the matching folder"

    def resolve_plan(self, engine="qdrive"):
        from src.quantum.moth_client import estimate_credits
        from src.store.cache import cached_result
        from src.targets.payloads import payload_sha256
        mismatch = self.calibration_mismatch()
        if mismatch:
            raise ValueError(mismatch)
        fname, eid = dict(qdrive=("payload_qdrive.json", "qdrive-api-v1"), graph_v1=("payload_graph_v1.json", "graph-v1"))[engine]
        path = os.path.join(self.calib_dir, fname)
        with open(path) as f:
            params = json.load(f)["params"]
        sha = payload_sha256({"engine": eid, "params": params})
        cached = cached_result(os.path.join(self.run_dir, engine), sha)
        credits = 0 if cached else estimate_credits(eid)
        plan = dict(engine=engine, engine_id=eid, payload=path, sha256=sha, credits=credits, cached=bool(cached),
                    n_targets=len(params.get("targets", params.get("operations", []))), allow_spend=self.allow_spend,
                    note=("A result for exactly this payload is already on disk: re-solving reuses it and costs nothing." if cached else
                          f"Sends this payload to Moth and spends {credits} credit(s)." + ("" if self.allow_spend else " Disabled: the server was started without --allow-spend (preview only).")))
        self._resolve_plan = plan
        return plan

    def resolve_confirm(self, engine, sha256, credits):
        with self.lock:
            plan = self.resolve_plan(engine)
            if sha256 != plan["sha256"] or credits != plan["credits"]:
                raise ValueError("the payload or its cost changed since it was shown: reopen the re-solve dialog and read it again")
            if plan["credits"] and not self.allow_spend:
                raise PermissionError("this server was started without --allow-spend, so it will not submit to Moth")
        fn = self._solve_fn
        if fn is None:
            from src.quantum import solver
            fn = dict(qdrive=solver.solve_qdrive, graph_v1=solver.solve_graph_v1)[engine]
        self.say(f"re-solving with {engine}: {plan['credits']} credit(s)")
        threading.Thread(target=self._resolve_worker, args=(fn, engine, plan), daemon=True).start()
        return dict(started=True, plan=plan)

    def _resolve_worker(self, fn, engine, plan):
        try:
            fn(self.calib_dir, self.run_dir, plan["credits"] or None)             # None = preview / cached: the solver sends nothing
            if engine == "qdrive":
                path = os.path.join(self.run_dir, "qdrive", "circuit.qasm")
                if os.path.exists(path):
                    self.set_source("circuit", self.pol_basis, path)
                    self.say("QDrive circuit loaded as the frame source")
            self.say(f"{engine} re-solve finished")
        except BaseException as e:                                           # noqa: BLE001 - report, never crash the show
            self.say(f"{engine} re-solve failed: {e}")
        finally:
            with self.lock:
                self.ui["resolve"] = False
                self._notify()

    # ------------------------------------------------------------------ views
    def live_caption(self):
        look = getattr(self, "current", None)
        if look is None:
            return None
        prov = self.prov if not self.knobs.noise else dict(self.prov, quantum_backed=False)
        if look.source == "relief":
            cap = caption_mod.describe_relief(look.draw, self.panel_names, look.draw.K, prov, control=look.control)
            if getattr(look.draw, "game", None) is not None and getattr(self, "_game", None) is not None:
                cap["game"] = self._game_view(look.draw.game)
            return cap
        return caption_mod.describe(look.draw, self.panel_names, look.draw.K, prov, noise=look.noise)

    def _game_view(self, rnd=None):
        g = getattr(self, "_game", None)
        if g is None:
            return None
        s = g.stats.summary()
        out = dict(summary=s, text=g.describe(), predicted=g.predicted, mermin=g.M, parties=g.parties)
        if rnd is not None:
            out["round"] = dict(inputs=list(rnd["inputs"]), outcomes=list(rnd["outcomes"]), win=rnd["win"],
                                text=("won" if rnd["win"] else "lost") + f" (inputs {''.join(map(str, rnd['inputs']))}, outcomes {' '.join('+' if v > 0 else '-' for v in rnd['outcomes'])})")
        return out

    def audience(self):
        dv = self.demo_view()
        if dv:
            beat = self._flat[dv["index"]][3].audience
            if beat["kind"] == "live":
                return dict(kind="live", caption=self.live_caption(), provenance_quantum=self.prov["quantum_backed"])
            return dict(beat, provenance_quantum=self.prov["quantum_backed"])
        return dict(kind="live", caption=self.live_caption(), provenance_quantum=self.prov["quantum_backed"])

    def engine_results(self):
        """Engines whose requested-vs-achieved table is on disk for this run: the evidence for the overlay's gap rings."""
        return [e for e in ("graph_v1", "qdrive") if os.path.exists(os.path.join(self.run_dir, e, "requested_vs_achieved.json"))]

    def _engine_achieved(self, engine):
        """{(a, b): achieved <ZZ>} for patch-patch edges from an engine's own result (graph-v1: exact tomography; QDrive: its circuit, exact)."""
        with open(os.path.join(self.run_dir, engine, "requested_vs_achieved.json")) as f:
            rows = json.load(f)["rows"]
        return {tuple(r["qubits"]): r["achieved"] for r in rows if r["kind"] == "patch-patch" and r["achieved"] is not None}

    def overlay_image(self, gaps="source"):
        """The patch-graph overlay as a PIL image (laptop only). gaps: 'source' compares the active source's own correlations with the
        requested ones; 'graph_v1' / 'qdrive' compare that ENGINE's returned correlations (the honest gap)."""
        from src.graph.edges import patch_edges
        look = getattr(self, "current", None)
        if self.family == "relief":
            rel = self.relief()
            return overlay_mod.render_relief_overlay(self.scene, rel["facets"], rel["spec"], getattr(self, "_canvas", None), look.draw if look else None,
                                                     domains=rel["ds"])
        targets = None
        tp = os.path.join(self.calib_dir, "targets.json")
        if os.path.exists(tp):
            with open(tp) as f:
                targets = json.load(f)
        requested = overlay_mod.requested_edge_values(targets, self.layout)
        achieved = None
        if requested is not None:
            if gaps in ("graph_v1", "qdrive") and gaps in self.engine_results():
                achieved = self._engine_achieved(gaps)
            else:
                achieved = overlay_mod.empirical_edge_correlations(self.pool, patch_edges(self.sampler.J))
        lit = look.draw.lit if look else np.zeros(self.sampler.n)
        img = overlay_mod.render_overlay(self.scene, self.sampler.cells, self.sampler.J, getattr(self, "_canvas", None), lit,
                                         look.draw.lamp if look else (1, 1), look.draw.pol if look else (1,) * self.scene.n_panels,
                                         requested, achieved)
        return img

    def coherence(self, n=4000):
        """Coherence metrics of the active pool (the spec §2.2 numbers), for the operator panel. For the relief family: the exact budget numbers."""
        if self.family == "relief":
            sp = self.relief()["spec"]
            pb = self._panel_budget()
            return dict(family="relief", kappa=self._kappa(), visibility=[round(pb[k]["visibility"], 4) for k in sorted(pb)],
                        sum_tau2=[round(pb[k]["sum_tau2"], 4) for k in sorted(pb)], facets=sp.F, polarity=sp.P, simulated=sp.n_sim, circuit=sp.n_full,
                        couplings=len(sp.edges))
        p = self.pool
        sel = slice(0, min(n, len(p.lamps)))
        return coherence_metrics(self.sampler, p.lamps[sel], p.shots[sel], p.pol[sel])

    def _engine_info(self):
        rel = self.relief()
        ds = rel["ds"]
        if ds is None:
            return dict(name="panel", text="per-panel relief: one polarity qubit per panel")
        from src.geometry.domains import frustration
        sp = rel["spec"]
        lk = sp.lock_array()
        fr, frl = frustration(ds), frustration(ds, lock=True, locked=None if lk is None else [d for d in range(sp.P) if lk[d]])
        return dict(name="domain", domains=ds.n_domains, domain_edges=len(ds.edges), creases=sum(1 for e in ds.edges if e[2] == "crease"),
                    backend=getattr(self.rstate, "backend", "exact") if self.rstate is not None else None, lock=self.relief_params["domain"]["lock"],
                    frustrated_cycles=fr["frustrated_cycles"], frustrated_cycles_with_lamp=frl["frustrated_cycles"],
                    text=f"domain relief: {ds.n_domains} depth domains, {len(ds.edges)} coupled edges, {fr['frustrated_cycles']} frustrated cycles "
                         f"({frl['frustrated_cycles']} with the lamp), lamp lock {self.relief_params['domain']['lock']}")

    def _kappa(self):
        rel = self.relief()
        if rel["ds"] is not None:                                           # per-domain budget: sum tau^2 = group_size tau^2
            return round(float(rel["ds"].params["tau"] * np.sqrt(rel["ds"].params["group_size"])), 4)
        return rel["facets"].kappa

    def _panel_budget(self):
        """{panel: dict(n, sum_tau2, visibility)}. For the domain engine the visibility is the mean over the panel's domains."""
        rel = self.relief()
        from src.geometry.facets import panel_budget
        if rel["ds"] is None:
            return panel_budget(rel["facets"])
        ds, sp = rel["ds"], rel["spec"]
        out = {}
        for k in range(self.scene.n_panels):
            idx = [d for d in range(ds.n_domains) if ds.domain_panel[d] == k]
            fac = [i for i in range(sp.F) if int(sp.facet_panel[i]) == k]
            out[k] = dict(n=len(fac), sum_tau2=float((sp.tau[fac] ** 2).sum()), visibility=float(np.mean([ds.visibility(d) for d in idx])) if idx else 1.0)
        return out

    def _budget_text(self):
        if self.family == "relief":
            b = self.budget
            if b.get("engine") == "domain":
                ds = self.relief()["ds"]
                return (f"{b['patches']} facet qubits + {b['panels']} domain polarity qubits = {b['simulated']} simulated qubits across {len(ds.edges)} domain edges "
                        f"({sum(1 for e in ds.edges if e[2] == 'crease')} creases); with the lamp (2) and observation (2) registers the circuit has {b['total']}")
            return (f"{b['patches']} facet qubits + {b['panels']} polarity qubits = {b['simulated']} simulated qubits; with the lamp (2) and observation (2) registers "
                    f"the circuit has {b['total']}")
        return format_budget(self.budget)

    def science(self):
        """What the science panel shows (laptop only): budget and visibility, the depth-sphere dots of the current frame, the controls, and
        the last witness certificate."""
        if self.family != "relief":
            return None
        rel = self.relief()
        pb = self._panel_budget()
        look = getattr(self, "current", None)
        sph = None
        if look is not None and look.source == "relief":
            sph = [caption_mod.panel_dot(look.draw.gamma, look.draw.chi, p) for p in look.draw.pol]
        return dict(kappa=self._kappa(), panels=[dict(panel=k, name=self.panel_names[k], facets=pb[k]["n"], sum_tau2=pb[k]["sum_tau2"], visibility=pb[k]["visibility"])
                                                for k in sorted(pb)], engine=self._engine_info(),
                    spheres=sph, controls=dict(noise=self.knobs.noise, dephased=self.knobs.dephased, light_interference=self.knobs.light_interference),
                    certificate=None if self.certificate is None else dict(provenance=self.certificate["provenance"], lines=self.certificate["lines"]),
                    game=self._game_view(getattr(look.draw, "game", None) if look is not None else None), evolution=getattr(self, "evolution", None),
                    couplings=len(rel["spec"].edges), entangle=self.relief_params["entangle"])

    def state(self):
        look = getattr(self, "current", None)
        c = self.cal
        relief = self.family == "relief"
        look_d = None if look is None else dict(id=look.id, lamp=list(look.draw.lamp), pol=list(look.draw.pol), K=look.draw.K, noise=look.noise,
                                                position=self.cursor + 1, history=len(self.history))
        if look is not None and relief:
            d = look.draw
            from src.quantum.relief_state import depth_word
            look_d.update(control=look.control, gamma_deg=float(np.degrees(d.gamma)), chi_deg=float(np.degrees(d.chi)), lean=list(d.lean),
                          words=[depth_word(v) for v in d.lean], lamp_mode=d.lamp_mode, world=d.world, obs=d.obs)
        out = dict(
            knobs=self.knobs.snapshot(relief),
            family=self.family, sources=list(SOURCES), science=self.science(),
            look=look_d,
            provenance=self.prov, pol_basis=self.pol_basis, source=self.source_kind, coupling=self.coupling,
            output=dict(self.output), size=list(self.size), pattern=self.pattern, align=None if not self.align else dict(
                quad=self.align["quad"], selected=self.align["selected"]),
            calibration=dict(method=c.method, rms_px=c.rms_px, margin_px=c.margin_px, projector=list(c.projector), notes=c.notes, path=self.calib_path,
                             saved=os.path.exists(self.calib_path)),
            glass=dict(self.glass), demo=self.demo_view(), ui=dict(self.ui), error=self.error, messages=list(self.messages)[-12:],
            scene=dict(panels=self.panel_names, patches=self.relief()["spec"].F if relief else self.sampler.n,
                       qubits=self.budget["total"] if relief else self.layout["n_qubits"], budget=self._budget_text(),
                       over_cap=self.budget["over_cap"], canvas=[self.scene.W, self.scene.H]),
            engine_results=self.engine_results(), allow_spend=self.allow_spend, audience=self.audience(), ladder=LADDER, patterns=list(patterns.ALL), seq=self.hub.seq,
        )
        return _clean(out)

    # ------------------------------------------------------------------ dispatch
    ACTIONS = ("next", "prev", "auto_set", "auto_toggle", "cycle", "hold_world", "hold_lamp", "release_lamp", "toggle_hold_lamp", "hold_pol",
               "hold_pol_all", "toggle_hold_pol", "pol_basis_toggle", "K", "K_step", "hardness_sweep_set", "blend", "blend_step", "gain", "fade",
               "blackout_set", "blackout_toggle", "noise_set", "noise_toggle", "overlay_toggle", "help_toggle", "resolve_open", "resolve_close",
               "demo_toggle", "demo_next", "demo_prev", "snapshot", "pattern", "pattern_cycle", "set_source", "align_start", "align_move",
               "align_nudge", "align_apply", "align_cancel", "align_reset", "set_margin", "dephased_set", "dephased_toggle", "light_interference_set",
               "light_interference_toggle", "contrast", "witness_run", "game_set", "game_toggle", "game_reset", "evolve_set", "evolve_step")

    def dispatch(self, action, /, **args):
        """Run an operator action by name. `action` is positional-only because some actions take an argument called `name`."""
        if action not in self.ACTIONS or (self.family == "relief" and action in LEGACY_ONLY):
            raise ValueError(f"unknown action {action!r}")
        fn = self.set_pattern if action == "pattern" else getattr(self, action)
        result = fn(**args)
        return result if isinstance(result, dict) else {}
