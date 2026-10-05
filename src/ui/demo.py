"""The five-minute demo flow (handoff §10.4) as data: steps made of beats. Each beat sets the projector and the operator's knobs
(`actions`, the same names Session.dispatch() takes) and says what the audience screen shows.

Honesty rule: slides are built from what is actually on disk and from the current provenance. A slide about the QDrive comparison
says so when QDrive has not returned a circuit for the scene; the "quantum" wording follows the source's provenance (ui.caption), and the
closing slide states what is NOT quantum-specific as plainly as what is.
"""
from dataclasses import dataclass, field

from src.ui.caption import provenance_line


@dataclass
class Beat:
    audience: dict                                    # {"kind": "live" | "slide" | "overlay", ...}
    actions: list = field(default_factory=list)       # [(action name, args dict), ...]
    notes: str = ""                                   # for the operator


@dataclass
class Step:
    title: str
    beats: list


def slide(title, bullets, image=None, footnote=""):
    return dict(kind="slide", title=title, bullets=list(bullets), image=image, footnote=footnote)


LIVE = dict(kind="live")


def _engine_lines(ab):
    if not ab:
        return ["No engine results on disk yet: nothing has been run on a Moth engine for this scene."]
    out = []
    for name, e in ab.get("engines", {}).items():
        out.append(f"{name}: rms error {e['rms']:.2f} over {e['n_achieved']} requested correlations; {e['n_gap_over_0p1']} off by more than 0.1")
    c = ab.get("credits", {})
    spent = sum(v for v in c.values())
    out.append(f"Credits spent on these runs: {spent}")
    for n in ab.get("notes", []):
        out.append(n)
    return out


def build_relief_steps(ctx):
    """The seven beats of the Superposed Relief demo (v2 handoff section 7). No beat chooses a look: the actions set the projector, the exposure and the
    controls, never the light, the depth or the observation, which come out of the circuit."""
    prov, cert = ctx["prov"], ctx.get("tomography")
    from src.quantum.relief_witness import visibility_table
    table = visibility_table()
    where = ("a circuit whose facet register Moth's QDrive prepared, executed gate by gate on the Aer simulator on this laptop" if prov.get("from_moth") and prov.get("executed") == "aer" else
             "a circuit returned by Moth's QDrive for the facet qubits" if prov.get("from_moth") else
             "a circuit executed gate by gate on the Aer simulator on this laptop; no Moth result is involved" if prov.get("executed") == "aer" else
             "a circuit simulated exactly on this laptop; no Moth result is involved")
    moth = ctx.get("moth_lines") or ["Nothing has been run on a Moth engine for this scene yet: no tomography, no QDrive circuit."]
    return [
        Step("1. One wall, flat", [
            Beat(slide("A wall has one shape",
                       ["Every pixel here is lit the same: the real window, flat.", "Light can make it look swollen, hollowed or creased. The depth is not in the wall; it is in the relationship between the light and the viewer's guess."],
                       footnote="The projector shows white-in-frame; the glass stays black."),
                 [("noise_set", dict(on=False)), ("dephased_set", dict(on=False)), ("auto_set", dict(on=False)), ("pattern", dict(name="white"))], "Point out the black panes."),
        ]),
        Step("2. The wall as qubits", [
            Beat(dict(kind="overlay", caption=ctx["budget"] + ". Each node is a patch whose state is its surface normal; the arrow is the slope direction, its length the tilt."),
                 [("pattern", dict(name=None)), ("auto_set", dict(on=True, seconds=4.0))], "Facet qubits, polarity qubits, lamp and observation registers."),
        ]),
        Step("3. Photons: Lambert is Born", [
            Beat(LIVE, [("K", dict(value=1)), ("hardness_sweep_set", dict(on=True)), ("auto_set", dict(on=True, seconds=2.5))],
                 "A patch is bright with the probability its normal faces the light (Lambert's cosine law is the Born rule). Hard grain with few photons, soft with many: exposure, not a scene property."),
        ]),
        Step("4. Bump, hollow and flat", [
            Beat(LIVE, [("hardness_sweep_set", dict(on=False)), ("K", dict(value=48)), ("auto_set", dict(on=True, seconds=3.5))],
                 "Watch the depth-sphere dot on the operator panel. Where the observation lands decides the frame: a pole is a definite bump or hollow, the equator is undecided and flat. Nothing is toggled."),
        ]),
        Step("5. Control B beside the quantum state", [
            Beat(slide("Same pictures, different correlations",
                       ["Control B replaces each depth qubit by a classical mixture of bump and hollow. Every patch keeps the same statistics.",
                        "What disappears is interference between the depths: the polarity outcome loses its bias (1+V)/2 and the stabilizer X(B) Z(facets) drops from 1 to 0.",
                        "Press W on the operator panel to run the witness: it certifies entanglement for the quantum state and not for the controls."],
                       footnote=where),
                 [("auto_set", dict(on=False))], "Run the witness (W) here."),
            Beat(LIVE, [("dephased_set", dict(on=True)), ("auto_set", dict(on=True, seconds=3.5))], "Control B on the wall."),
            Beat(LIVE, [("dephased_set", dict(on=False)), ("noise_set", dict(on=True))], "Control A: independent noise."),
        ]),
        Step("6. The visibility law", [
            Beat(slide("How finely can relief be specified?",
                       ["Interference between depth maps has visibility V = product of cos(tilt) over the facets, about exp(-sum tilt^2 / 2).",
                        "A fixed tilt per facet collapses: " + ", ".join(f"N={n}: {a:.2g}" for n, a, _ in table[:5]) + ".",
                        "Tilt = kappa / sqrt(N) keeps it constant: " + ", ".join(f"N={n}: {b:.2f}" for n, _, b in table[:7]) + ". Finer grain, more tentative depth.",
                        "A product relief is classically easy at any N; hardness would need entangling layers with geometry-set angles (an IQP-shaped circuit): a direction, not a claim."],
                       footnote="Verified in the simulator for N = 8 to 18 (tests); beyond that it is an extrapolation."),
                 [("noise_set", dict(on=False)), ("auto_set", dict(on=False))], "The scaling law."),
        ]),
        Step("7. Honest scorecard", [
            Beat(slide("What is and is not quantum here", [provenance_line(prov)] + moth + [
                "Simulable: at 18 simulated qubits everything here is simulated exactly on a laptop. No quantum advantage is claimed.",
                "Controls: the lamp and observation registers are uniform draws (physical randomness on a QPU, pseudo-random here); the quantum content is the facet register, the polarity superposition and the controlled operations between them."],
                        footnote=("Executed gate by gate on the Aer simulator on this laptop; every run is checked against the reference state (state.json, and the tests)." if prov.get("executed") == "aer"
                                  else "Local reference circuit, fidelity-checked against the Qiskit circuit and Aer in the tests.")),
                 [("noise_set", dict(on=False)), ("dephased_set", dict(on=False)), ("auto_set", dict(on=False))], "End on this slide."),
        ]),
    ]


def build_steps(ctx):
    if ctx.get("family") == "relief":
        return build_relief_steps(ctx)
    """ctx: dict with prov (provenance), budget (text), ab (ab_summary.json or None), qdrive_circuit (bool), comp (complementary
    report.json or None), can_complementary (bool), has_overlay_targets (bool). Returns list[Step]."""
    prov, ab, comp = ctx["prov"], ctx.get("ab"), ctx.get("comp")
    steps = [
        Step("1. The problem", [
            Beat(slide("Random shading that still reads as one light",
                       ["If every patch picks its own brightness independently you get noise, not relief.",
                        "A depth illusion needs the draw to agree with itself: coplanar patches agree, a crease flips the contrast, everything looks lit by one light."],
                       footnote="On the wall now: every outcome an independent coin flip."),
                 [("pattern", dict(name=None)), ("blackout_set", dict(on=False)), ("noise_set", dict(on=True)), ("auto_set", dict(on=True, seconds=3.0))],
                 "Independent noise on the wall. Press N (or Space) to swap in the correlated draw."),
            Beat(LIVE, [("noise_set", dict(on=False))], "Same geometry, same code, correlated draw: now it reads as a single light."),
        ]),
        Step("2. Geometry and the black-glass rule", [
            Beat(slide("The geometry sets the fences",
                       ["Panels, planes and creases come from the shapes drawn on the real window.",
                        "Solid lines: panels that may be lit. Dashed: glass, which may never be."],
                       footnote="The projector shows the outline."),
                 [("auto_set", dict(on=False)), ("pattern", dict(name="outline"))], "Check the outline sits on the real window edges."),
            Beat(slide("Glass is exactly black",
                       ["Full light on every projectable pixel: and the glass stays at zero.",
                        "Glass gets no qubits, no correlation targets, and no blur leaks into it; every frame is checked in projector space before it is sent."],
                       footnote="The projector shows white-in-frame."),
                 [("pattern", dict(name="white"))], "Point out the black panes."),
        ]),
        Step("3. The qubit graph", [
            Beat(dict(kind="overlay", caption=ctx["budget"]),
                 [("pattern", dict(name=None)), ("auto_set", dict(on=True, seconds=4.0))],
                 "Patches, two lamp qubits and one polarity qubit per panel; edge colour is the requested correlation."),
        ]),
        Step("4. Run: requested vs achieved", [
            Beat(slide("What was asked, and what the engine returned", _engine_lines(ab),
                       image="ab_requested_vs_achieved" if ab and ctx.get("ab_images") else None,
                       footnote="Every number is read back from the engine's own result, never assumed."),
                 [], "Per-engine table; the honest gap is the point."),
        ]),
        Step("5. Live frames: three knobs in turn", [
            Beat(LIVE, [("noise_set", dict(on=False)), ("auto_set", dict(on=True, seconds=3.0)), ("hardness_sweep_set", dict(on=False)), ("K", dict(value=40)),
                        ("release_lamp", {}), ("hold_pol_all", dict(value=-1))], "Knob 1, the lighting world (lamp qubits): the light direction changes, relief held sunk."),
            Beat(LIVE, [("hold_world", dict(l1=1, l2=1)), ("hold_pol_all", dict(value=None))], "Knob 2, relief polarity: the light is held; panels flip between sunk and raised."),
            Beat(LIVE, [("hold_pol_all", dict(value=-1)), ("hardness_sweep_set", dict(on=True))], "Knob 3, hardness: shots averaged sweeps 1 to 128, hard to soft. A free control, not a quantum claim."),
        ] + ([Beat(LIVE, [("hardness_sweep_set", dict(on=False)), ("K", dict(value=8)), ("release_lamp", {}), ("hold_pol_all", dict(value=None)),
                          ("set_source", dict(kind="complementary", pol_basis="x"))],
                   "The complementary experiment (UNTESTED stand-in): polarity is read in the X basis, so it is bound to the light, not free.")]
             if ctx.get("can_complementary") else [])),
        Step("6. QDrive vs graph-v1", [
            Beat(slide("The case for or against each engine",
                       _engine_lines(ab) + ([] if ctx.get("qdrive_circuit") else
                                            ["QDrive has not returned a circuit for this scene yet (small jobs have run, but no whole-scene job has finished), so there is no QDrive frame source to compare."]),
                       image="ab_frames" if ab and ctx.get("ab_images") else None,
                       footnote="graph-v1 returns exact tomography and the top 20 outcomes; QDrive returns a circuit that can be sampled for unlimited frames."),
                 [("auto_set", dict(on=True, seconds=4.0))], "Credits: graph-v1 5, QDrive 1."),
        ]),
        Step("7. What is quantum-specific, and what is not", [
            Beat(slide("Honest scorecard", _closing_bullets(prov, comp), footnote="No quantum advantage is claimed: everything here is classically simulable at this size."),
                 [("auto_set", dict(on=False))], "End on this slide."),
        ]),
    ]
    return steps


def _closing_bullets(prov, comp):
    b = [provenance_line(prov),
         "Not quantum-specific: a lighting world drawn in the Z basis is classically reproducible (the classical mock does exactly that), and hardness is sampling statistics.",
         "Quantum-specific candidate: relief polarity as a complementary observable. Lit/dark is read in Z and raised/sunk in X, so a definite light state leaves polarity undecided.",
         "Entanglement is the specification: the geometry's correlation targets are the program; at scale, frustrated loops are the part expected to become hard to sample classically (inferred, unverified)."]
    if comp:
        for name, e in comp.get("states", {}).items():
            for m in e.get("mermin", []):
                b.append(f"Local stand-in state (UNTESTED against Moth): {m['parties']}-party Mermin value {m['exact']:.2f}; the classical bound is {m['local_bound']}, the quantum maximum {m['quantum_max']}.")
                return b
    return b


def flatten(steps):
    """[(step index, beat index, step, beat)] in order."""
    return [(i, j, s, b) for i, s in enumerate(steps) for j, b in enumerate(s.beats)]
