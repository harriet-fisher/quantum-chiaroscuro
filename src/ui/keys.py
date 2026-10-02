"""Keyboard bindings for performance mode (handoff §10.2): one table, used by the operator page (serialised to JSON) and by tests.

Each binding is (keys, action, args, label, group). `keys` are KeyboardEvent.key values (letters are matched case-insensitively);
`action` names a Session.dispatch() action. A binding with `demo` set only applies when demo mode is on (and shadows the plain one).

`family` says where a binding exists: "both", "relief" (Superposed Relief, the performance mode) or "classical" (the pool-based rehearsal
sources). In the relief family there are NO keys that choose a look: the light, the depth decision and the observation axis are drawn
inside the circuit, so the lighting-world, relief-polarity and polarity-basis keys exist only for the classical sources (v2 handoff 4.5).
"""

BINDINGS = [
    # ---- looks
    dict(keys=[" ", "ArrowRight"], action="next", args={}, label="next look (a new draw)", group="Look"),
    dict(keys=[" ", "ArrowRight"], action="demo_next", args={}, label="next demo beat", group="Demo", demo=True),
    dict(keys=["ArrowLeft"], action="prev", args={}, label="previous look", group="Look"),
    dict(keys=["ArrowLeft"], action="demo_prev", args={}, label="previous demo beat", group="Demo", demo=True),
    dict(keys=["a"], action="auto_toggle", args={}, label="auto-cycle on / off", group="Look"),
    dict(keys=["r"], action="next", args={}, label="re-sample (new draw, same knobs)", group="Look"),
    # ---- lighting world (lamp qubits)
    dict(keys=["1"], action="hold_world", args=dict(l1=1, l2=1), label="hold: light from right, frontal", group="Lighting world", family="classical"),
    dict(keys=["2"], action="hold_world", args=dict(l1=-1, l2=1), label="hold: light from left, frontal", group="Lighting world", family="classical"),
    dict(keys=["3"], action="hold_world", args=dict(l1=1, l2=-1), label="hold: light from right, grazing", group="Lighting world", family="classical"),
    dict(keys=["4"], action="hold_world", args=dict(l1=-1, l2=-1), label="hold: light from left, grazing", group="Lighting world", family="classical"),
    dict(keys=["l"], action="toggle_hold_lamp", args={}, label="hold / release the current lighting world", group="Lighting world", family="classical"),
    dict(keys=["0"], action="release_lamp", args={}, label="lighting world: auto", group="Lighting world", family="classical"),
    # ---- relief polarity (one qubit per panel)
    dict(keys=["5"], action="hold_pol_all", args=dict(value=-1), label="hold: all panels sunk", group="Relief polarity", family="classical"),
    dict(keys=["6"], action="hold_pol_all", args=dict(value=1), label="hold: all panels raised", group="Relief polarity", family="classical"),
    dict(keys=["p"], action="toggle_hold_pol", args={}, label="hold / release the current polarity", group="Relief polarity", family="classical"),
    dict(keys=["7"], action="hold_pol_all", args=dict(value=None), label="polarity: auto", group="Relief polarity", family="classical"),
    dict(keys=["x"], action="pol_basis_toggle", args={}, label="read polarity in Z / X (complementary experiment)", group="Relief polarity", family="classical"),
    # ---- science panel (relief family): controls and experiments, labelled as such, never as looks
    dict(keys=["m"], action="dephased_toggle", args={}, label="control B: dephased depth (classical mixture of bump and hollow)", group="Science (relief)", family="relief"),
    dict(keys=["i"], action="light_interference_toggle", args={}, label="experiment: read the lamp register in X (light directions interfere)", group="Science (relief)", family="relief"),
    dict(keys=["w"], action="witness_run", args={}, label="witness run: measure the certificate on this state", group="Science (relief)", family="relief"),
    # ---- hardness and blend
    dict(keys=["["], action="K_step", args=dict(step=-1), label="harder (fewer shots averaged)", group="Hardness and blend"),
    dict(keys=["]"], action="K_step", args=dict(step=1), label="softer (more shots averaged)", group="Hardness and blend"),
    dict(keys=["-"], action="blend_step", args=dict(step=-1), label="less blend", group="Hardness and blend"),
    dict(keys=["=", "+"], action="blend_step", args=dict(step=1), label="more blend", group="Hardness and blend"),
    # ---- projector
    dict(keys=["b"], action="blackout_toggle", args={}, label="blackout (projector black)", group="Projector"),
    dict(keys=["n"], action="noise_toggle", args={}, label="swap the quantum draw for independent noise (the §2.2 test)", group="Projector"),
    dict(keys=["t"], action="pattern_cycle", args={}, label="test patterns: black, white, outline, dots, grid, edge, off", group="Projector"),
    dict(keys=["s"], action="snapshot", args={}, label="snapshot the projector frame for the photographed test", group="Projector"),
    # ---- panel
    dict(keys=["o"], action="overlay_toggle", args={}, label="patch-graph overlay (laptop only)", group="Panel"),
    dict(keys=["d"], action="demo_toggle", args={}, label="demo mode (the five-minute script)", group="Panel"),
    dict(keys=["v"], action="resolve_open", args={}, label="re-solve (shows the cost first; never spends without a click)", group="Panel"),
    dict(keys=["?", "h"], action="help_toggle", args={}, label="this help", group="Panel"),
]

DISPLAY = {" ": "Space", "ArrowRight": "→", "ArrowLeft": "←"}


def applies(b, family):
    return family is None or b.get("family", "both") in ("both", family)


def bindings_json(family=None):
    return [dict(b, demo=b.get("demo", False), family=b.get("family", "both")) for b in BINDINGS if applies(b, family)]


def help_rows(family=None):
    """[(group, [(keys text, label), ...])] in table order, for the help overlay and the runbook."""
    groups = {}
    for b in BINDINGS:
        if applies(b, family):
            groups.setdefault(b["group"], []).append((" / ".join(DISPLAY.get(k, k.upper() if len(k) == 1 else k) for k in b["keys"]), b["label"]))
    return list(groups.items())
