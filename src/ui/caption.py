"""Plain-language captions for the audience view (handoff §10.3: "each look is a measurement, and geometry made the draw coherent").

Everything is derived from the measurement outcomes of the current look (lamp bits, polarity bits, shots averaged K), plus the
provenance of the source those outcomes came from. The wording never claims more than the provenance supports: a classical
stand-in is called a rehearsal, a locally simulated state is said to be simulated, and nothing is called a Moth result unless it
came from a Moth job.
"""

POL_WORD = {1: "raised", -1: "sunk"}


def hardness_word(K):
    if K <= 2:
        return "hard, posterised"
    if K <= 8:
        return "crisp"
    if K <= 32:
        return "soft"
    return "very soft"


def provenance_line(prov):
    """One honest sentence about where the draws come from."""
    if prov.get("relief") and prov["quantum_backed"]:
        where = ("a circuit returned by Moth's QDrive for the facet qubits" if prov.get("from_moth")
                 else "a quantum circuit built and simulated exactly on this laptop (not a Moth result)")
        status = " This design is untested against a real Moth circuit." if prov.get("untested") else ""
        return (f"Each look is one run of {where}: every patch of the wall is a qubit whose state is its surface normal, each panel's depth is a qubit in "
                f"superposition of raised and sunk, and the light and the observation are drawn inside the circuit.{status}")
    if not prov["quantum_backed"]:
        return ("Rehearsal mode: the draws come from a classical stand-in sampler, with no quantum calls. "
                "The look is real; the claim that a quantum state drew it is not being made.")
    where = "a circuit returned by Moth's QDrive" if prov.get("from_moth") else "a quantum state built and simulated on this laptop (not a Moth result)"
    basis = " Relief polarity is read in the X basis, complementary to the lit/dark (Z) reading." if prov.get("pol_basis") == "x" else ""
    status = " This design is untested against a real Moth circuit." if prov.get("untested") else ""
    return f"Each look is one measurement of {where}.{basis}{status}"


def describe(draw, panel_names, K, prov, noise=False):
    """The audience caption for a FrameDraw. Returns a JSON-able dict."""
    lr = "right" if draw.lamp[0] > 0 else "left"
    fg = "frontal" if draw.lamp[1] > 0 else "grazing"
    panels = [dict(name=n, state=POL_WORD[int(p)], bit=int(p)) for n, p in zip(panel_names, draw.pol)]
    cap = dict(
        headline=f"Light from the {lr}, {fg}",
        panels=panels,
        hardness=f"{K} shot{'s' if K != 1 else ''} averaged: {hardness_word(K)}",
        lamp_bits=[dict(name="L1", meaning="light from the right" if draw.lamp[0] > 0 else "light from the left", bit=int(draw.lamp[0])),
                   dict(name="L2", meaning="frontal light" if draw.lamp[1] > 0 else "grazing light", bit=int(draw.lamp[1]))],
        provenance=provenance_line(prov),
        explain=("The geometry fixes where light may never fall (the glass) and how bright each plane may get; "
                 + ("the measurement chose the shading inside those fences" if prov["quantum_backed"] else "a classical stand-in sampler chose the shading inside those fences")
                 + ", and the correlations the geometry asked for are why the panels agree on one light."),
    )
    if noise:
        cap.update(headline="Independent noise, no correlations",
                   explain="Every patch, lamp and relief qubit is an independent coin flip. Same geometry, same fences, same compose code: the panels no longer agree on a light, and no relief reads.",
                   provenance="This is the baseline the correlated draw is compared against: plain random numbers, no quantum state.")
    return cap


def sphere_dot(gamma, chi):
    """Where the observation landed on a polarity qubit's Bloch sphere (the depth sphere): north pole = decided raised, south = decided sunk,
    equator = undecided. (x, z) in the unit disc seen from the side, the form the audience page draws."""
    import math
    return dict(x=math.sin(gamma) * math.cos(chi), z=math.cos(gamma), gamma_deg=round(math.degrees(gamma), 1), chi_deg=round(math.degrees(chi), 1))


def describe_relief(draw, panel_names, K, prov, control="coherent"):
    """The audience caption for a ReliefDraw. Same keys as describe() plus the depth-sphere dots."""
    from src.quantum.relief_state import WORLD_NAMES, depth_word
    if draw.lamp_mode == "z":
        lr = "right" if draw.lamp[0] > 0 else "left"
        fg = "frontal" if draw.lamp[1] > 0 else "grazing"
        headline = f"Light from the {lr}, {fg}"
        lamp_bits = [dict(name="L1", meaning="light from the right" if draw.lamp[0] > 0 else "light from the left", bit=int(draw.lamp[0])),
                     dict(name="L2", meaning="frontal light" if draw.lamp[1] > 0 else "grazing light", bit=int(draw.lamp[1]))]
    else:
        headline = "Light in superposition: the directions interfere"
        lamp_bits = [dict(name="L1", meaning=f"read in {draw.lamp_mode.upper()}", bit=1), dict(name="L2", meaning=f"read in {draw.lamp_mode.upper()}", bit=1)]
    panels = [dict(name=n, state=depth_word(lean), bit=int(p), lean=round(float(lean), 3)) for n, p, lean in zip(panel_names, draw.pol, draw.lean)]
    dot = sphere_dot(draw.gamma, draw.chi)
    if abs(dot["z"]) > 0.95:
        look = "The observation landed at a pole of the depth sphere: each panel's depth is decided, so the frame shows a definite bump or hollow."
    elif abs(dot["z"]) < 0.25:
        look = ("The observation landed on the equator of the depth sphere: raised and sunk are equally present, so no panel can show a bevel in this frame. "
                "It looks flat; the evidence that both depths are there is in the correlations, not in the picture.")
    else:
        look = "The observation landed between a pole and the equator: the more decided the depth, the stronger the bevel, and the less the two depths can interfere."
    cap = dict(
        headline=headline, panels=panels, hardness=f"{K} photon{'s' if K != 1 else ''} averaged: {hardness_word(K)}", lamp_bits=lamp_bits,
        provenance=provenance_line(prov), observation=dict(text=look, **dot),
        spheres=[dict(name=n, **dot, outcome=0 if p > 0 else 1) for n, p in zip(panel_names, draw.pol)],
        explain=("Lambert's cosine law is the Born rule: a patch is bright with the probability that its surface normal faces the light, so a frame is a measurement, "
                 "not a render. " + look))
    if control == "noise":
        cap.update(headline="Independent noise, no correlations",
                   explain="Every patch and every depth is an independent coin flip. Same geometry, same fences, same compose code: no depth reads and no light is shared.",
                   provenance="This is the baseline the quantum draw is compared against: plain random numbers, no quantum state.")
    elif control == "dephased":
        cap.update(headline="Control B: the depth is decided before anyone looks",
                   explain=("Each panel's depth is a classical mixture of raised and sunk instead of a superposition. Every patch has the same statistics as in the quantum "
                            "state, but the interference between the two depths is gone, and with it the correlations that certify entanglement."),
                   provenance="Control experiment: a classical mixture of depth maps, simulated on this laptop. Not the quantum state.")
    return cap
