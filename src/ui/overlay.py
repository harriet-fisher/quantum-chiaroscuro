"""Patch-graph overlay for the operator panel (handoff §10.2: "patch graph with correlation-coloured edges, current lamp bits,
achieved-vs-requested gaps"). Laptop only: it is never sent to the projector.

Drawn on the scene's canvas, over the dimmed current frame:
  nodes   one per patch qubit, filled by that patch's lit fraction in the current look
  edges   between 4-neighbour patches; colour = correlation (blue agree, orange disagree), width = |value|. The value is the
          REQUESTED <ZZ> from the calibration targets when they match this scene, otherwise the correlation the source actually has
  rings   red ring on an edge whose achieved value (measured from the source's own shots) is more than `gap` away from requested
  badges  the lamp bits (lighting world) and each panel's polarity bit for this look
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.graph.edges import patch_edges

BLUE, ORANGE, RED, WHITE = (76, 155, 232), (242, 163, 58), (255, 107, 94), (240, 240, 240)
GAP = 0.1


def empirical_edge_correlations(pool, edges):
    """Achieved <Z_a Z_b> for patch edges, from the shots a source actually produces."""
    return {e: float((pool.shots[:, e[0]] * pool.shots[:, e[1]]).mean()) for e in edges}


def requested_edge_values(targets, layout):
    """{(a, b): requested <ZZ>} for patch-patch edges from a calibration targets.json, or None when it is for a different layout."""
    if not targets:
        return None
    if targets["meta"].get("qubit_map") != {k: layout[k] for k in ("patches", "lamps", "polarity", "n_qubits")}:
        return None
    return {tuple(r["qubits"]): float(r["ZZ"]) for r in targets["relationships"] if r.get("kind") == "patch-patch"}


def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def render_overlay(scene, cells, J, look_img, lit, lamp, pol, requested=None, achieved=None, panel_names=None):
    """RGB PIL image at scene size. look_img: the current frame (float 0..1) or None; lit: per-patch lit fractions."""
    base = np.zeros((scene.H, scene.W, 3)) if look_img is None else np.repeat((np.asarray(look_img) * 0.38)[..., None], 3, axis=2)
    base[scene.blocked] = base[scene.blocked] * 0.5 + np.array([0.08, 0.17, 0.35]) * 0.5
    im = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    small, big = _font(max(11, scene.W // 90)), _font(max(14, scene.W // 60))
    for p in scene.panels:
        pts = [tuple(q) for q in p.polygon]
        d.line(pts + [pts[0]], fill=(255, 255, 255), width=1)
    centre = [((c["box"][0] + c["box"][1]) / 2, (c["box"][2] + c["box"][3]) / 2) for c in cells]
    gaps = 0
    for a, b in patch_edges(J):
        v = requested.get((a, b), requested.get((b, a))) if requested else None
        v = achieved[(a, b)] if v is None and achieved else (v if v is not None else float(J[a, b]))
        col = BLUE if v >= 0 else ORANGE
        d.line([centre[a], centre[b]], fill=col, width=max(1, int(1 + 4 * abs(v))))
        if requested is not None and achieved is not None and (a, b) in achieved and (a, b) in requested:
            if abs(achieved[(a, b)] - requested[(a, b)]) > GAP:
                mx, my = (centre[a][0] + centre[b][0]) / 2, (centre[a][1] + centre[b][1]) / 2
                r = max(6, scene.W // 150)
                d.ellipse([mx - r, my - r, mx + r, my + r], outline=RED, width=2)
                gaps += 1
    r = max(8, scene.W // 90)
    for i, ((cx, cy), f) in enumerate(zip(centre, lit)):
        g = int(40 + 215 * float(f))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(g, g, g), outline=WHITE)
        d.text((cx - r + 1, cy - r - 12), str(i), fill=(255, 214, 10), font=small)
    for k, p in enumerate(scene.panels):
        cx = sum(q[0] for q in p.polygon) / len(p.polygon)
        cy = sum(q[1] for q in p.polygon) / len(p.polygon)
        d.text((cx - 28, cy - 8), f"Q{k} {'raised' if pol[k] > 0 else 'sunk'}", fill=(95, 211, 141), font=big)
    d.text((10, 8), f"L1 {'right' if lamp[0] > 0 else 'left'} ({int(lamp[0]):+d})   L2 {'frontal' if lamp[1] > 0 else 'grazing'} ({int(lamp[1]):+d})", fill=(255, 214, 10), font=big)
    legend = "edges: blue agree, orange disagree, width = |<ZZ>|" + ("; red ring = achieved is >0.1 from requested" if requested is not None and achieved is not None else "")
    d.text((10, scene.H - 22), legend + (f"   [{gaps} gaps]" if requested is not None and achieved is not None else ""), fill=(200, 200, 205), font=small)
    return im


def render_relief_overlay(scene, fs, spec, look_img, draw, domains=None):
    """Operator overlay for the relief family (laptop only). Facets are tinted by this frame's lit fraction and drawn as nodes at their
    centroids with an arrow along their slope azimuth (length ~ tilt); coplanar couplings are blue, creases orange; each panel shows its
    depth word and a small depth sphere with a dot where the observation landed (north pole = decided raised, south = decided sunk).
    With `domains` (the domain engine) the polarity qubits are drawn too: a ring at each domain's centroid, filled yellow when this frame decided it
    raised and red when sunk, joined along the domain graph (blue coplanar, orange crease, thick = more shared boundary); a + or - marks the
    crater-gauge sign that couples the domain to the lamp."""
    import math
    from src.quantum.relief_state import depth_word
    base = np.zeros((scene.H, scene.W, 3)) if look_img is None else np.repeat((np.asarray(look_img) * 0.38)[..., None], 3, axis=2)
    base[scene.blocked] = base[scene.blocked] * 0.5 + np.array([0.08, 0.17, 0.35]) * 0.5
    im = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    small, big = _font(max(11, scene.W // 90)), _font(max(14, scene.W // 60))
    for p in scene.panels:
        pts = [tuple(q) for q in p.polygon]
        d.line(pts + [pts[0]], fill=(255, 255, 255), width=1)
    nq = fs.n_quantum
    for i, j, kind, n in fs.edges:
        if i >= nq or j >= nq:                                           # classical plateaus carry no qubit and no coupling
            continue
        a, b = fs.facets[i].anchor, fs.facets[j].anchor
        d.line([a, b], fill=BLUE if kind == "coplanar" else ORANGE, width=max(1, min(5, 1 + n // 250)))
    many = fs.n > 24
    r = max(8, scene.W // 90) if not many else max(5, scene.W // 170)
    for f, v in zip(fs.facets, draw.lit if draw is not None else np.zeros(fs.n)):
        cx, cy = f.anchor
        g = int(40 + 215 * float(v))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(g, g, g), outline=WHITE)
        if f.tau > 0.05:
            L = 6 * r * math.sin(f.tau)
            d.line([(cx, cy), (cx + L * math.cos(f.phi), cy + L * math.sin(f.phi))], fill=(95, 211, 141), width=2)
        if not many:
            d.text((cx - r + 1, cy - r - 12), f"{f.idx}{'b' if f.kind == 'band' else 'i'}", fill=(255, 214, 10), font=small)
    if domains is not None:
        pol = getattr(draw, "domain_pol", None) if draw is not None else None
        for a, b, kind, n in domains.edges:
            d.line([domains.domain_centroid[a], domains.domain_centroid[b]], fill=BLUE if kind == "coplanar" else ORANGE, width=3 if kind == "crease" else 2)
        R = max(9, scene.W // 80)
        for k, (cx, cy) in enumerate(domains.domain_centroid):
            fill = None if pol is None else ((255, 214, 10) if pol[k] > 0 else (255, 107, 94))
            d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=fill, outline=WHITE, width=2)
            sign = int(domains.lock_sign[k])
            if sign:
                d.text((cx - 3, cy - 6), "+" if sign > 0 else "-", fill=(20, 20, 24), font=small)
    if draw is not None:
        for k, p in enumerate(scene.panels):
            cx = sum(q[0] for q in p.polygon) / len(p.polygon)
            cy = sum(q[1] for q in p.polygon) / len(p.polygon)
            d.text((cx - 40, cy - 8), f"Q{k} {depth_word(draw.lean[k])}", fill=(95, 211, 141), font=big)
            R = max(16, scene.W // 48)
            sx, sy = cx, cy + 2.4 * R
            d.ellipse([sx - R, sy - R, sx + R, sy + R], outline=(200, 200, 205), width=1)
            d.line([(sx - R, sy), (sx + R, sy)], fill=(90, 90, 96))
            d.text((sx + R + 3, sy - R - 2), "raised", fill=(255, 216, 160), font=small)
            d.text((sx + R + 3, sy + R - 10), "sunk", fill=(143, 184, 255), font=small)
            dx, dy = R * math.sin(draw.gamma) * math.cos(draw.chi), -R * math.cos(draw.gamma)
            rr = max(4, R // 4)
            d.ellipse([sx + dx - rr, sy + dy - rr, sx + dx + rr, sy + dy + rr], fill=(255, 107, 94) if draw.pol[k] < 0 else (255, 214, 10))
        L1 = "light in superposition" if draw.lamp_mode != "z" else f"L1 {'right' if draw.lamp[0] > 0 else 'left'}   L2 {'frontal' if draw.lamp[1] > 0 else 'grazing'}"
        d.text((10, 8), f"{L1}   observation gamma {math.degrees(draw.gamma):.0f} deg   control: {draw.control}", fill=(255, 214, 10), font=big)
    d.text((10, scene.H - 22), "nodes: facet qubits (arrow = slope direction, length = tilt)   edges: blue coplanar, orange crease   sphere dot: where the observation landed"
           + ("   big rings: domain polarity qubits (yellow raised, red sunk; +/- lamp lock sign)" if domains is not None else ""), fill=(200, 200, 205), font=small)
    return im
