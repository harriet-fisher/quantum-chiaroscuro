#!/usr/bin/env python3
"""Photographed glass-stays-black check on the real surface (spec §13 Evaluation, handoff §10.1 step 5, §12).

The software guarantees that the glass pixels it SENDS are exactly 0 (projector.warp). Whether the glass really stays dark on
the wall also depends on calibration, projector creep, lens bloom and what the glass reflects, and only a photograph can say.
This tool turns the photograph into a number.

Take the photos from one fixed camera position (tripod, manual exposure and focus, room dimmed, nothing moves between them).
Show, in turn, with the operator panel's test-pattern buttons:

    black.jpg   projector black                          ambient light + the projector's black level
    white.jpg   "white in frame": full light on every projectable pixel, glass black
    dots.jpg    the nine-dot pattern                     only for the optical check and the frame fidelity (optional)
    frame.jpg   a real frame (optional), with frame_projector.png, the exact frame sent (the snapshot button writes it)

TWO CHECKS, because they answer different questions:

  PHYSICAL (the one that matters)  needs --truth: a labels.json traced on black.jpg itself (python -m src.capture.pen_tool draw
        black.jpg, mark the real glass and panels). Those polygons are where the glass physically is in the camera's view,
        independent of the projector, so any light the software sends onto the glass shows up as light inside them.
        This is the only check that catches a projector that has crept, a bad calibration, or glass that reflects.
  OPTICAL  needs dots.jpg. Registers the camera to the projector and measures light at the pixels the software SENT as
        black (its own labels, through the calibration). It cannot see misalignment (it measures what we sent, wherever it
        landed), only black level, bloom and scatter, so it is reported as a separate, explicitly weaker result.

    python -m src.projector.verify --photos DIR --truth truth_labels.json [--labels L --calib C]
    python -m src.projector.verify --selftest [--out runs/verify_selftest]      # synthetic camera: aligned, creep, leaky glass

Method: black is subtracted from white (ambient and black level cancel); light inside the glass, away from the glass edge by
--guard pixels (which absorbs camera blur), is compared with the light on the panels:

    leak_ratio = mean(glass) / mean(panel)        PASS needs leak_ratio <= 0.02 and p99(glass)/mean(panel) <= 0.06

Verdicts are PASS, FAIL or INCONCLUSIVE (overexposed, too dark for the noise, too little glass in view). Without --truth the
headline is NOT CHECKED: a report that cannot see misalignment does not get to say the glass is safe.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi

from src.mask.build_mask import poly_mask
from src.projector import patterns
from src.projector.calibrate import DotError, apply_homography, dots_homography, load_calibration, scale_matrix
from src.projector.warp import Warper, warp_image

LEAK_RATIO_MAX = 0.02
LEAK_P99_RATIO_MAX = 0.06
MIN_SNR = 12.0
MIN_ZONE_PIXELS = 400
SATURATION_MAX = 0.05
MAX_SIDE = 2400


# ------------------------------------------------------------------ photo handling
def srgb_to_linear(v):
    v = np.asarray(v, float)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(np.asarray(v, float), 0, 1)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def load_luminance(path, max_side=MAX_SIDE):
    """A photo as (linear luminance, encoded 0..1 values), both float arrays, upright, longest side at most max_side."""
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    except Exception as e:                                                       # noqa: BLE001 - friendly message
        hint = "\nPillow cannot read HEIC. Convert first:  sips -s format jpeg IN.heic --out out.jpg" if str(path).lower().endswith((".heic", ".heif")) else ""
        raise SystemExit(f"cannot open photo {path!r}: {e}{hint}")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    enc = np.asarray(im, float) / 255.0
    return srgb_to_linear(enc), enc


# ------------------------------------------------------------------ zones
def _stats(a, mask):
    v = a[mask]
    if v.size == 0:
        return dict(n=0, mean=None, p99=None, std=None)
    return dict(n=int(v.size), mean=float(v.mean()), p99=float(np.percentile(v, 99)), std=float(v.std()))


class Zones:
    """Boolean masks on some pixel grid: the panels (where light belongs), the glass (where it must not be) and the rest."""

    def __init__(self, panel, glass, guard_px, seen=None):
        self.shape = panel.shape
        dist = ndi.distance_transform_edt(~panel)
        seen = np.ones(panel.shape, bool) if seen is None else seen
        self.panel_core = ndi.binary_erosion(panel, iterations=int(guard_px), border_value=1) & seen
        self.glass = glass & ~panel
        self.glass_core = self.glass & (dist > guard_px) & seen
        self.glass_band = self.glass & (dist <= guard_px) & seen
        self.outside_core = ~glass & ~panel & (dist > guard_px) & seen
        self.panel = panel


def camera_zones(truth_labels, shape, guard_px):
    """Zones from labels traced on the camera's own photo: where the physical glass and panels are in its view."""
    H, W = shape
    sx, sy = W / truth_labels["image"]["width"], H / truth_labels["image"]["height"]
    if abs(sx / sy - 1) > 0.02:
        raise SystemExit(f"the truth labels were traced on a {truth_labels['image']['width']}x{truth_labels['image']['height']} image but the "
                         f"photos are {W}x{H}: not the same aspect ratio (different crop?)")
    poly = lambda p: [(x * sx, y * sy) for x, y in p]
    panels = np.zeros(shape, bool)
    for p in truth_labels["panels"]:
        panels |= poly_mask(poly(p["polygon"]), (W, H))
    blocked = np.zeros(shape, bool)
    for key in ("glass", "off_limits"):
        for g in truth_labels.get(key, []):
            blocked |= poly_mask(poly(g["polygon"]), (W, H))
    return Zones(panels & ~blocked, blocked, guard_px)


def projector_zones(scene, cal, guard_px):
    """Zones on the projector grid from the SOFTWARE's labels through the calibration: what was sent, not where the glass is."""
    size = tuple(cal.projector)
    lit0 = Warper(cal.H, scene.frame, size, margin_px=0).lit
    glass = warp_image(scene.blocked.astype(float), cal.H, size) >= 0.5
    return Zones(lit0, glass, guard_px)


# ------------------------------------------------------------------ analysis
def leak_report(A, zones, enc_white=None, thresholds=None):
    """Light in the glass relative to light on the panels, from a difference image A (white minus black, linear)."""
    th = dict(leak_ratio=LEAK_RATIO_MAX, leak_p99_ratio=LEAK_P99_RATIO_MAX, min_snr=MIN_SNR) | (thresholds or {})
    panel, glass = _stats(A, zones.panel_core), _stats(A, zones.glass_core)
    band, outside = _stats(A, zones.glass_band), _stats(A, zones.outside_core)
    sat = float((enc_white[zones.panel_core] >= 0.98).mean()) if enc_white is not None and zones.panel_core.any() else None
    rep = dict(panel=panel, glass_core=glass, glass_edge_band=band, outside=outside, saturated_fraction=sat, thresholds=th)
    reasons = []
    if panel["n"] < MIN_ZONE_PIXELS or not panel["mean"] or panel["mean"] <= 0:
        reasons.append("the panels are hardly visible in the photo (is the projector on, and the right part of the wall in view?)")
    if glass["n"] < MIN_ZONE_PIXELS:
        reasons.append(f"only {glass['n']} glass pixels are in view (need {MIN_ZONE_PIXELS}): move the camera, or enlarge the glass or lower --guard")
    if sat is not None and sat > SATURATION_MAX:
        reasons.append(f"{100 * sat:.0f}% of the panel pixels are overexposed in white.jpg: lower the exposure so the ratio is meaningful")
    if not reasons:
        rep["leak_ratio"] = max(glass["mean"], 0.0) / panel["mean"]
        rep["leak_p99_ratio"] = max(glass["p99"], 0.0) / panel["mean"]
        rep["edge_band_ratio"] = max(band["mean"], 0.0) / panel["mean"] if band["n"] else None
        rep["outside_ratio"] = max(outside["mean"], 0.0) / panel["mean"] if outside["n"] else None
        noise = glass["std"] / np.sqrt(max(glass["n"] / 25.0, 1.0))      # blur correlates pixels: roughly 25 per independent sample
        rep["snr"] = float(panel["mean"] / max(noise, 1e-9))
        if rep["snr"] < th["min_snr"]:
            reasons.append(f"signal-to-noise {rep['snr']:.1f} < {th['min_snr']}: too dark or too noisy; open the aperture or raise ISO")
    if reasons:
        rep["verdict"], rep["reasons"] = "INCONCLUSIVE", reasons
    elif rep["leak_ratio"] <= th["leak_ratio"] and rep["leak_p99_ratio"] <= th["leak_p99_ratio"]:
        rep["verdict"], rep["reasons"] = "PASS", []
    else:
        why = []
        if rep["leak_ratio"] > th["leak_ratio"]:
            why.append(f"mean light in the glass is {100 * rep['leak_ratio']:.1f}% of the panels' (limit {100 * th['leak_ratio']:.0f}%)")
        if rep["leak_p99_ratio"] > th["leak_p99_ratio"]:
            why.append(f"the brightest 1% of glass is {100 * rep['leak_p99_ratio']:.1f}% of the panels' (limit {100 * th['leak_p99_ratio']:.0f}%)")
        rep["verdict"], rep["reasons"] = "FAIL", why
    return rep


def fidelity(frame_lin, black_lin, H_cam2proj, size, frame_png, panel_core):
    """Correlation between the photographed light and the frame that was sent, over the panels (alignment and tone check)."""
    A = warp_image(frame_lin - black_lin, H_cam2proj, size)
    seen = warp_image(np.ones_like(frame_lin), H_cam2proj, size) >= 1 - 1e-6
    sent = (np.asarray(frame_png, float) / 255.0) ** 2.2                       # projectors apply roughly this gamma
    m = panel_core & seen
    r = float(np.corrcoef(A[m], sent[m])[0, 1]) if m.sum() > 100 else None
    return dict(correlation_with_sent=r, pixels=int(m.sum()))


# ------------------------------------------------------------------ annotated picture
def annotate(path, A, zones, rep, title):
    """Left: the light seen (white minus black) with the glass tinted blue and the panel edge in yellow; right: the same, stretched
    10x so that faint light in the glass shows."""
    ref = max(rep["panel"]["mean"] or 1e-6, 1e-6)
    tint = zones.glass.astype(float) * 0.35
    edge = zones.panel ^ ndi.binary_erosion(zones.panel, iterations=2)
    tiles = []
    for img in (np.clip(A / ref, 0, 1.2) / 1.2, np.clip(A / (0.1 * ref), 0, 1)):
        g = img * (1 - tint)
        rgb = np.stack([g, g, g + tint], axis=-1)
        rgb[edge] = (1.0, 0.85, 0.0)
        tiles.append(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)))
    w, h = tiles[0].size
    out = Image.new("RGB", (w * 2, h + 60), (22, 23, 26))
    out.paste(tiles[0], (0, 60)); out.paste(tiles[1], (w, 60))
    d = ImageDraw.Draw(out)
    colour = {"PASS": (95, 211, 141), "FAIL": (255, 107, 94), "INCONCLUSIVE": (255, 214, 10)}[rep["verdict"]]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.2f}%"
    d.text((10, 8), f"{title}: {rep['verdict']}   glass leak {pct(rep.get('leak_ratio'))} of panel light   p99 {pct(rep.get('leak_p99_ratio'))}", fill=colour)
    d.text((10, 30), "left: light seen (white - black), glass tinted blue, yellow = panel edge.   right: stretched 10x to show faint leaks", fill=(180, 180, 185))
    out.save(path)


# ------------------------------------------------------------------ the synthetic camera
def random_camera(projector, cam_size=(1600, 1000), rng=None, rotation_deg=12.0, keystone=0.0004):
    """A plausible camera homography proj -> cam: scale, rotation, a little perspective, centred."""
    rng = rng or np.random.default_rng(0)
    th = np.deg2rad(rng.uniform(-rotation_deg, rotation_deg))
    s = 0.72 * min(cam_size[0] / projector[0], cam_size[1] / projector[1])
    c, sn = np.cos(th), np.sin(th)
    R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]])
    Tn = np.array([[1, 0, -projector[0] / 2], [0, 1, -projector[1] / 2], [0, 0, 1.0]])
    P = np.array([[1, 0, 0], [0, 1, 0], [rng.uniform(-keystone, keystone), rng.uniform(-keystone, keystone), 1.0]])
    Tc = np.array([[1, 0, cam_size[0] / 2], [0, 1, cam_size[1] / 2], [0, 0, 1.0]])
    H = Tc @ scale_matrix(s) @ R @ P @ Tn
    return H / H[2, 2]


def simulate_photo(frame_u8, H_proj2cam, cam_size, ambient=0.03, black_level=0.012, blur_px=1.6, noise=0.002, gain=0.85,
                   gamma=2.2, shift_px=(0.0, 0.0), glass_reflect=0.0, glass_mask=None, rng=None):
    """What a camera sees of a projected frame: (linear luminance, sRGB-encoded luminance), both 0..1.
    shift_px moves the projected image on the wall (a projector that crept); glass_reflect makes the glass_mask pixels (on the
    wall, i.e. NOT shifted with the projector) glow with that fraction of the panel light (a leaky glass)."""
    rng = rng or np.random.default_rng(0)
    f = ndi.shift((np.asarray(frame_u8, float) / 255.0) ** gamma, (shift_px[1], shift_px[0]), order=1, mode="constant", cval=0.0)
    lin = ambient + black_level + gain * f
    if glass_reflect and glass_mask is not None:
        lin = lin + glass_reflect * gain * glass_mask * float(f.max())
    cam = warp_image(lin, H_proj2cam, cam_size, cval=ambient)
    lin = np.clip(ndi.gaussian_filter(cam, blur_px) + rng.normal(0, noise, cam.shape), 0, 1)
    return lin, linear_to_srgb(lin)


def labels_through(labels, H_canvas2cam, cam_size):
    """The labels' panel and glass polygons mapped into a camera view: the 'truth trace' a person would draw on that photo."""
    sx = labels["canvas"]["width"] / labels["image"]["width"]
    mp = lambda poly: [list(map(float, q)) for q in apply_homography(H_canvas2cam, [(x * sx, y * sx) for x, y in poly])]
    return dict(image=dict(width=cam_size[0], height=cam_size[1]), panels=[dict(polygon=mp(p["polygon"])) for p in labels["panels"]],
                glass=[dict(polygon=mp(g["polygon"])) for g in labels["glass"]], off_limits=[dict(polygon=mp(g["polygon"])) for g in labels["off_limits"]])


# ------------------------------------------------------------------ driver
def find_photo(folder, stem):
    hits = [h for h in sorted(glob.glob(os.path.join(folder, stem + ".*"))) if h.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"))]
    return hits[0] if hits else None


def run(black, white, dots=None, truth=None, labels=None, cal=None, frame=None, frame_png=None, out_dir=None, guard_px=8,
        optical_guard_px=10, thresholds=None, loader=load_luminance):
    """black / white / dots / frame are paths, or (linear, encoded) pairs when loader is None. Returns the report dict:
    verdict (physical check, or 'NOT CHECKED' without truth), plus 'physical', 'optical' and 'frame' sections when they ran."""
    get = loader or (lambda p: p)
    (b_lin, _), (w_lin, w_enc) = get(black), get(white)
    if b_lin.shape != w_lin.shape:
        raise SystemExit(f"black and white photos differ in size ({b_lin.shape} vs {w_lin.shape}): camera moved or re-cropped?")
    rep = dict(verdict="NOT CHECKED", reasons=["no --truth trace of the real glass was given, so misalignment cannot be seen"])
    diff = w_lin - b_lin
    if truth is not None:
        z = camera_zones(truth, b_lin.shape, guard_px)
        rep["physical"] = leak_report(diff, z, w_enc, thresholds)
        rep["verdict"], rep["reasons"] = rep["physical"]["verdict"], rep["physical"]["reasons"]
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            annotate(os.path.join(out_dir, "glass_check_physical.png"), diff, z, rep["physical"], "PHYSICAL")
    if dots is not None and labels is not None and cal is not None:
        from src.geometry.planes import build_scene
        d_lin, _ = get(dots)
        try:
            H, rms, _ = dots_homography(d_lin, b_lin, cal.projector)
        except DotError as e:
            rep["optical"] = dict(verdict="INCONCLUSIVE", reasons=[f"could not register the camera to the projector: {e}"])
        else:
            zp = projector_zones(build_scene(labels), cal, optical_guard_px)
            seen = warp_image(np.ones_like(diff), H, cal.projector) >= 1 - 1e-6
            zp = Zones(zp.panel, zp.glass, optical_guard_px, seen)
            enc_p = warp_image(w_enc, H, cal.projector)
            A = warp_image(diff, H, cal.projector)
            rep["optical"] = leak_report(A, zp, enc_p, thresholds)
            rep["optical"]["registration_rms_projector_px"] = float(rms)
            rep["optical"]["note"] = "measures pixels the software sent as black; blind to misalignment (see the physical check)"
            if frame is not None and frame_png is not None:
                rep["frame"] = fidelity(get(frame)[0], b_lin, H, cal.projector, frame_png, zp.panel_core)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                annotate(os.path.join(out_dir, "glass_check_optical.png"), A, zp, rep["optical"], "OPTICAL")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "glass_check.json"), "w") as f:
            json.dump(rep, f, indent=1)
    return rep


def print_report(rep):
    pct = lambda v: "n/a" if v is None else f"{100 * v:.2f}%"
    print(f"\nglass-stays-black check: {rep['verdict']}")
    for name in ("physical", "optical"):
        s = rep.get(name)
        if not s:
            continue
        print(f"  {name}: {s['verdict']}")
        if "leak_ratio" in s:
            th = s["thresholds"]
            print(f"    light in the glass: mean {pct(s['leak_ratio'])} of the panels' (limit {100 * th['leak_ratio']:.0f}%), brightest 1% "
                  f"{pct(s['leak_p99_ratio'])} (limit {100 * th['leak_p99_ratio']:.0f}%); edge band {pct(s.get('edge_band_ratio'))}; SNR {s['snr']:.0f}")
        for r in s.get("reasons", []):
            print("    -", r)
    if "frame" in rep:
        print(f"  real frame: correlation with what was sent {rep['frame']['correlation_with_sent']}")
    if rep["verdict"] == "NOT CHECKED":
        for r in rep["reasons"]:
            print("  -", r)


def selftest(out):
    """Synthetic camera, three cases through BOTH checks: aligned; projector crept 14 px (physical must FAIL, optical is blind and
    PASSES, which is the point of having two); glass that reflects 8% of the panel light (both FAIL)."""
    from src.capture.labels import bay_window_labels, fill_defaults, validate
    from src.geometry.planes import build_scene
    from src.projector.calibrate import labels_calibration
    labels = validate(fill_defaults(bay_window_labels()))
    scene = build_scene(labels)
    proj, cam = (1280, 720), (1600, 1000)
    cal = labels_calibration(labels, proj)
    warper = Warper(cal.H, scene.frame, proj, cal.margin_px)
    Hpc = random_camera(proj, cam, rng=np.random.default_rng(3))
    truth = labels_through(labels, Hpc @ cal.H, cam)           # the physical glass, as traced on the camera photo
    glass_p = warp_image(scene.blocked.astype(float), cal.H, proj) >= 0.5
    black_f, white_f, dots_f = patterns.black(proj), patterns.white(warper), patterns.dots(proj)
    cases = {"aligned": (dict(), "PASS", "PASS"),
             "projector_crept_14px": (dict(shift_px=(14.0, 6.0)), "FAIL", "PASS"),
             "glass_reflects_8pct": (dict(glass_reflect=0.08, glass_mask=glass_p), "FAIL", "FAIL")}
    ok = True
    for name, (kw, want_phys, want_opt) in cases.items():
        seeds = iter(range(100, 200))                                  # independent noise in every photo, as with a real camera
        shot = lambda f, **k: simulate_photo(f, Hpc, cam, rng=np.random.default_rng(next(seeds)), **k)
        creep = {k: v for k, v in kw.items() if k == "shift_px"}       # a crept projector shifts the dots too; glass reflection only the white
        rep = run(shot(black_f), shot(white_f, **kw), shot(dots_f, **creep), truth, labels, cal, out_dir=os.path.join(out, name), loader=None)
        got = (rep["verdict"], rep["optical"]["verdict"])
        ok &= got == (want_phys, want_opt)
        print(f"\n[{name}] expected physical {want_phys}, optical {want_opt}")
        print_report(rep)
    print("\nselftest:", "all cases behaved" if ok else "UNEXPECTED VERDICT")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", help="folder with black.*, white.* (and dots.*, frame.* + frame_projector.png)")
    ap.add_argument("--truth", help="labels.json traced on black.jpg with the pen tool: where the real glass and panels are")
    ap.add_argument("--labels", help="the scene labels (needed for the optical check)")
    ap.add_argument("--calib", help="calibration.json (needed for the optical check)")
    ap.add_argument("--out", help="default: <photos>/verify")
    ap.add_argument("--guard", type=int, default=8, help="photo pixels kept clear around the panels (absorbs camera blur)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        sys.exit(0 if selftest(a.out or "runs/verify_selftest") else 1)
    if not a.photos:
        ap.error("--photos is required (or --selftest)")
    paths = {s: find_photo(a.photos, s) for s in ("black", "white", "dots", "frame")}
    if not (paths["black"] and paths["white"]):
        raise SystemExit(f"missing in {a.photos}: black.jpg and white.jpg are required")
    from src.capture.labels import load_labels
    truth = load_labels(a.truth) if a.truth else None
    labels = load_labels(a.labels) if a.labels else None
    cal = load_calibration(a.calib) if a.calib else None
    png = None
    if paths["frame"] and os.path.exists(os.path.join(a.photos, "frame_projector.png")):
        png = np.array(Image.open(os.path.join(a.photos, "frame_projector.png")).convert("L"))
    out = a.out or os.path.join(a.photos, "verify")
    rep = run(paths["black"], paths["white"], paths["dots"], truth, labels, cal, paths["frame"], png, out, a.guard)
    print_report(rep)
    print(f"report and annotated pictures in {out}")
    sys.exit({"PASS": 0, "FAIL": 1}.get(rep["verdict"], 2))


if __name__ == "__main__":
    main()
