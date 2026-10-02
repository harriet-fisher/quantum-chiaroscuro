"""Projector calibration: homography from canvas (labels) space to projector pixels (handoff §7.2, §10.1 step 5, §11.11, §14 day 4).

Everything here is plain numpy (a normalised DLT), so there is no OpenCV dependency. Coordinates use the "pixel centre is
an integer index" convention everywhere (array element [j, i] sits at x = i, y = j), which is what PIL's polygon fill and
scipy's map_coordinates both use, so a polygon vertex lands where its filled pixels land.

Three ways to get the canvas -> projector homography H (Calibration.method):

  "scale"    labels drawn in projector space with no photo (source "projector"): the shapes already sit where the light
             lands, so H is only a uniform scale (plus centring) from the canvas to the projector's pixel grid. Needs nothing.
  "corners"  four landmarks. The operator drags four projected markers until each sits on its physical feature (a corner of
             the bay frame, say); H = DLT(landmark in canvas -> where the marker had to go). This is also the quick
             re-calibration after the projector has been bumped (§11.11): the landmarks are the original corners.
  "dots"     labels traced on a photo (source "photo"): the camera and projector are related by a homography because the
             wall is (close to) a plane. Photograph the projector's own dot pattern (patterns.dots) and the same scene with
             the projector black; detect_dots() finds the nine dots; H_cam->proj follows.

Calibration is saved as calibration.json (schema standing-light/calibration@1) and rescaled when the output window's real
pixel size differs from the size it was made at.
"""
import json
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from src.store.jsonfmt import dumps_compact

SCHEMA = "standing-light/calibration@1"
METHODS = ("scale", "corners", "dots")
DEFAULT_MARGIN_PX = 2          # projector pixels shaved off every lit region so a small misalignment cannot light the glass


# ------------------------------------------------------------------ homography maths
def _normalise(P):
    """Hartley normalisation: translate the centroid to 0 and scale to mean distance sqrt(2). Returns (Pn, T)."""
    c = P.mean(axis=0)
    d = np.sqrt(((P - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d if d > 0 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return (P - c) * s, T


def fit_homography(src, dst):
    """3x3 H with dst ~ H @ src (homogeneous), least squares over >= 4 correspondences (normalised DLT).
    Raises ValueError for fewer than 4 points or a degenerate (collinear) configuration."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError(f"need at least 4 matching (x, y) points, got {src.shape} and {dst.shape}")
    sn, Ts = _normalise(src)
    dn, Td = _normalise(dst)
    rows = []
    for (x, y), (u, v) in zip(sn, dn):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, sv, Vt = np.linalg.svd(np.array(rows))
    if sv[-2] < 1e-9 * max(sv[0], 1e-30):
        raise ValueError("the points are degenerate (collinear or repeated): cannot fit a homography")
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("degenerate homography (maps the origin to infinity)")
    return H / H[2, 2]


def apply_homography(H, pts):
    """Map an (N, 2) array (or a single (x, y)) through H."""
    P = np.atleast_2d(np.asarray(pts, float))
    q = np.c_[P, np.ones(len(P))] @ np.asarray(H, float).T
    out = q[:, :2] / q[:, 2:3]
    return out if np.ndim(pts) > 1 else out[0]


def reprojection_rms(H, src, dst):
    d = apply_homography(H, src) - np.asarray(dst, float)
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def scale_matrix(sx, sy=None, ox=0.0, oy=0.0):
    sy = sx if sy is None else sy
    return np.array([[sx, 0, ox], [0, sy, oy], [0, 0, 1.0]])


def rescale_to(H, from_size, to_size):
    """H maps canvas -> a projector grid of from_size; return the H for a grid of to_size (same physical image, resampled).
    Pixel-centre convention: x' = (x + 0.5) * s - 0.5."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    S = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])
    return S @ np.asarray(H, float)


def fit_inside(src_size, dst_size):
    """Uniform scale + centring that fits a src_size grid inside a dst_size grid (letterbox), pixel-centre convention."""
    s = min(dst_size[0] / src_size[0], dst_size[1] / src_size[1])
    ox = (dst_size[0] - src_size[0] * s) / 2 + 0.5 * s - 0.5
    oy = (dst_size[1] - src_size[1] * s) / 2 + 0.5 * s - 0.5
    return scale_matrix(s, s, ox, oy)


def quad_is_convex(q):
    """True when the 4 points, in order, form a simple convex quadrilateral (what a homography between two rectangles gives)."""
    q = np.asarray(q, float)
    z = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        z.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(v > 0 for v in z) or all(v < 0 for v in z)


# ------------------------------------------------------------------ the calibration record
@dataclass
class Calibration:
    projector: tuple                      # (w, h) of the projector pixel grid H maps into
    canvas: tuple                         # (w, h) of the canvas (labels working resolution) H maps from
    H: np.ndarray                         # 3x3, canvas -> projector
    method: str = "scale"
    landmarks_canvas: list = field(default_factory=list)       # method "corners": 4 (x, y) in canvas pixels
    landmarks_projector: list = field(default_factory=list)    # where each had to be projected (projector pixels)
    rms_px: float = 0.0                   # reprojection error over the correspondences, projector pixels
    margin_px: int = DEFAULT_MARGIN_PX
    created: str = ""
    notes: list = field(default_factory=list)

    def for_size(self, size):
        """The same calibration on a projector grid of `size` (e.g. when the output window's real size differs)."""
        size = (int(size[0]), int(size[1]))
        if size == tuple(self.projector):
            return self
        out = Calibration(**{**self.__dict__})
        out.projector, out.H = size, rescale_to(self.H, self.projector, size)
        out.landmarks_projector = [list(map(float, apply_homography(rescale_to(np.eye(3), self.projector, size), p)))
                                   for p in self.landmarks_projector]
        out.notes = self.notes + [f"rescaled from {self.projector[0]}x{self.projector[1]} to {size[0]}x{size[1]}"]
        return out

    def to_dict(self):
        return dict(schema=SCHEMA, method=self.method, projector=list(self.projector), canvas=list(self.canvas),
                    H=[[float(v) for v in row] for row in self.H],
                    landmarks_canvas=[list(map(float, p)) for p in self.landmarks_canvas],
                    landmarks_projector=[list(map(float, p)) for p in self.landmarks_projector],
                    rms_px=float(self.rms_px), margin_px=int(self.margin_px), created=self.created, notes=list(self.notes))


def _stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def scale_calibration(canvas, projector, margin_px=DEFAULT_MARGIN_PX, note=None):
    """Labels already in projector space (or synthetic): fit the canvas inside the projector, centred, aspect preserved."""
    notes = [] if note is None else [note]
    ca, pa = canvas[0] / canvas[1], projector[0] / projector[1]
    if abs(ca / pa - 1) > 0.02:
        notes.append(f"canvas aspect {ca:.3f} differs from the projector's {pa:.3f}: letterboxed (black bars carry no light)")
    return Calibration(tuple(projector), tuple(canvas), fit_inside(canvas, projector), "scale", margin_px=margin_px,
                       created=_stamp(), notes=notes)


def corner_calibration(canvas, projector, landmarks_canvas, landmarks_projector, margin_px=DEFAULT_MARGIN_PX):
    """Four (or more) landmarks: where each canvas point has to be projected to land on its physical feature."""
    lc, lp = np.asarray(landmarks_canvas, float), np.asarray(landmarks_projector, float)
    H = fit_homography(lc, lp)
    notes = []
    if len(lc) == 4 and not quad_is_convex(lp):
        raise ValueError("the four projector points do not form a convex quadrilateral in the same order as the landmarks "
                         "(two handles crossed over?)")
    return Calibration(tuple(projector), tuple(canvas), H, "corners", lc.tolist(), lp.tolist(), reprojection_rms(H, lc, lp),
                       margin_px, _stamp(), notes)


def default_landmarks(labels, canvas=None):
    """Four real vertices of the drawn panels, nearest the corners of their bounding box (TL, TR, BR, BL), in canvas pixels.
    These are the points the operator re-finds on the physical object when re-aligning after the projector moved."""
    cv = canvas or (labels["canvas"]["width"], labels["canvas"]["height"])
    sx = cv[0] / labels["image"]["width"]
    pts = np.array([[p[0] * sx, p[1] * sx] for panel in labels["panels"] for p in panel["polygon"]], float)
    x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
    out = []
    for corner in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        out.append(pts[np.argmin(((pts - corner) ** 2).sum(axis=1))].tolist())
    return out


def labels_calibration(labels, projector, photo_h=None, margin_px=DEFAULT_MARGIN_PX):
    """The starting calibration for a labels dict. Projector-space and synthetic labels need only a scale; photo labels need
    photo_h (camera -> projector, from dots_calibration) or they cannot be projected."""
    cv = (labels["canvas"]["width"], labels["canvas"]["height"])
    if labels.get("source", "photo") != "photo":
        note = None
        if labels.get("source") == "projector":
            iw, ih = labels["image"]["width"], labels["image"]["height"]
            if (iw, ih) != tuple(projector):
                note = f"labels were drawn on a {iw}x{ih} projector grid; the output is {projector[0]}x{projector[1]}"
        return scale_calibration(cv, projector, margin_px, note)
    if photo_h is None:
        raise ValueError("labels were drawn on a photo: calibrate the camera to the projector first (dots_calibration)")
    iw = labels["image"]["width"]
    canvas_to_image = scale_matrix(iw / cv[0])
    H = np.asarray(photo_h, float) @ canvas_to_image
    return Calibration(tuple(projector), cv, H / H[2, 2], "dots", margin_px=margin_px, created=_stamp(),
                       notes=["photo labels: canvas -> photo -> projector"])


# ------------------------------------------------------------------ dots (camera <-> projector)
DOT_GRID = (3, 3)
DOT_INSET = 0.12                  # outermost dots sit this fraction in from each edge of the projector frame
BIG_DOT_RATIO = 1.8               # the top-left dot has this many times the others' radius (fixes the orientation)


def dot_positions(projector):
    """The nine projector pixels the dot pattern is drawn at, row-major: index 0 is the top-left (big) dot."""
    w, h = projector
    xs = np.linspace(DOT_INSET * (w - 1), (1 - DOT_INSET) * (w - 1), DOT_GRID[0])
    ys = np.linspace(DOT_INSET * (h - 1), (1 - DOT_INSET) * (h - 1), DOT_GRID[1])
    return np.array([[x, y] for y in ys for x in xs])


class DotError(ValueError):
    pass


def detect_dots(photo_on, photo_off, expected=9):
    """Centres (x, y) and areas of the projected dots in a photo, found by differencing against the same photo with the
    projector black. Both arguments are 2-D luminance arrays of the same shape (any scale). Raises DotError with what was
    seen when the number of blobs is not `expected`."""
    a, b = np.asarray(photo_on, float), np.asarray(photo_off, float)
    if a.shape != b.shape or a.ndim != 2:
        raise DotError(f"the two photos must be the same size and greyscale, got {a.shape} and {b.shape}")
    d = ndi.gaussian_filter(np.clip(a - b, 0, None), 1.0)
    if d.max() <= 0:
        raise DotError("the 'dots on' photo is not brighter than the 'dots off' photo anywhere")
    lab, n = ndi.label(d > 0.35 * d.max())
    if n == 0:
        raise DotError("no dots found")
    idx = np.arange(1, n + 1)
    areas = ndi.sum(np.ones_like(d), lab, idx)
    keep = idx[areas >= max(6, 0.05 * np.median(areas))]
    if len(keep) != expected:
        raise DotError(f"found {len(keep)} bright blobs, expected {expected}. Keep the camera still between the two photos, "
                       f"lock exposure, dim the room, and make sure all nine dots land on a surface that shows them "
                       f"(glass swallows them).")
    cm = np.array(ndi.center_of_mass(d, lab, keep))[:, ::-1]                       # (x, y)
    return cm, np.array([areas[i - 1] for i in keep])


def order_dots(centres, areas):
    """Match detected dots to dot_positions() order using the big top-left dot and clockwise angle order about the centre dot.
    Works for any camera rotation as long as the camera is not mirrored."""
    c = np.asarray(centres, float)
    big = int(np.argmax(areas))
    rest = np.delete(np.arange(len(c)), big)
    if areas[big] < BIG_DOT_RATIO ** 2 * 0.6 * np.median(np.delete(areas, big)):
        raise DotError("cannot tell the large top-left dot from the others (is it hidden or merged with a neighbour?)")
    centre = c[np.argmin(((c - c.mean(axis=0)) ** 2).sum(axis=1))]
    ci = int(np.argmin(((c - centre) ** 2).sum(axis=1)))
    ring = [i for i in range(len(c)) if i != ci]
    ang = {i: np.arctan2(c[i, 1] - centre[1], c[i, 0] - centre[0]) for i in ring}      # y down => increasing angle is clockwise
    start = ang[big]
    ring.sort(key=lambda i: (ang[i] - start) % (2 * np.pi))
    # clockwise from top-left on screen: TL, T, TR, R, BR, B, BL, L  ->  row-major indices 0, 1, 2, 5, 8, 7, 6, 3
    row_major = {0: ring[0], 1: ring[1], 2: ring[2], 5: ring[3], 8: ring[4], 7: ring[5], 6: ring[6], 3: ring[7], 4: ci}
    return c[[row_major[k] for k in range(9)]]


def dots_homography(photo_on, photo_off, projector):
    """camera -> projector homography from a photo of dots_pattern and one with the projector black.
    Returns (H, rms_px_in_projector_pixels, camera_points)."""
    centres, areas = detect_dots(photo_on, photo_off)
    cam = order_dots(centres, areas)
    proj = dot_positions(projector)
    H = fit_homography(cam, proj)
    return H, reprojection_rms(H, cam, proj), cam


# ------------------------------------------------------------------ files
def save_calibration(cal, path):
    with open(path, "w") as f:
        f.write(dumps_compact(cal.to_dict()) + "\n")


def load_calibration(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unknown schema {d.get('schema')!r}, expected {SCHEMA!r}")
    if d["method"] not in METHODS:
        raise ValueError(f"{path}: method must be one of {METHODS}")
    H = np.array(d["H"], float)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError(f"{path}: H must be a finite 3x3 matrix")
    return Calibration(tuple(d["projector"]), tuple(d["canvas"]), H / H[2, 2], d["method"], d.get("landmarks_canvas", []),
                       d.get("landmarks_projector", []), d.get("rms_px", 0.0), int(d.get("margin_px", DEFAULT_MARGIN_PX)),
                       d.get("created", ""), d.get("notes", []))


# ------------------------------------------------------------------ command line
def main(argv=None):
    """Calibrate photo-traced labels (and inspect a calibration) from the command line.

        python -m src.projector.calibrate dots --on dots.jpg --off black.jpg --labels runs/scene/labels.json \\
               --projector 1920x1080 --out runs/scene/calibration.json
        python -m src.projector.calibrate info runs/scene/calibration.json

    dots: photograph the projector's nine-dot pattern (the operator panel's Photo test tab shows it) and the same view with the
    projector black, from the camera position the labels were traced from, then run this. Labels drawn directly on the projector's own
    screen (source "projector") need no calibration at all, and corner alignment is done live in the operator panel.
    """
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dots")
    d.add_argument("--on", required=True, help="photo with the dot pattern projected")
    d.add_argument("--off", required=True, help="photo of the same view with the projector black")
    d.add_argument("--labels", required=True)
    d.add_argument("--projector", default="1920x1080")
    d.add_argument("--out", required=True)
    d.add_argument("--margin", type=int, default=DEFAULT_MARGIN_PX)
    i = sub.add_parser("info")
    i.add_argument("calibration")
    a = ap.parse_args(argv)
    if a.cmd == "info":
        c = load_calibration(a.calibration)
        print(f"{a.calibration}: method {c.method}, canvas {c.canvas[0]}x{c.canvas[1]} -> projector {c.projector[0]}x{c.projector[1]}, "
              f"fit error {c.rms_px:.3f} px, margin {c.margin_px} px, created {c.created}")
        for n in c.notes:
            print("  note:", n)
        return
    from PIL import Image
    from src.capture.labels import load_labels
    from src.projector.verify import load_luminance
    try:
        proj = tuple(int(v) for v in a.projector.lower().split("x"))
        assert len(proj) == 2
    except (ValueError, AssertionError):
        raise SystemExit(f"--projector must look like 1920x1080, got {a.projector!r}")
    labels = load_labels(a.labels)
    on, _ = load_luminance(a.on)
    off, _ = load_luminance(a.off)
    if on.shape != off.shape:
        raise SystemExit(f"the two photos differ in size ({on.shape} vs {off.shape}): the camera must not move or change crop between them")
    with Image.open(a.on) as im:
        orig_w = im.size[0]
    iw = labels["image"]["width"]
    if abs((on.shape[1] / on.shape[0]) / (iw / labels["image"]["height"]) - 1) > 0.02:
        raise SystemExit(f"the labels were traced on a {iw}x{labels['image']['height']} photo but these photos are {on.shape[1]}x{on.shape[0]}: "
                         "they must come from the same camera position and crop")
    H, rms, _ = dots_homography(on, off, proj)
    photo_h = H @ scale_matrix(on.shape[1] / iw)               # label (original photo) pixels -> detected-photo pixels -> projector
    c = labels_calibration({**labels, "source": "photo"}, proj, photo_h, margin_px=a.margin)
    c.rms_px = rms
    save_calibration(c, a.out)
    print(f"wrote {a.out}: camera registered to the projector with {rms:.2f} px rms over the nine dots"
          + ("  (large: the wall may not be flat enough for one homography)" if rms > 3 else ""))


if __name__ == "__main__":
    main()
