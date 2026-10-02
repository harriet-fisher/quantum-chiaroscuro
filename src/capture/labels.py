"""labels.json: the hand-off between the pen tool and everything downstream (handoff §8, stage 'Label').

Polygons are stored in `image` pixel coordinates; `canvas` is the working resolution everything else is
computed at (polygons are scaled to it, uniformly, when a Scene is built).

`source` says what the coordinates mean:
  "photo"      drawn over a photo taken from the projector's position: camera space, needs a camera->projector
               warp (src/projector, handoff §14 day 4) before projecting;
  "projector"  drawn directly on the projector's own screen with no photo (pen tool, blank mode): `image` is the
               projector's pixel grid, so the shapes already sit where the light lands and NO warp is needed;
  "synthetic"  the built-in test bay window.

    {
      "schema": "standing-light/labels@1",
      "source": "photo",
      "image":  {"file": "bay.jpg", "width": 4032, "height": 3024},
      "canvas": {"width": 1200, "height": 900},
      "grid":   {"nx": 6, "ny": 4, "min_cover": 0.30},
      "panels": [{"id": 0, "name": "left", "plane_id": 0, "angle_deg": 35.0, "polygon": [[x, y], ...],
                  "window_near": [0.25, 0.80], "window_far": [0.10, 0.65], "depth_x": [70.0, 420.0]}, ...],
      "glass":      [{"polygon": [[x, y], ...]}, ...],
      "off_limits": [{"polygon": [[x, y], ...]}, ...]
    }

angle_deg is the yaw of the plane's normal about the vertical axis; positive tilts the normal toward +x
(image right). A bay window's left wing seen from inside is +35, the centre 0, the right wing -35.
Panels sharing a plane_id are coplanar (must share an angle). Glass and off_limits are impenetrable.
window_near / window_far / depth_x are optional; defaults come from bounds.windows.default_window.
"""
import json
import math

from src.bounds.windows import default_window
from src.mask.build_mask import scale_poly
from src.store.jsonfmt import dumps_compact

SCHEMA = "standing-light/labels@1"
SOURCES = ("photo", "projector", "synthetic")
CANVAS_WIDTH = 1200
GRID_DEFAULT = {"nx": 6, "ny": 4, "min_cover": 0.30}


def canvas_for(image_w, image_h, canvas_w=CANVAS_WIDTH):
    return {"width": int(canvas_w), "height": max(1, round(canvas_w * image_h / image_w))}


def bay_window_labels(glass_scale=0.45):
    """The synthetic bay window from mock/standing_light_mock.py, as a labels dict."""
    polys = {
        "left":   [(70, 70), (420, 150), (420, 560), (70, 640)],
        "centre": [(420, 150), (780, 150), (780, 560), (420, 560)],
        "right":  [(780, 150), (1130, 70), (1130, 640), (780, 560)],
    }
    angles = {"left": 35.0, "centre": 0.0, "right": -35.0}
    panels = []
    for k, (name, poly) in enumerate(polys.items()):
        p = dict(id=k, name=name, plane_id=k, angle_deg=angles[name], polygon=[[float(x), float(y)] for x, y in poly])
        p.update(default_window(p["angle_deg"], p["polygon"]))
        panels.append(p)
    return {
        "schema": SCHEMA,
        "source": "synthetic",
        "image": {"file": None, "width": 1200, "height": 700},
        "canvas": {"width": 1200, "height": 700},
        "grid": dict(GRID_DEFAULT),
        "panels": panels,
        "glass": [{"polygon": [list(pt) for pt in scale_poly(p["polygon"], glass_scale)]} for p in panels],
        "off_limits": [],
    }


def _check_poly(poly, where):
    if len(poly) < 3:
        raise ValueError(f"{where}: polygon needs at least 3 vertices")
    for pt in poly:
        if len(pt) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pt):
            raise ValueError(f"{where}: bad vertex {pt!r}")


def _check_window(w, where):
    if w is not None and not (len(w) == 2 and 0 <= w[0] < w[1] <= 1):
        raise ValueError(f"{where}: window must be [lo, hi] with 0 <= lo < hi <= 1, got {w!r}")


def validate(labels):
    """Raise ValueError describing the first problem found; return labels unchanged otherwise."""
    if labels.get("schema") != SCHEMA:
        raise ValueError(f"unknown schema {labels.get('schema')!r}, expected {SCHEMA!r}")
    if labels.get("source", "photo") not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {labels['source']!r}")
    img, cv, grid = labels["image"], labels["canvas"], labels["grid"]
    for name, d in (("image", img), ("canvas", cv)):
        if not (isinstance(d["width"], int) and isinstance(d["height"], int) and d["width"] > 0 and d["height"] > 0):
            raise ValueError(f"{name}: width and height must be positive integers")
    if not (isinstance(grid["nx"], int) and isinstance(grid["ny"], int) and grid["nx"] > 0 and grid["ny"] > 0):
        raise ValueError("grid: nx and ny must be positive integers")
    if not 0 < grid["min_cover"] <= 1:
        raise ValueError("grid: min_cover must be in (0, 1]")
    if not labels["panels"]:
        raise ValueError("at least one panel is required")
    plane_angle = {}
    for i, p in enumerate(labels["panels"]):
        where = f"panel {i} ({p.get('name', '?')})"
        if p["id"] != i:
            raise ValueError(f"{where}: id must equal its position in the list")
        _check_poly(p["polygon"], where)
        if not isinstance(p["plane_id"], int):
            raise ValueError(f"{where}: plane_id must be an integer")
        if not (isinstance(p["angle_deg"], (int, float)) and -89 <= p["angle_deg"] <= 89):
            raise ValueError(f"{where}: angle_deg must be a number in [-89, 89]")
        if plane_angle.setdefault(p["plane_id"], p["angle_deg"]) != p["angle_deg"]:
            raise ValueError(f"{where}: plane {p['plane_id']} already has angle {plane_angle[p['plane_id']]}")
        _check_window(p.get("window_near"), where)
        _check_window(p.get("window_far"), where)
    for key in ("glass", "off_limits"):
        for i, g in enumerate(labels[key]):
            _check_poly(g["polygon"], f"{key} {i}")
    return labels


def fill_defaults(labels):
    """Return a copy with window_near / window_far / depth_x filled in wherever they are missing."""
    out = json.loads(json.dumps(labels))
    out.setdefault("source", "photo" if out["image"].get("file") else "projector")
    if not out.get("canvas"):
        out["canvas"] = canvas_for(out["image"]["width"], out["image"]["height"])
    for p in out["panels"]:
        if p.get("window_near") is None:
            p.update(default_window(p["angle_deg"], p["polygon"]))
        else:
            p.setdefault("window_far", None)
            p.setdefault("depth_x", None)
    return out


def is_projector_space(labels):
    """True when the shapes are already in projector pixels (no camera-to-projector warp needed)."""
    return labels.get("source", "photo") != "photo"


def load_labels(path):
    with open(path) as f:
        return validate(fill_defaults(json.load(f)))


def save_labels(labels, path):
    with open(path, "w") as f:
        f.write(dumps_compact(labels) + "\n")
