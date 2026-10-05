"""Scene: everything the renderer and the patch graph need to know about the surface, built from labels."""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from src.bounds.windows import build_windows
from src.geometry.depth import depth_coordinate
from src.geometry.relief import relief_field
from src.mask.build_mask import build_masks


def normal_from_angle(angle_deg):
    """Plane normal (x, y, z) from the yaw angle; positive tilts the normal toward +x."""
    th = np.deg2rad(angle_deg)
    return (np.sin(th), 0.0, np.cos(th))


@dataclass
class Panel:
    idx: int
    name: str
    plane_id: int
    angle_deg: float
    polygon: list                    # canvas pixels
    normal: tuple
    window_near: tuple
    window_far: tuple | None
    depth_x: tuple | None            # canvas pixels


@dataclass
class Scene:
    W: int
    H: int
    panels: list
    region: np.ndarray               # panel index per pixel, -1 outside
    blocked: np.ndarray              # glass + off-limits
    frame: np.ndarray                # projectable
    impenetrable: np.ndarray         # ~frame: black, no qubits
    t: np.ndarray                    # depth coordinate
    lo: np.ndarray
    hi: np.ndarray
    relief: np.ndarray
    labels: dict = field(repr=False, default=None)

    @property
    def n_panels(self):
        return len(self.panels)

    def panel_adjacency(self, touch_px=6):
        """Pairs (a, b), a < b, of panels that share an edge: their regions come within touch_px of each other along a boundary at least about
        3 * touch_px pixels long (polarity-qubit chain). A corner where four panes meet (the middle of a bay-window wing's rail meets the edge
        between two wings) is a point, not an edge, and does not join the two panels that only touch there diagonally."""
        pairs = []
        need = 3 * touch_px * touch_px                      # a contact strip touch_px wide and 3 * touch_px long; a diagonal corner overlaps about touch_px**2 / 2
        for a in range(self.n_panels):
            grown = ndi.binary_dilation(self.region == a, iterations=touch_px)
            for b in range(a + 1, self.n_panels):
                if int((grown & (self.region == b)).sum()) >= need:
                    pairs.append((a, b))
        return pairs


def _canvas_polys(labels, size):
    W, H = size if size else (labels["canvas"]["width"], labels["canvas"]["height"])
    sx, sy = W / labels["image"]["width"], H / labels["image"]["height"]
    scale = lambda poly: [(p[0] * sx, p[1] * sy) for p in poly]
    polys = [scale(p["polygon"]) for p in labels["panels"]]
    blocked_polys = [scale(g["polygon"]) for key in ("glass", "off_limits") for g in labels[key]]
    return (W, H), polys, blocked_polys


def build_scene(labels, size=None):
    """labels must already be validated and default-filled (capture.labels.load_labels does both)."""
    (W, H), polys, blocked_polys = _canvas_polys(labels, size)
    sx = W / labels["image"]["width"]
    region, blocked, frame, impenetrable = build_masks((W, H), polys, blocked_polys)
    panels = []
    for k, (p, poly) in enumerate(zip(labels["panels"], polys)):
        panels.append(Panel(
            idx=k, name=p["name"], plane_id=p["plane_id"], angle_deg=p["angle_deg"], polygon=poly,
            normal=normal_from_angle(p["angle_deg"]),
            window_near=tuple(p["window_near"]),
            window_far=None if p["window_far"] is None else tuple(p["window_far"]),
            depth_x=None if p["depth_x"] is None else tuple(v * sx for v in p["depth_x"])))
    t = depth_coordinate(region, panels)
    lo, hi = build_windows(region, t, panels)
    relief = relief_field(region, frame, len(panels))
    return Scene(W, H, panels, region, blocked, frame, impenetrable, t, lo, hi, relief, labels)
