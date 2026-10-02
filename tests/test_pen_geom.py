"""Shared edges in the pen tool: the browser geometry (run under node) and what the backend does with shapes that share edges.

    python -m unittest tests.test_pen_geom -v
"""
import os
import shutil
import subprocess
import unittest

import numpy as np

from src.capture import pen_tool
from src.capture.labels import fill_defaults, validate
from src.geometry.planes import build_scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def labels_with(polys):
    panels = [dict(id=i, name=f"p{i}", plane_id=i, angle_deg=0.0, polygon=[list(map(float, v)) for v in poly]) for i, poly in enumerate(polys)]
    return validate(fill_defaults({"schema": "standing-light/labels@1", "source": "projector", "image": {"file": None, "width": 400, "height": 300},
                                   "canvas": {"width": 400, "height": 300}, "grid": {"nx": 4, "ny": 3, "min_cover": 0.3}, "panels": panels,
                                   "glass": [], "off_limits": []}))


class BrowserGeometry(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_node_suite(self):
        r = subprocess.run(["node", os.path.join(ROOT, "tests", "pen_geom.test.js")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("pen_geom tests passed", r.stdout)

    def test_the_page_serves_the_geometry_module(self):
        self.assertIn("/pen_geom.js", pen_tool.STATIC)
        self.assertTrue(os.path.exists(os.path.join(pen_tool.HERE, pen_tool.STATIC["/pen_geom.js"][0])))
        with open(os.path.join(pen_tool.HERE, "pen_tool.html")) as f:
            self.assertIn('src="/pen_geom.js"', f.read())


class BackendAcceptsSharedEdges(unittest.TestCase):
    # A, with a T-junction: B and C stack along A's right edge (welded, as the pen tool leaves them)
    A = [(20, 20), (200, 20), (200, 110), (200, 200), (20, 200)]
    B = [(200, 20), (380, 40), (380, 110), (200, 110)]
    C = [(200, 110), (380, 110), (380, 260), (200, 200)]

    def test_touching_shapes_leave_no_gap_and_are_adjacent(self):
        scene = build_scene(labels_with([self.A, self.B, self.C]))
        self.assertEqual(scene.panel_adjacency(), [(0, 1), (0, 2), (1, 2)])
        for x in (199, 200, 201):                              # columns straddling the shared edge, away from the corners
            col = scene.region[60:190, x]
            self.assertTrue(np.all(col >= 0), f"gap in column {x}")
        self.assertTrue(np.all(scene.region[30:100, 200] >= 0))

    def test_budget_and_export_work_on_shared_edge_shapes(self):
        summary, scene, cells = pen_tool.summarise(labels_with([self.A, self.B, self.C]))
        self.assertEqual(summary["panels"], 3)
        self.assertGreater(summary["patches"], 3)


if __name__ == "__main__":
    unittest.main()
