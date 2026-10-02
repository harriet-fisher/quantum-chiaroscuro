"""Homography, calibration, warp (exact black), patterns, dot registration and the photographed check.

    python -m unittest tests.test_projector -v
"""
import os
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi

from src.capture.labels import bay_window_labels, fill_defaults, validate
from src.geometry.planes import build_scene
from src.projector import calibrate as cal
from src.projector import patterns, verify
from src.projector.warp import GlassLeak, Warper, warp_image

LABELS = validate(fill_defaults(bay_window_labels()))
SCENE = build_scene(LABELS)


class Homography(unittest.TestCase):
    def test_recovers_a_known_homography_and_survives_noise(self):
        rng = np.random.default_rng(0)
        H = np.array([[1.2, 0.1, 30], [-0.05, 0.9, 12], [2e-4, -1e-4, 1]])
        src = rng.uniform(0, 1000, (12, 2))
        dst = cal.apply_homography(H, src)
        np.testing.assert_allclose(cal.fit_homography(src, dst), H / H[2, 2], rtol=1e-8, atol=1e-8)
        noisy = cal.fit_homography(src, dst + rng.normal(0, 0.3, dst.shape))
        self.assertLess(cal.reprojection_rms(noisy, src, dst), 1.0)

    def test_bad_input_is_rejected(self):
        with self.assertRaises(ValueError):
            cal.fit_homography([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            cal.fit_homography([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 2], [2, 4], [3, 6]])

    def test_rescale_follows_the_pixel_centre_convention(self):
        H = cal.fit_inside((1200, 700), (1920, 1080))
        p = np.array([100.0, 50.0])
        a = cal.apply_homography(H, p)
        b = cal.apply_homography(cal.rescale_to(H, (1920, 1080), (1280, 720)), p)
        np.testing.assert_allclose(b, (a + 0.5) * (1280 / 1920) - 0.5)

    def test_fit_inside_centres_and_letterboxes(self):
        H = cal.fit_inside((1200, 700), (1920, 1080))
        c = cal.apply_homography(H, [(0, 0), (1199, 699)])
        self.assertAlmostEqual((c[0, 0] + c[1, 0]) / 2, 959.5, delta=1.0)
        self.assertAlmostEqual((c[0, 1] + c[1, 1]) / 2, 539.5, delta=1.0)

    def test_crossed_corner_handles_are_refused(self):
        lc = [(0, 0), (100, 0), (100, 100), (0, 100)]
        with self.assertRaisesRegex(ValueError, "convex"):
            cal.corner_calibration((100, 100), (200, 200), lc, [(0, 0), (100, 100), (100, 0), (0, 100)])

    def test_save_load_and_for_size(self):
        c = cal.corner_calibration((1200, 700), (1920, 1080), cal.default_landmarks(LABELS),
                                   [(p[0] * 1.6 + 5, p[1] * 1.6 - 4) for p in cal.default_landmarks(LABELS)])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            cal.save_calibration(c, path)
            back = cal.load_calibration(path)
        np.testing.assert_allclose(back.H, c.H, atol=1e-9)
        small = back.for_size((1280, 720))
        np.testing.assert_allclose(cal.apply_homography(small.H, [10, 10]), (cal.apply_homography(c.H, [10, 10]) + 0.5) * (2 / 3) - 0.5, atol=1e-9)

    def test_default_landmarks_are_real_vertices(self):
        verts = {tuple(p) for panel in LABELS["panels"] for p in panel["polygon"]}
        for p in cal.default_landmarks(LABELS):
            self.assertIn(tuple(p), verts)

    def test_photo_labels_need_a_camera_calibration(self):
        photo = {**LABELS, "source": "photo"}
        with self.assertRaisesRegex(ValueError, "photo"):
            cal.labels_calibration(photo, (1920, 1080))


class WarpIsExactlyBlack(unittest.TestCase):
    def check(self, H, size, margin):
        w = Warper(H, SCENE.frame, size, margin)
        img = np.random.default_rng(1).random((SCENE.H, SCENE.W))
        out = w(img)
        self.assertTrue(np.all(out[w.must_be_black] == 0))
        self.assertGreater(out.max(), 100)
        self.assertGreater(w.n_lit, 0.1 * out.size)
        w.check(out)
        return w, out

    def test_scale_calibration(self):
        self.check(cal.fit_inside((SCENE.W, SCENE.H), (1920, 1080)), (1920, 1080), 2)

    def test_strong_keystone(self):
        src = [(0, 0), (SCENE.W, 0), (SCENE.W, SCENE.H), (0, SCENE.H)]
        H = cal.fit_homography(src, [(120, 60), (1750, 10), (1900, 1050), (30, 980)])
        self.check(H, (1920, 1080), 2)

    def test_lit_region_never_exceeds_the_drawn_panels_and_keeps_its_margin(self):
        H = cal.fit_inside((SCENE.W, SCENE.H), (1920, 1080))
        w = Warper(H, SCENE.frame, (1920, 1080), 3)
        drawn = warp_image(SCENE.frame.astype(float), H, (1920, 1080)) > 0.5
        self.assertFalse((w.lit & ~drawn).any())
        unmargined = Warper(H, SCENE.frame, (1920, 1080), 0).lit
        d = ndi.distance_transform_edt(unmargined)             # distance of each lit pixel to the nearest pixel that must be black
        self.assertGreaterEqual(d[w.lit].min(), 3.0)

    def test_check_catches_a_leak(self):
        w, out = self.check(cal.fit_inside((SCENE.W, SCENE.H), (1280, 720)), (1280, 720), 2)
        bad = out.copy()
        bad[w.must_be_black.nonzero()[0][0], w.must_be_black.nonzero()[1][0]] = 5
        with self.assertRaises(GlassLeak):
            w.check(bad)

    def test_gain_is_clamped_and_white_pattern_obeys_the_mask(self):
        H = cal.fit_inside((SCENE.W, SCENE.H), (1280, 720))
        w = Warper(H, SCENE.frame, (1280, 720), 2)
        self.assertEqual(w(np.ones((SCENE.H, SCENE.W)), gain=5).max(), 255)
        self.assertTrue(np.all(w.white()[w.must_be_black] == 0))


class Dots(unittest.TestCase):
    def photo(self, Hpc, size=(1280, 720), cam=(1600, 1000), seed=0):
        on, off = (verify.simulate_photo(f, Hpc, cam, rng=np.random.default_rng(seed + i))[0] for i, f in enumerate((patterns.dots(size), patterns.black(size))))
        return on, off

    def test_registration_is_subpixel_for_rotated_cameras(self):
        size, cam = (1280, 720), (1600, 1600)
        for seed, extra in [(1, 0), (2, 0), (3, 180), (4, 90), (5, 270)]:
            Hpc = verify.random_camera(size, cam, rng=np.random.default_rng(seed))
            th, c = np.deg2rad(extra), np.array(cam) / 2
            R = np.array([[np.cos(th), -np.sin(th), c[0] - np.cos(th) * c[0] + np.sin(th) * c[1]],
                          [np.sin(th), np.cos(th), c[1] - np.sin(th) * c[0] - np.cos(th) * c[1]], [0, 0, 1]])
            Hpc = R @ Hpc
            on, off = self.photo(Hpc, size, cam, seed=seed)
            H, rms, _ = cal.dots_homography(on, off, size)
            probe = np.array([[200, 150], [1000, 600], [640, 360]], float)
            err = np.hypot(*(cal.apply_homography(H, cal.apply_homography(Hpc, probe)) - probe).T)
            self.assertLess(err.max(), 1.5, f"camera rotated by {extra} deg")
            self.assertLess(rms, 1.5)

    def test_a_missing_dot_is_a_clear_error(self):
        size = (1280, 720)
        Hpc = verify.random_camera(size, rng=np.random.default_rng(1))
        dots = patterns.dots(size)
        x, y = cal.dot_positions(size)[5]
        dots[int(y) - 40:int(y) + 40, int(x) - 40:int(x) + 40] = 0
        on = verify.simulate_photo(dots, Hpc, (1600, 1000), rng=np.random.default_rng(0))[0]
        off = verify.simulate_photo(patterns.black(size), Hpc, (1600, 1000), rng=np.random.default_rng(1))[0]
        with self.assertRaisesRegex(cal.DotError, "found 8"):
            cal.dots_homography(on, off, size)


class PhotoRoute(unittest.TestCase):
    def test_photo_traced_labels_land_where_the_real_glass_is(self):
        """Synthetic camera photos (saved as JPEG) -> dots CLI -> calibration -> the traced panels and glass map back onto the projector
        grid where the reference geometry says they are."""
        from PIL import Image
        from src.capture.labels import load_labels, save_labels
        proj, cam = (1280, 720), (1600, 1000)
        Hpc = verify.random_camera(proj, cam, rng=np.random.default_rng(8))
        with tempfile.TemporaryDirectory() as d:
            shot = lambda f, k: verify.simulate_photo(f, Hpc, cam, rng=np.random.default_rng(k))[1]
            for name, f, k in (("dots", patterns.dots(proj), 1), ("black", patterns.black(proj), 2)):
                Image.fromarray((shot(f, k) * 255).astype(np.uint8)).save(os.path.join(d, name + ".jpg"), quality=95)
            ref_H = cal.fit_inside((1200, 700), proj)
            truth = verify.labels_through(LABELS, Hpc @ ref_H, cam)                 # what a person would trace on black.jpg
            panels = []
            for p, t in zip(LABELS["panels"], truth["panels"]):
                q = {k: v for k, v in p.items() if k not in ("window_near", "window_far", "depth_x")}
                panels.append(dict(q, polygon=t["polygon"]))
            traced = dict(schema=LABELS["schema"], source="photo", image=dict(file="black.jpg", width=cam[0], height=cam[1]),
                          canvas=dict(width=1200, height=750), grid=LABELS["grid"], panels=panels, glass=truth["glass"], off_limits=[])
            save_labels(traced, os.path.join(d, "labels.json"))
            cal.main(["dots", "--on", os.path.join(d, "dots.jpg"), "--off", os.path.join(d, "black.jpg"), "--labels", os.path.join(d, "labels.json"),
                      "--projector", "1280x720", "--out", os.path.join(d, "calibration.json"), "--margin", "0"])
            c = cal.load_calibration(os.path.join(d, "calibration.json"))
            scene = build_scene(load_labels(os.path.join(d, "labels.json")))
        got = Warper(c.H, scene.frame, proj, 0).lit
        want = Warper(ref_H, SCENE.frame, proj, 0).lit
        iou = (got & want).sum() / (got | want).sum()
        self.assertGreater(iou, 0.985)
        self.assertLess(c.rms_px, 1.0)
        self.assertEqual(c.method, "dots")
        with self.assertRaises(ValueError):                                          # without a camera calibration photo labels cannot be projected
            cal.labels_calibration({**LABELS, "source": "photo"}, proj)


class Patterns(unittest.TestCase):
    def test_scene_patterns_obey_the_glass_rule_and_calibration_patterns_are_drawn(self):
        size = (1280, 720)
        H = cal.fit_inside((SCENE.W, SCENE.H), size)
        w = Warper(H, SCENE.frame, size, 2)
        kw = dict(size=size, warper=w, H=H, panel_polys=[p.polygon for p in SCENE.panels],
                  glass_polys=[g["polygon"] for g in LABELS["glass"]], landmarks_canvas=cal.default_landmarks(LABELS))
        for name in patterns.ALL:
            img = patterns.render(name, **kw)
            self.assertEqual((img.dtype, img.shape), (np.uint8, (720, 1280)), name)
        self.assertTrue(np.all(patterns.render("white", **kw)[w.must_be_black] == 0))
        self.assertEqual(patterns.render("black", **kw).max(), 0)
        for name in ("outline", "align", "dots", "grid", "edge"):
            self.assertGreater(patterns.render(name, **kw).max(), 0, name)
        with self.assertRaises(KeyError):
            patterns.render("nope", **kw)


class PhotographedCheck(unittest.TestCase):
    def test_selftest_aligned_passes_creep_and_reflection_fail_and_optical_is_blind_to_creep(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(verify.selftest(d))
            self.assertTrue(os.path.exists(os.path.join(d, "aligned", "glass_check_physical.png")))

    def test_without_a_truth_trace_the_verdict_is_not_checked(self):
        proj, cam = (640, 360), (800, 500)
        c = cal.labels_calibration(LABELS, proj)
        w = Warper(c.H, SCENE.frame, proj, 2)
        Hpc = verify.random_camera(proj, cam, rng=np.random.default_rng(3))
        shot = lambda f, s: verify.simulate_photo(f, Hpc, cam, rng=np.random.default_rng(s))
        rep = verify.run(shot(patterns.black(proj), 1), shot(patterns.white(w), 2), loader=None)
        self.assertEqual(rep["verdict"], "NOT CHECKED")

    def test_overexposed_or_tiny_glass_is_inconclusive_not_pass(self):
        proj, cam = (640, 360), (800, 500)
        c = cal.labels_calibration(LABELS, proj)
        w = Warper(c.H, SCENE.frame, proj, 2)
        Hpc = verify.random_camera(proj, cam, rng=np.random.default_rng(3))
        truth = verify.labels_through(LABELS, Hpc @ c.H, cam)
        shot = lambda f, s, **k: verify.simulate_photo(f, Hpc, cam, rng=np.random.default_rng(s), **k)
        rep = verify.run(shot(patterns.black(proj), 1), shot(patterns.white(w), 2, gain=6.0), truth=truth, loader=None)
        self.assertEqual(rep["verdict"], "INCONCLUSIVE")
        self.assertIn("overexposed", " ".join(rep["reasons"]))


if __name__ == "__main__":
    unittest.main()
