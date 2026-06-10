# v2/tests/test_rectify.py
import unittest
import numpy as np, cv2
from v2.rectify import fit_tilt, rectify_disc

class TestRectify(unittest.TestCase):
    def _ellipse_pts(self, cx, cy, a, b, angle_deg):
        th = np.linspace(0, 2*np.pi, 72, endpoint=False)
        x, y = a*np.cos(th), b*np.sin(th)
        ar = np.deg2rad(angle_deg)
        xr = cx + x*np.cos(ar) - y*np.sin(ar)
        yr = cy + x*np.sin(ar) + y*np.cos(ar)
        return list(zip(xr.tolist(), yr.tolist()))
    def test_fit_detects_tilt(self):
        pts = self._ellipse_pts(200, 200, 100, 80, 30)
        t = fit_tilt(pts)
        self.assertIsNotNone(t)
        self.assertLess(abs(t["ratio"] - 0.8), 0.05)
        self.assertLess(min(abs(t["angle"] - 30), abs(t["angle"] - 210)) % 180, 8)
    def test_fit_circle_returns_none(self):
        pts = self._ellipse_pts(200, 200, 100, 99, 0)   # quasi cerchio
        self.assertIsNone(fit_tilt(pts))
    def test_rectify_makes_circle(self):
        img = np.zeros((400, 400, 3), np.uint8)
        cv2.ellipse(img, (200, 200), (100, 80), 30, 0, 360, (255, 255, 255), -1)
        pts = self._ellipse_pts(200, 200, 100, 80, 30)
        disc, new_r = rectify_disc(img, 200, 200, fit_tilt(pts))
        self.assertAlmostEqual(new_r, 100, delta=3)
        g = cv2.cvtColor(disc, cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero(g > 128)
        h = ys.max() - ys.min(); w = xs.max() - xs.min()
        self.assertLess(abs(h - w) / max(h, w), 0.05)    # tornato tondo

    def test_scene_tilt_agreeing(self):
        from v2.rectify import scene_tilt
        t = scene_tilt([{"angle":80.0,"ratio":0.8},{"angle":100.0,"ratio":0.84}])
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t["angle"], 90.0, delta=2)
    def test_scene_tilt_wraparound(self):
        from v2.rectify import scene_tilt
        t = scene_tilt([{"angle":5.0,"ratio":0.8},{"angle":175.0,"ratio":0.8}])
        self.assertIsNotNone(t)
        self.assertLess(min(t["angle"], 180 - t["angle"]), 6)   # ~0°/180°, NOT 90°
    def test_scene_tilt_disagreeing(self):
        from v2.rectify import scene_tilt
        self.assertIsNone(scene_tilt([{"angle":30.0,"ratio":0.8},{"angle":150.0,"ratio":0.8}]))

if __name__ == "__main__":
    unittest.main()
