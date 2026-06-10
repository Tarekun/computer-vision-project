# v2/tests/test_evalpos.py
import unittest
from v2.evalpos import circle_iou, match_scene, scene_metrics

class TestEval(unittest.TestCase):
    def test_iou_identical(self):
        self.assertAlmostEqual(circle_iou((100,100,50),(100,100,50)), 1.0, places=3)
    def test_iou_disjoint(self):
        self.assertEqual(circle_iou((0,0,10),(100,100,10)), 0.0)
    def test_match_and_metrics(self):
        gt = [{"cx":100,"cy":100,"r":50,"label":"5cent","sure":True},
              {"cx":300,"cy":300,"r":60,"label":"unknown","sure":True}]
        pred = [{"cx":102,"cy":101,"r":51,"pred":"10cent"},
                {"cx":600,"cy":600,"r":40,"pred":"1euro"}]
        m = match_scene(gt, pred)                      # IoU>0.5
        self.assertEqual(m["pairs"], [(0,0)])
        self.assertEqual(m["miss"], [1])
        self.assertEqual(m["fp"], [1])
        s = scene_metrics(gt, pred, m)
        self.assertEqual(s["n_eval"], 1)               # unknown fuori denominatore
        self.assertEqual(s["n_correct"], 0)
        self.assertEqual(s["stage"], ["famiglia"])     # 5cent(copper) vs 10cent(gold)

if __name__ == "__main__":
    unittest.main()
