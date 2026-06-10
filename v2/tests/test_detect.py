import unittest
import numpy as np, cv2
from v2.detect import candidates
from v2.validate import validate

def synth():
    img = np.full((400, 600, 3), 30, np.uint8)
    cv2.circle(img, (150, 200), 70, (180, 160, 140), -1)
    cv2.circle(img, (430, 180), 55, (190, 170, 150), -1)
    return img

class TestDetect(unittest.TestCase):
    def test_candidates_find_both(self):
        cs = candidates(synth())
        self.assertGreaterEqual(len(cs), 2)
        self.assertTrue(any(abs(c[0]-150) < 8 and abs(c[1]-200) < 8 for c in cs))
    def test_validator_accepts_real_rejects_ghost(self):
        img = synth()
        ok = validate(img, 150, 200, 70)
        self.assertTrue(ok["accept"])
        self.assertLess(abs(ok["r"] - 70), 5)
        ghost = validate(img, 300, 320, 60)     # nessun bordo coerente lì
        self.assertFalse(ghost["accept"])

if __name__ == "__main__":
    unittest.main()
