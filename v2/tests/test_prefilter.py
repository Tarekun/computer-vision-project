import unittest
import numpy as np
from v2.prefilter import VARIANTS, apply

class TestPrefilter(unittest.TestCase):
    def test_variants_run_and_preserve_shape(self):
        img = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
        for v in VARIANTS:
            out = apply(img, v)
            self.assertEqual(out.shape, img.shape, v)
            self.assertEqual(out.dtype, np.uint8, v)

if __name__ == "__main__":
    unittest.main()
