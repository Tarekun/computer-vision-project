import unittest, os
from v2.paths import TARGET_DIR, REF_DIR, target_images, image_num, is_dev

class TestPaths(unittest.TestCase):
    def test_dirs_exist(self):
        self.assertTrue(os.path.isdir(TARGET_DIR))
        self.assertTrue(os.path.isdir(REF_DIR))
    def test_target_list(self):
        imgs = target_images()
        self.assertEqual(len(imgs), 142)
        self.assertTrue(all(i.startswith("image_") for i in imgs))
    def test_image_num_and_split(self):
        self.assertEqual(image_num("image_85.jpg"), 85)
        self.assertFalse(is_dev("image_85.jpg"))   # dispari = sigillata
        self.assertTrue(is_dev("image_4.jpg"))     # pari = sviluppo

if __name__ == "__main__":
    unittest.main()
