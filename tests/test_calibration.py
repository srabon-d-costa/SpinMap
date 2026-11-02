"""Simple deterministic test; execute: python -m unittest discover -s tests -v"""
import unittest
import cv2
import numpy as np
from spinmap.calibrate_table import estimate_homography


class CalibrationTests(unittest.TestCase):
    def test_known_rectangle_maps_to_topdown(self):
        src = np.array([[20, 25], [220, 25], [220, 125], [20, 125]], np.float32)
        H = estimate_homography(src, (610, 1096))
        mapped = cv2.perspectiveTransform(src[None, :, :], H)[0]
        expected = np.array([[0, 0], [609, 0], [609, 1095], [0, 1095]])
        np.testing.assert_allclose(mapped, expected, atol=0.01)

    def test_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            estimate_homography([(0, 0)] * 3)

    def test_rejects_degenerate_polygon(self):
        with self.assertRaises(ValueError):
            estimate_homography([(0, 0), (1, 0), (1, 1), (0, 1)])


if __name__ == '__main__':
    unittest.main()
