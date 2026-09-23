from __future__ import annotations

import math
import unittest

from benchmark import knee_flexion_deg


class KneeGeometryTests(unittest.TestCase):
    def test_straight_leg_is_zero_flexion(self) -> None:
        self.assertAlmostEqual(
            knee_flexion_deg((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
            0.0,
        )

    def test_right_angle_is_ninety_flexion(self) -> None:
        self.assertAlmostEqual(
            knee_flexion_deg((0.0, 1.0), (0.0, 0.0), (1.0, 0.0)),
            90.0,
        )

    def test_angle_is_scale_and_translation_invariant(self) -> None:
        baseline = knee_flexion_deg((0.0, 1.0), (0.0, 0.0), (1.0, 0.0))
        transformed = knee_flexion_deg(
            (10.0, 23.0),
            (10.0, 20.0),
            (13.0, 20.0),
        )
        self.assertTrue(math.isclose(baseline, transformed, abs_tol=1e-9))

    def test_degenerate_segment_is_unavailable(self) -> None:
        self.assertIsNone(
            knee_flexion_deg((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
        )


if __name__ == "__main__":
    unittest.main()
