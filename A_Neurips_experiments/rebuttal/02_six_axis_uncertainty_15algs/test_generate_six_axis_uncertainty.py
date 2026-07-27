import importlib.util
from pathlib import Path
import unittest

import matplotlib.pyplot as plt


MODULE_PATH = Path(__file__).with_name("generate_six_axis_uncertainty.py")
SPEC = importlib.util.spec_from_file_location("six_axis_uncertainty", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SetF1Test(unittest.TestCase):
    def test_disjoint_nonempty_sets_have_zero_f1(self) -> None:
        self.assertEqual(MODULE.set_f1({"x1"}, {"x2"}), 0.0)

    def test_two_empty_sets_are_an_exact_match(self) -> None:
        self.assertEqual(MODULE.set_f1(set(), set()), 1.0)

    def test_partial_overlap_uses_harmonic_mean(self) -> None:
        self.assertAlmostEqual(
            MODULE.set_f1({"x1", "x2"}, {"x2", "x3"}),
            0.5,
        )


class SharedYAxisTest(unittest.TestCase):
    def test_shared_axes_are_inverted_exactly_once(self) -> None:
        figure, axes = plt.subplots(2, 3, sharey=True)
        try:
            MODULE.invert_shared_y_axis(axes)
            self.assertTrue(
                all(axis.get_ylim()[0] > axis.get_ylim()[1] for axis in axes.ravel())
            )
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
