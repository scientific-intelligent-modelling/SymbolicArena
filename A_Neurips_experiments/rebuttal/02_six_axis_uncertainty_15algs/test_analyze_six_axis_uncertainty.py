import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

import matplotlib.pyplot as plt
import pandas as pd


MODULE_PATH = Path(__file__).with_name("analyze_six_axis_uncertainty.py")
SPEC = importlib.util.spec_from_file_location("six_axis_uncertainty", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_components() -> pd.DataFrame:
    rows = []
    for algorithm, value in (("dso", 0.8), ("pysr", 0.6)):
        for task_index in range(10):
            rows.append(
                {
                    "algorithm": algorithm,
                    "dataset": f"task_{task_index:02d}",
                    "cohort": "unsafe-input-label",
                    "clean_seeds": 3,
                    "noise_levels": "0.01|0.05",
                    "ID_Q_component": value,
                }
            )
    return pd.DataFrame(rows)


class ComponentValidationTest(unittest.TestCase):
    def test_validation_replaces_input_cohort_with_public_labels(self) -> None:
        validated = MODULE.validate_components(
            make_components(),
            ("ID_Q",),
            expected_algorithms=2,
            expected_datasets=10,
        )

        self.assertEqual(set(validated["cohort"]), {"Original"})


class MultipleComparisonTest(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        adjusted = MODULE.holm_adjust([0.01, 0.04, 0.03])

        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_clear_paired_difference_is_distinguishable(self) -> None:
        components = MODULE.validate_components(
            make_components(),
            ("ID_Q",),
            expected_algorithms=2,
            expected_datasets=10,
        )

        _, pairwise = MODULE.bootstrap_and_pairwise(
            components,
            ("ID_Q",),
            n_bootstrap=2000,
            n_permutations=4096,
            seed=7,
        )
        result = MODULE.oriented_pair(pairwise, "ID_Q", "dso", "pysr")

        self.assertAlmostEqual(result["difference_a_minus_b"], 20.0)
        self.assertGreater(result["ci_low"], 0.0)
        self.assertLess(result["holm_adjusted_p"], 0.05)
        self.assertTrue(result["statistically_distinguishable"])

    def test_oriented_pair_reverses_difference_and_interval(self) -> None:
        pairwise = pd.DataFrame(
            [
                {
                    "axis": "ID_Q",
                    "algorithm_a": "dso",
                    "algorithm_b": "pysr",
                    "difference_a_minus_b": 2.0,
                    "ci_low": 1.0,
                    "ci_high": 3.0,
                    "bootstrap_probability_a_gt_b": 0.9,
                    "paired_sign_flip_p": 0.01,
                    "holm_adjusted_p": 0.02,
                    "ci_excludes_zero": True,
                    "statistically_distinguishable": True,
                    "tasks": 50,
                }
            ]
        )

        result = MODULE.oriented_pair(pairwise, "ID_Q", "pysr", "dso")

        self.assertEqual(result["difference_a_minus_b"], -2.0)
        self.assertEqual(result["ci_low"], -3.0)
        self.assertEqual(result["ci_high"], -1.0)
        self.assertAlmostEqual(
            result["bootstrap_probability_a_gt_b"],
            0.1,
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


class FigureExportTest(unittest.TestCase):
    def test_pdf_export_is_reproducible_across_wall_clock_time(self) -> None:
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "figure.png"
                MODULE.save_figure_variants(figure, output)
                first_pdf = output.with_suffix(".pdf").read_bytes()
                time.sleep(1.1)
                MODULE.save_figure_variants(figure, output)
                second_pdf = output.with_suffix(".pdf").read_bytes()

                self.assertEqual(first_pdf, second_pdf)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
