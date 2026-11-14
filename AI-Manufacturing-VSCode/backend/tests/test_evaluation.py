import unittest

import numpy as np

from evaluate_temporal_holdout import binary_metrics, calibration_bins


class TemporalEvaluationTests(unittest.TestCase):
    def test_binary_metrics_return_fixed_confusion_matrix_labels(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.7, 0.99])

        result = binary_metrics(labels, probabilities, 0.75)

        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["predicted_alerts"], 2)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_calibration_bins_cover_every_sample(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

        bins = calibration_bins(labels, probabilities, bin_count=3)

        self.assertEqual(sum(item["samples"] for item in bins), len(labels))
        self.assertEqual(
            [item["mean_predicted_probability"] for item in bins],
            [0.15000000000000002, 0.5, 0.8500000000000001],
        )
