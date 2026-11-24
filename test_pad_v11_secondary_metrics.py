"""Synthetic regression tests for paired resampling, biopsy membership and calibration."""

import json
import math
import unittest

import numpy as np

from pad_v11_secondary_metrics import reliability_bins, s1_summary, s3_summary, s6_summary


def lesion(patient, lesion_number, fold, label, a, b, n_images=1):
    return {"patient_id": patient, "lesion_id": str(lesion_number), "outer_fold": fold,
            "label": label, "p_A": a, "p_B": b, "n_images": n_images}


def explicit_auroc(rows, column):
    positives = [r[column] for r in rows if r["label"] == 1]
    negatives = [r[column] for r in rows if r["label"] == 0]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


class SecondaryMetricTests(unittest.TestCase):
    def test_paired_patient_draws_match_explicit_repetition_with_ties_and_undefined_auroc(self):
        rows = [
            lesion("d", 1, 1, 1, 0.2, 0.6), lesion("d", 2, 1, 1, 0.5, 0.2),
            lesion("d", 3, 1, 1, 0.4, 0.8), lesion("a", 2, 2, 0, 0.2, 0.2),
            lesion("c", 1, 3, 0, 0.9, 0.3), lesion("b", 1, 4, 1, 0.4, 0.8),
            lesion("b", 2, 4, 0, 0.2, 0.7),
        ]
        patients = sorted({r["patient_id"] for r in rows})
        rng = np.random.Generator(np.random.PCG64(20261125))
        auc_deltas, brier_deltas = [], []
        for _ in range(2000):
            draw = rng.choice(patients, size=len(patients), replace=True)
            expanded = [r for patient in draw for r in rows if r["patient_id"] == patient]
            a, b = (explicit_auroc(expanded, c) for c in ("p_A", "p_B"))
            if a is not None:
                auc_deltas.append(b - a)
            brier_deltas.append(np.mean([(r["p_B"] - r["label"])**2 - (r["p_A"] - r["label"])**2
                                         for r in expanded]))
        actual = s1_summary(rows)
        self.assertEqual((actual["n_keys"], actual["n_patients"]), (7, 4))
        self.assertEqual((actual["seed"], actual["resamples"]), (20261125, 2000))
        self.assertTrue(0 < len(auc_deltas) < 2000)
        for metric, expected in (("auroc", auc_deltas), ("brier", brier_deltas)):
            np.testing.assert_allclose(actual[metric]["ci_95"], np.percentile(expected, [2.5, 97.5]),
                                       rtol=0, atol=1e-14)
            self.assertEqual(actual[metric]["defined_draws"], len(expected))
            self.assertEqual(actual[metric]["undefined_draws"], 2000 - len(expected))
        for model in ("A", "B"):
            self.assertAlmostEqual(actual["auroc"][model], explicit_auroc(rows, "p_" + model))

    def test_entire_single_class_sample_produces_no_auroc_ci(self):
        rows = [lesion("a", 1, 1, 0, 0.2, 0.5), lesion("b", 1, 2, 0, 0.3, 0.7)]
        s1 = s1_summary(rows)
        self.assertEqual(s1["auroc"], {"A": None, "B": None, "delta": None, "ci_95": None,
                                      "defined_draws": 0, "undefined_draws": 2000})
        self.assertEqual(s1["brier"]["defined_draws"], 2000)
        json.dumps(s1, allow_nan=False)

    def test_duplicate_oof_key_cannot_substitute_for_missing_biopsy_key(self):
        record = lesion("a", 1, 1, 1, 0.4, 0.6)
        duplicate = [record, dict(record)]
        for calculate, args in ((s1_summary, (duplicate,)), (s6_summary, (duplicate,)),
                                (s3_summary, (duplicate, {("a", "1"), ("b", "1")}))):
            with self.subTest(analysis=calculate.__name__), self.assertRaisesRegex(ValueError, "Repeated lesion key"):
                calculate(*args)

    def test_s3_uses_exact_biopsy_keys_original_folds_and_equal_key_weights(self):
        rows = [lesion("a", 1, 1, 1, 0.5, 0.8, 4), lesion("a", 2, 1, 0, 0.5, 0.3),
                lesion("b", 1, 2, 1, 0.25, 0.9, 2), lesion("b", 2, 2, 0, 0.8, 0.9)]
        result = s3_summary(rows, {("a", "1"), ("a", "2"), ("b", "1")})
        group = result["pooled"]
        self.assertEqual((group["n_keys"], group["n_patients"], group["positive"], group["negative"],
                          group["n_images"]), (3, 2, 2, 1, 7))
        self.assertAlmostEqual(group["log_loss"]["A"], -(2 * math.log(0.5) + math.log(0.25)) / 3)
        self.assertEqual(group["auroc"], {"A": 0.25, "B": 1.0, "delta": 0.75})
        self.assertEqual([f["n_keys"] for f in result["folds"]], [2, 1])
        self.assertIsNone(result["folds"][1]["auroc"]["delta"])
        self.assertIsNotNone(result["folds"][1]["brier"]["delta"])
        with self.assertRaisesRegex(ValueError, "membership"):
            s3_summary(rows, {("absent", "1")})

    def test_reliability_bins_decimal_boundaries_empty_bins_and_endpoint_one(self):
        bins = reliability_bins([0, 1, 1, 0, 1, 0, 1],
                                [0, np.nextafter(0.3, 0), 0.3, 0.6, 0.7, 0.9, 1.0])
        self.assertEqual([r["count"] for r in bins], [1, 0, 1, 1, 0, 0, 1, 1, 0, 2])
        self.assertEqual(bins[9]["observed"], 0.5)
        self.assertEqual(bins[9]["mean_predicted"], 0.95)
        self.assertIsNone(bins[1]["mean_predicted"])
        self.assertIsNone(bins[1]["observed"])
        self.assertEqual(bins[1]["positive"], 0)

    def test_calibration_joint_unpenalised_coefficients_against_two_exact_group_rates(self):
        labels = [0] * 7 + [1] + [0] + [1] * 3
        rows = [lesion("p", i, 1, y, 0.2 if i < 8 else 0.8, 0.8 if i < 8 else 0.2)
                for i, y in enumerate(labels)]
        result = s6_summary(rows)
        alpha = (math.log(1/7) + math.log(3)) / 2
        slope = (math.log(3) - math.log(1/7)) / (2 * math.log(4))
        for model, beta in (("A", slope), ("B", -slope)):
            self.assertAlmostEqual(result[model]["intercept"], alpha, places=10)
            self.assertAlmostEqual(result[model]["slope"], beta, places=10)
            self.assertEqual(sum(r["count"] for r in result[model]["bins"]), 12)
        self.assertAlmostEqual(result["A"]["mean_predicted"], (0.2 * 8 + 0.8 * 4) / 12)
        json.dumps(result, allow_nan=False)

    def test_calibration_rejects_infinite_logits_and_unidentified_coefficients(self):
        for probs in ((0.0, 1.0), (0.4, 0.4), (0.2, 0.8)):
            rows = [lesion("p", i, 1, i, p, p) for i, p in enumerate(probs)]
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                s6_summary(rows)


if __name__ == "__main__":
    unittest.main()
