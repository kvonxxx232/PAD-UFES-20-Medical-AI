"""Synthetic lesion-level tests for the prespecified V1.1 primary statistics."""

import json
import math
import unittest

import numpy as np

from pad_v11_primary import lesion_losses, primary_report


def synthetic_oof():
    rows = []
    for fold in range(1, 6):
        for patient_index in (0, 1):
            for lesion_index in range(fold if patient_index == 0 else 1):
                label = (lesion_index + patient_index) % 2
                probability = 0.12 + 0.09 * fold + 0.08 * patient_index + 0.015 * lesion_index
                rows.append({
                    "patient_id": f"example-{fold}-{patient_index}",
                    "lesion_id": str(lesion_index),
                    "outer_fold": fold,
                    "n_images": 1 + patient_index + lesion_index,
                    "label": label,
                    "p_A": probability,
                    "p_B": 0.85 - probability,
                })
    return rows


def one_loss(row, model):
    prob = row["p_" + model]
    return -math.log(prob if row["label"] else 1 - prob)


class PrimaryMetricTests(unittest.TestCase):
    def test_lesion_loss_clips_boundaries_and_uses_natural_log(self):
        losses = lesion_losses([1, 0, 1, 0], [0, 1, 0.3, 0.3])
        expected = [-math.log(1e-7), -math.log(1 - (1 - 1e-7)), -math.log(0.3), -math.log(0.7)]
        np.testing.assert_allclose(losses, expected, rtol=0, atol=1e-14)

    def test_invalid_labels_and_probabilities_refused(self):
        for labels, probs in [
            ([], []), ([0, 1], [0.5]), ([0], [[0.5]]), ([2], [0.5]),
            ([1], [np.nan]), ([1], [np.inf]), ([1], [-0.01]), ([1], [1.01]),
        ]:
            with self.subTest(labels=labels, probs=probs), self.assertRaises(ValueError):
                lesion_losses(labels, probs)

    def test_pooled_lesion_means_fold_directions_and_row_order(self):
        rows = synthetic_oof()
        result = primary_report(rows)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["n_keys"], 20)
        self.assertEqual(result["n_patients"], 10)
        self.assertEqual(result["n_images"], sum(row["n_images"] for row in rows))
        for model in ("A", "B"):
            self.assertAlmostEqual(result["log_loss_" + model], np.mean([one_loss(r, model) for r in rows]))
        self.assertEqual(result["delta_log_loss"], result["log_loss_B"] - result["log_loss_A"])
        fold_deltas = []
        for info in result["folds"]:
            part = [r for r in rows if r["outer_fold"] == info["outer_fold"]]
            expected = np.mean([one_loss(r, "B") - one_loss(r, "A") for r in part])
            self.assertEqual(info["n_keys"], len(part))
            self.assertAlmostEqual(info["delta_log_loss"], expected)
            fold_deltas.append(expected)
        self.assertNotAlmostEqual(result["delta_log_loss"], np.mean(fold_deltas))
        self.assertEqual(result["n_folds_favouring_B"], sum(v < 0 for v in fold_deltas))
        self.assertEqual(result, primary_report(rows[::-1]))

    def test_paired_cluster_interval_matches_explicit_lesion_expansion(self):
        rows = synthetic_oof()
        patients = sorted({r["patient_id"] for r in rows})
        groups = {p: [r for r in rows if r["patient_id"] == p] for p in patients}
        rng = np.random.Generator(np.random.PCG64(20261125))
        draws = []
        for _ in range(2000):
            selected = rng.choice(patients, size=len(patients), replace=True)
            repeated = [r for p in selected for r in groups[p]]
            draws.append(sum(one_loss(r, "B") - one_loss(r, "A") for r in repeated) / len(repeated))
        expected = np.percentile(draws, [2.5, 97.5], method="linear")
        report = primary_report(rows)["bootstrap"]
        np.testing.assert_allclose(report["ci_95"], expected, rtol=0, atol=1e-14)
        self.assertAlmostEqual(report["width"], expected[1] - expected[0])
        self.assertEqual((report["resamples"], report["seed"]), (2000, 20261125))

    def test_ci_interpretations_use_unrounded_bounds_including_touching_zero(self):
        rows = synthetic_oof()
        for row in rows:
            row["p_A"] = row["p_B"] = 0.5
        result = primary_report(rows)
        self.assertEqual(result["bootstrap"]["ci_95"], [0, 0])
        self.assertEqual(result["n_folds_favouring_B"], 0)
        self.assertIn("inconclusive", result["bootstrap"]["interpretation"])
        rows[0]["p_B"] = 0.4
        result = primary_report(rows)
        self.assertLess(result["bootstrap"]["ci_95"][0], 0)
        self.assertEqual(result["bootstrap"]["ci_95"][1], 0)
        self.assertIn("inconclusive", result["bootstrap"]["interpretation"])
        for row in rows:
            row["p_B"] = 0.5 + (2 * row["label"] - 1) * 1e-9
        self.assertIn("lower", primary_report(rows)["bootstrap"]["interpretation"])
        for row in rows:
            row["p_A"], row["p_B"] = row["p_B"], row["p_A"]
        self.assertIn("higher", primary_report(rows)["bootstrap"]["interpretation"])

    def test_incomplete_ambiguous_or_invalid_oof_refused(self):
        rows = synthetic_oof()
        bad_inputs = [
            rows + [rows[0]],
            rows + [dict(rows[0], lesion_id="another", outer_fold=2)],
            [r for r in rows if r["outer_fold"] != 5],
            [dict(r, label=0) if r["outer_fold"] == 2 else r for r in rows],
        ]
        for change in [dict(patient_id=""), dict(lesion_id=None), dict(outer_fold=0),
                       dict(n_images=0), dict(n_images=1.5), dict(p_A=np.nan)]:
            bad_inputs.append([dict(rows[0], **change), *rows[1:]])
        for index, bad in enumerate(bad_inputs):
            with self.subTest(index=index), self.assertRaises(ValueError):
                primary_report(bad)


if __name__ == "__main__":
    unittest.main()
