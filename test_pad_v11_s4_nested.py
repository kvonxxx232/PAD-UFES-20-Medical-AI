"""Synthetic checks of clinical scaling on the exact complete-case primary splits."""

from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import pad_v11_nested as primary
import pad_v11_s4_nested as s4
from test_pad_v11_nested import embeddings, synthetic_photos


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.photos = synthetic_photos()
        self.images = embeddings(self.photos)
        self.plan = primary.plan_splits(self.photos)

    def test_factor_one_reproduces_entire_primary_fit(self):
        basic, primary_grid, primary_fits = primary.run_nested(self.photos, self.images)
        with patch.object(primary, "LogisticRegression", wraps=primary.LogisticRegression) as lr:
            actual, grid, fits = s4.run_factor(self.photos, self.images, self.plan, 1)
        self.assertEqual(actual, basic)
        self.assertEqual(len(lr.call_args_list), 260)
        self.assertTrue(all(call.kwargs["fit_intercept"] is True for call in lr.call_args_list))
        for log, baseline in ((grid, primary_grid), (fits, primary_fits)):
            self.assertTrue(all(row["factor"] == 1 for row in log))
            self.assertEqual([{k: v for k, v in row.items() if k != "factor"} for row in log], baseline)

    def test_scaling_of_train_and_held_columns_and_reselection_for_each_factor(self):
        partitions = {}
        for split in self.plan:
            partitions[split.fold, None] = split.outer
            partitions.update({(split.fold, i): part for i, part in enumerate(split.inner, 1)})
        choices = {1: {"A": .001, "B": 1.}, 3: {"A": .01, "B": .1}, 10: {"A": 10., "B": .01}}
        originals = self.images.copy()
        for factor in s4.FACTORS:
            calls = []

            def inspect(x, held_x, y, weights, c, context, diagnostics):
                model, outer, inner = (context[k] for k in ("model", "outer_fold", "inner_fold"))
                self.assertEqual(factor, context["factor"])
                part = partitions[outer, inner]
                fit = [self.photos[i] for i in part.train]
                sites = sorted({p.site for p in fit})
                ages = np.array([p.age for p in fit])
                center, scale = ages.mean(), ages.std(ddof=0) or 1.
                counts = Counter(p.key for p in fit)
                np.testing.assert_array_equal(y, [p.label for p in fit])
                np.testing.assert_array_equal(weights, [1. / counts[p.key] for p in fit])
                for matrix, indices in ((x, part.train), (held_x, part.held_out)):
                    expected = []
                    for i in indices:
                        p = self.photos[i]
                        clinical = [(p.age - center) / scale, *(p.site == site for site in sites)]
                        if model == "B":
                            clinical.extend(p.symptoms)
                        expected.append([*self.images[i], *(factor * v for v in clinical)])
                    # Exact equality checks the unchanged image block, held-site handling,
                    # scaled age and symptoms, and that no extra intercept column is present.
                    np.testing.assert_array_equal(matrix, expected)
                calls.append((outer, inner, model, c))
                correct = .85 if c == choices[factor][model] else .55
                return np.where(held_x[:, 0] == 1, correct, 1 - correct)

            with patch.object(s4, "_fit_predict", side_effect=inspect):
                _, grid, _ = s4.run_factor(self.photos, self.images, self.plan, factor)
            selected = [r for r in grid if r["selected"]]
            self.assertEqual(len(selected), 10)
            self.assertTrue(all(r["C"] == choices[factor][r["model"]] for r in selected))
            refits = [(outer, model, c) for outer, inner, model, c in calls if inner is None]
            self.assertCountEqual(refits, [(r["outer_fold"], r["model"], r["C"]) for r in selected])
            self.assertEqual(len(calls), 260)
            np.testing.assert_array_equal(self.images, originals)

    def test_leaked_inner_patient_and_unspecified_factor_rejected_before_fit(self):
        split = self.plan[0]
        part = split.inner[0]
        # Swap one photo on each side, preserving total photo coverage but leaking a patient.
        bad = replace(part, train=(part.held_out[0], *part.train[1:]),
                      held_out=(part.train[0], *part.held_out[1:]))
        plan = [replace(split, inner=(bad, *split.inner[1:])), *self.plan[1:]]
        with patch.object(s4, "_fit_predict") as fit:
            with self.assertRaisesRegex(ValueError, "patient or linked component"):
                s4.run_factor(self.photos, self.images, plan, 1)
            for factor in (0, 2, True):
                with self.assertRaisesRegex(ValueError, "multipliers"):
                    s4.run_factor(self.photos, self.images, self.plan, factor)
            fit.assert_not_called()

    def test_iteration_cap_stops_before_any_predictions_and_reports_factor(self):
        def cap(estimator, x, y, sample_weight):
            estimator.n_iter_ = [estimator.max_iter]
        with (patch.object(primary.LogisticRegression, "fit", cap),
              patch.object(primary.LogisticRegression, "predict_proba") as predict,
              self.assertRaises(primary.FitFailure) as failed):
            s4.run_factor(self.photos, self.images, self.plan, 10)
        predict.assert_not_called()
        record, = failed.exception.diagnostics
        self.assertEqual(record["factor"], 10)
        self.assertTrue(record["iteration_cap_reached"])
        self.assertFalse(record["converged"])


if __name__ == "__main__":
    unittest.main()
