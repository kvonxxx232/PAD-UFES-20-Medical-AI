"""Synthetic complete-case fixtures only; independent checks of the nested primary fits."""

from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

import pad_v11_nested as nested
from pad_v11_nested import Photo


def synthetic_photos():
    photos = []
    for component in range(20):
        fold = component % 5 + 1
        # Linked synthetic patients with multiple evaluated lesions and photos.
        for member in range(2):
            patient = f"patient-{component}-{member}"
            lesions = 6 if component == 2 else 2
            for lesion in range(lesions):
                label = lesion % 2
                key = f"lesion-{lesion}"
                for view in range(1 + int(lesion == 1 and member == 1)):
                    photos.append(Photo(
                        f"{patient}-{key}-view{view}", patient, key, f"group-{component}", fold,
                        label, 116 if fold == 1 else 40 + component, "NECK" if fold == 1 else "ARM",
                        (label, member, component % 2)))
    photos.reverse()  # The sorted-key inner splitter must not depend on input file order.
    return photos


def embeddings(photos):
    return np.array([[float(p.label), i / len(photos)] for i, p in enumerate(photos)])


class NestedTests(unittest.TestCase):
    def test_saved_outer_inner_memberships_and_shared_components(self):
        photos = synthetic_photos()
        plans = nested.plan_splits(photos)
        for plan in plans:
            train = [photos[i] for i in plan.outer.train]
            test = [photos[i] for i in plan.outer.held_out]
            self.assertEqual({p.fold for p in test}, {plan.fold})
            self.assertFalse({p.patient_id for p in train} & {p.patient_id for p in test})
            self.assertFalse({p.group for p in train} & {p.group for p in test})
            by_key = {p.key: p for p in train}
            ordered = [by_key[k] for k in sorted(by_key)]
            reference = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261124)
            expected = reference.split(np.zeros(len(ordered)), [p.label for p in ordered],
                                       [p.group for p in ordered])
            for part, (fit_indices, val_indices) in zip(plan.inner, expected, strict=True):
                fit, val = [photos[i] for i in part.train], [photos[i] for i in part.held_out]
                self.assertEqual({p.key for p in fit}, {ordered[i].key for i in fit_indices})
                self.assertEqual({p.key for p in val}, {ordered[i].key for i in val_indices})
                self.assertFalse({p.group for p in fit} & {p.group for p in val})
                self.assertFalse(({p.patient_id for p in fit} | {p.patient_id for p in val})
                                 & {p.patient_id for p in test})
                self.assertEqual({p.label for p in val}, {0, 1})
            self.assertCountEqual([i for part in plan.inner for i in part.held_out], plan.outer.train)

    def test_invalid_or_infeasible_inputs_stop_before_any_fit(self):
        photos = synthetic_photos()
        first = photos[0]
        for title, changed in [
            ("group across folds", [replace(first, fold=1), *photos[1:]]),
            ("patient in different groups", [replace(first, group="different"), *photos[1:]]),
            ("incomplete symptom", [replace(first, symptoms=(0, 1, None)), *photos[1:]]),
            ("repeated photo ID", photos + [first]),
            ("one class in validation", [replace(p, label=0) if p.fold == 1 else p for p in photos]),
        ]:
            with self.subTest(title=title), patch.object(nested.LogisticRegression, "fit") as fit:
                with self.assertRaises(ValueError):
                    nested.run_nested(changed, embeddings(changed))
                fit.assert_not_called()

    def test_constant_age_and_unseen_sites(self):
        p = synthetic_photos()[0]
        encoder = nested.ClinicalEncoder.fit([p, p])
        held = [replace(p, age=p.age + 4, site="FOOT")]
        self.assertEqual(encoder.age_scale, 1.)
        expected = np.array([[-4., 9., 4., 0., *p.symptoms]])
        np.testing.assert_array_equal(encoder.transform(held, np.array([[-4., 9.]]), "B"), expected)

    def test_all_fits_use_only_their_training_preprocessor_and_weights(self):
        photos = synthetic_photos()
        vectors = embeddings(photos)
        expected = {}
        for plan in nested.plan_splits(photos):
            expected[plan.fold, None] = plan.outer
            for index, part in enumerate(plan.inner, 1):
                expected[plan.fold, index] = part
        calls = Counter()

        def inspect(train_x, held_x, labels, weights, c, context, diagnostics):
            outer, inner, model = context["outer_fold"], context["inner_fold"], context["model"]
            calls[outer, inner, model, c] += 1
            part = expected[outer, inner]
            train = [photos[i] for i in part.train]
            ages = [p.age for p in train]
            mu, scale = np.mean(ages), np.std(ages) or 1.
            vocabulary = sorted({p.site for p in train})
            for design, indices in [(train_x, part.train), (held_x, part.held_out)]:
                rows = []
                for i in indices:
                    p = photos[i]
                    row = [*vectors[i], (p.age - mu) / scale, *[p.site == v for v in vocabulary]]
                    if model == "B":
                        row.extend(p.symptoms)
                    rows.append(row)
                np.testing.assert_allclose(design, rows)
            np.testing.assert_array_equal(labels, [p.label for p in train])
            counts = Counter(p.key for p in train)
            np.testing.assert_allclose(weights, [1 / counts[p.key] for p in train])
            return np.full(len(held_x), 0.5)

        with patch.object(nested, "_fit_predict", side_effect=inspect):
            _, tuning, _ = nested.run_nested(photos, vectors)
        self.assertEqual(sum(calls.values()), 260)
        self.assertEqual(set(calls.values()), {1})
        self.assertEqual([r["C"] for r in tuning if r["selected"]], [0.001] * 10)

    def test_inner_selection_uses_pooled_equal_key_loss_and_unrounded_ties(self):
        photos = synthetic_photos()
        target = nested.plan_splits(photos)[0]
        sizes = [len({photos[i].key for i in part.held_out}) for part in target.inner]
        largest = int(np.argmax(sizes)) + 1
        pooled = (max(sizes) * 1. + (sum(sizes) - max(sizes)) * .1) / sum(sizes)
        fold_mean = (1. + 4 * .1) / 5
        self.assertGreater(pooled, fold_mean)
        constant = (pooled + fold_mean) / 2

        def prescribed(train_x, held_x, labels, weights, c, context, diagnostics):
            loss = 2.
            if context["model"] == "A":
                if c == 0.01:
                    loss = 1. if context["inner_fold"] == largest else .1
                if c == 0.1:
                    loss = constant
            elif c == 10:
                loss = .01
            correct_class_prob = np.exp(-loss)
            return np.where(held_x[:, 0] == 1, correct_class_prob, 1 - correct_class_prob)

        with patch.object(nested, "_fit_predict", side_effect=prescribed):
            _, tuning, _ = nested.run_nested(photos, embeddings(photos))
        grid = {r["C"]: r["pooled_log_loss"] for r in tuning if r["outer_fold"] == 1 and r["model"] == "A"}
        self.assertAlmostEqual(grid[0.01], pooled)
        self.assertAlmostEqual(grid[0.1], constant)
        selected = {r["model"]: r["C"] for r in tuning if r["outer_fold"] == 1 and r["selected"]}
        self.assertEqual(selected, {"A": 0.1, "B": 10})
        near_tie = dict.fromkeys(nested.C_GRID, 1.)
        near_tie[10] = np.nextafter(1., 0.)
        self.assertEqual(nested.select_c(near_tie), 10)
        self.assertEqual(nested.select_c(dict.fromkeys(nested.C_GRID, 1.)), .001)

    def test_probability_aggregation_and_signed_extremes(self):
        p = synthetic_photos()[0]
        q = replace(p, patient_id="second")   # Reused lesion ID stays separate.
        means = nested.mean_probabilities([p, q, p, p], [.02, 1., .4, .9])
        self.assertAlmostEqual(means[p.key], .44)
        self.assertEqual(means[q.key], 1.)
        with self.assertRaises(ValueError):
            nested.mean_probabilities([p], [np.nan])

    def test_failed_warning_or_iteration_cap_blocks_all_predictions(self):
        photos = synthetic_photos()
        for signal in ["warning", "cap", "fit_error"]:
            def fake_fit(estimator, x, y, sample_weight):
                if signal == "fit_error":
                    raise RuntimeError("synthetic fit failure")
                estimator.n_iter_ = [estimator.max_iter if signal == "cap" else 2]
                if signal == "warning":
                    warnings.warn("synthetic warning", ConvergenceWarning)

            with (self.subTest(signal=signal), patch.object(nested.LogisticRegression, "fit", fake_fit),
                  patch.object(nested.LogisticRegression, "predict_proba") as predict):
                with self.assertRaises(nested.FitFailure) as caught:
                    nested.run_nested(photos, embeddings(photos))
                predict.assert_not_called()
                log, = caught.exception.diagnostics
                self.assertFalse(log["converged"])
                self.assertEqual(log["iteration_cap_reached"], signal == "cap")
                self.assertEqual([w["category"] for w in log["warnings"]],
                                 ["ConvergenceWarning"] if signal == "warning" else [])
                self.assertEqual(log["error"] is not None, signal == "fit_error")

    def test_full_synthetic_fit_has_paired_oof_and_complete_diagnostics(self):
        photos = synthetic_photos()
        rows, scores, fits = nested.run_nested(photos, embeddings(photos))
        counts = Counter(p.key for p in photos)
        self.assertEqual([(r["patient_id"], r["lesion_id"]) for r in rows], sorted(counts))
        for row in rows:
            self.assertEqual(row["n_images"], counts[row["patient_id"], row["lesion_id"]])
            for field in ("p_A", "p_B"):
                self.assertTrue(0 < row[field] < 1)
        self.assertEqual(len(scores), 50)
        self.assertEqual(sum(r["selected"] for r in scores), 10)
        self.assertEqual(len(fits), 260)
        self.assertEqual(sum(r["inner_fold"] is None for r in fits), 10)
        self.assertTrue(all(r["converged"] and r["n_iter"] and not r["iteration_cap_reached"] for r in fits))


if __name__ == "__main__":
    unittest.main()
