"""Synthetic checks for all-key training and complete-only S2 prediction and tuning."""

from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import pad_v11_s2_nested as s2
from pad_v11_nested import FitFailure, Photo
from test_pad_v11_nested import synthetic_photos


def synthetic_s2():
    photos = synthetic_photos()
    complete = {p.key for p in photos}
    # Some incomplete keys share a patient with a complete key. Others share
    # only a component, and five whole components have no complete keys.
    for group in range(20):
        fold = group % 5 + 1
        for patient in (f"patient-{group}-0", f"new-{group}"):
            for j in (0, 1):
                photos.append(Photo(f"{patient}-incomplete-{j}", patient, "incomplete",
                                    f"group-{group}", fold, group % 2, 51, "LIP", (j, 0, 1)))
    for fold in range(1, 6):
        key = f"only-incomplete-{fold}"
        photos.append(Photo(key + "-img", key, "unknown", key, fold, fold % 2, 23, "FACE", (0, 1, 0)))
    return photos, complete


class SecondaryNestedTests(unittest.TestCase):
    def test_partitions_against_independent_all_key_split(self):
        photos, complete = synthetic_s2()
        representative = {p.key: p for p in photos}
        plans = s2.plan_s2(photos, complete)
        self.assertEqual(len(plans), 5)
        for plan in plans:
            train = sorted(k for k, p in representative.items() if p.fold != plan.fold)
            test = representative.keys() - set(train)
            self.assertEqual({photos[i].key for i in plan.outer.train}, set(train))
            self.assertEqual({photos[i].key for i in plan.outer.held_out}, test & complete)
            outer_patients = {k[0] for k in test}
            outer_groups = {representative[k].group for k in test}
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261124)
            ref = splitter.split(np.zeros(len(train)), [2*representative[k].label + int(k in complete) for k in train],
                                 [representative[k].group for k in train])
            for (tr, val), inner in zip(ref, plan.inner, strict=True):
                fit_keys, val_keys = {train[i] for i in tr}, {train[i] for i in val}
                self.assertEqual({photos[i].key for i in inner.train}, fit_keys)
                self.assertEqual({photos[i].key for i in inner.held_out}, val_keys & complete)
                self.assertTrue({k[0] for k in fit_keys}.isdisjoint(k[0] for k in val_keys))
                self.assertTrue({representative[k].group for k in fit_keys}.isdisjoint(
                    representative[k].group for k in val_keys))
                self.assertTrue(outer_patients.isdisjoint(k[0] for k in fit_keys | val_keys))
                self.assertTrue(outer_groups.isdisjoint(representative[k].group for k in fit_keys | val_keys))
        self.assertCountEqual([i for plan in plans for i in plan.outer.held_out],
                              [i for i, p in enumerate(photos) if p.key in complete])

    def test_fit_weights_preprocessing_and_scored_objective(self):
        photos, complete = synthetic_s2()
        indices = np.arange(len(photos))[:, None]
        partitions = {}
        for split in s2.plan_s2(photos, complete):
            partitions[split.fold, None] = split.outer
            partitions.update({(split.fold, i): p for i, p in enumerate(split.inner, 1)})
        calls = Counter()

        def predict(model, c, i):
            # Unequal risks within multi-photo keys stress aggregation before scoring.
            good = c == (0.01 if model == "A" else 1)
            correct = (0.7 if good else 0.3) + (i % 3) / 20
            return correct if photos[i].label else 1 - correct

        def inspect(x, vx, y, weights, c, context, diagnostics):
            fold, inner, model = context["outer_fold"], context["inner_fold"], context["model"]
            calls[fold, inner, model, c] += 1
            part = partitions[fold, inner]
            train = [photos[i] for i in part.train]
            self.assertTrue(any(p.key not in complete for p in train))
            self.assertTrue(all(photos[i].key in complete for i in part.held_out))
            np.testing.assert_array_equal(y, [p.label for p in train])
            counts = Counter(p.key for p in train)
            np.testing.assert_array_equal(weights, [1/counts[p.key] for p in train])
            ages, sites = [p.age for p in train], sorted({p.site for p in train})
            mu, scale = np.mean(ages), np.std(ages) or 1.
            for matrix, subset in [(x, part.train), (vx, part.held_out)]:
                expected = []
                for i in subset:
                    p = photos[i]
                    row = [i, (p.age - mu)/scale, *[p.site == site for site in sites]]
                    if model == "B":
                        row.extend(p.symptoms)
                    expected.append(row)
                np.testing.assert_allclose(matrix, expected)
            return np.array([predict(model, c, i) for i in part.held_out])

        with patch.object(s2, "_fit_predict", side_effect=inspect):
            oof, scores, _ = s2.run_s2_nested(photos, indices, complete)
        self.assertEqual(len(calls), 260)
        self.assertEqual(set(calls.values()), {1})
        for score in scores:
            fold, model, c = score["outer_fold"], score["model"], score["C"]
            # Independent, direct probability means and losses on complete training keys.
            probs = {}
            for i, p in enumerate(photos):
                if p.fold != fold and p.key in complete:
                    probs.setdefault((p.key, p.label), []).append(predict(model, c, i))
            loss = [-np.log(np.mean(v) if y else 1-np.mean(v)) for (_, y), v in probs.items()]
            self.assertEqual(score["n_keys"], len(probs))
            self.assertAlmostEqual(score["pooled_log_loss"], np.mean(loss))
            self.assertEqual(score["selected"], c == (0.01 if model == "A" else 1))
        for row in oof:
            ids = [i for i,p in enumerate(photos) if p.key == (row["patient_id"], row["lesion_id"])]
            for model, c in [("A", 0.01), ("B", 1)]:
                self.assertAlmostEqual(row["p_" + model], np.mean([predict(model, c, i) for i in ids]))

    def test_failed_checks_stop_before_fitting(self):
        photos, complete = synthetic_s2()
        a = photos[-1]  # From a group without any complete keys.
        wrong_fold = replace(a, image_id="other-image", lesion_id="other-lesion", fold=1)
        invalid_sets = [(photos + [wrong_fold], complete), (photos, complete | {("unknown", "key")}),
                        (photos, {p.key for p in photos if p.key in complete and p.label == 1})]
        for current, keys in invalid_sets:
            with self.subTest(), patch.object(s2, "_fit_predict") as fit:
                with self.assertRaises(ValueError):
                    s2.run_s2_nested(current, np.zeros((len(current), 2)), keys)
                fit.assert_not_called()

    def test_synthetic_real_heads_and_failure_stop(self):
        photos, complete = synthetic_s2()
        rng = np.random.default_rng(53)
        vectors = rng.normal(size=(len(photos), 3))
        rows, grid, diagnostics = s2.run_s2_nested(photos, vectors, complete)
        self.assertCountEqual([(r["patient_id"],r["lesion_id"]) for r in rows], complete)
        self.assertTrue(all(r["converged"] for r in diagnostics))
        self.assertEqual(len(diagnostics), 260)
        self.assertEqual(len(grid), 50)
        self.assertEqual(sum(r["selected"] for r in grid), 10)
        failure = FitFailure([{"converged": False}])
        with patch.object(s2, "_fit_predict", side_effect=failure) as fit:
            with self.assertRaises(FitFailure) as caught:
                s2.run_s2_nested(photos, vectors, complete)
            fit.assert_called_once()
            self.assertEqual(caught.exception.diagnostics, [{"converged": False}])


if __name__ == "__main__":
    unittest.main()
