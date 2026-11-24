"""Verify S2's independent partition audit and descriptive pairing with H1."""

from collections import Counter
from dataclasses import replace
import math
import unittest

from pad_ufes20_s2 import compare_oof, partition_audit
from pad_v11_s2_nested import plan_s2
from test_pad_v11_s2_nested import synthetic_s2


class S2RunnerTests(unittest.TestCase):
    def setUp(self):
        self.photos, self.complete = synthetic_s2()
        self.plan = plan_s2(self.photos, self.complete)
        sample = {p.key: p for p in self.photos if p.key in self.complete}
        counts = Counter(p.key for p in self.photos)
        self.h1 = [{"patient_id": p.patient_id, "lesion_id": p.lesion_id, "n_images": counts[k],
                    "label": p.label, "outer_fold": p.fold, "p_A": .5, "p_B": .5} for k, p in sample.items()]

    def test_all_keys_in_training_only_complete_keys_scored(self):
        counts = partition_audit(self.photos, self.complete, self.h1, self.plan)
        self.assertEqual(sum(f["scored_complete_keys"] for f in counts), len(self.h1))
        for f in counts:
            self.assertGreater(f["train_incomplete_keys"], 0)
            self.assertEqual(sum(r["held_keys"] for r in f["inner"]), f["train_keys"])
            self.assertEqual(sum(r["scored_complete_keys"] for r in f["inner"]),
                             f["train_keys"] - f["train_incomplete_keys"])
        outer = self.plan[0].outer
        # An unscored, incomplete outer-test photo must never be moved into training.
        leaked = next(i for i, p in enumerate(self.photos) if p.fold == 1 and p.key not in self.complete)
        bad = replace(self.plan[0], outer=replace(outer, train=(*outer.train, leaked)))
        with self.assertRaises(ValueError):
            partition_audit(self.photos, self.complete, self.h1, [bad, *self.plan[1:]])

    def test_paired_outputs_and_separate_h1_reference(self):
        rows = [dict(r, p_B=.75 if r["label"] else .25) for r in reversed(self.h1)]
        report = compare_oof(rows, self.h1)
        self.assertAlmostEqual(report["h1"]["delta_log_loss"], 0.)
        self.assertAlmostEqual(report["s2"]["delta_log_loss"], -math.log(.75) + math.log(.5))
        self.assertEqual(report["folds_favouring_B"], 5)
        for field, wrong in (("n_images", 0), ("outer_fold", 0), ("label", 3),
                             ("patient_id", "wrong"), ("lesion_id", "wrong")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_oof([dict(rows[0], **{field: wrong}), *rows[1:]], self.h1)
        with self.assertRaises(ValueError):
            compare_oof(rows + [rows[0]], self.h1)


if __name__ == "__main__":
    unittest.main()
