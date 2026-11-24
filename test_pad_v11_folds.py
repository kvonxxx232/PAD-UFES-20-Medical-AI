"""Protocol V1.1 outer-fold tests on fictional patients, without model fitting."""

import random
import unittest

from sklearn.model_selection import StratifiedGroupKFold

from pad_ufes20_folds import generate_outer_folds, split_table_bytes


def synthetic_keys():
    records = {}
    for i in range(48):
        patient = f"example-{i:02d}"
        # Each of the first four components connects two different patients.
        group = f"linked-{i // 2}" if i < 8 else patient
        for lesion, stratum in (("10", i % 4), ("2", (i + 1 + (i // 4) % 3) % 4)):
            records[patient, lesion] = group, stratum // 2, bool(stratum % 2)
    return records


class OuterFoldTests(unittest.TestCase):
    def setUp(self):
        self.keys = synthetic_keys()
        # Reused lesion IDs under different patients are distinct keys.
        self.quarantine = {("outside-cohort", "10"), ("example-00", "quarantined")}

    def test_assignment_coverage_and_patient_component_disjointness(self):
        assigned, summaries = generate_outer_folds(self.keys, self.quarantine)
        self.assertEqual(set(assigned), set(self.keys))
        self.assertTrue(self.quarantine.isdisjoint(assigned))
        self.assertEqual(set(assigned.values()), {1, 2, 3, 4, 5})
        self.assertEqual(len(summaries), 5)
        for fold in range(1, 6):
            validation = {key for key in assigned if assigned[key] == fold}
            training = set(assigned) - validation
            self.assertTrue({k[0] for k in training}.isdisjoint(k[0] for k in validation))
            self.assertTrue({self.keys[k][0] for k in training}.isdisjoint(
                self.keys[k][0] for k in validation
            ))
            for population in (training, validation):
                self.assertEqual({self.keys[k][1] for k in population if self.keys[k][2]}, {0, 1})

    def test_matches_frozen_splitter_seed_strata_and_string_sort(self):
        ordered = sorted(self.keys)
        groups = [self.keys[k][0] for k in ordered]
        strata = [2 * self.keys[k][1] + int(self.keys[k][2]) for k in ordered]
        expected = {}
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261124)
        for fold, (_, held_out) in enumerate(splitter.split(ordered, strata, groups), 1):
            for i in held_out:
                expected[ordered[i]] = fold
        actual, _ = generate_outer_folds(self.keys, self.quarantine)
        self.assertEqual(actual, expected)

    def test_shuffled_input_reproduces_assignments_summaries_and_bytes(self):
        original, summaries = generate_outer_folds(self.keys, self.quarantine)
        rows = list(self.keys.items())
        random.Random(99).shuffle(rows)
        repeated, repeated_summaries = generate_outer_folds(dict(rows), self.quarantine)
        self.assertEqual(original, repeated)
        self.assertEqual(summaries, repeated_summaries)
        self.assertEqual(split_table_bytes(original), split_table_bytes(repeated))

    def test_assignment_file_is_lexicographic_utf8_csv_with_lf(self):
        assignment = {("Å", "3"): 5, ("a", "2"): 2, ("a", "10"): 1}
        self.assertEqual(
            split_table_bytes(assignment),
            "patient_id,lesion_id,fold\na,10,1\na,2,2\nÅ,3,5\n".encode("utf-8"),
        )

    def test_quarantine_leak_refuses_split(self):
        with self.assertRaises(ValueError):
            generate_outer_folds(self.keys, self.quarantine | {("example-00", "10")})

    def test_inconsistent_component_for_one_patient_refuses_split(self):
        key = ("example-00", "2")
        _, label, complete = self.keys[key]
        self.keys[key] = "unrelated-group", label, complete
        with self.assertRaises(ValueError):
            generate_outer_folds(self.keys, self.quarantine)

    def test_missing_primary_class_refuses_split_even_when_all_keys_have_both_classes(self):
        keys = {k: (group, label, label == 0 and complete)
                for k, (group, label, complete) in self.keys.items()}
        with self.assertRaises(ValueError):
            generate_outer_folds(keys, self.quarantine)

    def test_global_class_counts_cannot_hide_a_fold_without_primary_positives(self):
        keys = {k: (group, label, label == 0 and complete)
                for k, (group, label, complete) in self.keys.items()}
        # Six positive complete keys but only one patient can supply them.
        for i in range(6):
            keys["one-positive-patient", str(i)] = ("one-positive-patient", 1, True)
        with self.assertRaises(ValueError):
            generate_outer_folds(keys, self.quarantine)


if __name__ == "__main__":
    unittest.main()
