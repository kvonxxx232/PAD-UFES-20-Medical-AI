"""Synthetic checks of S7 denominators, within-key aggregation and Table 1."""

import json
import unittest

from pad_v11_secondary_inputs import _clinical_cohorts
from pad_v11_secondary_metrics import s3_summary
from pad_v11_secondary_missingness import s7_missingness
from pad_v11_secondary_report import ReportComparison, _s7_table1
from test_pad_v11_secondary_inputs import CURATED, KEYS, METADATA


def photo(patient, lesion, image, **changes):
    return {"patient_id": patient, "lesion_id": lesion, "img_id": image,
            "grew": "False", "changed": "False", "bleed": "False",
            "age": "40", "region": "ARM", "diagnostic": "ACK",
            "biopsed": "False", "gender": "", **changes}


class MissingnessTests(unittest.TestCase):
    def test_three_biopsy_statuses_count_each_key_once_and_keep_conflicts_in_s7(self):
        photos = [
            photo("alpha", "one", "a", biopsed="True"),
            photo("alpha", "one", "b", biopsed="True"),
            photo("beta", "one", "c", biopsed="False"),
            photo("beta", "one", "d", biopsed="False"),
            photo("beta", "two", "e", biopsed="False", diagnostic="BCC"),
            photo("beta", "two", "f", biopsed="True", diagnostic="BCC"),
            photo("alpha", "two", "g", biopsed="True", diagnostic="MEL"),
            photo("alpha", "two", "h", biopsed="False", diagnostic="MEL", bleed="UNK"),
        ]
        result = s7_missingness(photos)
        complete, missing = result["groups"]["complete"], result["groups"]["missing"]
        for group, expected in ((complete, (1, 1, 1)), (missing, (0, 0, 1))):
            counts = (group["biopsied"], group["not_biopsied"], group["biopsy_conflicting"])
            self.assertEqual(counts, expected)
            self.assertEqual(sum(counts), group["keys"])
        table = result["table1"]
        self.assertEqual(table["Biopsied"], ["1 (33.3%)", "0 (0.0%)"])
        self.assertEqual(table["Biopsy status, False"], ["1 (33.3%)", "0 (0.0%)"])
        self.assertEqual(table["Biopsy status, Conflicting"], ["1 (33.3%)", "1 (100.0%)"])
        self.assertEqual(table["Photographs"], ["6", "2"])
        self.assertEqual(complete["diagnosis"], {"ACK": 2, "BCC": 1})
        self.assertEqual(missing["diagnosis"], {"MEL": 1})
        self.assertEqual(result["missing_fraction"], 1 / 4)
        self.assertEqual(result["symptom_missingness"]["bleed"], {"keys": 1, "fraction": 1 / 4})
        self.assertEqual(result["unknown_field_counts"][1], {
            "unknown_fields": 1, "keys": 1, "positive": 1, "negative": 0, "positive_fraction": 1.0})
        self.assertEqual(result["patient_concordance"], {
            "eligible_patients": 2, "all_complete": 1, "all_missing": 0, "mixed": 1,
            "concordant_patients": 1, "fraction": 0.5})
        self.assertEqual(result, s7_missingness(photos[::-1]))

    def test_s3_excludes_complete_conflict_while_s7_retains_it(self):
        retained, biopsy_keys, cohorts = _clinical_cohorts(METADATA, CURATED, KEYS)
        self.assertEqual(biopsy_keys, {("one", "first")})
        oof = [{"patient_id": patient, "lesion_id": "first", "outer_fold": 1,
                "n_images": 2, "label": label, "p_A": 0.4, "p_B": 0.6}
               for patient, label in (("one", 1), ("two", 0))]
        s3 = s3_summary(oof, biopsy_keys)["pooled"]
        self.assertEqual((s3["n_keys"], s3["n_images"], s3["positive"]), (1, 2, 1))
        self.assertEqual(cohorts["curated"]["keys"], 3)
        table = s7_missingness(retained)["table1"]
        self.assertEqual(table["Lesion keys"], ["2", "1"])
        self.assertEqual(table["Biopsy status, Conflicting"], ["1 (50.0%)", "0 (0.0%)"])
        self.assertEqual(table["Biopsied"], ["1 (50.0%)", "1 (100.0%)"])

    def test_no_conflicts_report_explicit_zeros_for_both_groups(self):
        result = s7_missingness([photo("a", "x", "i", biopsed="True"),
                                 photo("b", "x", "j", biopsed="False", grew="UNK")])
        self.assertEqual(result["table1"]["Biopsy status, Conflicting"], ["0 (0.0%)", "0 (0.0%)"])
        self.assertEqual(result["table1"]["Biopsy status, False"], ["0 (0.0%)", "1 (100.0%)"])
        for group in result["groups"].values():
            self.assertEqual(group["biopsy_conflicting"], 0)

    def test_report_may_omit_only_a_conflict_row_with_zero_in_both_groups(self):
        def markdown(result, omit=None):
            lines = ["| Characteristic | Complete | Missing |", "| --- | ---: | ---: |"]
            for label, (complete, missing) in result["table1"].items():
                if label != omit:
                    lines.append(f"| {label} | {complete} | {missing} |")
            return "\n".join(lines)

        photos = [photo("a", "x", "i", biopsed="True"),
                  photo("b", "x", "j", biopsed="False", grew="UNK")]
        conflict = "Biopsy status, Conflicting"
        no_conflict = s7_missingness(photos)
        full = markdown(no_conflict)
        archived = markdown(no_conflict, omit=conflict)
        for valid in (full, archived):
            _s7_table1(ReportComparison(valid), no_conflict)
        for invalid in (markdown(no_conflict, omit="Biopsied"),
                        archived.replace("| Lesion keys | 1 | 1 |", "| Lesion keys | 2 | 1 |"),
                        full.replace(f"| {conflict} | 0 (0.0%)", f"| {conflict} | 1 (100.0%)")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _s7_table1(ReportComparison(invalid), no_conflict)
        for extra in (photo("a", "x", "k", biopsed="False"),
                      photo("b", "x", "k", biopsed="True", grew="UNK")):
            with self.subTest(extra=extra["patient_id"]):
                result = s7_missingness([*photos, extra])
                _s7_table1(ReportComparison(markdown(result)), result)
                with self.assertRaises(ValueError):
                    _s7_table1(ReportComparison(markdown(result, omit=conflict)), result)

    def test_unknown_union_over_photos_and_lesion_denominators(self):
        photos = [
            photo("alpha", "one", "a", grew="UNK", diagnostic="BCC"),
            photo("alpha", "one", "b", grew="UNK", changed="", diagnostic="BCC"),
            photo("alpha", "one", "c", changed="UNK", diagnostic="BCC"),
            photo("alpha", "two", "d"), photo("beta", "one", "e", bleed="UNK"),
        ]
        result = s7_missingness(photos)
        self.assertEqual(result["symptom_missingness"]["grew"], {"keys": 1, "fraction": 1 / 3})
        self.assertEqual(result["growth_change_only"], {"keys": 1, "fraction_of_missing": 0.5})
        strata = result["unknown_field_counts"]
        self.assertEqual(strata[2], {"unknown_fields": 2, "keys": 1, "positive": 1, "negative": 0,
                                     "positive_fraction": 1.0})
        self.assertEqual(strata[0]["keys"], 1)
        self.assertIsNone(strata[3]["positive_fraction"])
        self.assertEqual(result["groups"]["missing"]["photos"], 4)
        self.assertEqual(result["groups"]["missing"]["positive_fraction"], 0.5)
        self.assertNotIn("alpha", json.dumps(result, allow_nan=False))
        self.assertEqual(result, s7_missingness(photos[::-1]))

    def test_patient_concordance_counts_patients_once(self):
        photos = [
            photo("one", "a", "a"), photo("one", "b", "b"),
            photo("two", "a", "c", grew="UNK"), photo("two", "b", "d", bleed="UNK"),
            photo("three", "a", "e"), photo("three", "b", "f", changed="UNK"),
            photo("three", "c", "g"), photo("four", "a", "h"), photo("four", "a", "i"),
        ]
        result = s7_missingness(photos)
        self.assertEqual(result["patient_concordance"], {
            "eligible_patients": 3, "all_complete": 1, "all_missing": 1, "mixed": 1,
            "concordant_patients": 2, "fraction": 2 / 3,
        })
        self.assertEqual(result["table1"]["Patients"], ["3", "2"])

    def test_table1_uses_keys_unrecorded_sex_and_sample_sd(self):
        photos = [
            photo("a", "q", "a", age="20", gender="FEMALE"),
            photo("a", "q", "b", age="20", gender="FEMALE"),
            photo("b", "q", "c", age="40"),
            photo("c", "q", "d", age="40", grew="UNK", gender="MALE", biopsed="True"),
            photo("d", "q", "e", age="60", grew="UNK", biopsed="True"),
        ]
        result = s7_missingness(photos)
        table = result["table1"]
        self.assertEqual(table["Age, years, mean (SD)"], ["30.00 (14.14)", "50.00 (14.14)"])
        self.assertEqual(table["Age, median [Q1, Q3]"], ["30 [25, 35]", "50 [45, 55]"])
        self.assertEqual(table["Sex, unrecorded"], ["1 (50.0%)", "1 (50.0%)"])
        self.assertEqual(table["Biopsied"], ["0 (0.0%)", "2 (100.0%)"])
        self.assertEqual(table["Photographs per key, mean (SD)"], ["1.500 (0.707)", "1.000 (0.000)"])
        self.assertAlmostEqual(result["standardised_mean_differences"]["age"], 2 ** 0.5)
        self.assertAlmostEqual(result["standardised_mean_differences"]["photos_per_key"], -1.0)

    def test_undefined_denominators_are_null(self):
        result = s7_missingness([photo("p", "a", "i")])
        self.assertIsNone(result["patient_concordance"]["fraction"])
        self.assertIsNone(result["groups"]["missing"]["positive_fraction"])
        self.assertIsNone(result["standardised_mean_differences"]["age"])
        json.dumps(result, allow_nan=False)

    def test_repeated_image_and_conflicting_sex_do_not_select_first_record(self):
        first = photo("p", "a", "i", gender="MALE")
        with self.assertRaisesRegex(ValueError, "unique"):
            s7_missingness([first, first])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            s7_missingness([first, photo("p", "a", "j", gender="FEMALE")])


if __name__ == "__main__":
    unittest.main()
