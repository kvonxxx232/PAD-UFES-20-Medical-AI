"""Synthetic checks for the secondary analyses' private input gate."""

from pathlib import Path
import unittest
from unittest.mock import patch

from pad_v11_secondary_inputs import _clinical_cohorts, load_secondary_inputs


METADATA = (
    "patient_id,lesion_id,img_id,gender,age,region,diagnostic,biopsed,grew,changed,bleed\n"
    "one,first,photo-a,FEMALE,50,ARM,BCC,True,True,False,False\n"
    "one,first,photo-b,FEMALE,50,ARM,BCC,True,True,False,False\n"
    "one,second,photo-c,FEMALE,50,ARM,NEV,True,True,False,False\n"
    "one,second,photo-d,FEMALE,50,ARM,NEV,True,True,UNK,False\n"
    "two,first,photo-e,MALE,61,EAR,NEV,True,False,True,False\n"
    "two,first,photo-f,MALE,61,EAR,NEV,False,False,True,False\n"
    "three,excluded,photo-g,MALE,31,HAND,MEL,True,False,True,True\n"
).encode()
CURATED = (
    "patient_id,lesion_id,s3_eligible,biopsy_conflict,unknown_symptoms\n"
    "one,first,True,False,\n"
    "one,second,False,False,changed\n"
    "two,first,False,True,\n"
).encode()
KEYS = {("one", "first"): ("one", 1, True), ("one", "second"): ("one", 0, False),
        ("two", "first"): ("two", 0, True)}


class SecondaryInputTests(unittest.TestCase):
    def test_cohorts_use_ordered_keys_all_photo_biopsy_and_any_photo_unknown(self):
        rows, biopsy, cohorts = _clinical_cohorts(METADATA, CURATED, KEYS)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["patient_id"] != "three" for row in rows))
        self.assertEqual(biopsy, {("one", "first")})
        self.assertEqual(cohorts["primary"], {
            "keys": 2, "patients": 2, "images": 4, "positive": 1, "negative": 1})
        self.assertEqual(cohorts["incomplete"]["keys"], 1)
        self.assertEqual(cohorts["s3"]["images"], 2)

    def test_duplicate_retained_key_stops(self):
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            _clinical_cohorts(METADATA, CURATED + b"one,first,True,False,\n", KEYS)

    def test_unexpected_or_missing_metadata_or_retained_key_stops(self):
        for bad_meta, bad_curated in (
            (METADATA, CURATED + b"three,excluded,True,False,\n"),
            (METADATA, CURATED.replace(b"one,first,True,False,\n", b"")),
            (METADATA.replace(b"two,first,", b"other,first,"), CURATED),
        ):
            with self.subTest(), self.assertRaisesRegex(ValueError, "coverage"):
                _clinical_cohorts(bad_meta, bad_curated, KEYS)

    def test_s3_missingness_and_label_flags_must_match_original_photos(self):
        for field in [(b"True,False,\n", b"False,False,\n"), (b"changed", b"grew"),
                      (b"False,True,", b"False,False,")]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Clinical flags"):
                _clinical_cohorts(METADATA, CURATED.replace(*field), KEYS)
        with self.assertRaisesRegex(ValueError, "binary label"):
            _clinical_cohorts(METADATA, CURATED, {**KEYS, ("one", "first"): ("one", 0, True)})

    def test_tampered_h1_summary_rejected_before_patient_files_are_read(self):
        original_read = Path.read_bytes
        reads = []

        def wrong_summary(path):
            reads.append(path.name)
            if path.name == "protocol_v1.1.md":
                return original_read(path)
            if path.name == "summary.json":
                return b'{"status":"complete"}'
            self.fail("Read patient data before validating the H1 summary")

        with patch.object(Path, "read_bytes", wrong_summary):
            with self.assertRaisesRegex(ValueError, "Stage 4B summary SHA-256 mismatch"):
                load_secondary_inputs(*(Path("synthetic") for _ in range(5)))
        self.assertEqual(reads, ["protocol_v1.1.md", "summary.json"])


if __name__ == "__main__":
    unittest.main()
