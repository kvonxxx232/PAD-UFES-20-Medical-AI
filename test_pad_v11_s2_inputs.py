"""Synthetic tests for per-photo S2 encoding and pairing with primary OOF evidence."""

import unittest

from pad_v11_s2_inputs import _aligned_reference, _curated_photos


FIELDS = "img_id,patient_id,lesion_id,age,region,grew,changed,bleed\n"
CLINICAL = FIELDS + (
    "i1,p1,k,30,ARM,True,False,False\n"
    "i2,p2,k,40,EAR,UNK,True,False\n"
    "i3,p2,k,40,EAR,True,,False\n"
    "i4,p3,k,20,FACE,False,False,False\n"
    "i5,quarantined,k,32,FACE,True,False,True\n"
)
KEYS = {("p1", "k"): ("p1", 0, True), ("p2", "k"): ("p2", 1, False),
        ("p3", "k"): ("p3", 1, True)}
FOLDS = {("p1", "k"): 1, ("p2", "k"): 1, ("p3", "k"): 2}
OOF_HEADER = "patient_id,lesion_id,outer_fold,label,n_images,p_A,p_B\n"
R1 = "p1,k,1,0,1,0.1,0.2\n"
R3 = "p3,k,2,1,1,0.9,0.8\n"


class S2InputTests(unittest.TestCase):
    def setUp(self):
        self.photos, self.complete = _curated_photos(CLINICAL.encode(), KEYS, FOLDS)

    def test_per_photo_unknowns_and_whole_key_eligibility(self):
        self.assertEqual([p.image_id for p in self.photos], ["i1", "i2", "i3", "i4"])
        self.assertEqual([p.symptoms for p in self.photos],
                         [(1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0)])
        self.assertEqual(self.complete, {("p1", "k"), ("p3", "k")})
        # The omission is image-specific. A whitespace-only response behaves like a blank.
        rows, complete = _curated_photos(CLINICAL.replace("True,,False", "True, ,False").encode(),
                                         KEYS, FOLDS)
        self.assertEqual((rows, complete), (self.photos, self.complete))

    def test_unrecognized_token_and_inconsistent_completeness_stop(self):
        with self.assertRaisesRegex(ValueError, "Unrecognised"):
            _curated_photos(CLINICAL.replace("UNK", "other").encode(), KEYS, FOLDS)
        with self.assertRaisesRegex(ValueError, "completeness"):
            _curated_photos(CLINICAL.encode(), {**KEYS, ("p2", "k"): ("p2", 1, True)}, FOLDS)

    def test_original_primary_row_pairing(self):
        rows = _aligned_reference((OOF_HEADER + R3 + R1).encode(), self.photos, self.complete)
        self.assertEqual([r["patient_id"] for r in rows], ["p3", "p1"])
        self.assertEqual((rows[0]["outer_fold"], rows[0]["label"], rows[0]["p_A"], rows[0]["p_B"]),
                         (2, 1, 0.9, 0.8))

    def test_missing_extra_duplicate_or_discordant_oof_rows_stop(self):
        altered = {"missing": R3, "extra incomplete": R1 + R3 + R3.replace("p3", "p2"),
                   "duplicate": R1 + R3 + R1, "patient": R1 + R3.replace("p3", "p7"),
                   "lesion": R1 + R3.replace(",k,", ",other,"), "fold": R1 + R3.replace(",2,", ",3,"),
                   "label": R1 + R3.replace(",2,1,", ",2,0,"),
                   "images": R1 + R3.replace(",1,0.9", ",2,0.9"),
                   "probability": R1 + R3.replace("0.8", "nan")}
        for name, records in altered.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                _aligned_reference((OOF_HEADER + records).encode(), self.photos, self.complete)


if __name__ == "__main__":
    unittest.main()
