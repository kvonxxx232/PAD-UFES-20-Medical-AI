"""Synthetic image and clinical-record tests for the V1.1 curation rules."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from pad_v11_images import audit_images, phash64
from pad_v11_metadata import MetadataError, assess_metadata


def patterned_image():
    width, height = 63, 47
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend((
                (17 * x + 13 * y + x * y) % 256,
                (7 * x + 5 * y) % 256,
                (11 * x + 3 * y + x * y) % 256,
            ))
    return Image.frombytes("RGB", (width, height), bytes(pixels))


class ImageAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parts = [Path(temporary.name) / name for name in ("one", "two", "three")]
        for directory in self.parts:
            directory.mkdir()

    def test_cross_key_pair_precedence_and_review_denominator(self):
        image = patterned_image()
        # A nested archive layout and the order of metadata rows must not affect IDs.
        original = self.parts[0] / "nested" / "z-original.png"
        original.parent.mkdir()
        image.save(original)
        contents = original.read_bytes()
        (self.parts[0] / "a-foreign.png").write_bytes(contents)
        (self.parts[1] / "s-same.png").write_bytes(contents)
        info = PngImagePlugin.PngInfo()
        info.add_text("synthetic", "different PNG encoding")
        image.save(self.parts[2] / "m-reencoded.png", pnginfo=info)
        keys = {
            "z-original.png": ("patient-a", "one"),
            "s-same.png": ("patient-a", "one"),
            "a-foreign.png": ("patient-b", "one"),
            "m-reencoded.png": ("patient-a", "two"),
        }
        audit = audit_images(keys, self.parts)
        self.assertEqual(audit.confirmed, {
            ("a-foreign.png", "s-same.png"): "sha256",
            ("a-foreign.png", "z-original.png"): "sha256",
            ("a-foreign.png", "m-reencoded.png"): "rgb_equal",
            ("m-reencoded.png", "s-same.png"): "rgb_equal",
            ("m-reencoded.png", "z-original.png"): "rgb_equal",
        })
        self.assertEqual(audit.candidates, {
            ("a-foreign.png", "m-reencoded.png"): 0,
            ("m-reencoded.png", "s-same.png"): 0,
            ("m-reencoded.png", "z-original.png"): 0,
        })
        self.assertEqual(set(audit.file_hashes), set(keys))
        self.assertEqual(audit.file_hashes["z-original.png"], hashlib.sha256(contents).hexdigest())
        self.assertNotEqual(audit.file_hashes["m-reencoded.png"], audit.file_hashes["z-original.png"])

    def test_phash_convention_and_inclusive_review_cutoff(self):
        image = patterned_image()
        # Computed independently from the explicit type-II cosine basis.
        self.assertEqual(phash64(image), 0xF2A40E50017BABFB)
        keys = {"base.png": ("synthetic-a", "one")}
        image.save(self.parts[0] / "base.png")
        for size, filename in ((4, "eight.png"), (5, "ten.png")):
            variant = image.copy()
            variant.paste((255, 255, 255), (9, 7, 9 + size, 7 + size))
            variant.save(self.parts[1] / filename)
            keys[filename] = ("synthetic-a", filename)
        audit = audit_images(keys, self.parts)
        self.assertEqual(audit.confirmed, {})
        self.assertEqual(audit.candidates[("base.png", "eight.png")], 8)
        self.assertNotIn(("base.png", "ten.png"), audit.candidates)

    def test_equal_rgb_byte_stream_with_different_dimensions_is_only_a_candidate(self):
        for index, (name, size) in enumerate((("wide.png", (8, 2)), ("tall.png", (2, 8)))):
            Image.new("RGB", size, (70, 70, 70)).save(self.parts[index] / name)
        keys = {"wide.png": ("person-a", "one"), "tall.png": ("person-a", "two")}
        audit = audit_images(keys, self.parts)
        self.assertEqual(audit.confirmed, {})
        self.assertEqual(audit.candidates, {("tall.png", "wide.png"): 0})

    def test_different_source_modes_can_decode_to_equal_native_rgb(self):
        image = patterned_image().convert("L")
        image.save(self.parts[0] / "gray.png")
        image.convert("RGB").save(self.parts[1] / "rgb.png")
        keys = {"gray.png": ("person-a", "one"), "rgb.png": ("person-b", "two")}
        audit = audit_images(keys, self.parts)
        self.assertEqual(audit.confirmed, {("gray.png", "rgb.png"): "rgb_equal"})
        self.assertEqual(audit.candidates, {("gray.png", "rgb.png"): 0})

    def test_missing_extra_ambiguous_and_unreadable_images_fail(self):
        keys = {"only.png": ("synthetic-a", "one")}
        image = patterned_image()
        with self.assertRaises(ValueError):
            audit_images(keys, self.parts)
        image.save(self.parts[0] / "only.png")
        extra = self.parts[2] / "extra.png"
        image.save(extra)
        with self.assertRaises(ValueError):
            audit_images(keys, self.parts)
        extra.rename(self.parts[2] / "only.png")
        with self.assertRaises(ValueError):
            audit_images(keys, self.parts)
        (self.parts[2] / "only.png").unlink()
        (self.parts[0] / "only.png").write_bytes(b"broken image")
        with self.assertRaises(ValueError):
            audit_images(keys, self.parts)


def synthetic_row(name, **fields):
    row = {
        "img_id": name, "patient_id": "synthetic-patient", "lesion_id": "synthetic-lesion",
        "age": "40", "region": "ARM", "diagnostic": "ACK", "biopsed": "True",
        "grew": "True", "changed": "False", "bleed": "False",
    }
    row.update(fields)
    return row


class MetadataTests(unittest.TestCase):
    def assess_pair(self, first, second):
        rows = [synthetic_row("left.png", **first), synthetic_row("right.png", **second)]
        return assess_metadata(rows)[("synthetic-patient", "synthetic-lesion")]

    def error_fields(self, first, second):
        with self.assertRaises(MetadataError) as error:
            self.assess_pair(first, second)
        for issue in error.exception.issues:
            self.assertEqual(issue["patient_id"], "synthetic-patient")
            self.assertEqual(issue["lesion_id"], "synthetic-lesion")
        return {issue["field"] for issue in error.exception.issues}

    def test_complete_consistent_and_binary_mapping(self):
        for diagnosis in ("ACK", "NEV", "SEK", "BCC", "SCC", "MEL"):
            with self.subTest(diagnosis=diagnosis):
                result = self.assess_pair({"diagnostic": diagnosis}, {"diagnostic": diagnosis})
                self.assertEqual(result.positive, diagnosis in ("BCC", "SCC", "MEL"))
                self.assertTrue(result.complete)
                self.assertTrue(result.biopsy_confirmed)
                self.assertEqual(result.image_count, 2)
                self.assertFalse(result.symptom_conflicts)

    def test_contradiction_vs_ordinary_missingness(self):
        for missing in ("UNK", ""):
            with self.subTest(missing=missing):
                result = self.assess_pair({}, {"grew": missing})
                self.assertFalse(result.complete)
                self.assertFalse(result.symptom_conflicts)
                self.assertEqual(set(result.unknown_symptoms), {"grew"})
        result = self.assess_pair({}, {"grew": "False", "changed": "True", "bleed": "UNK"})
        self.assertFalse(result.complete)
        self.assertEqual(set(result.symptom_conflicts), {"grew", "changed"})
        self.assertEqual(set(result.unknown_symptoms), {"bleed"})

    def test_biopsy_disagreement_excludes_s3_only(self):
        result = self.assess_pair({}, {"biopsed": "False"})
        self.assertTrue(result.complete)
        self.assertFalse(result.biopsy_confirmed)
        self.assertTrue(result.biopsy_conflict)
        self.assertFalse(result.symptom_conflicts)

    def test_conflicting_detailed_diagnoses_in_same_or_different_class(self):
        fields = self.error_fields({"diagnostic": "BCC"}, {"diagnostic": "SCC"})
        self.assertIn("diagnostic", fields)
        self.assertNotIn("binary_label", fields)
        fields = self.error_fields({"diagnostic": "BCC"}, {"diagnostic": "NEV"})
        self.assertTrue({"diagnostic", "binary_label"}.issubset(fields))

    def test_invalid_missing_and_discordant_fields_stop_with_field_details(self):
        for field, bad in (
            ("age", "-1"), ("age", "121"), ("age", "2.5"), ("age", ""), ("age", "41"),
            ("region", "OTHER"), ("region", ""), ("region", "BACK"),
            ("diagnostic", ""), ("diagnostic", "NOT_A_CLASS"),
            ("biopsed", "UNK"), ("grew", "yes"),
        ):
            with self.subTest(field=field, value=bad):
                self.assertIn(field, self.error_fields({}, {field: bad}))
        for age in ("0", "120"):
            with self.subTest(valid_age=age):
                self.assertTrue(self.assess_pair({"age": age}, {"age": age}).complete)


if __name__ == "__main__":
    unittest.main()
