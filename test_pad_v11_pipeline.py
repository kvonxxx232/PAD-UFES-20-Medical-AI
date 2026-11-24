"""End-to-end Stage 2 tests with temporary PNGs and fictional records."""

import csv
import hashlib
import io
import json
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image, PngImagePlugin

import pad_ufes20_curate as pipeline


class CurationPipelineTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.metadata = self.root / "input.csv"
        self.dirs = [self.root / f"part-{i}" for i in range(3)]
        for directory in self.dirs:
            directory.mkdir()
        self.rows = []
        self.output = self.root / "audit"

        self.add_photo("a", "dup-a", 1)
        self.add_photo("a", "dup-a", 2, grew="True")
        self.add_photo("b", "dup-b", 1)
        self.add_photo("a", "kept-a", 3)
        self.add_photo("b", "kept-b", 4, grew="UNK")
        self.add_photo("c", "symptoms", 5)
        self.add_photo("c", "symptoms", 6, changed="True")
        self.add_photo("d", "biopsy", 7)
        self.add_photo("d", "biopsy", 8, biopsed="False")
        self.add_photo("e", "rgb-a", 9)
        self.add_photo("e", "rgb-b", 9, comment="alternate encoding")
        self.write_metadata()

    def add_photo(self, patient, lesion, colour, comment=None, **overrides):
        number = len(self.rows)
        name = f"fixture-{number}.png"
        # Solid but different colours intentionally generate visual-review candidates.
        im = Image.new("RGB", (12, 10), (colour * 20, 71, 113))
        info = PngImagePlugin.PngInfo()
        if comment:
            info.add_text("note", comment)
        im.save(self.dirs[number % len(self.dirs)] / name, pnginfo=info)
        row = dict(
            img_id=name, patient_id=patient, lesion_id=lesion, age="52",
            region="ARM", diagnostic="NEV", biopsed="True", grew="False",
            changed="False", bleed="False",
        )
        row.update(overrides)
        self.rows.append(row)

    def write_metadata(self):
        with self.metadata.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def run_cli(self, output=None):
        args = [
            "--metadata", str(self.metadata), "--images", *map(str, self.dirs),
            "--output-dir", str(output or self.output),
        ]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return pipeline.main(args)

    def read_csv(self, name):
        with (self.output / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_complete_run_and_grouped_quarantine(self):
        # The CLI always enforces the published reference for real use.
        with patch.multiple(pipeline, IMAGE_REFERENCE={}, COHORT_REFERENCE={}):
            self.assertEqual(self.run_cli(), 0)
        manifest = {
            (row["patient_id"], row["lesion_id"]): set(row["rule_ids"].split("|"))
            for row in self.read_csv("quarantine_manifest.csv")
        }
        self.assertEqual(manifest, {
            ("a", "dup-a"): {"sha256", "symptom_conflict"},
            ("b", "dup-b"): {"sha256"},
            ("c", "symptoms"): {"symptom_conflict"},
            ("e", "rgb-a"): {"rgb_equal"},
            ("e", "rgb-b"): {"rgb_equal"},
        })
        retained = {
            (row["patient_id"], row["lesion_id"]): row
            for row in self.read_csv("curated_keys.csv")
        }
        self.assertEqual(set(retained), {("a", "kept-a"), ("b", "kept-b"), ("d", "biopsy")})
        a, b, d = (retained[k] for k in (("a", "kept-a"), ("b", "kept-b"), ("d", "biopsy")))
        self.assertEqual(a["patient_group"], b["patient_group"])
        self.assertNotEqual(a["patient_group"], d["patient_group"])
        self.assertEqual((a["primary_eligible"], a["s3_eligible"]), ("True", "True"))
        self.assertEqual((b["primary_eligible"], b["s3_eligible"]), ("False", "False"))
        self.assertEqual((d["primary_eligible"], d["s3_eligible"]), ("True", "False"))

        statuses = [r["status"] for r in self.read_csv("phash_candidates.csv")]
        self.assertIn("pending_review", statuses)
        self.assertEqual(statuses.count("confirmed_rgb"), 1)
        self.assertEqual(set(statuses), {"pending_review", "confirmed_rgb"})
        self.assertTrue((self.output / "summary.json").is_file())
        self.assertFalse((self.output / "errors.json").exists())

    def test_fatal_diagnosis_on_quarantined_key_preserves_prior_image_evidence(self):
        self.rows[1]["diagnostic"] = "ACK"  # Same binary class, different detailed diagnosis.
        self.write_metadata()
        with patch.multiple(pipeline, IMAGE_REFERENCE={}, COHORT_REFERENCE={}):
            self.assertNotEqual(self.run_cli(), 0)
        self.assertEqual(len(self.read_csv("confirmed_pairs.csv")), 2)
        with (self.output / "errors.json").open(encoding="utf-8") as f:
            issues = json.load(f)["issues"]
        self.assertIn(("a", "dup-a", "diagnostic"), {
            (issue["patient_id"], issue["lesion_id"], issue["field"]) for issue in issues
        })
        self.assertFalse((self.output / "summary.json").exists())

    def test_reference_mismatch_fails_without_success_summary(self):
        self.assertNotEqual(self.run_cli(), 0)
        self.assertFalse((self.output / "summary.json").exists())
        with (self.output / "errors.json").open(encoding="utf-8") as stream:
            self.assertIn("Frozen regression check failed", json.load(stream)["error"])

    def test_reference_serialization_uses_sorted_distinct_string_tuples(self):
        tuples = [("two", "2"), ("one", "é"), ("two", "10"), ("one", "é")]
        expected = "one\té\ntwo\t10\ntwo\t2\n".encode("utf-8")
        self.assertEqual(pipeline.set_sha256(tuples), hashlib.sha256(expected).hexdigest())

    def test_refuse_public_output_in_git_worktree(self):
        repo = self.root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
        (repo / ".gitignore").write_text("private/\n", encoding="utf-8")
        self.assertNotEqual(self.run_cli(repo / "publishable-output"), 0)
        self.assertFalse((repo / "publishable-output").exists())
        with patch.multiple(pipeline, IMAGE_REFERENCE={}, COHORT_REFERENCE={}):
            self.assertEqual(self.run_cli(repo / "private" / "audit"), 0)
        listed = subprocess.check_output(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "private/"]
        )
        self.assertEqual(listed, b"")

    def test_invalid_image_identifiers_fail_before_clinical_audit(self):
        for name in ("", "../outside.png", "ambiguous\tfield.png", self.rows[1]["img_id"]):
            with self.subTest(name=name):
                self.rows[0]["img_id"] = name
                self.write_metadata()
                with self.assertRaises(ValueError):
                    pipeline.image_index(self.metadata.read_bytes())

    def test_existing_output_and_input_overlap_are_rejected(self):
        self.output.mkdir()
        sentinel = self.output / "preserve-me.txt"
        sentinel.write_text("original", encoding="utf-8")
        with patch.multiple(pipeline, IMAGE_REFERENCE={}, COHORT_REFERENCE={}):
            self.assertNotEqual(self.run_cli(), 0)
            self.assertEqual(list(self.output.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")
            nested = self.dirs[0] / "new-audit"
            self.assertNotEqual(self.run_cli(nested), 0)
            self.assertFalse(nested.exists())


if __name__ == "__main__":
    unittest.main()
