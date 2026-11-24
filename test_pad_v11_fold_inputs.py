"""Synthetic checks for binding a review record to the Stage 2 queue."""

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pad_v11_fold_inputs as inputs


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.queue = (
            "image_a,image_b,hamming_distance,status\n"
            "a.png,b.png,4,pending_review\n"
            "a.png,c.png,8,pending_review\n"
            "b.png,c.png,0,confirmed_rgb\n"
        ).encode()
        self.confirmed = {("b.png", "c.png"): "rgb_equal"}
        source = {
            "session": "fictional-session", "call_id": "fictional-call", "message_id": "fictional-message",
            "reported_utc": "2000-01-01T00:00:00Z", "source_line": 1, "parent_delivery_line": 2,
            "report": "One stripe versus two stripes. One dot versus two dots.",
        }
        records = []
        for i, (name, distance, evidence) in enumerate((
            ("b.png", 4, "One stripe versus two stripes."), ("c.png", 8, "One dot versus two dots."),
        )):
            records.append({
                "blind_id": f"example-{i}", "image_a": "a.png", "image_b": name,
                "hamming_distance": distance, "reviewer": "review-agent", "source_locator": f"item {i}",
                "evidence": evidence, "disposition": "Visually similar but distinct",
            })
        self.bundle = {
            "schema_version": 1, "stage2_queue_sha256": hashlib.sha256(self.queue).hexdigest(),
            "human_signoff": False, "review_nature": "AI-assisted label-blinded visual review",
            "primary_session": "primary", "pair_mapping_call": "crosswalk", "sources": {"review-agent": source},
            "records": records,
        }

    def verify(self, bundle):
        return inputs.validate_review(self.queue, json.dumps(bundle).encode(), self.confirmed)

    def test_complete_coverage_and_deterministic_rgb(self):
        self.assertEqual(self.verify(self.bundle), 2)
        with self.assertRaises(ValueError):
            inputs.validate_review(self.queue, json.dumps(self.bundle).encode(), {})

    def test_missing_repeated_and_extra_pairs(self):
        first, second = self.bundle["records"]
        for bad_rows in ([first], [first, first], [first, second, dict(first, image_b="d.png")]):
            with self.subTest(rows=len(bad_rows)), self.assertRaises(ValueError):
                self.verify(dict(self.bundle, records=bad_rows))

    def test_review_fields_must_identify_pair_source_evidence_and_disposition(self):
        for field, value in (
            ("image_a", "different.png"), ("hamming_distance", 6), ("disposition", "Uncertain"),
            ("reviewer", "absent-agent"), ("source_locator", ""), ("evidence", " "),
            ("evidence", "absent from the historical note"), ("blind_id", "example-1"),
        ):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(self.bundle)
                changed["records"][0][field] = value
                with self.assertRaises(ValueError):
                    self.verify(changed)

    def test_missing_provenance_changed_queue_or_claimed_human_signoff(self):
        for field, value in (
            ("stage2_queue_sha256", "different"), ("sources", {}), ("human_signoff", True),
            ("review_nature", "Human review"), ("primary_session", ""), ("pair_mapping_call", ""),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(dict(self.bundle, **{field: value}))
        changed = copy.deepcopy(self.bundle)
        del changed["sources"]["review-agent"]["message_id"]
        with self.assertRaises(ValueError):
            self.verify(changed)

    def test_load_stops_on_unverified_review_before_reading_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            contents = dict.fromkeys(inputs.STAGE2_FILES, b"")
            contents["confirmed_pairs.csv"] = b"image_a,image_b,rule_id\nb.png,c.png,rgb_equal\n"
            contents["phash_candidates.csv"] = self.queue
            summary = {
                "status": "passed", "protocol": "protocol-v1.1-frozen", "image_audit": {},
                "output_sha256": {n: hashlib.sha256(v).hexdigest() for n, v in contents.items()},
            }
            for name, raw in contents.items():
                (root / name).write_bytes(raw)
            (root / "review.json").write_text(json.dumps(self.bundle), encoding="utf-8")
            ref = {"pair_set_sha256": inputs.set_sha256(self.confirmed)}
            summary["image_audit"] = ref
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with patch.object(inputs, "IMAGE_REFERENCE", ref):
                with self.assertRaisesRegex(ValueError, "verified recovery"):
                    inputs.load_curation(root, root / "no-metadata", root / "review.json")


if __name__ == "__main__":
    unittest.main()
