"""Synthetic reference checks at the H1/S4 boundary."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pad_v11_s4_inputs as inputs
from pad_v11_nested import Photo


def digest(data):
    return hashlib.sha256(data).hexdigest()


class S4InputsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.photos = [Photo("synthetic", "patient", "lesion", "group", 2, 1,
                             37, "ARM", (1, 0, 1))]
        self.shared = {"primary": {"keys": 1}}
        self.loaded = self.enterContext(patch.object(
            inputs, "load_inputs", return_value=(self.photos, {}, self.shared)))
        raw = (b"patient_id,lesion_id,outer_fold,label,n_images,p_A,p_B\n"
               b"patient,lesion,2,1,1,0.4,0.6\n")
        (self.root / "oof_predictions.csv").write_bytes(raw)
        source = {**self.shared, "encoder_checkpoint_sha256": inputs.CHECKPOINT_SHA256}
        outputs = {"oof_predictions.csv": digest(raw)}
        summary = {"status": "complete", "analysis": inputs.ANALYSIS, "inputs": source,
                   "protocol_sha256": inputs.PROTOCOL_SHA256, "output_sha256": outputs,
                   "primary": {"delta_log_loss": -0.4054651081081643}}
        summary_bytes = json.dumps(summary).encode()
        (self.root / "summary.json").write_bytes(summary_bytes)
        record = {"input_provenance": source, "code_commit": inputs.H1_COMMIT,
                  "commit_at_end": inputs.H1_COMMIT, "exit_code": 0,
                  "summary_sha256": digest(summary_bytes), "output_sha256": outputs}
        record_bytes = json.dumps(record).encode()
        (self.root / "run_provenance.json").write_bytes(record_bytes)
        for name, data in (("H1_OOF_SHA256", raw), ("H1_SUMMARY_SHA256", summary_bytes),
                           ("H1_PROVENANCE_SHA256", record_bytes)):
            self.enterContext(patch.object(inputs, name, digest(data)))

    def load(self):
        return inputs.load_s4_inputs(self.root, self.root, self.root, self.root, [], self.root)

    def test_preserves_photo_order_and_unrounded_original_delta(self):
        photos, _, rows, delta, provenance = self.load()
        self.assertIs(photos, self.photos)
        self.assertEqual(delta, -0.4054651081081643)
        self.assertEqual(rows[0]["outer_fold"], 2)
        self.assertEqual(rows[0]["label"], 1)
        self.assertEqual(rows[0]["p_A"], 0.4)
        self.assertEqual(provenance["h1_oof_sha256"], inputs.H1_OOF_SHA256)

    def test_rejects_changed_oof_or_input_population(self):
        self.loaded.return_value = self.photos, {}, {"primary": {"keys": 99}}
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.load()
        self.loaded.return_value = self.photos, {}, self.shared
        with (self.root / "oof_predictions.csv").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.load()
