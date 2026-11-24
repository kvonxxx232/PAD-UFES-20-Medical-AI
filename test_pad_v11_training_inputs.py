"""Synthetic fixtures for the Stage 4 input and feature boundary."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import pad_v11_training_inputs as inputs
from pad_ufes20_folds import split_table_bytes


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class VerifiedInputsTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.parts = [self.root / str(i) for i in range(3)]
        for directory in self.parts:
            directory.mkdir()
        self.metadata = self.root / "metadata.csv"
        self.keys, self.folds = {}, {}
        self.quarantine = {("synth-x", "quarantined")}
        clinical = ["img_id,patient_id,lesion_id,age,region,grew,changed,bleed"]
        inventory = ["img_id,file_sha256"]

        def add_image(i, key, symptom="False"):
            basename = f"synthetic-{i}.png"
            path = self.parts[i % 3] / basename
            Image.new("RGB", (8, 8), (i, 20, 50)).save(path)
            clinical.append(f"{basename},{key[0]},{key[1]},38,ARM,True,{symptom},True")
            inventory.append(f"{basename},{digest(path.read_bytes())}")

        for i in range(10):
            key = (f"synth-{i}", "complete")
            self.keys[key] = (key[0], i % 2, True)
            self.folds[key] = 1 + i // 2
            add_image(i, key)
        add_image(10, ("synth-0", "complete"))
        incomplete = ("synth-0", "incomplete")
        self.keys[incomplete] = ("synth-0", 1, False)
        self.folds[incomplete] = 1
        add_image(11, incomplete, "UNK")
        add_image(12, next(iter(self.quarantine)))

        self.metadata.write_text("\n".join(clinical) + "\n", encoding="utf-8")
        (self.root / "image_inventory.csv").write_text("\n".join(inventory) + "\n", encoding="utf-8")
        (self.root / "summary.json").write_text('{"source_archive_sha256":null}', encoding="utf-8")
        provenance = {
            "metadata_sha256": digest(self.metadata.read_bytes()),
            "stage2_summary_sha256": digest((self.root / "summary.json").read_bytes()),
            "stage2_output_sha256": {"image_inventory.csv": digest((self.root / "image_inventory.csv").read_bytes())},
        }
        split = split_table_bytes(self.folds)
        (self.root / "splits.csv").write_bytes(split)
        summary = {"protocol": "protocol-v1.1-frozen", "level": "outer", "splitter": "StratifiedGroupKFold",
                   "n_splits": 5, "seed": 20261124, "shuffle": True,
                   "stratum": "2 * binary_label + int(symptom_complete)",
                   "splits_sha256": digest(split), "input_provenance": provenance}
        summary_raw = json.dumps(summary).encode()
        (self.root / "fold_summary.json").write_bytes(summary_raw)
        primary_counts = {"keys": 10, "patients": 10, "positive": 5, "negative": 5, "images": 11}
        self.enterContext(patch.object(inputs, "SPLITS_SHA256", digest(split)))
        self.enterContext(patch.object(inputs, "FOLD_SUMMARY_SHA256", digest(summary_raw)))
        self.enterContext(patch.dict(inputs.COHORT_REFERENCE, {"primary": primary_counts}))
        self.enterContext(patch.object(inputs, "load_curation", return_value=(self.keys, self.quarantine, provenance)))

    def load(self):
        return inputs.load_inputs(self.root, self.metadata, self.root / "review.json", self.root, self.parts)

    def test_primary_includes_all_photos_and_correct_features(self):
        photos, paths, provenance = self.load()
        self.assertEqual((len(photos), len(paths), provenance["verified_pngs"]), (11, 11, 13))
        self.assertEqual({p.lesion_id for p in photos}, {"complete"})
        self.assertEqual(sum(p.patient_id == "synth-0" for p in photos), 2)
        self.assertTrue(all((p.age, p.site, p.symptoms) == (38, "ARM", (1, 0, 1)) for p in photos))
        self.assertTrue(all(p.fold == self.folds[p.patient_id, p.lesion_id] for p in photos))
        self.assertEqual(provenance["primary"]["positive"], 5)
        self.assertIsNone(provenance["source_archive_sha256"])

    def test_even_ineligible_png_and_metadata_are_authenticated(self):
        path = self.parts[0] / "synthetic-12.png"
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "PNG differs"):
            self.load()
        path.write_bytes(original)
        self.metadata.write_bytes(self.metadata.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "metadata SHA-256"):
            self.load()

    def test_changed_outer_file_refused_and_component_isolation_checked(self):
        path = self.root / "splits.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "assignments SHA-256"):
            self.load()
        self.keys["synth-2", "complete"] = ("synth-0", 0, True)
        with self.assertRaisesRegex(ValueError, "crosses saved outer"):
            inputs._fold_assignments(split_table_bytes(self.folds), self.keys, self.quarantine)

    def test_image_changed_after_preflight_is_rejected_at_decode(self):
        photos, paths, _ = self.load()
        paths[photos[0].image_id].path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "PNG at extraction SHA-256"):
            inputs._ImageRecords(photos, paths)[0]

    def test_encoder_matches_v1_rgb_transform_and_512_features_on_synthetic_png(self):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
        from pad_ufes20_full import Photos

        photos, paths, _ = self.load()
        path = self.root / "synthetic-weights.pth"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(117)
            reference = resnet18(weights=None)
        torch.save(reference.state_dict(), path)
        with self.assertRaisesRegex(ValueError, "published V1.0"):
            inputs.verify_checkpoint(path)
        # A test-only checkpoint allows a fully offline test with a real ResNet18.
        with patch.object(inputs, "CHECKPOINT_SHA256", digest(path.read_bytes())):
            self.assertEqual(inputs.verify_checkpoint(path), inputs.CHECKPOINT_SHA256)
            subset = [photos[-1], photos[0]]
            actual = inputs.extract_embeddings(subset, paths, path, batch_size=2)
            reference.fc = torch.nn.Identity()
            reference.eval()
            images = Photos([(paths[p.image_id].path,) for p in subset], ResNet18_Weights.DEFAULT.transforms())
            with torch.inference_mode():
                expected = reference(torch.stack([images[i] for i in range(len(images))])).numpy()
            self.assertEqual(actual.shape, (2, 512))
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
