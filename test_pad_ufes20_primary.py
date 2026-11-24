"""Synthetic integration and private output tests for the Stage 4A entry point."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import pad_ufes20_primary as runner
from pad_v11_nested import FitFailure, Photo


def synthetic_primary():
    photos, vectors = [], []
    rng = np.random.default_rng(42)
    for fold in range(1, 6):
        for family in range(2):
            for member in range(2):
                group = f"unit-{fold}-{family}"
                patient = f"{group}-{member}"
                for label in (0, 1):
                    # Deliberately reuse lesion numbers across patients and vary photos per key.
                    symptoms = (family, member, (fold + label) % 2)
                    for shot in range(1 + member):
                        photos.append(Photo(f"{patient}-{label}-{shot}", patient, str(label),
                                            group, fold, label, 30 + fold + family,
                                            ("ARM", "BACK")[family], symptoms))
                        vectors.append(rng.normal(size=4))
    return photos, np.array(vectors)


class PrimaryRunTest(unittest.TestCase):
    def test_primary_run_from_cli_and_no_overwrite(self):
        photos, embeddings = synthetic_primary()
        with TemporaryDirectory(prefix="stage4a-") as temp:
            # Exercise Git ignore protection in a throwaway worktree with synthetic data.
            root = self.private_project(temp)
            output = root / "private" / "synthetic-run"
            fake_metadata = root / "synthetic.csv"
            checkpoint = root / "synthetic.pth"
            provenance = {"primary": {"keys": 40}, "splits_sha256": "synthetic-split"}
            args = ["train", "--metadata", str(fake_metadata), "--images", "a", "b", "c",
                    "--checkpoint", str(checkpoint), "--output-dir", str(output)]
            with (patch.object(runner, "ROOT", root),
                  patch.object(runner, "verified_environment", return_value={"test": "synthetic"}),
                  patch.object(runner, "verify_checkpoint", return_value="synthetic-checkpoint"),
                  patch.object(runner, "load_inputs", return_value=(photos, {}, provenance)),
                  patch.object(runner, "extract_embeddings", return_value=embeddings) as extract,
                  redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(args), 0)
                extract.assert_called_once_with(photos, {}, checkpoint, batch_size=32, device="cpu")
                before = {path.name: path.read_bytes() for path in output.iterdir()}
                with self.assertRaisesRegex(ValueError, "already exists"):
                    runner.train_primary(photos, {}, checkpoint, provenance, {}, output, [])
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
                self.assertEqual(extract.call_count, 1)

            summary = json.loads(before[runner.SUMMARY])
            self.assertEqual(summary["inputs"]["encoder_checkpoint_sha256"], "synthetic-checkpoint")
            self.assertEqual(summary["primary"]["n_patients"], 20)
            self.assertEqual(summary["primary"]["n_images"], 60)
            self.assertEqual(summary["primary"]["bootstrap"]["resamples"], 2000)
            self.assertEqual(summary["n_fits"], 260)
            self.assertEqual(len(summary["inner_cv"]["selected"]), 10)
            for name, digest in summary["output_sha256"].items():
                self.assertEqual(runner.file_sha256(output / name), digest)
            with (output / runner.OOF).open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), runner.OOF_FIELDS)
                rows = list(reader)
            expected = Counter(p.key for p in photos)
            self.assertEqual(len(rows), len(expected))
            self.assertEqual({(r["patient_id"], r["lesion_id"]): int(r["n_images"]) for r in rows},
                             expected)
            with (output / runner.GRID).open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 50)
            self.assertEqual(len(json.loads(before[runner.FITS])), 260)
            self.assertTrue(all(float(r[p]) >= 0 and float(r[p]) <= 1
                                for r in rows for p in ("p_A", "p_B")))
            self.assertEqual(output.stat().st_mode & 0o077, 0)

    @staticmethod
    def private_project(temp):
        root = Path(temp).resolve() / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        (root / ".gitignore").write_text("private/\n", encoding="utf-8")
        return root

    def test_destination_restrictions_and_failure_evidence(self):
        photos, embeddings = synthetic_primary()
        with TemporaryDirectory(prefix="stage4a-") as temp:
            root = self.private_project(temp)
            with patch.object(runner, "ROOT", root):
                protected = root / "private" / "inputs"
                for invalid in (root / "public", Path(temp), protected / "overlap"):
                    with self.assertRaises(ValueError):
                        runner.reserve_private_run(invalid, [protected])
                out = root / "private" / "failed"
                diagnostics = [{"converged": False, "warnings": ["synthetic warning"]}]
                with (patch.object(runner, "extract_embeddings", return_value=embeddings),
                      patch.object(runner, "run_nested", side_effect=FitFailure(diagnostics)),
                      self.assertRaises(FitFailure)):
                    runner.train_primary(photos, {}, Path("test"), {}, {}, out, [])
                self.assertEqual([p.name for p in out.iterdir()], [runner.FITS])
                self.assertEqual(json.loads((out / runner.FITS).read_bytes()), diagnostics)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    runner.reserve_private_run(out, [])

    def test_preflight_never_extracts_or_fits(self):
        photos, _ = synthetic_primary()
        provenance = {"primary": {"keys": 40}, "splits_sha256": "synthetic"}
        stream = io.StringIO()
        with (patch.object(runner, "verified_environment", return_value={}),
              patch.object(runner, "verify_checkpoint", return_value="synthetic"),
              patch.object(runner, "load_inputs", return_value=(photos, {}, provenance)),
              patch.object(runner, "train_primary") as train,
              patch.object(runner, "extract_embeddings") as extract,
              redirect_stdout(stream)):
            result = runner.main(["preflight", "--metadata", "mock", "--images", "a", "b", "c",
                                  "--checkpoint", "mock"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stream.getvalue())["inner_partitions_verified"], 25)
        train.assert_not_called()
        extract.assert_not_called()

    def test_numeric_reproduction_uses_unrounded_delta_and_matching_inputs(self):
        from copy import deepcopy

        reference = {"status": "complete", "analysis": runner.ANALYSIS,
                     "protocol_sha256": runner.PROTOCOL_SHA256, "inputs": {"splits_sha256": "fixture"},
                     "primary": {"delta_log_loss": 0.0}}
        repeated = deepcopy(reference)
        for delta, expected in [(0.001, True), (np.nextafter(0.001, 1), False), (-0.002, False)]:
            repeated["primary"]["delta_log_loss"] = delta
            report = runner.compare_runs(reference, repeated)
            self.assertEqual(report["within_tolerance"], expected)
            self.assertEqual(report["absolute_difference"], abs(delta))
        repeated["inputs"]["splits_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "input provenance"):
            runner.compare_runs(reference, repeated)

    def test_missing_png_does_not_expose_private_filename(self):
        errors = io.StringIO()
        with (patch.object(runner, "verified_environment", return_value={}),
              patch.object(runner, "verify_checkpoint", return_value="synthetic"),
              patch.object(runner, "load_inputs", side_effect=FileNotFoundError("synthetic-subject.png")),
              redirect_stderr(errors)):
            result = runner.main(["preflight", "--metadata", "mock", "--images", "a", "b", "c",
                                  "--checkpoint", "mock"])
        self.assertEqual(result, 1)
        self.assertNotIn("synthetic-subject", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
