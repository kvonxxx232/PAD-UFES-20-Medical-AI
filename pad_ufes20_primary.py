"""Run the frozen V1.1 complete-case primary comparison using the saved Stage 3 folds.

preflight verifies the private inputs, local checkpoint and split feasibility without
extracting features or fitting heads. train makes one shared feature pass, then
writes lesion-level OOF predictions and aggregate results in a fresh ignored directory.
compare reads two completed summaries and checks the protocol's numerical rerun tolerance.
"""

import argparse
import csv
import json
from importlib.metadata import version
import math
from pathlib import Path
import platform
import subprocess
import sys

from pad_ufes20_curate import file_sha256, reserve_output, write_csv, write_json
from pad_ufes20_folds import N_SPLITS, SEED
from pad_v11_nested import C_GRID, FitFailure, plan_splits, run_nested
from pad_v11_primary import primary_report
from pad_v11_training_inputs import extract_embeddings, load_inputs, verify_checkpoint


ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "private"
PROTOCOL_SHA256 = "f208036381abd1c355d0b4be2e096b0cc01f4b826c5d4889b40f3ca0d763a39d"
ANALYSIS = "V1.1 primary complete-case A/B"
OOF_FIELDS = ("patient_id", "lesion_id", "outer_fold", "label", "n_images", "p_A", "p_B")
GRID_FIELDS = ("outer_fold", "model", "C", "pooled_log_loss", "selected", "n_keys")
OOF = "oof_predictions.csv"
GRID = "inner_scores.csv"
FITS = "fit_diagnostics.json"
SUMMARY = "summary.json"


def verified_environment():
    if file_sha256(ROOT / "protocol_v1.1.md") != PROTOCOL_SHA256:
        raise ValueError("Working protocol differs from the verified frozen protocol.")
    packages = ("scikit-learn", "numpy", "scipy", "Pillow", "torch", "torchvision")
    versions = {name: version(name) for name in packages}
    if versions["scikit-learn"] != "1.9.1":
        raise ValueError("The frozen primary protocol specifies scikit-learn 1.9.1.")
    return dict(versions, Python=platform.python_version())


def reserve_private_run(output, inputs):
    """Require Git protection for every output and reserve the directory exclusively."""
    target = output.resolve()
    if not target.is_relative_to(ROOT) or target == ROOT:
        raise ValueError("Choose a new Git-ignored private directory in this project.")
    git = ["git", "-C", str(ROOT)]
    repo = subprocess.run([*git, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if repo.returncode or Path(repo.stdout.strip()).resolve() != ROOT:
        raise ValueError("Cannot verify this project's Git ignore rules.")
    names = [str((target / name).relative_to(ROOT)) for name in (OOF, GRID, FITS, SUMMARY)]
    ignored = subprocess.run([*git, "check-ignore", "-z", "--stdin"], capture_output=True,
                             input=b"".join(name.encode() + b"\0" for name in names))
    if ignored.returncode or set(ignored.stdout.split(b"\0")[:-1]) != {s.encode() for s in names}:
        raise ValueError("Every run artifact must be Git-ignored and untracked.")
    return reserve_output(target, inputs)


def write_records(path, fields, records):
    write_csv(path, fields, ([row[k] for k in fields] for row in records))


def train_primary(photos, images, checkpoint, provenance, versions, output, inputs):
    """Run primary training. Only summary.json marks a successfully completed run."""
    dest = reserve_private_run(output, inputs)
    embeddings = extract_embeddings(photos, images, checkpoint, batch_size=32, device="cpu")
    try:
        oof, tuning, diagnostics = run_nested(photos, embeddings)
    except FitFailure as exc:
        write_json(dest / FITS, exc.diagnostics)
        raise
    write_json(dest / FITS, diagnostics)
    report = primary_report(oof)
    write_records(dest / OOF, OOF_FIELDS, oof)
    write_records(dest / GRID, GRID_FIELDS, tuning)
    summary = {
        "analysis": ANALYSIS, "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256, "versions": versions,
        "inputs": provenance,
        "encoder": {"architecture": "frozen ResNet18", "embedding_size": embeddings.shape[1],
                    "weights": "ResNet18_Weights.IMAGENET1K_V1", "device": "cpu", "batch_size": 32},
        "inner_cv": {"splitter": "StratifiedGroupKFold", "folds": N_SPLITS, "shuffle": True,
                     "seed": SEED, "C_grid": list(C_GRID),
                     "selected": [r for r in tuning if r["selected"]]},
        "n_fits": len(diagnostics),
        "n_fits_with_warnings": sum(bool(r["warnings"]) for r in diagnostics),
        "primary": report,
        "output_sha256": {name: file_sha256(dest / name) for name in (OOF, GRID, FITS)},
    }
    write_json(dest / SUMMARY, summary)
    return summary


def compare_runs(reference, rerun):
    """Apply the numerical reproduction check to comparable completed primary runs."""
    if any(s["status"] != "complete" or s["analysis"] != ANALYSIS
           or s["protocol_sha256"] != PROTOCOL_SHA256 for s in (reference, rerun)):
        raise ValueError("Two completed frozen primary run summaries are required.")
    if reference["inputs"] != rerun["inputs"]:
        raise ValueError("The two runs have different input provenance.")
    original, repeated = (float(s["primary"]["delta_log_loss"]) for s in (reference, rerun))
    if not (math.isfinite(original) and math.isfinite(repeated)):
        raise ValueError("Expected finite primary Delta Log Loss values.")
    difference = abs(repeated - original)
    return {"delta_reference": original, "delta_rerun": repeated, "absolute_difference": difference,
            "tolerance": 0.001, "within_tolerance": difference <= 0.001}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    inputs = argparse.ArgumentParser(add_help=False)
    inputs.add_argument("--metadata", type=Path, required=True)
    inputs.add_argument("--images", type=Path, nargs=3, required=True, metavar=("PART1", "PART2", "PART3"))
    inputs.add_argument("--checkpoint", type=Path, required=True,
                        help="Local V1.0 ImageNet ResNet18 checkpoint.")
    inputs.add_argument("--curation", type=Path, default=PRIVATE / "v1.1-curation")
    inputs.add_argument("--review", type=Path, default=PRIVATE / "v1.1-review" / "recovered_review.json")
    inputs.add_argument("--fold-dir", type=Path, default=PRIVATE / "v1.1-outer-folds")
    actions.add_parser("preflight", parents=[inputs], help="Read-only input and split checks.")
    train = actions.add_parser("train", parents=[inputs], help="Fit and evaluate primary A and B.")
    train.add_argument("--output-dir", type=Path, default=PRIVATE / "v1.1-primary",
                       help="Fresh, Git-ignored private run directory.")
    compare = actions.add_parser("compare", help="Check numerical reproduction of Delta.")
    compare.add_argument("reference_summary", type=Path)
    compare.add_argument("rerun_summary", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "compare":
            if args.reference_summary.samefile(args.rerun_summary):
                raise ValueError("Select the separate reference and rerun summaries.")
            result = compare_runs(json.loads(args.reference_summary.read_bytes()),
                                  json.loads(args.rerun_summary.read_bytes()))
            print(json.dumps(result, sort_keys=True))
            return 0 if result["within_tolerance"] else 1
        versions = verified_environment()
        checkpoint_hash = verify_checkpoint(args.checkpoint)
        photos, images, provenance = load_inputs(
            args.curation, args.metadata, args.review, args.fold_dir, args.images)
        provenance["encoder_checkpoint_sha256"] = checkpoint_hash
        if args.action == "preflight":
            plan = plan_splits(photos)
            print(json.dumps({
                "status": "preflight passed", "primary": provenance["primary"],
                "outer_folds_verified": len(plan),
                "inner_partitions_verified": sum(len(part.inner) for part in plan),
                "splits_sha256": provenance["splits_sha256"], "checkpoint_sha256": checkpoint_hash,
            }, sort_keys=True))
            return 0
        summary = train_primary(
            photos, images, args.checkpoint, provenance, versions,
            args.output_dir, [*args.images, args.curation, args.review, args.fold_dir])
    except (OSError, KeyError) as exc:
        # Path and dictionary-key exceptions can contain original patient or image IDs.
        print(f"Primary analysis stopped ({type(exc).__name__}). Check local inputs, schema "
              "and filesystem access.", file=sys.stderr)
        return 1
    except (ValueError, csv.Error, RuntimeError) as exc:
        print(f"Primary analysis stopped: {exc}", file=sys.stderr)
        return 1
    print(f"Primary analysis complete: {summary['primary']['n_keys']} paired lesion predictions; "
          f"{summary['n_fits_with_warnings']} fits with warnings. See the private summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
