"""Run the frozen V1.1 curation, splits, H1 and prespecified secondary analyses."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

# A dry run must not populate the checkout with bytecode.
sys.dont_write_bytecode = True

from pad_ufes20_curate import file_sha256, image_index, write_json
from pad_ufes20_primary import PRIVATE, ROOT, reserve_private_run, verified_environment
from pad_v11_fold_inputs import REVIEW_SHA256
from pad_v11_images import _inventory
from pad_v11_s2_inputs import H1_OOF_SHA256, H1_PROVENANCE_SHA256, H1_SUMMARY_SHA256
from pad_v11_training_inputs import FOLD_SUMMARY_SHA256, SPLITS_SHA256, verify_checkpoint


def check_hash(path, expected, label):
    if file_sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch.")


def inventory_hash(metadata, image_dirs):
    """Recreate the canonical Stage 2 inventory in memory without clinical fields."""
    index = image_index(metadata)
    try:
        paths = _inventory(index, image_dirs)
    except ValueError:
        raise ValueError("The three image directories do not match the metadata inventory.") from None
    data = io.StringIO(newline="")
    writer = csv.writer(data, lineterminator="\n")
    writer.writerow(("img_id", "patient_id", "lesion_id", "file_sha256"))
    for name in sorted(index):
        writer.writerow((name, *index[name], file_sha256(paths[name])))
    return hashlib.sha256(data.getvalue().encode("utf-8")).hexdigest(), len(paths)


def preflight(args):
    output = args.output_dir
    if (output.exists() or output.is_symlink() or PRIVATE not in output.resolve().parents
            or any(output.resolve().is_relative_to(p) for p in [args.primary_reference, *args.images])):
        raise ValueError("Choose an unused output directory under private/, separate from input directories.")
    relative = output.resolve().relative_to(ROOT)
    git = ["git", "--no-optional-locks", "-C", str(ROOT)]
    ignored = subprocess.run([*git, "check-ignore", "-q", "--", str(relative) + "/"])
    if ignored.returncode or subprocess.check_output([*git, "ls-files", "-z", "--", str(relative)]):
        raise ValueError("The output directory must be untracked and Git-ignored.")

    archive = args.primary_reference
    check_hash(archive / "summary.json", H1_SUMMARY_SHA256, "Original H1 summary")
    check_hash(archive / "oof_predictions.csv", H1_OOF_SHA256, "Original H1 OOF")
    check_hash(archive / "run_provenance.json", H1_PROVENANCE_SHA256, "Original H1 run record")
    reference = json.loads((archive / "summary.json").read_bytes())
    inputs = reference["inputs"]
    versions = verified_environment()
    if versions != reference["versions"]:
        raise ValueError("Install the Python and package versions recorded in the original H1 run.")
    check_hash(args.review, REVIEW_SHA256, "Recovered label-blinded review")
    checkpoint_hash = verify_checkpoint(args.checkpoint)
    raw = args.metadata.read_bytes()
    if hashlib.sha256(raw).hexdigest() != inputs["metadata_sha256"]:
        raise ValueError("Original metadata SHA-256 mismatch.")
    digest, count = inventory_hash(raw, args.images)
    if digest != inputs["stage2_output_sha256"]["image_inventory.csv"] or count != inputs["verified_pngs"]:
        raise ValueError("Per-image checksums or inventory count differ from the original inputs.")
    report_hash = file_sha256(args.report)
    return {
        "software": versions, "platform": platform.platform(), "python": sys.executable,
        "git_commit": subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip(),
        "protocol_sha256": reference["protocol_sha256"], "launcher_sha256": file_sha256(Path(__file__)),
        "metadata_sha256": inputs["metadata_sha256"], "verified_pngs": count,
        "image_inventory_sha256": digest, "checkpoint_sha256": checkpoint_hash,
        "report_sha256": report_hash, "review_sha256": REVIEW_SHA256,
        "historical_h1_provenance_sha256": H1_PROVENANCE_SHA256,
        "historical_source_archive_sha256": inputs["source_archive_sha256"],
    }


def run_step(output, name, script, *arguments, stdout_name=None):
    """Log exact private arguments and times. A failing process halts the chain."""
    command = [sys.executable, "-B", "-s", str(ROOT / script), *map(str, arguments)]
    stdout_name = stdout_name or name + ".stdout.log"
    with (output / "commands.jsonl").open("a", encoding="utf-8") as journal:
        def record(**fields):
            journal.write(json.dumps({"step": name, "utc": datetime.now(timezone.utc).isoformat(),
                                      **fields}) + "\n")
            journal.flush()

        record(argv=command, stdout=stdout_name)
        print(f"Running {name}", flush=True)
        environment = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                           GIT_OPTIONAL_LOCKS="0")
        environment.pop("PYTHONPATH", None)
        with ((output / stdout_name).open("x") as stdout,
              (output / (name + ".stderr.log")).open("x") as stderr):
            proc = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        record(exit_code=proc.returncode)
        if proc.returncode:
            raise ValueError(f"{name} stopped with exit code {proc.returncode}; inspect the private logs.")


def run(args, evidence):
    output = reserve_private_run(args.output_dir, [args.primary_reference, *args.images])
    write_json(output / "inputs.json", evidence)
    curation = output / "curation"
    folds = output / "outer-folds"
    primary = output / "h1"
    shared = ["--metadata", args.metadata, "--curation", curation, "--review", args.review]
    clinical = [*shared, "--fold-dir", folds]
    fit = [*clinical, "--images", *args.images, "--checkpoint", args.checkpoint]
    run_step(output, "curation", "pad_ufes20_curate.py", "--metadata", args.metadata,
             "--images", *args.images, "--output-dir", curation)
    run_step(output, "folds", "pad_ufes20_folds.py", *shared, "--output-dir", folds)
    run_step(output, "verify-folds", "pad_ufes20_folds.py", "--verify", *shared, "--output-dir", folds)
    check_hash(folds / "splits.csv", SPLITS_SHA256, "Regenerated split")
    check_hash(folds / "fold_summary.json", FOLD_SUMMARY_SHA256, "Regenerated split summary")
    run_step(output, "h1-preflight", "pad_ufes20_primary.py", "preflight", *fit)
    run_step(output, "h1-train", "pad_ufes20_primary.py", "train", *fit, "--output-dir", primary)
    run_step(output, "compare-h1", "pad_ufes20_primary.py", "compare",
             args.primary_reference / "summary.json", primary / "summary.json", stdout_name="h1-comparison.json")

    # The secondary CLIs authenticate exact historical hashes in addition to the
    # primary numerical tolerance. The historical record describes the original run.
    view = output / "authenticated-h1"
    sources = {"summary.json": (primary, H1_SUMMARY_SHA256),
               "oof_predictions.csv": (primary, H1_OOF_SHA256),
               "run_provenance.json": (args.primary_reference, H1_PROVENANCE_SHA256)}
    view.mkdir(mode=0o700)
    manifest = {}
    for name, (directory, digest) in sources.items():
        source = directory / name
        check_hash(source, digest, f"Secondary authentication for {name}")
        (view / name).symlink_to(os.path.relpath(source, view))
        manifest[name] = {"source": str(source), "sha256": digest}
    write_json(output / "h1-reference-sources.json", manifest)

    for analysis in ("s2", "s4"):
        script = f"pad_ufes20_{analysis}.py"
        arguments = [*fit, "--primary-dir", view]
        run_step(output, analysis + "-preflight", script, "preflight", *arguments)
        run_step(output, analysis + "-train", script, "train", *arguments, "--output-dir", output / analysis)
    check_hash(args.report, evidence["report_sha256"], "Technical report")
    run_step(output, "secondary", "pad_ufes20_secondary.py", *clinical, "--primary-dir", view,
             "--report", args.report, stdout_name="secondary.json")
    secondary = json.loads((output / "secondary.json").read_bytes())
    write_json(output / "completed.json", {
        "status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "h1_comparison": json.loads((output / "h1-comparison.json").read_bytes()),
        "secondary_report_check": secondary["report_comparison"],
        "outputs_sha256": {name: file_sha256(output / name) for name in (
            "curation/summary.json", "outer-folds/splits.csv", "outer-folds/fold_summary.json",
            "h1/summary.json", "h1/oof_predictions.csv", "s2/summary.json", "s4/summary.json",
            "s4/s1/summary.json", "s4/s3/summary.json", "s4/s10/summary.json",
            "secondary.json", "inputs.json", "h1-comparison.json", "h1-reference-sources.json",
            "commands.jsonl")},
    })
    print("Completed curation, folds, H1, S2, S4 and secondary report checks.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and destination without writing outputs.")
    parser.add_argument("--metadata", required=True, type=Path, help="Original V1 metadata CSV")
    parser.add_argument("--images", required=True, type=Path, nargs=3, metavar=("PART1", "PART2", "PART3"))
    parser.add_argument("--checkpoint", required=True, type=Path, help="Frozen ImageNet ResNet18 checkpoint")
    parser.add_argument("--review", required=True, type=Path, help="Private recovered label-blinded review JSON")
    parser.add_argument("--primary-reference", required=True, type=Path, help="Sealed historical H1 archive directory")
    parser.add_argument("--report", required=True, type=Path, help="English V1.1 technical report to verify")
    parser.add_argument("--output-dir", required=True, type=Path, help="Fresh, Git-ignored directory under private/")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        for key in ("metadata", "checkpoint", "review", "primary_reference", "report"):
            setattr(args, key, getattr(args, key).resolve(strict=True))
        args.images = [path.resolve(strict=True) for path in args.images]
        evidence = preflight(args)
        if args.dry_run:
            print(json.dumps(evidence, indent=2, sort_keys=True))
            return 0
        run(args, evidence)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        # OS exceptions may contain original image or patient identifiers.
        print(str(exc) if isinstance(exc, ValueError) else f"Reproduction stopped ({type(exc).__name__}).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
