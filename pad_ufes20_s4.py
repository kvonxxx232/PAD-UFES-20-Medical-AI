"""Frozen S4 clinical-feature scaling on the original complete-case H1 partitions."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from pad_ufes20_curate import file_sha256, write_json
from pad_ufes20_primary import (
    FITS, GRID, GRID_FIELDS, OOF, OOF_FIELDS, PRIVATE, PROTOCOL_SHA256, ROOT, SUMMARY,
    reserve_private_run, verified_environment, write_records,
)
from pad_ufes20_s2 import partition_audit, source_digests as shared_source_digests
from pad_v11_nested import C_GRID, FitFailure, plan_splits
from pad_v11_primary import _loss_summary, lesion_losses
from pad_v11_s4_inputs import load_s4_inputs
from pad_v11_s4_nested import FACTORS, run_factor
from pad_v11_training_inputs import extract_embeddings, verify_checkpoint


def source_digests():
    sources = shared_source_digests()
    for name in ("pad_ufes20_s4.py", "pad_v11_s4_inputs.py", "pad_v11_s4_nested.py"):
        sources[name] = file_sha256(ROOT / name)
    return sources


def audited_plan(photos, reference):
    plan = tuple(plan_splits(photos))
    counts = partition_audit(photos, {p.key for p in photos}, reference, plan)
    # Ordered photo IDs specify the index space for the shared, in-memory fit plan.
    membership = {"photos": [p.image_id for p in photos],
                  "partitions": [[p.train, p.held_out] for f in plan for p in (f.outer, *f.inner)]}
    digest = hashlib.sha256(json.dumps(membership, separators=(",", ":")).encode()).hexdigest()
    return plan, counts, digest


def score_factor(rows, reference_rows):
    fields = ("patient_id", "lesion_id", "outer_fold", "label", "n_images")
    def identity(table):
        return sorted(tuple(r[f] for f in fields) for r in table)
    if identity(rows) != identity(reference_rows):
        raise ValueError("Outer evaluation keys, folds, labels or image counts differ from H1.")
    def losses(table):
        labels = [r["label"] for r in table]
        return _loss_summary(*(lesion_losses(labels, [r[f] for r in table]) for f in ("p_A", "p_B")))
    folds = []
    for fold in range(1, 6):
        held = [r for r in rows if r["outer_fold"] == fold]
        folds.append({"outer_fold": fold, "keys": len(held), **losses(held)})
    return {**losses(rows), "folds": folds, "folds_favouring_B": sum(f["delta_log_loss"] < 0 for f in folds)}


def run_series(dest, photos, embeddings, plan, reference_rows, reference_delta, protected):
    reports = []
    for factor in FACTORS:
        folder = reserve_private_run(dest / f"s{factor}", protected)
        print(f"Starting prespecified factor {factor}.", flush=True)
        try:
            rows, grid, diagnostics = run_factor(photos, embeddings, plan, factor)
        except FitFailure as exc:
            write_json(folder / FITS, exc.diagnostics)
            raise
        write_json(folder / FITS, diagnostics)
        if not all(f["converged"] and not f["iteration_cap_reached"] and f["error"] is None
                   for f in diagnostics) or len(diagnostics) != 260:
            raise ValueError("Incomplete or unconfirmed fitting diagnostics.")
        result = score_factor(rows, reference_rows)
        write_records(folder / OOF, OOF_FIELDS, rows)
        write_records(folder / GRID, ("factor", *GRID_FIELDS), grid)
        iterations = [i for f in diagnostics for i in f["n_iter"]]
        report = {"factor": factor, **result, "selected_C": [r for r in grid if r["selected"]],
                  "n_fits": len(diagnostics), "fits_with_warnings": sum(bool(f["warnings"]) for f in diagnostics),
                  "min_iterations": min(iterations), "max_iterations": max(iterations),
                  "output_sha256": {name: file_sha256(folder / name) for name in (OOF, GRID, FITS)}}
        if factor == 1:
            difference = abs(result["delta_log_loss"] - reference_delta)
            reproduced = difference <= 0.001
            report["reproduction"] = {"reference_delta": reference_delta, "absolute_difference": difference,
                                      "tolerance": 0.001, "passed": reproduced}
            if not reproduced:
                write_json(folder / SUMMARY, dict(report, status="stopped: H1 reproduction mismatch"))
                raise ValueError("Factor 1 failed the frozen H1 numerical reproduction check.")
        write_json(folder / SUMMARY, dict(report, status="complete"))
        reports.append(report)
        print(f"Factor {factor} complete; {len(diagnostics)} fits confirmed.", flush=True)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train"))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images", nargs=3, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--curation", type=Path, default=PRIVATE / "v1.1-curation")
    parser.add_argument("--review", type=Path, default=PRIVATE / "v1.1-review" / "recovered_review.json")
    parser.add_argument("--fold-dir", type=Path, default=PRIVATE / "v1.1-outer-folds")
    parser.add_argument("--primary-dir", type=Path, default=PRIVATE / "v1.1-primary-stage4b")
    parser.add_argument("--output-dir", type=Path, default=PRIVATE / "v1.1-secondary-s4")
    args = parser.parse_args(argv)
    try:
        versions = verified_environment()
        checkpoint_digest = verify_checkpoint(args.checkpoint)
        photos, images, reference, delta, provenance = load_s4_inputs(
            args.curation, args.metadata, args.review, args.fold_dir, args.images, args.primary_dir)
        plan, counts, plan_hash = audited_plan(photos, reference)
        if args.action == "preflight":
            print(json.dumps({"primary": provenance["primary"], "scored_keys_by_outer_fold":
                              [f["scored_complete_keys"] for f in counts], "plan_sha256": plan_hash,
                              "inner_partitions": sum(len(f.inner) for f in plan), "reference_delta": delta}))
            return 0
        protected = [args.primary_dir, args.curation, args.fold_dir, args.review, *args.images]
        dest = reserve_private_run(args.output_dir, protected)
        started = datetime.now(timezone.utc).isoformat()
        sources = source_digests()
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        embeddings = extract_embeddings(photos, images, args.checkpoint, batch_size=32, device="cpu")
        embeddings.setflags(write=False)
        encoder = {"weights": "frozen ResNet18 IMAGENET1K_V1", "checkpoint_sha256": checkpoint_digest,
                   "device": "cpu", "batch_size": 32, "embedding_shape": list(embeddings.shape),
                   "embedding_dtype": str(embeddings.dtype),
                   "row_major_embedding_sha256": hashlib.sha256(embeddings.tobytes(order="C")).hexdigest()}
        reports = run_series(dest, photos, embeddings, plan, reference, delta, protected)
        if source_digests() != sources:
            raise ValueError("Source files changed during S4.")
        write_json(dest / SUMMARY, {
            "analysis": "S4: prespecified descriptive clinical-feature scaling sensitivity",
            "status": "complete", "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "versions": versions, "inputs": provenance, "encoder": encoder, "source_sha256": sources,
            "git_base_commit": commit, "protocol_sha256": PROTOCOL_SHA256, "partition_audit": counts,
            "plan_sha256": plan_hash, "inner_seed": 20261124, "inner_stratification": "binary label",
            "C_grid": list(C_GRID), "factors": reports,
            "factor_summary_sha256": {str(s): file_sha256(dest / f"s{s}" / SUMMARY) for s in FACTORS},
        })
    except (OSError, KeyError) as exc:
        print(f"S4 stopped ({type(exc).__name__}). Check private inputs and filesystem access.", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, csv.Error) as exc:
        print(f"S4 stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
