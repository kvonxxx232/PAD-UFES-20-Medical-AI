"""Frozen V1.1 secondary analysis S2 on all curated training keys.

The preflight command checks the fixed outer assignments, all inner partitions
and H1 test membership without fitting. Train writes one fresh private run.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from pad_ufes20_curate import file_sha256, write_json
from pad_ufes20_folds import N_SPLITS, SEED
from pad_ufes20_primary import (
    FITS, GRID, GRID_FIELDS, OOF, OOF_FIELDS, PRIVATE, PROTOCOL_SHA256, ROOT, SUMMARY,
    reserve_private_run, verified_environment, write_records,
)
from pad_v11_nested import C_GRID, FitFailure
from pad_v11_primary import _loss_summary, lesion_losses
from pad_v11_s2_inputs import load_s2_inputs
from pad_v11_s2_nested import plan_s2, run_s2_nested
from pad_v11_training_inputs import extract_embeddings, verify_checkpoint


def partition_audit(photos, complete_keys, h1_rows, plan):
    """Check the plan against the original rows, including unscored held-out keys."""
    h1_folds = {(r["patient_id"], r["lesion_id"]): r["outer_fold"] for r in h1_rows}
    if set(h1_folds) != complete_keys or len(h1_rows) != len(complete_keys):
        raise ValueError("Primary eligibility differs from the recorded H1 cohort.")
    if [s.fold for s in plan] != list(range(1, N_SPLITS + 1)):
        raise ValueError("Unexpected outer partition numbering.")
    indices = set(range(len(photos)))

    def inspect(part, pool):
        train = set(part.train)
        held = pool - train
        if not train <= pool or len(train) != len(part.train):
            raise ValueError("Repeated or out-of-partition training row.")
        scored = {i for i in held if photos[i].key in complete_keys}
        if set(part.held_out) != scored or len(part.held_out) != len(scored):
            raise ValueError("Validation scoring must cover exactly the complete held-out photos.")
        for field in ("patient_id", "group"):
            if ({getattr(photos[i], field) for i in train}
                    & {getattr(photos[i], field) for i in held}):
                raise ValueError("Patient or linked patient group crosses an S2 partition.")
        if any({photos[i].label for i in side} != {0, 1} for side in (train, scored)):
            raise ValueError("An S2 training or scoring set lacks a class.")
        train_keys = {photos[i].key for i in train}
        return held, {
            "train_keys": len(train_keys), "train_incomplete_keys": len(train_keys - complete_keys),
            "train_photos": len(train), "held_keys": len({photos[i].key for i in held}),
            "scored_complete_keys": len({photos[i].key for i in scored}), "scored_photos": len(scored),
        }

    records = []
    for split in plan:
        outer_train = set(split.outer.train)
        outer_held, counts = inspect(split.outer, indices)
        if outer_held != {i for i, p in enumerate(photos) if p.fold == split.fold}:
            raise ValueError("Outer patient partition differs from the saved Stage 3 split.")
        if {photos[i].key for i in split.outer.held_out} != {k for k, v in h1_folds.items() if v == split.fold}:
            raise ValueError("An outer scoring fold differs from the H1 keys.")
        validation_count = Counter()
        inner_counts = []
        for number, part in enumerate(split.inner, 1):
            held, stats = inspect(part, outer_train)
            validation_count.update(held)
            inner_counts.append({"inner_fold": number, **stats})
        if len(inner_counts) != N_SPLITS or validation_count != Counter(outer_train):
            raise ValueError("Inner folds must hold out every outer-training photo exactly once.")
        records.append({"outer_fold": split.fold, **counts, "inner": inner_counts})
    return records


def compare_oof(oof, h1):
    """Compare the two runs on the same labelled evaluation set, with equal lesion weights."""
    def identity(rows):
        return sorted((r["patient_id"], r["lesion_id"], r["outer_fold"], r["label"], r["n_images"])
                      for r in rows)

    if len({(r["patient_id"], r["lesion_id"]) for r in oof}) != len(oof) or identity(oof) != identity(h1):
        raise ValueError("S2 OOF identities, folds, labels or image counts disagree with H1.")

    def scores(rows):
        labels = [r["label"] for r in rows]
        return _loss_summary(*(lesion_losses(labels, [r[p] for r in rows]) for p in ("p_A", "p_B")))

    folds = []
    for fold in range(1, N_SPLITS + 1):
        s2 = [r for r in oof if r["outer_fold"] == fold]
        baseline = [r for r in h1 if r["outer_fold"] == fold]
        folds.append({"outer_fold": fold, "keys": len(s2), "patients": len({r["patient_id"] for r in s2}),
                      **scores(s2), "h1_delta_log_loss": scores(baseline)["delta_log_loss"]})
    return {"s2": scores(oof), "h1": scores(h1), "folds": folds, "matched_keys": len(oof),
            "folds_favouring_B": sum(f["delta_log_loss"] < 0 for f in folds)}


def source_digests():
    names = ("pad_ufes20_s2", "pad_v11_s2_inputs", "pad_v11_s2_nested", "pad_ufes20_primary",
             "pad_v11_primary", "pad_v11_nested", "pad_v11_training_inputs", "pad_ufes20_full",
             "pad_ufes20_curate", "pad_v11_images", "pad_v11_metadata", "pad_ufes20_folds",
             "pad_v11_fold_inputs")
    return {name + ".py": file_sha256(ROOT / (name + ".py")) for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train"))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs=3, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--curation", type=Path, default=PRIVATE / "v1.1-curation")
    parser.add_argument("--review", type=Path, default=PRIVATE / "v1.1-review" / "recovered_review.json")
    parser.add_argument("--fold-dir", type=Path, default=PRIVATE / "v1.1-outer-folds")
    parser.add_argument("--primary-dir", type=Path, default=PRIVATE / "v1.1-primary-stage4b")
    parser.add_argument("--output-dir", type=Path, default=PRIVATE / "v1.1-secondary-s2")
    args = parser.parse_args(argv)
    try:
        versions = verified_environment()
        checkpoint_sha = verify_checkpoint(args.checkpoint)
        photos, images, complete, h1, provenance = load_s2_inputs(
            args.curation, args.metadata, args.review, args.fold_dir, args.images, args.primary_dir)
        partitions = partition_audit(photos, complete, h1, plan_s2(photos, complete))
        if args.action == "preflight":
            print(json.dumps({"curated": provenance["curated"], "primary": provenance["primary"],
                              "verified_partitions": partitions, "splits_sha256": provenance["splits_sha256"]}))
            return 0
        sources = source_digests()
        base_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dest = reserve_private_run(args.output_dir, [args.curation, args.review, args.fold_dir,
                                                     args.primary_dir, *args.images])
        started = datetime.now(timezone.utc).isoformat()
        embeddings = extract_embeddings(photos, images, args.checkpoint, batch_size=32, device="cpu")
        try:
            oof, grid, diagnostics = run_s2_nested(photos, embeddings, complete)
        except FitFailure as exc:
            write_json(dest / FITS, exc.diagnostics)
            raise
        write_json(dest / FITS, diagnostics)
        report = compare_oof(oof, h1)
        if source_digests() != sources:
            raise ValueError("An S2 source file changed during the run.")
        write_records(dest / OOF, OOF_FIELDS, oof)
        write_records(dest / GRID, GRID_FIELDS, grid)
        write_json(dest / SUMMARY, {
            "analysis": "S2: all-curated training, held-out complete-case evaluation",
            "role": "prespecified descriptive secondary analysis", "status": "complete",
            "protocol_sha256": PROTOCOL_SHA256, "git_base_commit": base_commit, "source_sha256": sources,
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "versions": versions, "inputs": provenance,
            "encoder": {"weights": "frozen ResNet18 IMAGENET1K_V1", "checkpoint_sha256": checkpoint_sha,
                        "device": "cpu", "batch_size": 32, "embedding_size": embeddings.shape[1]},
            "inner_cv": {"splitter": "StratifiedGroupKFold", "n_splits": N_SPLITS, "shuffle": True,
                         "seed": SEED, "stratum": "2 * binary_label + int(symptom_complete)",
                         "C_grid": list(C_GRID), "selected": [r for r in grid if r["selected"]]},
            "symptoms_B": "photo-level True=1; False and unknown=0; no missingness indicators",
            "partition_audit": partitions, "results": report, "n_fits": len(diagnostics),
            "n_fits_with_warnings": sum(bool(r["warnings"]) for r in diagnostics),
            "output_sha256": {name: file_sha256(dest / name) for name in (FITS, OOF, GRID)},
        })
    except (KeyError, OSError) as exc:
        print(f"S2 stopped ({type(exc).__name__}). Check private inputs and filesystem access.", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, csv.Error) as exc:
        print(f"S2 stopped: {exc}", file=sys.stderr)
        return 1
    print(f"S2 complete. {report['matched_keys']} paired test keys matched H1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
