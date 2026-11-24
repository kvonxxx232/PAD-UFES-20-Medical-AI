"""Authenticate the saved Stage 2/3/4B evidence used by S1, S3, S6 and S7."""

from dataclasses import dataclass
import json
from pathlib import Path

from pad_ufes20_curate import COHORT_REFERENCE, population, require_match, select_columns
from pad_ufes20_primary import ANALYSIS, PROTOCOL_SHA256
from pad_v11_fold_inputs import _unique, load_curation
from pad_v11_metadata import FIELDS, Key, assess_metadata
from pad_v11_s2_inputs import (
    H1_COMMIT, H1_OOF_SHA256, H1_PROVENANCE_SHA256, H1_SUMMARY_SHA256,
    _aligned_reference, _curated_photos,
)
from pad_v11_training_inputs import FOLD_SUMMARY_SHA256, SPLITS_SHA256, _fold_assignments, _snapshot


@dataclass(repr=False)
class SecondaryInputs:
    oof_rows: list[dict]
    s3_keys: set[Key]
    metadata_rows: list[dict[str, str]]
    provenance: dict


def _clinical_cohorts(metadata, curated, keys):
    """Reconcile the clinical strata with Stage 2, preserving unknowns on every photo."""
    key_fields = ("patient_id", "lesion_id")
    fields = (*key_fields, "img_id", "gender", *FIELDS)
    rows = [r for r in select_columns(metadata, fields) if (r["patient_id"], r["lesion_id"]) in keys]
    assessed = assess_metadata(rows)
    saved_fields = (*key_fields, "s3_eligible", "biopsy_conflict", "unknown_symptoms")
    saved = _unique(select_columns(curated, saved_fields), key_fields)
    if assessed.keys() != keys.keys() or saved.keys() != keys.keys():
        raise ValueError("Secondary metadata or retained-key coverage differs from Stage 2.")
    complete = {key for key, a in assessed.items() if a.complete}
    s3 = {key for key in complete if assessed[key].biopsy_confirmed}
    for key, a in assessed.items():
        row = saved[key]
        if (keys[key][1:] != (int(a.positive), a.complete) or a.symptom_conflicts
                or row["s3_eligible"] != str(key in s3)
                or row["biopsy_conflict"] != str(a.biopsy_conflict)
                or row["unknown_symptoms"] != "|".join(a.unknown_symptoms)):
            raise ValueError("Clinical flags or binary label disagree with the retained cohort.")
    cohorts = {name: population(group, assessed) for name, group in (
        ("curated", keys), ("primary", complete), ("incomplete", keys.keys() - complete), ("s3", s3))}
    return rows, s3, cohorts


def load_secondary_inputs(curation_dir: Path, metadata: Path, review: Path,
                          fold_dir: Path, primary_dir: Path) -> SecondaryInputs:
    """Read private snapshots and return aligned data in memory plus aggregate-only provenance."""
    protocol = Path(__file__).resolve().with_name("protocol_v1.1.md")
    _snapshot(protocol, PROTOCOL_SHA256, "Frozen protocol")
    h1 = json.loads(_snapshot(primary_dir / "summary.json", H1_SUMMARY_SHA256, "Stage 4B summary"))
    run = json.loads(_snapshot(primary_dir / "run_provenance.json", H1_PROVENANCE_SHA256,
                               "Stage 4B run provenance"))
    if (h1["status"] != "complete" or h1["analysis"] != ANALYSIS
            or h1["protocol_sha256"] != PROTOCOL_SHA256 or run["exit_code"] != 0
            or run["code_commit"] != H1_COMMIT or run["commit_at_end"] != H1_COMMIT
            or run["summary_sha256"] != H1_SUMMARY_SHA256 or run["input_provenance"] != h1["inputs"]
            or run["output_sha256"] != h1["output_sha256"]
            or h1["output_sha256"]["oof_predictions.csv"] != H1_OOF_SHA256):
        raise ValueError("Stage 4B provenance or recorded OOF digest disagrees with the frozen run.")

    fold_summary = json.loads(_snapshot(fold_dir / "fold_summary.json", FOLD_SUMMARY_SHA256,
                                        "Fixed fold summary"))
    lineage = fold_summary["input_provenance"]
    h1_inputs = h1["inputs"]
    if (h1_inputs["fold_summary_sha256"] != FOLD_SUMMARY_SHA256
            or h1_inputs["splits_sha256"] != SPLITS_SHA256
            or fold_summary["splits_sha256"] != SPLITS_SHA256
            or any(h1_inputs[k] != v for k, v in lineage.items())):
        raise ValueError("Stage 4B and fixed folds disagree on their input lineage.")
    _snapshot(curation_dir / "summary.json", lineage["stage2_summary_sha256"], "Stage 2 summary")
    keys, quarantine, observed_lineage = load_curation(curation_dir, metadata, review)
    if observed_lineage != lineage or len(quarantine) != h1_inputs["quarantined_keys"]:
        raise ValueError("Stage 2 inputs or quarantine count differ from the original run.")
    folds = _fold_assignments(_snapshot(fold_dir / "splits.csv", SPLITS_SHA256, "Fixed folds"),
                              keys, quarantine)

    clinical = _snapshot(metadata, lineage["metadata_sha256"], "Original metadata")
    curated = _snapshot(curation_dir / "curated_keys.csv",
                        lineage["stage2_output_sha256"]["curated_keys.csv"], "Stage 2 retained keys")
    rows, s3, cohorts = _clinical_cohorts(clinical, curated, keys)
    for name, counts in cohorts.items():
        require_match(counts, COHORT_REFERENCE[name])
    if h1_inputs["primary"] != cohorts["primary"] or h1_inputs["curated_keys"] != len(keys):
        raise ValueError("Primary or curated population differs from the Stage 4B run.")

    # Use the same complete-case row alignment as S2. This never reads image pixels.
    photos, complete = _curated_photos(clinical, keys, folds)
    oof = _snapshot(primary_dir / "oof_predictions.csv", H1_OOF_SHA256, "Stage 4B OOF predictions")
    predictions = _aligned_reference(oof, photos, complete)
    provenance = {
        **lineage, "quarantined_keys": len(quarantine), "cohorts": cohorts,
        "protocol_sha256": PROTOCOL_SHA256, "splits_sha256": SPLITS_SHA256,
        "fold_summary_sha256": FOLD_SUMMARY_SHA256, "h1_summary_sha256": H1_SUMMARY_SHA256,
        "h1_run_provenance_sha256": H1_PROVENANCE_SHA256, "h1_oof_sha256": H1_OOF_SHA256,
        "h1_code_commit": H1_COMMIT, "recorded_checkpoint_sha256": h1_inputs["encoder_checkpoint_sha256"],
    }
    return SecondaryInputs(predictions, s3, rows, provenance)
