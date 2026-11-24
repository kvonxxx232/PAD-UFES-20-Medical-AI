"""Authenticate the S2 training cohort, fixed folds and original primary test population."""

from collections import Counter
import json
import math

from pad_ufes20_curate import COHORT_REFERENCE, require_match, select_columns
from pad_ufes20_primary import ANALYSIS, OOF_FIELDS, PROTOCOL_SHA256
from pad_v11_fold_inputs import load_curation
from pad_v11_metadata import BOOLEAN, SYMPTOMS
from pad_v11_nested import Photo
from pad_v11_training_inputs import (
    CHECKPOINT_SHA256, FOLD_SUMMARY_SHA256, SPLITS_SHA256,
    _fold_assignments, _snapshot, _verified_images,
)


H1_SUMMARY_SHA256 = "c6362c4f367ce128525070e0af44a715434906e130ee1d8ebc0ab9ea1cb32f46"
H1_OOF_SHA256 = "4687d825180a99967a3e4a9f32267526e5f8f4c0be23d0cc394fc2badff1ce3b"
H1_PROVENANCE_SHA256 = "2e330227a2cc725c393b0e60eab66fa8d95ed49f58eb401e514600d18b05a169"
H1_COMMIT = "c62f33e1ce752df0f235a5f664a3425d8b6bafe5"


def _curated_photos(clinical, keys, folds):
    """Encode each photo separately and verify completeness before discarding UNK."""
    columns = ("img_id", "patient_id", "lesion_id", "age", "region", *SYMPTOMS)
    photos = []
    incomplete = set()
    for row in select_columns(clinical, columns):
        key = row["patient_id"], row["lesion_id"]
        if key not in keys:
            continue
        group, label, _ = keys[key]
        values = tuple(row[field] for field in SYMPTOMS)
        for value in values:
            if value not in BOOLEAN:
                if value != "UNK" and value.strip():
                    raise ValueError("Unrecognised S2 symptom token.")
                incomplete.add(key)
        photos.append(Photo(row["img_id"], *key, group, folds[key], label, int(row["age"]),
                            row["region"], tuple(int(value == "True") for value in values)))
    complete = {key for key, (_, _, eligible) in keys.items() if eligible}
    if ({p.key for p in photos} != keys.keys() or incomplete != keys.keys() - complete
            or len({p.image_id for p in photos}) != len(photos)):
        raise ValueError("S2 metadata coverage, image uniqueness or completeness differs from curation.")
    photos.sort(key=lambda p: (p.patient_id, p.lesion_id, p.image_id))
    return photos, complete


def _population(photos):
    labels = {p.key: p.label for p in photos}
    return {"keys": len(labels), "patients": len({p.patient_id for p in photos}),
            "images": len(photos), "positive": sum(labels.values()),
            "negative": len(labels) - sum(labels.values())}


def _aligned_reference(raw, photos, complete):
    """Require every original H1 row to match a unique complete key, fold, label and count."""
    counts = Counter(p.key for p in photos if p.key in complete)
    expected = {p.key: (p.fold, p.label, counts[p.key]) for p in photos if p.key in complete}
    rows = []
    for row in select_columns(raw, OOF_FIELDS):
        key = row["patient_id"], row["lesion_id"]
        for field in ("outer_fold", "label", "n_images"):
            row[field] = int(row[field])
        if expected.pop(key, None) != (row["outer_fold"], row["label"], row["n_images"]):
            raise ValueError("Primary OOF membership, fold, label or photo count disagrees with S2.")
        for field in ("p_A", "p_B"):
            row[field] = float(row[field])
            if not math.isfinite(row[field]) or not 0 <= row[field] <= 1:
                raise ValueError("Invalid primary reference probability.")
        rows.append(row)
    if expected:
        raise ValueError("Missing primary OOF record for an S2 evaluation key.")
    return rows


def load_s2_inputs(curation, metadata, review, fold_dir, image_dirs, primary_dir):
    keys, quarantine, lineage = load_curation(curation, metadata, review)
    split_summary = json.loads(_snapshot(fold_dir / "fold_summary.json", FOLD_SUMMARY_SHA256,
                                         "Fixed outer-fold summary"))
    if split_summary["input_provenance"] != lineage or split_summary["splits_sha256"] != SPLITS_SHA256:
        raise ValueError("Fixed outer-fold lineage differs from the verified S2 inputs.")
    folds = _fold_assignments(_snapshot(fold_dir / "splits.csv", SPLITS_SHA256,
                                        "Fixed outer-fold assignments"), keys, quarantine)

    clinical = _snapshot(metadata, lineage["metadata_sha256"], "Original metadata")
    photos, complete = _curated_photos(clinical, keys, folds)
    primary = _population([p for p in photos if p.key in complete])
    curated = _population(photos)
    incomplete = _population([p for p in photos if p.key not in complete])
    for name, observed in (("curated", curated), ("primary", primary), ("incomplete", incomplete)):
        require_match(observed, COHORT_REFERENCE[name])

    inventory = _snapshot(curation / "image_inventory.csv",
                          lineage["stage2_output_sha256"]["image_inventory.csv"], "Stage 2 PNG inventory")
    images = _verified_images(inventory, image_dirs)
    stage2 = json.loads(_snapshot(curation / "summary.json", lineage["stage2_summary_sha256"],
                                  "Stage 2 curation summary"))
    shared_inputs = {
        **lineage, "source_archive_sha256": stage2["source_archive_sha256"],
        "fold_summary_sha256": FOLD_SUMMARY_SHA256, "splits_sha256": SPLITS_SHA256,
        "verified_pngs": len(images), "curated_keys": len(keys), "quarantined_keys": len(quarantine),
        "primary": primary, "encoder_checkpoint_sha256": CHECKPOINT_SHA256,
    }
    summary = json.loads(_snapshot(primary_dir / "summary.json", H1_SUMMARY_SHA256, "Stage 4B summary"))
    record = json.loads(_snapshot(primary_dir / "run_provenance.json", H1_PROVENANCE_SHA256,
                                  "Stage 4B run provenance"))
    if (summary["status"] != "complete" or summary["analysis"] != ANALYSIS
            or summary["protocol_sha256"] != PROTOCOL_SHA256
            or record["exit_code"] != 0 or record["code_commit"] != H1_COMMIT
            or record["commit_at_end"] != H1_COMMIT or record["summary_sha256"] != H1_SUMMARY_SHA256
            or summary["output_sha256"]["oof_predictions.csv"] != H1_OOF_SHA256
            or record["output_sha256"] != summary["output_sha256"]
            or summary["inputs"] != shared_inputs or record["input_provenance"] != shared_inputs):
        raise ValueError("S2 data, fixed split or Stage 4B reference provenance mismatch.")
    raw_oof = _snapshot(primary_dir / "oof_predictions.csv", H1_OOF_SHA256, "Stage 4B OOF")
    h1_rows = _aligned_reference(raw_oof, photos, complete)
    provenance = {**shared_inputs, "curated": curated, "incomplete": incomplete,
                  "h1_summary_sha256": H1_SUMMARY_SHA256, "h1_oof_sha256": H1_OOF_SHA256,
                  "h1_run_provenance_sha256": H1_PROVENANCE_SHA256, "h1_code_commit": H1_COMMIT}
    return photos, {p.image_id: images[p.image_id] for p in photos}, complete, h1_rows, provenance
