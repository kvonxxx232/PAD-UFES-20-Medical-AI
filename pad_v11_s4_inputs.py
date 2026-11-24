"""Use the original H1 complete-case inputs and authenticate its factor-one reference."""

import json
import math

from pad_ufes20_primary import ANALYSIS, PROTOCOL_SHA256
from pad_v11_s2_inputs import (
    H1_COMMIT, H1_OOF_SHA256, H1_PROVENANCE_SHA256, H1_SUMMARY_SHA256, _aligned_reference,
)
from pad_v11_training_inputs import CHECKPOINT_SHA256, _snapshot, load_inputs


def load_s4_inputs(curation, metadata, review, fold_dir, image_dirs, primary_dir):
    """Return the H1-ordered photos, verified PNGs, aligned H1 OOF rows, Delta and provenance.

    The caller authenticates its local encoder checkpoint with verify_checkpoint.
    Only the original H1 photos are sent to the encoder. H1 OOF probabilities
    are reference data and must stay outside feature construction and tuning.
    """
    photos, images, inputs = load_inputs(curation, metadata, review, fold_dir, image_dirs)
    source = {**inputs, "encoder_checkpoint_sha256": CHECKPOINT_SHA256}
    summary = json.loads(_snapshot(primary_dir / "summary.json", H1_SUMMARY_SHA256, "H1 summary"))
    record = json.loads(_snapshot(primary_dir / "run_provenance.json", H1_PROVENANCE_SHA256,
                                  "H1 run provenance"))
    if (summary["status"] != "complete" or summary["analysis"] != ANALYSIS
            or summary["protocol_sha256"] != PROTOCOL_SHA256
            or summary["inputs"] != source or record["input_provenance"] != source
            or record["exit_code"] != 0 or record["code_commit"] != H1_COMMIT
            or record["commit_at_end"] != H1_COMMIT or record["summary_sha256"] != H1_SUMMARY_SHA256
            or summary["output_sha256"]["oof_predictions.csv"] != H1_OOF_SHA256
            or record["output_sha256"] != summary["output_sha256"]):
        raise ValueError("S4 inputs or reference provenance disagree with the original H1 run.")

    raw_oof = _snapshot(primary_dir / "oof_predictions.csv", H1_OOF_SHA256, "H1 OOF predictions")
    try:
        rows = _aligned_reference(raw_oof, photos, {p.key for p in photos})
    except ValueError as exc:
        raise ValueError("H1 OOF rows could not be paired with the S4 complete-case cohort.") from exc
    delta = summary["primary"]["delta_log_loss"]
    if not math.isfinite(delta):
        raise ValueError("The original H1 Delta must be finite.")
    provenance = {**source, "h1_summary_sha256": H1_SUMMARY_SHA256, "h1_oof_sha256": H1_OOF_SHA256,
                  "h1_run_provenance_sha256": H1_PROVENANCE_SHA256, "h1_code_commit": H1_COMMIT}
    return photos, images, rows, delta, provenance
