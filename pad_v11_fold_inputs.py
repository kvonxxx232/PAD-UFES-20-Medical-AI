"""Verified, read-only inputs for the frozen V1.1 outer split."""

import hashlib
import json
from pathlib import Path

from pad_ufes20_curate import (
    COHORT_REFERENCE, IMAGE_REFERENCE, image_index, patient_groups, population,
    require_match, select_columns, set_sha256,
)
from pad_v11_metadata import FIELDS, assess_metadata


# Digest of the private recovery record verified against the original review session.
# Only the digest is stored with code. The report and pair identities stay private.
REVIEW_SHA256 = "f264121493e53b0602c0c35f847224518d466ee486466f9cea09dfc3aa4e69a9"
STAGE2_FILES = (
    "image_inventory.csv", "confirmed_pairs.csv", "duplicate_keys.csv",
    "phash_candidates.csv", "quarantine_manifest.csv", "curated_keys.csv",
)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        if not all(key) or key in result:
            raise ValueError(f"Missing or duplicate key for {fields}.")
        result[key] = row
    return result


def validate_review(queue_raw: bytes, review_raw: bytes, confirmed: dict) -> int:
    """Match every pending cross-key pair to its documented historical disposition."""
    review = json.loads(review_raw)
    if (review.get("stage2_queue_sha256") != _digest(queue_raw)
            or review.get("schema_version") != 1 or review.get("human_signoff") is not False
            or review.get("review_nature") != "AI-assisted label-blinded visual review"
            or not review.get("primary_session") or not review.get("pair_mapping_call")):
        raise ValueError("Review provenance or queue checksum is missing or inconsistent.")
    fields = ("image_a", "image_b")
    queue = _unique(select_columns(queue_raw, (*fields, "hamming_distance", "status")), fields)
    pending = {}
    rgb = set()
    for pair, row in queue.items():
        distance = int(row["hamming_distance"])
        if pair[0] >= pair[1] or not 0 <= distance <= 8:
            raise ValueError("Invalid pHash candidate pair or distance.")
        if row["status"] == "confirmed_rgb" and confirmed.get(pair) == "rgb_equal":
            rgb.add(pair)
        elif row["status"] == "pending_review" and pair not in confirmed:
            pending[pair] = distance
        else:
            raise ValueError("Review queue status disagrees with deterministic image audit.")
    if rgb != {p for p, rule in confirmed.items() if rule == "rgb_equal"}:
        raise ValueError("RGB-only candidate evidence is incomplete.")

    records = _unique(review["records"], fields)
    if set(records) != set(pending):
        raise ValueError("Recovered records do not cover exactly the pending pairs.")
    seen_ids = set()
    for pair, entry in records.items():
        blind_id = entry.get("blind_id")
        if not blind_id or blind_id in seen_ids:
            raise ValueError("Missing or repeated historical blind-review ID.")
        seen_ids.add(blind_id)
        if (entry.get("hamming_distance") != pending[pair]
                or entry.get("disposition") != "Visually similar but distinct"):
            raise ValueError("Pair distance or disposition differs from the frozen review.")
        source = review["sources"].get(entry.get("reviewer"))
        evidence = entry.get("evidence")
        if (not source or not all(source.get(k) for k in (
                "session", "call_id", "message_id", "source_line", "parent_delivery_line", "reported_utc"))
                or not entry.get("source_locator") or not isinstance(evidence, str)
                or not evidence.strip() or evidence not in source.get("report", "")):
            raise ValueError("Pair-specific evidence or source attribution is incomplete.")
    return len(records)


def _stage2_files(directory, summary):
    if summary.get("status") != "passed" or summary.get("protocol") != "protocol-v1.1-frozen":
        raise ValueError("A successful frozen-protocol Stage 2 run is required.")
    snapshots = {}
    for name in STAGE2_FILES:
        raw = (directory / name).read_bytes()
        if _digest(raw) != summary.get("output_sha256", {}).get(name):
            raise ValueError(f"Stage 2 checksum mismatch: {name}")
        snapshots[name] = raw
    return snapshots


def _load_curation(curation, metadata, review):
    summary_raw = (curation / "summary.json").read_bytes()
    summary = json.loads(summary_raw)
    files = _stage2_files(curation, summary)
    require_match(summary["image_audit"], IMAGE_REFERENCE)
    pair_fields = ("image_a", "image_b")
    pair_rows = select_columns(files["confirmed_pairs.csv"], (*pair_fields, "rule_id"))
    confirmed = {p: row["rule_id"] for p, row in _unique(pair_rows, pair_fields).items()}
    if set_sha256(confirmed) != IMAGE_REFERENCE["pair_set_sha256"]:
        raise ValueError("Confirmed pairs disagree with the frozen audit.")

    review_raw = review.read_bytes()
    if _digest(review_raw) != REVIEW_SHA256:
        raise ValueError("The historical review file is absent or differs from the verified recovery.")
    resolved = validate_review(files["phash_candidates.csv"], review_raw, confirmed)
    if resolved != IMAGE_REFERENCE["pending_review"]:
        raise ValueError("Unexpected number of recovered pending candidates.")

    # Read the reference clinical labels only after the review gate has passed.
    raw = metadata.read_bytes()
    if _digest(raw) != COHORT_REFERENCE["metadata_sha256"]:
        raise ValueError("Metadata SHA-256 differs from verified V1.0 input.")
    require_match(summary["cohorts"], COHORT_REFERENCE)
    index = image_index(files["image_inventory.csv"])
    if index != image_index(raw):
        raise ValueError("Image-to-key mapping differs from Stage 2.")
    groups = patient_groups(index, confirmed)
    duplicates = {index[n] for p in confirmed for n in p}
    key_fields = ("patient_id", "lesion_id")
    dup_rows = select_columns(files["duplicate_keys.csv"], key_fields)
    if (set(_unique(dup_rows, key_fields)) != duplicates
            or set_sha256(duplicates) != IMAGE_REFERENCE["duplicate_key_set_sha256"]):
        raise ValueError("Duplicate endpoints disagree with the frozen key set.")

    assessed = assess_metadata(list(select_columns(raw, (*key_fields, *FIELDS))))
    quarantine = set(_unique(select_columns(files["quarantine_manifest.csv"], key_fields), key_fields))
    if quarantine != duplicates | {key for key, a in assessed.items() if a.symptom_conflicts}:
        raise ValueError("Quarantine manifest differs from duplicate and symptom conflict rules.")
    columns = (*key_fields, "patient_group", "primary_eligible", "image_count")
    retained = _unique(select_columns(files["curated_keys.csv"], columns), key_fields)
    if set(retained) != assessed.keys() - quarantine:
        raise ValueError("Retained keys are missing, unexpected or quarantined.")
    keys = {}
    for key, row in retained.items():
        group = groups[key[0]]
        a = assessed[key]
        if (row["patient_group"] != group or row["primary_eligible"] != str(a.complete)
                or row["image_count"] != str(a.image_count)):
            raise ValueError("Retained key attributes disagree with curation or patient components.")
        keys[key] = (group, int(a.positive), a.complete)
    primary = {key for key, (_, _, complete) in keys.items() if complete}
    if (population(keys, assessed) != COHORT_REFERENCE["curated"]
            or population(primary, assessed) != COHORT_REFERENCE["primary"]):
        raise ValueError("Population differs from verified curated or primary cohort.")
    provenance = {
        "stage2_summary_sha256": _digest(summary_raw),
        "stage2_output_sha256": {name: _digest(files[name]) for name in STAGE2_FILES},
        "metadata_sha256": _digest(raw), "review_sha256": _digest(review_raw),
        "recovered_pending_pairs": resolved, "review_nature": "AI-assisted label-blinded visual review",
        "human_signoff": False,
    }
    return keys, quarantine, provenance


def load_curation(curation: Path, metadata: Path, review: Path):
    """Return verified key attributes, quarantined keys and non-identifying provenance.

    Key attributes are (pre-quarantine patient component, binary label, symptom complete).
    Each private input is hashed and parsed from the same byte snapshot.
    """
    try:
        return _load_curation(curation, metadata, review)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed Stage 2 output, review provenance or original metadata.") from exc
