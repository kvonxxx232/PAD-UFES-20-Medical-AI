"""Reproduce frozen V1.1 curation and write private local evidence and cohort counts."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy
import PIL
import scipy

from pad_v11_images import audit_images
from pad_v11_metadata import FIELDS, MetadataError, assess_metadata


# These are comparison values from the frozen audit, never selection criteria.
IMAGE_REFERENCE = {
    "images": 2298, "original_keys": 1891, "original_patients": 1373,
    "confirmed_pairs": 16, "sha256_pairs": 15, "rgb_only_pairs": 1,
    "duplicate_images": 32, "duplicate_patients": 13, "duplicate_keys": 26,
    "sha256_keys": 24, "rgb_only_keys": 2, "cross_patient_pairs": 0,
    "phash_candidates": 35, "pending_review": 34,
    "pair_set_sha256": "e0456e137f1c3c3c1b8f0fa20f0c5aa6908e4111dec340d8963bd53abaf3fff8",
    "duplicate_key_set_sha256": "2d1a9611aff51a12845f8b27f389d6fd46ab7aca577bed5c357ee828592ae25d",
}
COHORT_REFERENCE = {
    "metadata_sha256": "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527",
    "quarantined_keys": 26, "quarantined_images": 42, "symptom_conflict_keys": 0,
    "biopsy_conflict_keys": 0, "any_unknown_keys": 389, "duplicate_and_unknown_keys": 2,
    "curated": {"keys": 1865, "patients": 1368, "images": 2256, "positive": 816, "negative": 1049},
    "primary": {"keys": 1478, "patients": 1115, "images": 1854, "positive": 475, "negative": 1003},
    "incomplete": {"keys": 387, "patients": 318, "images": 402, "positive": 341, "negative": 46},
    "s3": {"keys": 638, "patients": 508, "images": 903, "positive": 475, "negative": 163},
}


def set_sha256(pairs):
    """Hash sorted, distinct string tuples as UTF-8, tab-separated, LF-terminated."""
    digest = hashlib.sha256()
    for left, right in sorted(set(pairs)):
        digest.update(f"{left}\t{right}\n".encode("utf-8"))
    return digest.hexdigest()


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_match(actual, reference):
    differences = {k: {"expected": v, "observed": actual[k]}
                   for k, v in reference.items() if actual[k] != v}
    if differences:
        raise ValueError("Frozen regression check failed: " + json.dumps(differences, sort_keys=True))


def select_columns(raw, columns):
    """Project one immutable CSV snapshot. The image phase requests identifier columns only."""
    records = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    header = next(records, [])
    if len(header) != len(set(header)) or not set(columns) <= set(header):
        raise ValueError(f"Duplicate header or missing required columns: {sorted(set(columns) - set(header))}")
    indices = [header.index(c) for c in columns]
    for row in records:
        if len(row) != len(header):
            raise ValueError(f"Malformed metadata row ending at line {records.line_num}")
        yield dict(zip(columns, (row[i] for i in indices)))


def image_index(raw):
    index = {}
    for row in select_columns(raw, ("img_id", "patient_id", "lesion_id")):
        for field, value in row.items():
            if not value or value != value.strip() or not value.isprintable():
                raise ValueError(f"Invalid {field} in metadata: {value!r}")
        name = row["img_id"]
        if Path(name).name != name or "\\" in name or not name.lower().endswith(".png"):
            raise ValueError(f"Expected a PNG basename for img_id: {name!r}")
        if name in index:
            raise ValueError(f"Repeated image ID in metadata: {name!r}")
        index[name] = row["patient_id"], row["lesion_id"]
    if not index:
        raise ValueError("Empty metadata")
    return index


def write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def write_json(path, value):
    # A final summary is the completion marker; an interrupted write must not publish it.
    tmp = path.with_suffix(".tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    tmp.rename(path)


def reserve_output(destination, image_dirs):
    """Use a new owner-only directory. Within a Git tree require an ignored directory."""
    target = destination.resolve()
    if target.exists():
        raise ValueError("Output directory already exists; choose a new run directory.")
    if ".git" in target.parts or any(target.is_relative_to(d.resolve()) for d in image_dirs):
        raise ValueError("Output directory overlaps Git internals or an image input.")
    parent = next(p for p in target.parents if p.exists())
    repo = subprocess.run(["git", "-C", str(parent), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True)
    if repo.returncode == 0:
        relative = target.relative_to(Path(repo.stdout.strip()).resolve())
        git = ["git", "-C", repo.stdout.strip()]
        ignored = subprocess.run([*git, "check-ignore", "-q", "--", str(relative) + "/"])
        tracked = subprocess.check_output([*git, "ls-files", "-z", "--", str(relative)])
        if ignored.returncode != 0 or tracked:
            raise ValueError("Use a Git-ignored output directory such as private/<run-name>.")
    elif any((p / ".git").exists() for p in (parent, *parent.parents)):
        raise ValueError("Could not verify Git ignore protection for this output path.")
    target.mkdir(parents=True, mode=0o700, exist_ok=False)
    return target


def record_image_audit(out, index, audit):
    """Persist the blinded evidence before clinical reconciliation begins."""
    reasons = defaultdict(set)
    for (a, b), rule in audit.confirmed.items():
        reasons[index[a]].add(rule)
        reasons[index[b]].add(rule)
    write_csv(out / "image_inventory.csv", ("img_id", "patient_id", "lesion_id", "file_sha256"),
              ((n, *index[n], audit.file_hashes[n]) for n in sorted(index)))
    write_csv(out / "confirmed_pairs.csv", ("image_a", "image_b", "rule_id"),
              ((a, b, rule) for (a, b), rule in sorted(audit.confirmed.items())))
    write_csv(out / "duplicate_keys.csv", ("patient_id", "lesion_id", "rule_ids"),
              ((*k, "|".join(sorted(reasons[k]))) for k in sorted(reasons)))
    # No clinical fields, prior dispositions or inferred human judgments enter this queue.
    write_csv(out / "phash_candidates.csv",
              ("image_a", "image_b", "hamming_distance", "status", "reviewer", "evidence", "visual_judgment"),
              ((a, b, d, "confirmed_rgb" if (a, b) in audit.confirmed else "pending_review", "", "", "")
               for (a, b), d in sorted(audit.candidates.items())))
    rules = Counter(audit.confirmed.values())
    stats = {
        "images": len(index), "original_keys": len(set(index.values())),
        "original_patients": len({p for p, _ in index.values()}),
        "confirmed_pairs": len(audit.confirmed), "sha256_pairs": rules["sha256"],
        "rgb_only_pairs": rules["rgb_equal"], "duplicate_keys": len(reasons),
        "duplicate_patients": len({p for p, _ in reasons}),
        "duplicate_images": len({n for pair in audit.confirmed for n in pair}),
        "sha256_keys": sum("sha256" in r for r in reasons.values()),
        "rgb_only_keys": sum("rgb_equal" in r for r in reasons.values()),
        "cross_patient_pairs": sum(index[a][0] != index[b][0] for a, b in audit.confirmed),
        "phash_candidates": len(audit.candidates),
        "pending_review": len(audit.candidates.keys() - audit.confirmed.keys()),
        "phash_distance_histogram": dict(Counter(audit.candidates.values())),
        "pair_set_sha256": set_sha256(audit.confirmed),
        "duplicate_key_set_sha256": set_sha256(reasons),
    }
    return reasons, stats


def patient_groups(index, confirmed):
    """Build components on original patients, before dropping quarantined keys."""
    parent = {patient: patient for patient, _ in index.values()}

    def root(patient):
        while parent[patient] != patient:
            parent[patient] = parent[parent[patient]]
            patient = parent[patient]
        return patient

    for a, b in confirmed:
        left, right = sorted((root(index[a][0]), root(index[b][0])))
        parent[right] = left
    return {p: root(p) for p in parent}


def population(keys, assessments):
    positives = sum(assessments[k].positive for k in keys)
    return {"keys": len(keys), "patients": len({p for p, _ in keys}),
            "images": sum(assessments[k].image_count for k in keys),
            "positive": positives, "negative": len(keys) - positives}


def run_pipeline(metadata, image_dirs, out, archives):
    raw = metadata.read_bytes()
    index = image_index(raw)
    audit = audit_images(index, image_dirs)
    reasons, image_stats = record_image_audit(out, index, audit)
    require_match(image_stats, IMAGE_REFERENCE)
    duplicates = set(reasons)
    groups = patient_groups(index, audit.confirmed)

    rows = list(select_columns(raw, ("patient_id", "lesion_id", *FIELDS)))
    assessed = assess_metadata(rows)
    conflicts = {k for k, a in assessed.items() if a.symptom_conflicts}
    unknown = {k for k, a in assessed.items() if a.unknown_symptoms}
    for key in conflicts:
        reasons[key].add("symptom_conflict")
    retained = assessed.keys() - reasons.keys()
    primary = {k for k in retained if assessed[k].complete}
    s3 = {k for k in primary if assessed[k].biopsy_confirmed}
    stats = {
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "quarantined_keys": len(reasons),
        "quarantined_images": sum(assessed[k].image_count for k in reasons),
        "symptom_conflict_keys": len(conflicts),
        "duplicate_and_symptom_conflict_keys": len(duplicates & conflicts),
        "biopsy_conflict_keys": sum(a.biopsy_conflict for a in assessed.values()),
        "any_unknown_keys": len(unknown), "duplicate_and_unknown_keys": len(duplicates & unknown),
        "quarantine_reason_combinations": dict(Counter("|".join(sorted(r)) for r in reasons.values())),
        "curated": population(retained, assessed), "primary": population(primary, assessed),
        "incomplete": population(retained - primary, assessed), "s3": population(s3, assessed),
    }
    require_match(stats, COHORT_REFERENCE)
    write_csv(out / "quarantine_manifest.csv",
              ("patient_id", "lesion_id", "rule_ids", "conflicting_symptoms", "image_count"),
              ((*k, "|".join(sorted(reasons[k])), "|".join(assessed[k].symptom_conflicts),
                assessed[k].image_count) for k in sorted(reasons)))
    write_csv(out / "curated_keys.csv",
              ("patient_id", "lesion_id", "patient_group", "image_count", "primary_eligible",
               "s3_eligible", "biopsy_conflict", "unknown_symptoms"),
              ((*k, groups[k[0]], assessed[k].image_count, k in primary, k in s3,
                assessed[k].biopsy_conflict, "|".join(assessed[k].unknown_symptoms)) for k in sorted(retained)))
    summary = {
        "protocol": "protocol-v1.1-frozen", "status": "passed", "image_audit": image_stats,
        "cohorts": stats, "source_archive_sha256": [file_sha256(p) for p in archives] if archives else None,
        "output_sha256": {p.name: file_sha256(p) for p in sorted(out.glob("*.csv"))},
        "versions": {"Python": platform.python_version(), "Pillow": PIL.__version__,
                     "NumPy": numpy.__version__, "SciPy": scipy.__version__},
    }
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="A successful run ends with summary.json. On a validation failure, inspect "
               "errors.json locally and rerun in a fresh directory after investigation. "
               "The review queue contains no imputed human judgments. Without --archives, "
               "original archive checksums are unavailable; per-PNG hashes are always recorded.",
    )
    parser.add_argument("--metadata", type=Path, required=True, help="Original PAD-UFES-20 metadata CSV")
    parser.add_argument("--images", nargs=3, type=Path, required=True, metavar=("PART1", "PART2", "PART3"))
    parser.add_argument("--archives", nargs=3, type=Path, metavar=("ZIP1", "ZIP2", "ZIP3"),
                        help="Original archives in part order, when available, for provenance checksums")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "private" / "v1.1-curation",
                        help="Fresh local directory (default: private/v1.1-curation). Within Git it must be ignored.")
    args = parser.parse_args(argv)
    try:
        out = reserve_output(args.output_dir, args.images)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Cannot reserve private output directory: {exc}", file=sys.stderr)
        return 2
    try:
        summary = run_pipeline(args.metadata, args.images, out, args.archives)
    except (OSError, ValueError, csv.Error) as exc:
        details = {"status": "failed", "error": str(exc)}
        if isinstance(exc, MetadataError):
            details["issues"] = exc.issues
        write_json(out / "errors.json", details)
        print("Curation stopped. See errors.json in the private run directory.", file=sys.stderr)
        return 1
    print(f"Curation passed: {summary['cohorts']['curated']['keys']} retained keys, "
          f"{summary['cohorts']['primary']['keys']} primary-eligible. See local summary.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
