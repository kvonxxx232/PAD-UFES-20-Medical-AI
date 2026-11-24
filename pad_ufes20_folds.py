"""Generate or verify the frozen V1.1 outer split. This stage does not fit models."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from pad_ufes20_curate import COHORT_REFERENCE, reserve_output, write_json
from pad_v11_fold_inputs import load_curation


SEED = 20261124
N_SPLITS = 5
PRIVATE = Path(__file__).resolve().parent / "private"


def generate_outer_folds(keys, quarantined):
    """Split sorted lesion keys; keep entire patient components on one side."""
    if not keys or keys.keys() & quarantined:
        raise ValueError("Empty cohort or a quarantined key in split input.")
    ordered = sorted(keys)
    patient_group = {}
    labels, complete, groups = [], [], []
    for key in ordered:
        patient, _ = key
        group, label, eligible = keys[key]
        if not group or patient_group.setdefault(patient, group) != group:
            raise ValueError("A patient has a missing or inconsistent component.")
        if label not in (0, 1) or type(eligible) is not bool:
            raise ValueError("Invalid binary label or symptom completeness.")
        groups.append(group)
        labels.append(label)
        complete.append(eligible)
    if len(set(groups)) < N_SPLITS:
        raise ValueError("Too few independent patient groups for five folds.")

    strata = np.array([2 * y + int(c) for y, c in zip(labels, complete)])
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    assignments = {}
    stats = []
    splits = splitter.split(np.zeros(len(ordered), dtype=np.uint8), strata, groups)
    for fold, (train, test) in enumerate(splits, start=1):
        test_patients = {ordered[i][0] for i in test}
        test_groups = {groups[i] for i in test}
        if test_patients.intersection(ordered[i][0] for i in train):
            raise ValueError("Patient crosses training and outer validation.")
        if test_groups.intersection(groups[i] for i in train):
            raise ValueError("Patient component crosses training and outer validation.")
        for partition in (train, test):
            if {labels[i] for i in partition if complete[i]} != {0, 1}:
                raise ValueError("An outer partition lacks a class among symptom-complete keys.")
        for i in test:
            key = ordered[i]
            if key in assignments:
                raise ValueError("A key has more than one held-out assignment.")
            assignments[key] = fold
        counts = Counter(strata[test].tolist())
        stats.append({
            "fold": fold, "patients": len(test_patients), "groups": len(test_groups),
            "keys": len(test), "negative": counts[0] + counts[1], "positive": counts[2] + counts[3],
            "incomplete": counts[0] + counts[2], "complete": counts[1] + counts[3],
            "incomplete_negative": counts[0], "complete_negative": counts[1],
            "incomplete_positive": counts[2], "complete_positive": counts[3],
        })
    if assignments.keys() != keys.keys() or len(stats) != N_SPLITS:
        raise ValueError("The five validation folds do not exactly cover the retained keys.")
    if sum(f["patients"] for f in stats) != len(patient_group):
        raise ValueError("A patient appears in multiple validation folds.")
    if sum(f["groups"] for f in stats) != len(set(groups)):
        raise ValueError("A component appears in multiple validation folds.")
    return assignments, stats


def split_table_bytes(assignments):
    """Canonical private table: header and sorted string keys, UTF-8 and LF."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("patient_id", "lesion_id", "fold"))
    for key in sorted(assignments):
        writer.writerow((*key, assignments[key]))
    return output.getvalue().encode("utf-8")


def run(curation, metadata, review, output, verify=False):
    keys, quarantine, provenance = load_curation(curation, metadata, review)
    assignments, folds = generate_outer_folds(keys, quarantine)
    for field in ("patients", "keys", "positive", "negative"):
        if sum(f[field] for f in folds) != COHORT_REFERENCE["curated"][field]:
            raise ValueError("Outer split totals differ from the frozen cohort.")
    if sum(f["complete"] for f in folds) != COHORT_REFERENCE["primary"]["keys"]:
        raise ValueError("Primary-eligible total differs from the frozen preflight.")
    for f in folds:
        if not all(f[k] for k in ("complete_positive", "complete_negative",
                                  "incomplete_positive", "incomplete_negative")):
            raise ValueError("A frozen cohort validation fold lacks a preflight stratum.")
    table = split_table_bytes(assignments)
    digest = hashlib.sha256(table).hexdigest()
    summary = {
        "protocol": "protocol-v1.1-frozen", "level": "outer", "splitter": "StratifiedGroupKFold",
        "seed": SEED, "shuffle": True, "n_splits": N_SPLITS,
        "stratum": "2 * binary_label + int(symptom_complete)", "splits_sha256": digest,
        "versions": {"Python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "NumPy": np.__version__},
        "input_provenance": provenance, "folds": folds,
    }
    if verify:
        if (output / "splits.csv").read_bytes() != table:
            raise ValueError("Regenerated fold table differs from the saved bytes.")
        if json.loads((output / "fold_summary.json").read_bytes()) != summary:
            raise ValueError("Regenerated provenance or fold summary differs.")
    else:
        dest = reserve_output(output, [curation])
        with (dest / "splits.csv").open("xb") as stream:
            stream.write(table)
        write_json(dest / "fold_summary.json", summary)
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Hash-verified metadata used by Stage 2 (needed to recover labels).")
    parser.add_argument("--curation", type=Path, default=PRIVATE / "v1.1-curation")
    parser.add_argument("--review", type=Path, default=PRIVATE / "v1.1-review" / "recovered_review.json")
    parser.add_argument("--output-dir", type=Path, default=PRIVATE / "v1.1-outer-folds")
    parser.add_argument("--verify", action="store_true",
                        help="Repeat the computation in memory and compare with the saved private outputs.")
    args = parser.parse_args(argv)
    try:
        digest = run(args.curation, args.metadata, args.review, args.output_dir, args.verify)
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print(f"Outer split stopped: {exc}", file=sys.stderr)
        return 1
    print(f"Outer split {'verified' if args.verify else 'saved'}; splits.csv SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
