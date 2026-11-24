"""S7 lesion-level summaries of symptom recording in the full curated cohort."""

from collections import Counter, defaultdict
from math import sqrt

import numpy as np

from pad_v11_metadata import DIAGNOSES, REGIONS, SYMPTOMS, assess_metadata


BIOPSY_CONFLICT_ROW = "Biopsy status, Conflicting"


def _fraction(count, denominator):
    return count / denominator if denominator else None


def _continuous(values):
    if not values:
        return dict.fromkeys(("mean", "sd", "q1", "median", "q3"))
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "q1": float(q1), "median": float(median), "q3": float(q3)}


def _group_summary(records):
    n = len(records)
    positive = sum(row["positive"] for row in records)
    biopsied = sum(row["biopsied"] for row in records)
    conflicting = sum(row["biopsy_conflict"] for row in records)
    photos = [row["photos"] for row in records]
    return {
        "keys": n, "patients": len({row["patient"] for row in records}),
        "photos": sum(photos), "positive": positive, "negative": n - positive,
        "positive_fraction": _fraction(positive, n),
        # Biopsy tokens have been validated. Remaining keys have only False photos.
        "biopsied": biopsied, "not_biopsied": n - biopsied - conflicting,
        "biopsy_conflicting": conflicting,
        "age": _continuous([row["age"] for row in records]),
        "photos_per_key": _continuous(photos), "single_photo_keys": photos.count(1),
        "sex": dict(Counter(row["sex"] for row in records)),
        "region": dict(Counter(row["region"] for row in records)),
        "diagnosis": dict(Counter(row["diagnosis"] for row in records)),
    }


def _smd(complete, missing):
    """Missing minus Complete, with the unweighted average of group sample variances."""
    if complete["sd"] is None or missing["sd"] is None:
        return None
    scale = sqrt((complete["sd"] ** 2 + missing["sd"] ** 2) / 2)
    return (missing["mean"] - complete["mean"]) / scale if scale else None


def _table_column(group):
    """Use the precision of the technical report while retaining raw statistics separately."""
    def count_cell(count):
        fraction = _fraction(count, group["keys"])
        return f"{count:,} ({fraction:.1%})" if fraction is not None else f"{count:,} (undefined)"

    def fmt(value, spec):
        return format(value, spec) if value is not None else "undefined"

    def mean_sd(stats, spec):
        return f"{fmt(stats['mean'], spec)} ({fmt(stats['sd'], spec)})"

    age = group["age"]
    result = {
        "Lesion keys": f"{group['keys']:,}", "Patients": f"{group['patients']:,}",
        "Photographs": f"{group['photos']:,}", "Cancer-positive keys": count_cell(group["positive"]),
        "Age, years, mean (SD)": mean_sd(age, ".2f"),
        "Age, median [Q1, Q3]": f"{fmt(age['median'], 'g')} [{fmt(age['q1'], 'g')}, {fmt(age['q3'], 'g')}]",
    }
    for sex in ("female", "male", "unrecorded"):
        result[f"Sex, {sex}"] = count_cell(group["sex"].get(sex, 0))
    result["Biopsied"] = count_cell(group["biopsied"])
    result["Biopsy status, False"] = count_cell(group["not_biopsied"])
    result[BIOPSY_CONFLICT_ROW] = count_cell(group["biopsy_conflicting"])
    for diagnosis in sorted(DIAGNOSES):
        result[f"Diagnosis, {diagnosis}"] = count_cell(group["diagnosis"].get(diagnosis, 0))
    result["Photographs per key, mean (SD)"] = mean_sd(group["photos_per_key"], ".3f")
    result["Keys with one photograph"] = count_cell(group["single_photo_keys"])
    for region in sorted(REGIONS):
        result[f"Site, {region.lower()}"] = count_cell(group["region"].get(region, 0))
    return result


def s7_missingness(curated_rows):
    """Summarise verified, retained photo-level metadata, without returning any identifiers."""
    rows = list(curated_rows)
    if not rows:
        raise ValueError("The S7 curated cohort is empty.")
    if (len({row["img_id"] for row in rows}) != len(rows)
            or any(not row["img_id"] for row in rows)):
        raise ValueError("S7 requires a unique nonempty image ID for every metadata row.")
    assessments = assess_metadata(rows)
    sexes = defaultdict(set)
    examples = {}
    for row in rows:
        key = row["patient_id"], row["lesion_id"]
        sex = row["gender"].strip().lower()
        if sex not in ("female", "male", ""):
            raise ValueError("Unrecognised recorded sex in S7 metadata.")
        sexes[key].add(sex or "unrecorded")
        examples[key] = row

    cohorts = {"complete": [], "missing": []}
    unknown_count = Counter()
    positive_count = Counter()
    per_field = Counter()
    growth_change_only = 0
    patient_status = defaultdict(list)
    for key, assessment in assessments.items():
        if len(sexes[key]) != 1 or assessment.symptom_conflicts:
            raise ValueError("S7 cannot report ambiguous key attributes or quarantined symptom conflicts.")
        # These attributes were checked for within-key agreement above.
        row = examples[key]
        unknown = assessment.unknown_symptoms
        k = len(unknown)
        unknown_count[k] += 1
        positive_count[k] += int(assessment.positive)
        per_field.update(unknown)
        growth_change_only += set(unknown) == {"grew", "changed"}
        patient_status[key[0]].append(bool(unknown))
        cohorts["missing" if unknown else "complete"].append({
            "patient": key[0], "photos": assessment.image_count, "positive": assessment.positive,
            "biopsied": assessment.biopsy_confirmed, "biopsy_conflict": assessment.biopsy_conflict,
            "age": int(row["age"]),
            "region": row["region"], "diagnosis": row["diagnostic"], "sex": next(iter(sexes[key])),
        })

    total = len(assessments)
    groups = {name: _group_summary(records) for name, records in cohorts.items()}
    complete, missing = groups["complete"], groups["missing"]
    complete_col, missing_col = _table_column(complete), _table_column(missing)
    statuses = [status for status in patient_status.values() if len(status) >= 2]
    all_missing = sum(all(status) for status in statuses)
    all_complete = sum(not any(status) for status in statuses)
    concordant = all_complete + all_missing
    return {
        "groups": groups, "missing_fraction": missing["keys"] / total,
        "table1": {label: [cell, missing_col[label]] for label, cell in complete_col.items()},
        "unknown_field_counts": [
            {"unknown_fields": k, "keys": unknown_count[k], "positive": positive_count[k],
             "negative": unknown_count[k] - positive_count[k],
             "positive_fraction": _fraction(positive_count[k], unknown_count[k])}
            for k in range(len(SYMPTOMS) + 1)
        ],
        "symptom_missingness": {
            field: {"keys": per_field[field], "fraction": per_field[field] / total} for field in SYMPTOMS
        },
        "growth_change_only": {
            "keys": growth_change_only, "fraction_of_missing": _fraction(growth_change_only, missing["keys"]),
        },
        "patient_concordance": {
            "eligible_patients": len(statuses), "all_complete": all_complete, "all_missing": all_missing,
            "mixed": len(statuses) - concordant, "concordant_patients": concordant,
            "fraction": _fraction(concordant, len(statuses)),
        },
        "standardised_mean_differences": {
            field: _smd(complete[field], missing[field]) for field in ("age", "photos_per_key")
        },
    }
