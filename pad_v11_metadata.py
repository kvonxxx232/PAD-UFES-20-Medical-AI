"""Clinical metadata checks for use after the label-blinded V1.1 image audit."""

from collections import defaultdict
from dataclasses import dataclass


Key = tuple[str, str]
SYMPTOMS = ("grew", "changed", "bleed")
REGIONS = frozenset(
    "ABDOMEN ARM BACK CHEST EAR FACE FOOT FOREARM HAND LIP NECK NOSE SCALP THIGH".split()
)
POSITIVE = frozenset(("BCC", "SCC", "MEL"))
DIAGNOSES = POSITIVE.union(("ACK", "NEV", "SEK"))
BOOLEAN = frozenset(("True", "False"))
FIELDS = ("age", "region", "diagnostic", "biopsed", *SYMPTOMS)


@dataclass(frozen=True)
class KeyAssessment:
    positive: bool
    complete: bool
    biopsy_confirmed: bool
    biopsy_conflict: bool
    symptom_conflicts: tuple[str, ...]
    unknown_symptoms: tuple[str, ...]
    image_count: int


class MetadataError(ValueError):
    """Record-level issues are for the local report only."""

    def __init__(self, issues: list[dict]):
        super().__init__(f"Metadata validation found {len(issues)} issue(s).")
        self.issues = issues


def assess_metadata(rows: list[dict[str, str]]) -> dict[Key, KeyAssessment]:
    """Validate every original key. Return assessments only if all fatal checks pass."""
    grouped = defaultdict(list)
    issues = []
    assessments = {}

    def report(key, field, values, reason):
        issues.append({
            "patient_id": key[0], "lesion_id": key[1], "field": field,
            "values": sorted(values), "reason": reason,
        })

    for row in rows:
        key = (row.get("patient_id", ""), row.get("lesion_id", ""))
        absent = [name for name in FIELDS if row.get(name) is None]
        if not all(key) or absent:
            report(key, "row", absent, "Missing key identifier or required column")
            continue
        grouped[key].append(row)

    for key, records in sorted(grouped.items()):
        values = {field: {row[field] for row in records} for field in FIELDS}
        age_values = values["age"]
        # Accept integer year values, compare numerically and preserve raw strings in reports.
        years = set()
        for token in sorted(age_values):
            try:
                if not token.isascii() or not token.isdecimal():
                    raise ValueError
                year = int(token)
                if year > 120:
                    raise ValueError
            except ValueError:
                report(key, "age", [token], "Missing or invalid integer age (expected 0–120)")
            else:
                years.add(year)
        if len(years) > 1:
            report(key, "age", age_values, "Conflicting ages")

        for field, allowed in (("region", REGIONS), ("diagnostic", DIAGNOSES), ("biopsed", BOOLEAN)):
            invalid = values[field] - allowed
            if invalid:
                report(key, field, invalid, "Missing or invalid value")
            if field != "biopsed" and len(values[field] & allowed) > 1:
                report(key, field, values[field], "Conflicting values")

        diagnoses = values["diagnostic"] & DIAGNOSES
        labels = {int(diagnosis in POSITIVE) for diagnosis in diagnoses}
        if len(labels) > 1:
            report(key, "binary_label", labels, "Detailed diagnoses map to conflicting binary labels")

        conflicts = []
        unknown = []
        for field in SYMPTOMS:
            observed = values[field]
            missing = {token for token in observed if token == "UNK" or not token.strip()}
            invalid = observed - BOOLEAN - missing
            if invalid:
                report(key, field, invalid, "Unrecognised symptom token")
            if BOOLEAN <= observed:
                conflicts.append(field)
            if missing:
                unknown.append(field)

        assessments[key] = KeyAssessment(
            positive=labels == {1},
            complete=not (conflicts or unknown),
            biopsy_confirmed=values["biopsed"] == {"True"},
            biopsy_conflict=BOOLEAN <= values["biopsed"],
            symptom_conflicts=tuple(conflicts),
            unknown_symptoms=tuple(unknown),
            image_count=len(records),
        )

    if issues:
        raise MetadataError(issues)
    return assessments
