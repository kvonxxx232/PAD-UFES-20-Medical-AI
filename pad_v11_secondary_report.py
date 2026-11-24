"""Check secondary aggregates against the actual Markdown report at its printed precision."""

import re

from pad_ufes20_curate import IMAGE_REFERENCE
from pad_v11_secondary_missingness import BIOPSY_CONFLICT_ROW


def _normalise(text):
    return " ".join(text.replace("−", "-").replace("**", "").split())


NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


class ReportComparison:
    def __init__(self, markdown):
        self.lines = [_normalise(line) for line in markdown.splitlines()]
        self.table_count = 0
        self.table_rows = 0
        self.paragraph_count = 0

    def _position(self, prefix):
        matches = [i for i, line in enumerate(self.lines) if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Report check requires one occurrence of: {prefix}")
        return matches[0]

    def table(self, headers, rows, *, omittable_row=None):
        header = "| " + " | ".join(headers) + " |"
        index = self._position(header)
        start = index + 2
        if (self.lines[index] != header or index + 1 >= len(self.lines)
                or not re.fullmatch(r"[| :\-]+", self.lines[index + 1])):
            raise ValueError(f"Malformed report table: {headers[0]}")
        end = start
        while end < len(self.lines) and self.lines[end].startswith("|"):
            end += 1
        if omittable_row is not None and end - start == len(rows) - 1:
            rows = [row for row in rows if row[0] != omittable_row]
        if end - start != len(rows):
            raise ValueError(f"Report row count differs: {headers[0]}")
        for number, (actual, row) in enumerate(zip(self.lines[start:end], rows), 1):
            expected = [_normalise(str(value)) for value in row]
            if [cell.strip() for cell in actual.strip("|").split("|")] != expected:
                raise ValueError(f"Report value or label differs: {headers[0]}, row {number}")
        self.table_count += 1
        self.table_rows += len(rows)

    def paragraph(self, prefix, numbers):
        line = self.lines[self._position(prefix)]
        actual = [x.replace(",", "") for x in NUMBER.findall(line)]
        expected = [str(x).replace(",", "") for x in numbers]
        if actual != expected:
            raise ValueError(f"Report narrative numbers differ: {prefix}")
        self.paragraph_count += 1


def _six(value, signed=False):
    return format(value, "+.6f" if signed else ".6f")


def _pct(value, digits=2):
    return f"{value:.{digits}%}"


def _s1(check, result):
    rows = []
    for field, name in (("auroc", "AUROC"), ("brier", "Brier score")):
        metric = result[field]
        bounds = " to ".join(_six(v, signed=True) for v in metric["ci_95"])
        rows.append([name, _six(metric["A"]), _six(metric["B"]),
                     _six(metric["delta"], signed=True), bounds])
    check.table(["Pooled measure", "A", "B", "B - A", "Descriptive 95% CI for B - A"], rows)
    check.paragraph("S1 used the original", [result["resamples"], result["auroc"]["defined_draws"]])


def _s3(check, result, complete):
    pooled = result["pooled"]
    rows = []
    for field, name in (("auroc", "AUROC"), ("log_loss", "Log loss"), ("brier", "Brier score")):
        metric = pooled[field]
        rows.append([name, _six(metric["A"]), _six(metric["B"]), _six(metric["delta"], True)])
    check.table(["Metric on biopsy-confirmed complete keys", "A", "B", "B - A"], rows)
    rows = [[r["outer_fold"], r["n_keys"], r["positive"], r["negative"],
             *[_six(r[m]["delta"], True) for m in ("auroc", "log_loss", "brier")]]
            for r in result["folds"]]
    check.table(["Outer fold", "Keys", "Positive", "Negative", "Δ AUROC", "Δ log loss", "Δ Brier"], rows)
    check.paragraph("S3 used the existing", [
        pooled["positive"], pooled["negative"], complete["negative"], pooled["n_keys"],
        pooled["n_patients"], _pct(complete["positive_fraction"]), _pct(pooled["positive_fraction"]),
    ])
    check.paragraph("| Biopsy-confirmed and complete, S3 subset |", [
        pooled["n_patients"], pooled["n_keys"], pooled["n_images"], pooled["positive"], pooled["negative"],
    ])


def _s6(check, result):
    rows = []
    for name in ("A", "B"):
        model = result[name]
        rows.append([name, _six(model["intercept"], True), _six(model["slope"], True),
                     _pct(model["mean_predicted"], 4), _pct(model["observed"], 4)])
    check.table(["Model", "Calibration intercept", "Calibration slope",
                 "Mean predicted positive-class probability", "Observed positive fraction"], rows)

    rows = []
    a_bins, b_bins = result["A"]["bins"], result["B"]["bins"]
    if len(a_bins) != 10 or len(b_bins) != 10:
        raise ValueError("S6 must return ten fixed bins per model.")
    for a, b in zip(a_bins, b_bins):
        if (a["lower"], a["upper"]) != (b["lower"], b["upper"]):
            raise ValueError("S6 bin edges differ between models.")
        close = "]" if a["upper"] == 1 else ")"
        row = [f"[{a['lower']:.1f}, {a['upper']:.1f}{close}"]
        for entry in (a, b):
            row.extend([entry["count"],
                        _pct(entry["mean_predicted"]) if entry["count"] else "—",
                        _pct(entry["observed"]) if entry["count"] else "—"])
        rows.append(row)
    check.table(["Probability bin", "A keys", "A mean predicted", "A observed",
                 "B keys", "B mean predicted", "B observed"], rows)

    highest = []
    for bins in (a_bins, b_bins):
        top = bins[-1]
        highest.extend([_pct(top["mean_predicted"]), top["positive"], top["count"], _pct(top["observed"])])
    check.paragraph("Both slopes were close", [*highest, a_bins[4]["lower"], a_bins[4]["upper"]])


def _s7_table1(check, result):
    # The archived Table 1 omits the conflict row because both counts were zero.
    zero_conflicts = all(group["biopsy_conflicting"] == 0 for group in result["groups"].values())
    check.table(["Characteristic", "Complete", "Missing"],
                [[label, *cells] for label, cells in result["table1"].items()],
                omittable_row=BIOPSY_CONFLICT_ROW if zero_conflicts else None)


def _s7(check, result):
    complete, missing = (result["groups"][k] for k in ("complete", "missing"))
    total = complete["keys"] + missing["keys"]
    counts = result["unknown_field_counts"]
    rows = [[r["unknown_fields"], f"{r['keys']:,}", f"{r['positive']:,}", f"{r['negative']:,}",
             _pct(r["positive_fraction"])] for r in counts]
    check.table(["Unknown fields per key", "Keys", "Positive", "Negative", "Positive fraction"], rows)
    _s7_table1(check, result)

    check.paragraph("S7 used all", [
        total, missing["keys"], _pct(result["missing_fraction"]), _pct(missing["positive_fraction"]),
        _pct(complete["positive_fraction"]), missing["biopsied"], complete["biopsied"], complete["keys"],
        _pct(complete["biopsied"] / complete["keys"]), 1,  # Table 1
    ])
    fields = result["symptom_missingness"]
    field_values = []
    for symptom in ("grew", "changed", "bleed"):
        field_values.extend([fields[symptom]["keys"], _pct(fields[symptom]["fraction"])])
    pattern = result["growth_change_only"]
    check.paragraph("Of the", [total, *field_values, pattern["keys"], missing["keys"],
                               _pct(pattern["fraction_of_missing"]), counts[1]["keys"], counts[3]["keys"]])
    c = result["patient_concordance"]
    check.paragraph(f"Among {c['eligible_patients']:,} patients",
                    [c["eligible_patients"], c["all_complete"], c["all_missing"], c["mixed"],
                     c["concordant_patients"], c["eligible_patients"], _pct(c["fraction"])])
    check.paragraph("Categorical entries are counts", [c["mixed"], 25, 75])
    smd = result["standardised_mean_differences"]
    check.paragraph("The descriptive standardised mean differences",
                    [f"{smd['age']:+.3f}", f"{smd['photos_per_key']:+.3f}"])


def _cohorts_and_sources(check, s7, provenance):
    complete, missing = (s7["groups"][name] for name in ("complete", "missing"))
    check.paragraph("The image audit identified", [
        IMAGE_REFERENCE["confirmed_pairs"], provenance["quarantined_keys"], complete["keys"] + missing["keys"],
        _pct(complete["positive_fraction"]), _pct(missing["positive_fraction"]),
    ])
    for name, prefix in (("curated", "| Curated after quarantine |"),
                         ("primary", "| Symptom-complete, H1 population |"),
                         ("incomplete", "| Symptom-incomplete, retained |")):
        group = provenance["cohorts"][name]
        check.paragraph(prefix, [group[k] for k in ("patients", "keys", "images", "positive", "negative")])
    check.paragraph("Documentation and selection.", [missing["positive"], missing["negative"]])
    digests = [
        ("Frozen protocol text", provenance["protocol_sha256"]),
        ("Original metadata used in the audit", provenance["metadata_sha256"]),
        ("Confirmed image-pair set", IMAGE_REFERENCE["pair_set_sha256"]),
        ("Duplicate-key union", IMAGE_REFERENCE["duplicate_key_set_sha256"]),
        ("Fixed V1.1 outer split table", provenance["splits_sha256"]),
        ("ImageNet ResNet18 checkpoint", provenance["recorded_checkpoint_sha256"]),
        ("Original H1 paired OOF file", provenance["h1_oof_sha256"]),
    ]
    check.table(["Reproduction object", "SHA-256"], [[name, f"`{digest}`"] for name, digest in digests])


def compare_report(markdown, results, provenance):
    """Fail closed on an absent table, a changed value, or a different analysis population."""
    check = ReportComparison(markdown)
    _s1(check, results["S1"])
    _s3(check, results["S3"], results["S7"]["groups"]["complete"])
    _s6(check, results["S6"])
    _s7(check, results["S7"])
    _cohorts_and_sources(check, results["S7"], provenance)
    return {"status": "matched_at_reported_precision", "tables": check.table_count,
            "table_data_rows": check.table_rows, "additional_numeric_lines": check.paragraph_count}
