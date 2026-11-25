"""Reproduce Figure 5 in the V1.1 project environment.

python -B docs/figures/plot_figure_5_symptom_missingness.py --metadata /path/to/metadata.csv

Uses the verified Stage 2 key lists, their source metadata, the archived S7
analysis and the SVG/PNG rendering helpers supplied alongside Figure 1.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from plot_figure_1_cohort_flow import INK, MUTED, SUMMARY_SHA256, TEAL, element, render_png, text

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))
from pad_ufes20_curate import select_columns
from pad_ufes20_primary import PROTOCOL_SHA256
from pad_v11_fold_inputs import _unique
from pad_v11_metadata import FIELDS
from pad_v11_secondary_missingness import s7_missingness
from pad_v11_secondary_report import ReportComparison, _s7
from pad_v11_training_inputs import _snapshot


STEM = "figure_5_symptom_missingness_and_case_composition"
SLATE, GRID = "#39556B", "#D7E0E3"
KEY_FIELDS = ("patient_id", "lesion_id")


def verified_s7(curation, metadata, report):
    _snapshot(PROJECT / "protocol_v1.1.md", PROTOCOL_SHA256, "Frozen protocol")
    summary = json.loads(_snapshot(curation / "summary.json", SUMMARY_SHA256, "Stage 2 summary"))
    if summary["status"] != "passed" or summary["protocol"] != "protocol-v1.1-frozen":
        raise ValueError("Expected the passed frozen-protocol curation.")
    hashes = summary["output_sha256"]
    kept = _snapshot(curation / "curated_keys.csv", hashes["curated_keys.csv"], "Curated keys")
    excluded = _snapshot(curation / "quarantine_manifest.csv", hashes["quarantine_manifest.csv"], "Quarantine")
    retained = _unique(select_columns(kept, (*KEY_FIELDS, "image_count")), KEY_FIELDS)
    quarantine = _unique(select_columns(excluded, KEY_FIELDS), KEY_FIELDS)
    cohorts = summary["cohorts"]
    if retained.keys() & quarantine.keys() or len(quarantine) != cohorts["quarantined_keys"]:
        raise ValueError("Stage 2 quarantine and retained keys do not reconcile.")

    raw = _snapshot(metadata, cohorts["metadata_sha256"], "Stage 2 source metadata")
    photos = [r for r in select_columns(raw, (*KEY_FIELDS, "img_id", "gender", *FIELDS))
              if (r["patient_id"], r["lesion_id"]) in retained]
    coverage = Counter((r["patient_id"], r["lesion_id"]) for r in photos)
    if coverage != {key: int(r["image_count"]) for key, r in retained.items()}:
        raise ValueError("Metadata photos do not match the retained key inventory.")
    result = s7_missingness(photos)
    for name, reference in (("complete", "primary"), ("missing", "incomplete")):
        group = result["groups"][name]
        observed = {k: group[k] for k in ("keys", "patients", "positive", "negative")}
        observed["images"] = group["photos"]
        if observed != cohorts[reference]:
            raise ValueError(f"{name} group differs from Stage 2.")
    if sum(row["keys"] for row in result["unknown_field_counts"]) != cohorts["curated"]["keys"]:
        raise ValueError("Unknown symptom strata do not cover the curated cohort.")
    # Compare all of S7 (including every archived Table 1 row) before rendering.
    check = ReportComparison(report.read_text(encoding="utf-8"))
    _s7(check, result)
    return result, check


def line(group, x1, y1, x2, y2, color, width=1):
    element(group, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width)


def panel(svg, letter, title, axis_y, axis_x, count_label, rows, left, right, caption_x):
    group = element(svg, "g", id=f"panel-{letter}")
    first = letter == "A"
    color = SLATE if first else TEAL
    top, bottom = 222, 448
    text(group, caption_x, 169, letter, 22, color, bold=True)
    text(group, caption_x+32, 169, title, 19, bold=True)
    text(group, caption_x, 192, ("One point per symptom-count group" if first else
                                "Biopsy recorded on every photo of a key"), 15, MUTED)
    for step in range(5):
        frac = step / 4
        y = bottom - (bottom-top)*frac
        line(group, left, y, right, y, GRID if step else MUTED)
        text(group, left-10, y+5, f"{frac:.0%}", 15, MUTED, text_anchor="end")
    line(group, left, top, left, bottom, MUTED)
    mid_y, axis_label_x = (top + bottom) / 2, left - 62
    text(group, axis_label_x, mid_y, axis_y, 16, bold=True, text_anchor="middle",
         transform=f"rotate(-90 {axis_label_x} {mid_y})")
    text(group, (left+right)/2, 505, axis_x, 17, bold=True, text_anchor="middle")
    text(group, caption_x, 549, count_label, 16, MUTED, bold=True)

    spacing = (right - left) / len(rows)
    for i, (category, numerator, denominator) in enumerate(rows):
        frac = numerator / denominator
        x = left + (i + 0.5)*spacing
        y = bottom - frac*(bottom-top)
        if first:
            # The two small groups identified in S7 receive open markers.
            fill = "white" if category in ("1", "3") else color
            element(group, "circle", id=f"estimate-{letter}-{i}", cx=x, cy=y, r=7,
                    fill=fill, stroke=color, stroke_width=2.5)
        else:
            w = spacing * 0.47
            element(group, "rect", id=f"estimate-{letter}-{i}", x=x-w/2, y=y,
                    width=w, height=bottom-y, fill=color)
        label_y = y - (14 if first else 8)
        if first:
            element(group, "rect", x=x-37, y=label_y-17, width=74, height=21, fill="white")
        text(group, x, label_y, f"{frac:.2%}", 19, color, bold=True,
             text_anchor="middle")
        text(group, x, 474, category, 18, text_anchor="middle")
        text(group, x, 580, f"{numerator:,} / {denominator:,}", 19, color,
             bold=True, text_anchor="middle", id=f"counts-{letter}-{i}")


def make_figure(result):
    counts = result["unknown_field_counts"]
    groups = result["groups"]
    rows_a = [(str(r["unknown_fields"]), r["positive"], r["keys"]) for r in counts]
    rows_b = [(k.capitalize(), groups[k]["biopsied"], groups[k]["keys"]) for k in ("complete", "missing")]
    sizes = {r["unknown_fields"]: r["keys"] for r in counts}
    n = sum(sizes.values())
    svg = ET.Element("svg", {"xmlns":"http://www.w3.org/2000/svg", "viewBox":"0 0 1120 700",
                             "width":"1120", "height":"700", "role":"img",
                             "aria-labelledby":"title description"})
    element(svg, "title", id="title").text = "Figure 5. Symptom missingness and case composition"
    element(svg, "desc", id="description").text = (
        f"Descriptive S7 analysis of {n:,} curated lesion keys. Cancer-positive keys by number of unknown "
        "symptoms: " + ", ".join(f"{cat} unknown, {pos} of {total}" for cat, pos, total in rows_a) +
        ". Biopsy-confirmed keys: " + ", ".join(f"{cat}, {pos} of {total}" for cat, pos, total in rows_b) + ".")
    element(svg, "rect", x=0, y=0, width=1120, height=700, fill="white")
    text(svg, 46, 35, "PAD-UFES-20 V1.1  ·  FIGURE 5", 13, TEAL, bold=True, letter_spacing=1.2)
    text(svg, 44, 81, "Symptom missingness and case composition", 37, bold=True)
    text(svg, 46, 111, f"Prespecified descriptive analysis (S7)  ·  {n:,} curated lesion keys", 18, MUTED)
    line(svg, 46, 129, 1074, 129, GRID)
    element(svg, "rect", x=46, y=524, width=1028, height=71, fill="#F1F6F6")
    panel(svg, "A", "Cancer-positive fraction by number of unknown symptoms", "Cancer-positive fraction",
          "Number of unknown symptoms per key", "Cancer-positive / total lesion keys", rows_a, 108, 605, 52)
    panel(svg, "B", "Biopsy status by symptom completeness", "Biopsy-confirmed fraction",
          "Symptom completeness", "Biopsy-confirmed / total lesion keys", rows_b, 743, 1068, 652)
    text(svg, 46, 625, "Missing means that growth, change or bleeding was unknown on at least one photo of the lesion.", 16, MUTED)
    text(svg, 46, 651, f"Open points mark the small groups with one or three unknown symptoms "
         f"({sizes[1]:,} and {sizes[3]:,} keys, respectively).", 16, MUTED)
    text(svg, 46, 677, "These proportions describe the recorded cases. They cannot establish why symptoms were missing.", 16, MUTED)
    ET.indent(svg, space="  ")
    return ET.tostring(svg, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True, help="Source metadata verified by Stage 2")
    parser.add_argument("--curation", type=Path, default=PROJECT / "private/v1.1-curation")
    parser.add_argument("--report", type=Path, default=PROJECT / "docs/technical_report_v1.1.md")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    result, check = verified_s7(args.curation, args.metadata, args.report)
    data = make_figure(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as tmp:
        svg = Path(tmp) / (STEM + ".svg")
        png = svg.with_suffix(".png")
        svg.write_bytes(data)
        render_png(svg, png)
        svg.replace(args.output_dir / svg.name)
        png.replace(args.output_dir / png.name)
    print(f"S7 matched {check.table_rows} table rows and {check.paragraph_count} numeric passages. "
          f"Wrote {STEM}.svg and .png.")


if __name__ == "__main__":
    main()
