"""Reproduce Figure 3 from the saved H1, S2 and S4 summaries and OOF files.

From the project environment:
    python -B docs/figures/plot_figure_3_sensitivity.py

Use --private-dir for a relocated run archive and --output-dir for another
destination. PNG uses the neighbouring Figure 1 renderer: rsvg-convert,
CairoSVG, or Java 11+ with an existing Batik installation on BATIK_CLASSPATH.
"""

import argparse
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from plot_figure_1_cohort_flow import INK, MUTED, TEAL, element, render_png, text

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))
from pad_ufes20_curate import select_columns
from pad_ufes20_primary import OOF, OOF_FIELDS, PROTOCOL_SHA256
from pad_ufes20_s4 import score_factor
from pad_v11_s2_inputs import H1_OOF_SHA256, H1_PROVENANCE_SHA256, H1_SUMMARY_SHA256
from pad_v11_s4_nested import FACTORS
from pad_v11_training_inputs import _snapshot


# Fingerprints of the complete, reviewed sensitivity-run summaries. Each
# authenticates its OOF file(s), via the individual factor summaries for S4.
S2_SUMMARY_SHA256 = "98797586f833d6bfe1fd492d48b40eacb96f6b87b86b3384ff596c67ca962851"
S4_SUMMARY_SHA256 = "4b45665b7600ccfd5bef4300316980099d1887e6435c2d42bbfed74abe8a0c83"
STEM = "figure_3_training_population_and_clinical_feature_scaling_sensitivity"
SLATE, GRID = "#39556B", "#D7E0E3"


def read_oof(folder, summary):
    raw = _snapshot(folder / OOF, summary["output_sha256"][OOF], "OOF predictions")
    rows = list(select_columns(raw, OOF_FIELDS))
    for row in rows:
        for field in ("label", "outer_fold", "n_images"):
            row[field] = int(row[field])
        for field in ("p_A", "p_B"):
            row[field] = float(row[field])
    return rows


def check_metrics(observed, pooled, folds):
    def check_losses(actual, reference):
        for key in ("log_loss_A", "log_loss_B", "delta_log_loss"):
            if not math.isclose(actual[key], reference[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"OOF {key} differs from its saved run summary.")

    check_losses(observed, pooled)
    for actual, ref in zip(observed["folds"], folds, strict=True):
        if actual["outer_fold"] != ref["outer_fold"] or actual["keys"] != ref.get("keys", ref.get("n_keys")):
            raise ValueError("Per-fold OOF coverage differs from its saved run summary.")
        check_losses(actual, ref)


def verified_scores(private):
    _snapshot(PROJECT / "protocol_v1.1.md", PROTOCOL_SHA256, "Frozen protocol")
    h1_dir, s2_dir, s4_dir = (private / name for name in
                              ("v1.1-primary-stage4b", "v1.1-secondary-s2", "v1.1-secondary-s4"))
    h1 = json.loads(_snapshot(h1_dir / "summary.json", H1_SUMMARY_SHA256, "H1 summary"))
    run = json.loads(_snapshot(h1_dir / "run_provenance.json", H1_PROVENANCE_SHA256, "H1 provenance"))
    s2 = json.loads(_snapshot(s2_dir / "summary.json", S2_SUMMARY_SHA256, "S2 summary"))
    s4 = json.loads(_snapshot(s4_dir / "summary.json", S4_SUMMARY_SHA256, "S4 summary"))
    if any(s["status"] != "complete" or s["protocol_sha256"] != PROTOCOL_SHA256 for s in (h1, s2, s4)):
        raise ValueError("Run status or frozen protocol mismatch.")
    if (run["exit_code"] != 0 or run["summary_sha256"] != H1_SUMMARY_SHA256
            or run["output_sha256"] != h1["output_sha256"] or h1["output_sha256"][OOF] != H1_OOF_SHA256):
        raise ValueError("H1 output provenance mismatch.")
    for other in (s2, s4):
        lineage = other["inputs"]
        if (lineage["h1_oof_sha256"] != H1_OOF_SHA256 or lineage["h1_summary_sha256"] != H1_SUMMARY_SHA256
                or lineage["splits_sha256"] != h1["inputs"]["splits_sha256"]):
            raise ValueError("The sensitivity run references a different H1 run or outer split.")

    ref_rows = read_oof(h1_dir, h1)
    keys = {(r["patient_id"], r["lesion_id"]) for r in ref_rows}
    patient_folds = {(r["patient_id"], r["outer_fold"]) for r in ref_rows}
    patients = {r["patient_id"] for r in ref_rows}
    n = len(ref_rows)
    if (len(keys) != n or n != h1["primary"]["n_keys"] or len(patient_folds) != len(patients)
            or len(patients) != h1["primary"]["n_patients"] or n != s2["results"]["matched_keys"]):
        raise ValueError("Invalid H1 key coverage or patient grouping.")
    # score_factor checks H1 key, patient, fold, label and image-count alignment,
    # then uses the original clipped, equal-key log loss, pooled and per fold.
    scores = {"H1": score_factor(ref_rows, ref_rows),
              "S2": score_factor(read_oof(s2_dir, s2), ref_rows)}
    check_metrics(scores["H1"], h1["primary"], h1["primary"]["folds"])
    check_metrics(scores["S2"], s2["results"]["s2"], s2["results"]["folds"])
    if [r["factor"] for r in s4["factors"]] != list(FACTORS):
        raise ValueError("Unexpected clinical-feature scaling factors.")
    for summary in s4["factors"]:
        factor = summary["factor"]
        folder = s4_dir / f"s{factor}"
        digest = s4["factor_summary_sha256"][str(factor)]
        saved = json.loads(_snapshot(folder / "summary.json", digest, f"S4 factor {factor} summary"))
        if saved != dict(summary, status="complete"):
            raise ValueError("Top-level and per-factor S4 summaries disagree.")
        if factor == 1 and saved["output_sha256"][OOF] != H1_OOF_SHA256:
            raise ValueError("The archived factor-one predictions differ from H1.")
        score = score_factor(read_oof(folder, saved), ref_rows)
        check_metrics(score, saved, saved["folds"])
        scores[f"s{factor}"] = score
    return n, scores


def number(delta):
    return f"{delta:.6f}".replace("-", "−")


def line(svg, x1, y1, x2, y2, color, width=1):
    element(svg, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width)


def figure(n, scores):
    svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "width": "1200", "height": "765",
                             "viewBox": "0 0 1200 765", "role": "img", "aria-labelledby": "title desc"})
    element(svg, "title", id="title").text = "Figure 3. Training population and clinical-feature scaling sensitivity"
    element(svg, "desc", id="desc").text = (
        f"Pooled out-of-fold differences in log loss on the same {n:,} symptom-complete lesion keys. "
        "Negative values favour Model B. Panel A compares training populations. Panel B varies the clinical "
        "feature multiplier. Exact differences for the five outer folds appear below their pooled estimates.")
    element(svg, "rect", x=0, y=0, width=1200, height=765, fill="white")
    text(svg, 46, 35, "PAD-UFES-20 V1.1  ·  FIGURE 3", 13, TEAL, bold=True, letter_spacing=1.2)
    text(svg, 44, 81, "Training population and clinical-feature scaling sensitivity", 37, bold=True)
    text(svg, 46, 113, f"All settings use the same {n:,} symptom-complete outer-test lesion keys. "
         "Negative Δ log loss favours Model B.", 18, MUTED)
    line(svg, 46, 132, 1154, 132, GRID)

    panels = (
        ("A", "Training population sensitivity", "S2 adds symptom-incomplete keys to each training fold.", 48, 137, 559,
         (("H1", "H1", "Complete cases"), ("S2", "S2", "Full curated cohort"))),
        ("B", "Clinical-feature scaling sensitivity", "S4 scales clinical features during complete-case training.", 635, 700, 1152,
         (("s1", "s = 1", "Primary setting"), ("s3", "s = 3", "Sensitivity"), ("s10", "s = 10", "Sensitivity"))),
    )
    # One shared scale; extend it to the next tick beyond the observed values.
    tick_step = 0.02
    estimates = [row["delta_log_loss"] for row in scores.values()]
    lo = min(-1, math.floor(min(estimates) / tick_step))
    hi = max(0, math.ceil(max(estimates) / tick_step))

    def y(value):
        return 233 + 172 * (hi - value / tick_step) / (hi - lo)

    element(svg, "rect", x=46, y=493, width=1108, height=34, fill="#EAF3F3")
    text(svg, 59, 516, "Outer fold", 16, bold=True)
    for fold in range(1, 6):
        baseline = 554 + (fold - 1)*28
        text(svg, 59, baseline, f"Fold {fold}", 16, MUTED)
        line(svg, 46, baseline+8, 1154, baseline+8, GRID)

    for letter, title, subtitle, start, left, right, settings in panels:
        panel = element(svg, "g", id=f"panel-{letter}")
        text(panel, start, 173, letter, 23, TEAL, bold=True)
        text(panel, start+33, 173, title, 23, bold=True)
        text(panel, start, 199, subtitle, 16, MUTED)
        for tick in range(lo, hi+1):
            val = tick*tick_step
            line(panel, left, y(val), right, y(val), MUTED if tick == 0 else GRID, 1.8 if tick == 0 else 1)
            label = f"{val:.2f}".replace("-", "−") if tick else "0"
            text(panel, left-11, y(val)+5, label, 15, MUTED, text_anchor="end")
        if letter == "A":
            text(panel, 56, 322, "Pooled Δ log loss", 17, bold=True, text_anchor="middle",
                 transform="rotate(-90 56 322)")

        for i, (key, label, detail) in enumerate(settings):
            x = left + (i + 0.5)*(right-left)/len(settings)
            score = scores[key]
            delta = score["delta_log_loss"]
            color = TEAL if key in ("H1", "s1") else SLATE
            # Bars start at zero. Their labels use the original unrounded delta.
            top, bottom = sorted([y(0), y(delta)])
            element(panel, "rect", id=f"estimate-{key}", x=x-21, y=top, width=42,
                    height=bottom-top, fill=color)
            label_y = y(delta)+23 if delta < 0 else y(delta)-10
            element(panel, "rect", x=x-52, y=label_y-17, width=104, height=21, fill="white")
            text(panel, x, label_y, number(delta), 19, color, bold=True,
                 text_anchor="middle", id=f"pooled-{key}")
            text(panel, x, 447, label, 20, color, bold=True, text_anchor="middle")
            text(panel, x, 470, detail, 16, MUTED, text_anchor="middle")
            text(panel, x, 516, label, 16, INK, bold=True, text_anchor="middle")
            for fold in score["folds"]:
                index = fold["outer_fold"]
                text(panel, x, 554+(index-1)*28, number(fold["delta_log_loss"]), 17, color,
                     text_anchor="middle", id=f"fold-{key}-{index}")

    text(svg, 46, 706, "Δ log loss = LogLoss(B) − LogLoss(A). Model B adds growth, change and bleeding to image, age and site.", 16, MUTED)
    text(svg, 46, 733, "S4 at s = 1 reproduces H1. S2 and S4 are prespecified descriptive sensitivity analyses.", 16, MUTED)
    ET.indent(svg, space="  ")
    return ET.tostring(svg, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-dir", type=Path, default=PROJECT / "private")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    n, scores = verified_scores(args.private_dir)
    drawing = figure(n, scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as temp:
        svg = Path(temp) / (STEM + ".svg")
        png = svg.with_suffix(".png")
        svg.write_bytes(drawing)
        render_png(svg, png)
        svg.replace(args.output_dir / svg.name)
        png.replace(args.output_dir / png.name)
    print(f"Verified 5 OOF files on {n:,} aligned keys, including all 25 fold deltas.")
    print("Pooled deltas:", ", ".join(f"{key} {number(score['delta_log_loss'])}" for key, score in scores.items()))


if __name__ == "__main__":
    main()
