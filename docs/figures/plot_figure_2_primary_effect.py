"""Reproduce Figure 2 from the archived H1 predictions and run records.

From the project root, in the V1.1 environment:
    python -B docs/figures/plot_figure_2_primary_effect.py --output-dir /path/to/new-output

The output directory may exist, but the two Figure 2 filenames must be free.
PNG rendering uses the Figure 1 renderer (rsvg-convert, CairoSVG, or Java 11+
with an existing Batik installation supplied via BATIK_CLASSPATH).
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from plot_figure_1_cohort_flow import INK, MUTED, TEAL, element, render_png, text


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))
from pad_ufes20_curate import select_columns
from pad_ufes20_primary import ANALYSIS, OOF, OOF_FIELDS, PROTOCOL_SHA256
from pad_v11_primary import primary_report
from pad_v11_s2_inputs import H1_COMMIT, H1_OOF_SHA256, H1_PROVENANCE_SHA256, H1_SUMMARY_SHA256
from pad_v11_training_inputs import _snapshot


STEM = "figure_2_primary_effect_by_outer_fold"


def verified_report(primary_dir):
    """Recompute the plotted values and check them against the authenticated H1 archive."""
    _snapshot(PROJECT / "protocol_v1.1.md", PROTOCOL_SHA256, "Frozen protocol")
    summary = json.loads(_snapshot(primary_dir / "summary.json", H1_SUMMARY_SHA256, "H1 summary"))
    run = json.loads(_snapshot(primary_dir / "run_provenance.json", H1_PROVENANCE_SHA256, "H1 provenance"))
    if (summary["status"] != "complete" or summary["analysis"] != ANALYSIS
            or summary["protocol_sha256"] != PROTOCOL_SHA256 or run["exit_code"] != 0
            or run["code_commit"] != H1_COMMIT or run["summary_sha256"] != H1_SUMMARY_SHA256
            or run["input_provenance"] != summary["inputs"] or run["output_sha256"] != summary["output_sha256"]
            or summary["output_sha256"][OOF] != H1_OOF_SHA256):
        raise ValueError("The H1 summary and run provenance disagree.")

    raw = _snapshot(primary_dir / OOF, H1_OOF_SHA256, "H1 lesion-level predictions")
    rows = list(select_columns(raw, OOF_FIELDS))
    for row in rows:
        for name in ("outer_fold", "n_images", "label"):
            row[name] = int(row[name])
        for name in ("p_A", "p_B"):
            row[name] = float(row[name])
    result = primary_report(rows)
    saved = summary["primary"]
    for field in ("n_keys", "n_patients", "n_images", "n_folds_favouring_B"):
        if result[field] != saved[field]:
            raise ValueError(f"H1 aggregate differs for {field}.")
    for observed, expected in zip([result, *result["folds"]], [saved, *saved["folds"]], strict=True):
        for field in ("outer_fold", "n_keys", "n_patients"):
            if observed.get(field) != expected.get(field):
                raise ValueError(f"H1 fold coverage differs for {field}.")
        for field in ("log_loss_A", "log_loss_B", "delta_log_loss"):
            if not math.isclose(observed[field], expected[field], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"H1 OOF {field} differs from the archived estimate.")
    boot, reference = result["bootstrap"], saved["bootstrap"]
    for field in ("unit", "paired", "seed", "resamples", "generator", "percentile_method"):
        if boot[field] != reference[field]:
            raise ValueError(f"H1 patient bootstrap differs for {field}.")
    if not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
               for a, b in zip(boot["ci_95"], reference["ci_95"], strict=True)):
        raise ValueError("Recomputed paired patient-bootstrap interval differs from H1.")
    return result


def label(parent, x, y, words, size, color=INK, *, bold=False, anchor="start", **attrs):
    text(parent, x, y, words, size, color, bold=bold, text_anchor=anchor, **attrs)


def line(parent, x1, y1, x2, y2, color, width, **attrs):
    element(parent, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width, **attrs)


def format_delta(value):
    return f"{value:.6f}".replace("-", "−")


def figure(report):
    """Draw the reviewed forest plot from aggregate lesion-level estimates."""
    fold_rows = report["folds"]
    pooled = report["delta_log_loss"]
    bootstrap = report["bootstrap"]
    lower, upper = bootstrap["ci_95"]
    interval = f"[{format_delta(lower)}, {format_delta(upper)}]"
    svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "width": "2000", "height": "1240",
                             "viewBox": "0 0 1000 620", "role": "img", "aria-labelledby": "title description"})
    element(svg, "title", id="title").text = "Figure 2. Primary effect by outer fold"
    fold_description = " ".join(f"Fold {row['outer_fold']}: {format_delta(row['delta_log_loss'])}."
                                for row in fold_rows)
    element(svg, "desc", id="description").text = (
        "Lesion-level out-of-fold log loss differences for Model B minus Model A. Negative values favour Model B. "
        f"{fold_description} Pooled difference: {format_delta(pooled)}; paired patient-bootstrap 95% "
        f"confidence interval {interval}.")
    element(svg, "rect", x=0, y=0, width=1000, height=620, fill="#FFFFFF")
    label(svg, 42, 35, "PAD-UFES-20 V1.1  ·  FIGURE 2", 13, TEAL, bold=True, letter_spacing=1.2)
    label(svg, 40, 81, "Primary effect by outer fold", 37, bold=True)
    label(svg, 42, 111, f"{report['n_keys']:,} symptom-complete lesion keys from "
          f"{report['n_patients']:,} patients", 19, MUTED)
    label(svg, 42, 159, "Evaluation set", 16, bold=True)
    label(svg, 192, 159, "Negative values favour Model B", 16, MUTED)
    label(svg, 720, 159, "Estimate", 16, bold=True)
    line(svg, 42, 172, 958, 172, "#C6D1D6", 1)
    element(svg, "rect", x=42, y=401, width=916, height=70, fill="#EAF3F3")

    # Preserve the reviewed axis limits and plot width. Place the marks using
    # the unrounded estimates, independently of their six-decimal labels.
    left, right = 192, 690
    axis_min, axis_max = -0.10, 0.02

    def scale(value):
        return round(left + (value - axis_min) * (right - left) / (axis_max - axis_min), 5)

    for hundredths in range(-10, 3, 2):
        tick = hundredths / 100
        x = scale(tick)
        if hundredths == 0:
            line(svg, x, 173, x, 480, "#52636B", 1.65, stroke_dasharray="5 4", id="zero-reference")
        else:
            line(svg, x, 173, x, 479, "#D7E0E3", 1.0)
        line(svg, x, 480, x, 486, "#8B9BA3", 1.1)
        tick_label = f"{tick:+.2f}".replace("-", "−") if hundredths else "0"
        label(svg, x, 506, tick_label, 16, MUTED if hundredths else INK,
              bold=hundredths == 0, anchor="middle")
    line(svg, left, 480, right, 480, "#8B9BA3", 1)

    for i, row in enumerate(fold_rows):
        fold, delta = row["outer_fold"], row["delta_log_loss"]
        y = 200 + 44*i
        group = element(svg, "g", id=f"fold-{fold}")
        label(group, 42, y+6, f"Fold {fold}", 19)
        element(group, "circle", cx=scale(delta), cy=y, r=6.5, fill="#39556B")
        label(group, 720, y+6, format_delta(delta), 19)

    group = element(svg, "g", id="pooled")
    label(group, 54, 442, "Pooled", 21, TEAL, bold=True)
    low_x, high_x = scale(lower), scale(upper)
    line(group, low_x, 435, high_x, 435, TEAL, 3.5, id="pooled-ci")
    line(group, low_x, 427, low_x, 443, TEAL, 2.5, id="ci-low")
    line(group, high_x, 427, high_x, 443, TEAL, 2.5, id="ci-high")
    x = scale(pooled)
    points = f"{x},426.5 {round(x+8.5, 5)},435 {x},443.5 {round(x-8.5, 5)},435"
    element(group, "polygon", points=points, fill=TEAL, id="pooled-point")
    label(group, 720, 431, format_delta(pooled), 22, TEAL, bold=True)
    label(group, 720, 454, f"95% CI {interval}", 15.5)
    label(svg, (left+right)/2, 539, "Δ log loss (Model B − Model A)", 19, bold=True, anchor="middle")
    line(svg, 42, 555, 958, 555, "#C6D1D6", 1)
    label(svg, 42, 581, "Model A uses image features, age and anatomical site. "
          "Model B adds growth, change and bleeding.", 16, MUTED)
    label(svg, 42, 604, f"Only the pooled interval is shown ({bootstrap['resamples']:,} "
          "paired patient-cluster bootstrap resamples).", 16, MUTED)
    ET.indent(svg, space="  ")
    return ET.tostring(svg, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary-dir", type=Path, default=PROJECT / "private/v1.1-primary-stage4b",
                        help="Archived H1 run directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for new copies of the SVG and PNG")
    args = parser.parse_args()
    svg = args.output_dir / (STEM + ".svg")
    png = svg.with_suffix(".png")
    if svg.exists() or png.exists():
        parser.error("Figure 2 already exists at this destination. Choose another output directory.")
    report = verified_report(args.primary_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as temp:
        staged_svg, staged_png = Path(temp)/svg.name, Path(temp)/png.name
        staged_svg.write_bytes(figure(report))
        render_png(staged_svg, staged_png)
        # link() publishes a complete file and rejects an existing name, even
        # if another invocation created it after the initial existence check.
        os.link(staged_svg, svg)
        try:
            os.link(staged_png, png)
        except OSError:
            svg.unlink()
            raise
    print(f"Wrote Figure 2 from {report['n_keys']:,} lesion keys and "
          f"{report['bootstrap']['resamples']:,} paired patient-cluster resamples.")


if __name__ == "__main__":
    main()
