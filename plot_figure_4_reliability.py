"""Regenerate Figure 4 from the original H1 lesion-level OOF file.

Run in the project Python environment:
    python -B plot_figure_4_reliability.py

PNG rendering uses rsvg-convert on PATH, CairoSVG, or an existing Apache Batik
installation supplied via BATIK_CLASSPATH (Batik jars plus dependencies, Java 11+).
No packages, fonts or weights are downloaded.
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from pad_v11_s2_inputs import H1_OOF_SHA256, H1_SUMMARY_SHA256
from pad_v11_secondary_metrics import reliability_bins
from pad_v11_secondary_report import ReportComparison


PROJECT = Path(__file__).resolve().parent
RUN = PROJECT / "private" / "v1.1-primary-stage4b"
STEM = "figure_4_reliability_diagrams"
WIDTH, HEIGHT = 1000, 830
INK, MUTED, GRID = "#182A37", "#43545F", "#D7E0E3"
TEAL, SLATE = "#125C69", "#39556B"


def verified_bytes(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return raw


def load_bins(oof, summary_file, report_file):
    summary = json.loads(verified_bytes(summary_file, H1_SUMMARY_SHA256))
    if summary["status"] != "complete" or summary["output_sha256"]["oof_predictions.csv"] != H1_OOF_SHA256:
        raise ValueError("Run record does not authenticate the completed H1 predictions.")
    rows = list(csv.DictReader(io.StringIO(verified_bytes(oof, H1_OOF_SHA256).decode("utf-8"))))
    n = len(rows)
    keys = {(r["patient_id"], r["lesion_id"]) for r in rows}
    if n != summary["primary"]["n_keys"] or len(keys) != n:
        raise ValueError("OOF rows do not cover the original unique lesion keys.")

    labels = [int(row["label"]) for row in rows]
    bins = {model: reliability_bins(labels, [float(row["p_" + model]) for row in rows])
            for model in ("A", "B")}
    for entries in bins.values():
        if len(entries) != 10 or sum(b["count"] for b in entries) != n:
            raise ValueError("Incomplete S6 bin coverage.")
    # The report is a check at printed precision. The figure retains the original bin means.
    table = []
    for a, b in zip(bins["A"], bins["B"]):
        close = "]" if a["upper"] == 1 else ")"
        row = [f"[{a['lower']:.1f}, {a['upper']:.1f}{close}"]
        for entry in (a, b):
            row.append(entry["count"])
            row.extend(f"{entry[field]:.2%}" if entry["count"] else "—"
                       for field in ("mean_predicted", "observed"))
        table.append(row)
    ReportComparison(report_file.read_text(encoding="utf-8")).table(
        ["Probability bin", "A keys", "A mean predicted", "A observed", "B keys", "B mean predicted", "B observed"],
        table,
    )
    return n, bins


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {name.replace("_", "-"): str(value) for name, value in attrs.items()})


def label(parent, x, y, words, size, color=INK, *, bold=False, anchor="start", **attrs):
    node = element(parent, "text", x=x, y=y, font_family="Arial, Helvetica, sans-serif",
                   font_size=size, font_weight="bold" if bold else "normal", text_anchor=anchor,
                   fill=color, **attrs)
    node.text = words


def line(parent, x1, y1, x2, y2, color, width=1, **attrs):
    return element(parent, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color,
                   stroke_width=width, **attrs)


def draw_panel(svg, model, bins, left, peak):
    panel = element(svg, "g", id=f"model-{model}")
    color = SLATE if model == "A" else TEAL
    features = {"A": "Image features, age and anatomical site",
                "B": "Model A plus growth, change and bleeding"}
    label(panel, left, 160, "Model " + model, 24, color, bold=True)
    label(panel, left, 184, features[model], 15.5, MUTED)
    top, side = 207, 386
    bottom = top + side
    for tick in range(6):
        value = tick / 5
        x, y = left + side * value, bottom - side * value
        line(panel, x, top, x, bottom, GRID)
        line(panel, left, y, left + side, y, GRID)
        label(panel, left - 12, y + 5, f"{value:.1f}", 15, MUTED, anchor="end")
        label(panel, x, bottom + 22, f"{value:.1f}", 15, MUTED, anchor="middle")
    line(panel, left, top, left, bottom, MUTED)
    line(panel, left, bottom, left + side, bottom, MUTED)
    line(panel, left, bottom, left + side, top, "#647883", 1.6,
         stroke_dasharray="6 5", id=f"ideal-{model}")

    # An empty bin gets a zero count and no point. Draw lines only between adjacent occupied bins.
    previous = None
    for b in bins:
        if not b["count"]:
            previous = None
            continue
        point = left + side * b["mean_predicted"], bottom - side * b["observed"]
        if previous is not None:
            line(panel, *previous, *point, color, 2.2)
        previous = point
    for i, b in enumerate(bins):
        if b["count"]:
            dot = element(panel, "circle", cx=left+side*b["mean_predicted"], cy=bottom-side*b["observed"],
                          r=5.4, fill=color, stroke="white", stroke_width=1.5, id=f"point-{model}-{i}")
            element(dot, "title").text = (f"{b['count']} lesion keys, mean predicted {b['mean_predicted']:.2%}, "
                                         f"observed {b['observed']:.2%}")

    hist = element(panel, "g", id=f"counts-{model}")
    baseline, bin_width = 755, side / len(bins)
    line(hist, left, baseline, left+side, baseline, MUTED)
    for i, b in enumerate(bins):
        center = left + (i + 0.5) * bin_width
        height = 48 * b["count"] / peak
        element(hist, "rect", x=center-bin_width*0.39, y=baseline-height, width=bin_width*0.78,
                height=height, fill=color, fill_opacity=0.72)
        label(hist, center, baseline-height-5, str(b["count"]), 14, color, bold=True,
              anchor="middle", id=f"n-{model}-{i}")
    for tick in range(6):
        value = tick / 5
        label(hist, left+side*value, baseline+21, f"{value:.1f}", 14, MUTED, anchor="middle")


def make_svg(n, bins):
    svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "width": str(WIDTH),
                             "height": str(HEIGHT), "viewBox": f"0 0 {WIDTH} {HEIGHT}",
                             "role": "img", "aria-labelledby": "title description"})
    element(svg, "title", id="title").text = "Figure 4. Reliability diagrams for Models A and B"
    element(svg, "desc", id="description").text = (
        f"Two calibration panels for {n:,} symptom-complete lesion keys. Each occupied bin is plotted "
        "at its mean predicted positive-class probability and observed positive fraction. "
        "The dashed diagonal marks equality. Aligned bars give lesion counts per bin.")
    element(svg, "rect", x=0, y=0, width=WIDTH, height=HEIGHT, fill="white")
    label(svg, 42, 35, "PAD-UFES-20 V1.1  ·  FIGURE 4", 13, TEAL, bold=True, letter_spacing=1.2)
    label(svg, 40, 81, "Reliability diagrams for Models A and B", 37, bold=True)
    label(svg, 42, 111, f"{n:,} symptom-complete lesion keys  ·  Original out-of-fold predictions", 17, MUTED)
    line(svg, 726, 105, 760, 105, "#647883", 1.6, stroke_dasharray="6 5")
    label(svg, 770, 111, "Ideal calibration (y = x)", 16, MUTED)
    line(svg, 42, 129, 958, 129, GRID)
    peak = max(b["count"] for entries in bins.values() for b in entries)
    draw_panel(svg, "A", bins["A"], 98, peak)
    draw_panel(svg, "B", bins["B"], 558, peak)
    label(svg, 29, 400, "Observed positive fraction", 18, bold=True,
          anchor="middle", transform="rotate(-90 29 400)")
    label(svg, 521, 646, "Mean predicted positive-class probability", 18, bold=True, anchor="middle")
    line(svg, 42, 662, 958, 662, GRID)
    label(svg, 42, 689, "Lesion keys per probability bin", 17, bold=True)
    line(svg, 42, 789, 958, 789, GRID)
    label(svg, 42, 815, "Each bin includes its lower edge. The last bin also includes 1.", 15, MUTED)
    ET.indent(svg, space="  ")
    return ET.tostring(svg, encoding="utf-8", xml_declaration=True)


def render_png(svg, png):
    width = 2400
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", str(width), "-o", str(png), str(svg)], check=True)
    elif importlib.util.find_spec("cairosvg"):
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=width)
    elif os.environ.get("BATIK_CLASSPATH") and shutil.which("java"):
        # Source-file mode keeps the Batik adapter temporary and requires no build step.
        adapter = svg.parent / "FigureRaster.java"
        adapter.write_text("""import java.io.*;
import java.nio.file.*;
import org.apache.batik.transcoder.*;
import org.apache.batik.transcoder.image.PNGTranscoder;
class FigureRaster {
    public static void main(String[] args) throws Exception {
        PNGTranscoder renderer = new PNGTranscoder();
        renderer.addTranscodingHint(PNGTranscoder.KEY_WIDTH, Float.valueOf(args[2]));
        try (OutputStream out = Files.newOutputStream(Path.of(args[1]))) {
            renderer.transcode(new TranscoderInput(Path.of(args[0]).toUri().toString()), new TranscoderOutput(out));
        }
    }
}
""", encoding="utf-8")
        subprocess.run(["java", "-Djava.awt.headless=true",
                        "-Djavax.xml.parsers.SAXParserFactory=com.sun.org.apache.xerces.internal.jaxp.SAXParserFactoryImpl",
                        "-Dorg.xml.sax.driver=com.sun.org.apache.xerces.internal.parsers.SAXParser",
                        "--class-path", os.environ["BATIK_CLASSPATH"], str(adapter), str(svg), str(png), str(width)],
                       check=True)
    else:
        raise RuntimeError("PNG rendering needs rsvg-convert, CairoSVG, or Java 11+ with BATIK_CLASSPATH set.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oof", type=Path, default=RUN / "oof_predictions.csv")
    parser.add_argument("--run-summary", type=Path, default=RUN / "summary.json")
    parser.add_argument("--report", type=Path, default=PROJECT / "docs" / "technical_report_v1.1.md")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "docs" / "figures")
    args = parser.parse_args()
    n, bins = load_bins(args.oof, args.run_summary, args.report)
    svg_bytes = make_svg(n, bins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as tmp:
        svg = Path(tmp) / (STEM + ".svg")
        png = svg.with_suffix(".png")
        svg.write_bytes(svg_bytes)
        render_png(svg, png)
        for path in (svg, png):
            path.replace(args.output_dir / path.name)
    print(f"Verified {n:,} OOF keys and both ten-bin S6 summaries. Wrote {STEM}.svg and .png.")


if __name__ == "__main__":
    main()
