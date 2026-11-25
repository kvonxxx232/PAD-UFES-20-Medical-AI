"""Draw Figure 1 from the authenticated aggregate Stage 2 summary.

From the project root: python3 -B docs/figures/plot_figure_1_cohort_flow.py
PNG requires rsvg-convert, CairoSVG, or Java 11+ with BATIK_CLASSPATH set to
the installed Batik jars and their dependencies. The aggregate Stage 2
summary supplies every cohort count.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
STEM = "figure_1_dataset_curation_and_cohort_flow"
# Recorded in the authenticated Stage 3 fold summary, input_provenance.stage2_summary_sha256.
SUMMARY_SHA256 = "626482b360ab6b0d4375ebf5d294ef8a127ed1a107046ca0933c62e5543afd7d"
INK, MUTED, TEAL = "#182A37", "#43545F", "#125C69"
EDGE, ARROW = "#A6B8BE", "#536C78"


def check_row(markdown, label, expected):
    rows = [line.strip("|").split("|") for line in markdown.splitlines() if line.startswith("|")]
    matches = [r for r in rows if r[0].strip() == label]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous reference row: {label}")
    actual = [int(v.strip().replace(",", "")) for v in matches[0][1:]]
    if actual[:len(expected)] != expected:
        raise ValueError(f"Reference counts differ for {label}")


def read_flow(summary_path, protocol_path, report_path):
    raw = summary_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUMMARY_SHA256:
        raise ValueError("Stage 2 summary SHA-256 differs from the archived input.")
    summary = json.loads(raw)
    if summary["status"] != "passed" or summary["protocol"] != "protocol-v1.1-frozen":
        raise ValueError("The Stage 2 summary must have passed under the frozen protocol.")
    audit, cohorts = summary["image_audit"], summary["cohorts"]
    curated, complete, incomplete, subset = (cohorts[k] for k in ("curated", "primary", "incomplete", "s3"))
    excluded_keys = audit["original_keys"] - curated["keys"]
    excluded_images = audit["images"] - curated["images"]
    pairs = audit["sha256_pairs"] + audit["rgb_only_pairs"]
    if (pairs != audit["confirmed_pairs"] or excluded_keys != audit["duplicate_keys"]
            or excluded_keys != cohorts["quarantined_keys"] or excluded_images != cohorts["quarantined_images"]
            or cohorts["symptom_conflict_keys"] != 0):
        raise ValueError("Whole-key quarantine does not reconcile with the duplicate audit.")
    for population in (curated, complete, incomplete, subset):
        if population["keys"] != population["positive"] + population["negative"]:
            raise ValueError("Binary class counts do not account for all keys.")
    for field in ("keys", "images", "positive", "negative"):
        if curated[field] != complete[field] + incomplete[field]:
            raise ValueError(f"Completeness split does not reconcile for {field}.")
    overlap = complete["patients"] + incomplete["patients"] - curated["patients"]
    if not 0 <= overlap <= min(complete["patients"], incomplete["patients"]):
        raise ValueError("Invalid overlap of patients between the symptom groups.")
    if any(subset[k] > complete[k] for k in ("keys", "images", "patients", "positive", "negative")):
        raise ValueError("S3 counts exceed the symptom-complete parent.")

    protocol = protocol_path.read_text(encoding="utf-8")
    report = report_path.read_text(encoding="utf-8")
    for digest in (audit["pair_set_sha256"], audit["duplicate_key_set_sha256"], cohorts["metadata_sha256"]):
        if digest not in protocol or digest not in report:
            raise ValueError("A curation reference digest is missing from the protocol or report.")
    check_row(report, "Original inventory", [audit["original_patients"], audit["original_keys"], audit["images"]])
    references = (
        (curated, "Curated cohort after quarantine", "Curated after quarantine"),
        (complete, "Primary symptom-complete cohort", "Symptom-complete, H1 population"),
        (incomplete, "Symptom-incomplete portion of the curated cohort", "Symptom-incomplete, retained"),
        (subset, "S3 biopsy-confirmed subset of the primary cohort", "Biopsy-confirmed and complete, S3 subset"),
    )
    for pop, protocol_row, report_row in references:
        check_row(protocol, protocol_row, [pop[k] for k in ("keys", "patients", "images", "positive", "negative")])
        check_row(report, report_row, [pop[k] for k in ("patients", "keys", "images", "positive", "negative")])
    return audit, cohorts, excluded_keys, excluded_images, overlap


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})


def text(parent, x, y, value, size, color=INK, *, bold=False, **attrs):
    node = element(parent, "text", x=x, y=y, font_family="Arial, Helvetica, sans-serif",
                   font_size=size, font_weight="bold" if bold else "normal", fill=color, **attrs)
    node.text = value


def box(root, name, x, y, w, h, *, fill="white", stroke=EDGE):
    group = element(root, "g", id=name)
    element(group, "rect", x=x, y=y, width=w, height=h, rx=6, fill=fill, stroke=stroke, stroke_width=1.4)
    return group


def arrow(root, name, path):
    element(root, "path", id=name, d=path, fill="none", stroke=ARROW, stroke_width=2,
            stroke_linejoin="round", marker_end="url(#arrowhead)")


def inventory_line(keys, patients, images):
    return f"{keys:,} keys  ·  {patients:,} patients  ·  {images:,} images"


def draw_group(root, name, x, y, title, pop, *, primary=False):
    g = box(root, name, x, y, 498, 108, fill="#EAF3F3" if primary else "white",
            stroke=TEAL if primary else EDGE)
    text(g, x+20, y+30, title, 22, TEAL if primary else INK, bold=True)
    text(g, x+20, y+59, f"{pop['keys']:,} keys", 22, INK, bold=True)
    text(g, x+178, y+59, f"{pop['positive']:,} positive  ·  {pop['negative']:,} negative", 19)
    text(g, x+20, y+87, f"{pop['patients']:,} patients  ·  {pop['images']:,} images", 17, MUTED)


def figure(audit, cohorts, excluded_keys, excluded_images, overlap):
    svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "viewBox": "0 0 1120 735",
                             "width": "1120", "height": "735", "role": "img",
                             "aria-labelledby": "title description"})
    element(svg, "title", id="title").text = "Figure 1. Dataset curation and cohort flow"
    element(svg, "desc", id="description").text = (
        "Duplicate-image quarantine removes whole patient–lesion keys from the original inventory. "
        "The retained keys split into symptom-complete and symptom-incomplete groups. "
        "The biopsy-confirmed S3 subset is drawn exclusively from the complete group.")
    marker = element(element(svg, "defs"), "marker", id="arrowhead", markerWidth=8, markerHeight=8,
                     refX=7, refY=4, orient="auto", markerUnits="userSpaceOnUse")
    element(marker, "path", d="M 0 0 L 7 4 L 0 8 Z", fill=ARROW)
    element(svg, "rect", x=0, y=0, width=1120, height=735, fill="white")
    text(svg, 46, 35, "PAD-UFES-20 V1.1  ·  FIGURE 1", 13, TEAL, bold=True, letter_spacing=1.2)
    text(svg, 44, 81, "Dataset curation and cohort flow", 37, bold=True)
    text(svg, 46, 111, "A lesion key groups all photographs of one patient–lesion combination.", 18, MUTED)

    arrow(svg, "original-to-curated", "M 297 220 V 269")
    arrow(svg, "whole-key-exclusions", "M 297 244 H 621")
    arrow(svg, "curated-to-complete", "M 297 352 V 414")
    arrow(svg, "curated-to-incomplete", "M 297 382 H 823 V 414")
    arrow(svg, "complete-to-s3", "M 297 523 V 564")
    for y in (244, 382):
        element(svg, "circle", cx=297, cy=y, r=3.1, fill=ARROW)

    original = box(svg, "original-inventory", 48, 138, 498, 82)
    text(original, 68, 170, "Original inventory", 23, bold=True)
    text(original, 68, 200, inventory_line(audit["original_keys"], audit["original_patients"], audit["images"]), 20)
    audited = box(svg, "duplicate-audit", 622, 162, 450, 160, fill="#F5F7F8")
    text(audited, 642, 194, "Duplicate-image audit", 22, bold=True)
    text(audited, 642, 223, f"{audit['sha256_pairs']:,} SHA-256-identical pairs", 18)
    rgb = audit["rgb_only_pairs"]
    text(audited, 642, 249, f"{rgb:,} additional RGB-identical {'pair' if rgb == 1 else 'pairs'}", 18)
    element(audited, "line", x1=642, y1=265, x2=1052, y2=265, stroke=EDGE)
    text(audited, 642, 292, f"{excluded_keys:,} keys quarantined  ·  {excluded_images:,} images excluded", 18, bold=True)
    curated = cohorts["curated"]
    kept = box(svg, "curated-cohort", 48, 270, 498, 82, fill="#F0F6F6", stroke=TEAL)
    text(kept, 68, 302, "Curated cohort", 23, TEAL, bold=True)
    text(kept, 68, 332, inventory_line(curated["keys"], curated["patients"], curated["images"]), 20)
    draw_group(svg, "symptom-complete", 48, 415, "Symptom-complete (H1)", cohorts["primary"], primary=True)
    draw_group(svg, "symptom-incomplete", 574, 415, "Symptom-incomplete", cohorts["incomplete"])
    draw_group(svg, "s3-subset", 48, 565, "S3 biopsy-confirmed subset", cohorts["s3"])
    element(svg, "line", x1=594, y1=583, x2=594, y2=652, stroke=EDGE, stroke_width=2)
    text(svg, 615, 601, "Patient overlap", 18, bold=True)
    text(svg, 615, 628, f"{overlap:,} patients contributed keys to both groups.", 18, MUTED)
    text(svg, 615, 651, "Each key belongs to exactly one symptom group.", 18, MUTED)
    element(svg, "line", x1=46, y1=692, x2=1073, y2=692, stroke="#D7E0E3")
    text(svg, 46, 719, "Positive and negative counts refer to the study’s binary cancer label.", 16, MUTED)
    ET.indent(svg, space="  ")
    return ET.tostring(svg, encoding="utf-8", xml_declaration=True)


def render_png(svg, png):
    width = 2400
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", str(width), "-o", str(png), str(svg)], check=True)
    elif importlib.util.find_spec("cairosvg"):
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=width)
    elif os.getenv("BATIK_CLASSPATH") and shutil.which("java"):
        adapter = svg.parent / "FlowRaster.java"
        adapter.write_text("""import java.nio.file.*;
import java.io.OutputStream;
import org.apache.batik.transcoder.*;
import org.apache.batik.transcoder.image.PNGTranscoder;
class FlowRaster {
    public static void main(String[] args) throws Exception {
        PNGTranscoder t = new PNGTranscoder();
        t.addTranscodingHint(PNGTranscoder.KEY_WIDTH, Float.valueOf(args[2]));
        try (OutputStream out = Files.newOutputStream(Path.of(args[1]))) {
            t.transcode(new TranscoderInput(Path.of(args[0]).toUri().toString()), new TranscoderOutput(out));
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
        raise RuntimeError("PNG rendering requires an existing rsvg-convert, CairoSVG, or Java/Batik installation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=PROJECT / "private/v1.1-curation/summary.json")
    parser.add_argument("--protocol", type=Path, default=PROJECT / "protocol_v1.1.md")
    parser.add_argument("--report", type=Path, default=PROJECT / "docs/technical_report_v1.1.md")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    drawing = figure(*read_flow(args.summary, args.protocol, args.report))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as temp:
        svg = Path(temp) / (STEM + ".svg")
        png = svg.with_suffix(".png")
        svg.write_bytes(drawing)
        render_png(svg, png)
        svg.replace(args.output_dir / svg.name)
        png.replace(args.output_dir / png.name)
    print(f"Verified Stage 2 and the reference tables. Wrote {STEM}.svg and .png.")


if __name__ == "__main__":
    main()
