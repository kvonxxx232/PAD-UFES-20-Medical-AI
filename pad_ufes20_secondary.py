"""Reproduce S1, S3, S6 and S7 from saved V1.1 inputs and check the technical report.

From the project root in the V1.1 environment:
    python -B pad_ufes20_secondary.py --metadata /path/to/verified/metadata.csv

The project-local defaults point to existing private artefacts. The original
metadata supplies photo-level clinical fields absent from the curated key list.
Successful stdout is a single aggregate JSON document. Input or report mismatches
produce a nonzero exit status and no results on stdout.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

from pad_v11_secondary_inputs import load_secondary_inputs
from pad_v11_secondary_metrics import s1_summary, s3_summary, s6_summary
from pad_v11_secondary_missingness import s7_missingness
from pad_v11_secondary_report import compare_report


PROJECT = Path(__file__).resolve().parent
PRIVATE = PROJECT / "private"


def run(args):
    report = args.report.read_bytes()
    inputs = load_secondary_inputs(args.curation, args.metadata, args.review,
                                   args.fold_dir, args.primary_dir)
    results = {
        "S1": s1_summary(inputs.oof_rows),
        "S3": s3_summary(inputs.oof_rows, inputs.s3_keys),
        "S6": s6_summary(inputs.oof_rows),
        "S7": s7_missingness(inputs.metadata_rows),
    }
    comparison = compare_report(report.decode("utf-8"), results, inputs.provenance)
    return {
        "analysis": "V1.1 prespecified descriptive secondary analyses",
        "report_comparison": comparison,
        "provenance": {
            **inputs.provenance, "report_sha256": hashlib.sha256(report).hexdigest(),
            "software": {"python": sys.version.split()[0], **{
                package: version(package) for package in ("numpy", "scipy", "scikit-learn")
            }},
        },
        **results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, type=Path, help="Original metadata verified by Stage 2")
    parser.add_argument("--curation", type=Path, default=PRIVATE / "v1.1-curation")
    parser.add_argument("--review", type=Path, default=PRIVATE / "v1.1-review" / "recovered_review.json")
    parser.add_argument("--fold-dir", type=Path, default=PRIVATE / "v1.1-outer-folds")
    parser.add_argument("--primary-dir", type=Path, default=PRIVATE / "v1.1-primary-stage4b")
    parser.add_argument("--report", type=Path, default=PROJECT / "docs" / "technical_report_v1.1.md")
    args = parser.parse_args(argv)
    try:
        output = json.dumps(run(args), ensure_ascii=False, allow_nan=False, indent=2)
    except OSError as exc:
        print(f"Secondary reproduction stopped: could not read a required input ({type(exc).__name__}).",
              file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"Secondary reproduction stopped: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
