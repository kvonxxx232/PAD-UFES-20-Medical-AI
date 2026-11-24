"""Report checks and CLI failure behaviour on synthetic aggregates only."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pad_ufes20_secondary as cli
from pad_v11_secondary_report import ReportComparison


class SecondaryReportTests(unittest.TestCase):
    def test_table_preserves_signed_values_column_order_and_display_precision(self):
        table = "| Measure | A | B | Δ |\n| --- | ---: | ---: | ---: |\n| Loss | 0.20 | 0.15 | −0.05 |"
        headers, rows = ["Measure", "A", "B", "Δ"], [["Loss", "0.20", "0.15", "-0.05"]]
        ReportComparison(table).table(headers, rows)
        for old, new in [("−0.05", "+0.05"), ("0.20", "0.2"), ("0.15", "0.16"), ("A | B", "B | A")]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                ReportComparison(table.replace(old, new)).table(headers, rows)

    def test_missing_extra_or_truncated_table_rejected(self):
        header = "| Group | Count |"
        body = "| --- | --- |\n| Sample | 2 |"
        for text in (header, header + "\n| --- | --- |", header + "\n" + body + "\n| Other | 3 |",
                     header + "\n" + body + "\n\n" + header + "\n" + body):
            with self.subTest(), self.assertRaises(ValueError):
                ReportComparison(text).table(["Group", "Count"], [["Sample", "2"]])

    def test_narrative_checks_denominators_signs_and_prevalences(self):
        line = "S7 included **1,234** keys, 10/50 (20.00%) in the example, with SMD −0.030."
        expected = [1234, 10, 50, "20.00%", "-0.030"]
        ReportComparison(line).paragraph("S7 included", expected)
        for old, new in (("50", "51"), ("20.00%", "20.01%"), ("−0.030", "+0.030")):
            with self.subTest(), self.assertRaises(ValueError):
                ReportComparison(line.replace(old, new)).paragraph("S7 included", expected)

    def test_report_mismatch_aborts_cli_without_emitting_even_partial_aggregates(self):
        metric = dict(A=0.5, B=0.5, delta=0.0, ci_95=[0.0, 0.0], defined_draws=8)
        inputs = SimpleNamespace(oof_rows=[], s3_keys=set(), metadata_rows=[], provenance={})
        table = ("| Pooled measure | A | B | B − A | Descriptive 95% CI for B − A |\n"
                 "| --- | --- | --- | --- | --- |\n"
                 "| AUROC | 0.500001 | 0.500000 | +0.000000 | +0.000000 to +0.000000 |\n"
                 "| Brier score | 0.500000 | 0.500000 | +0.000000 | +0.000000 to +0.000000 |\n")
        with TemporaryDirectory() as tmp:
            report = Path(tmp) / "synthetic_report.md"
            report.write_text(table, encoding="utf-8")
            stdout, stderr = StringIO(), StringIO()
            with (patch.object(cli, "load_secondary_inputs", return_value=inputs),
                  patch.object(cli, "s1_summary", return_value={"auroc": metric, "brier": metric, "resamples": 8}),
                  patch.object(cli, "s3_summary", return_value={}), patch.object(cli, "s6_summary", return_value={}),
                  patch.object(cli, "s7_missingness", return_value={}),
                  redirect_stdout(stdout), redirect_stderr(stderr)):
                status = cli.main(["--metadata", "unused.csv", "--report", str(report)])
            self.assertNotEqual(status, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("Report value or label differs", stderr.getvalue())
            self.assertEqual(report.read_text(encoding="utf-8"), table)
            self.assertEqual(list(Path(tmp).iterdir()), [report])


if __name__ == "__main__":
    unittest.main()
