"""S4 must stop before another multiplier after a failed reproduction or fit."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pad_ufes20_s4 as runner


def synthetic_rows():
    return [{"patient_id": f"synthetic-{fold}-{y}", "lesion_id": "1", "outer_fold": fold,
             "label": y, "n_images": 1, "p_A": .5, "p_B": .5}
            for fold in range(1, 6) for y in (0, 1)]


class ReproductionGateTests(unittest.TestCase):
    def test_failed_factor_one_blocks_later_factors_and_preserves_evidence(self):
        rows = synthetic_rows()
        diagnostics = [{"converged": True, "iteration_cap_reached": False, "error": None,
                        "n_iter": [2], "warnings": []} for _ in range(260)]
        with TemporaryDirectory() as temp:
            dest = Path(temp)
            def reserve(folder, protected):
                folder.mkdir()
                return folder
            with (patch.object(runner, "reserve_private_run", reserve),
                  patch.object(runner, "run_factor", return_value=(rows, [], diagnostics)) as fit,
                  redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "reproduction")):
                runner.run_series(dest, [], None, [], rows, .00100001, [])
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(fit.call_args.args[-1], 1)
            self.assertEqual({p.name for p in dest.iterdir()}, {"s1"})
            report = json.loads((dest / "s1" / runner.SUMMARY).read_bytes())
            self.assertFalse(report["reproduction"]["passed"])
            self.assertTrue((dest / "s1" / runner.FITS).is_file())

    def test_reference_membership_and_raw_lesion_scores(self):
        rows = synthetic_rows()
        self.assertEqual(runner.score_factor(rows, rows)["delta_log_loss"], 0.)
        with self.assertRaisesRegex(ValueError, "differ from H1"):
            runner.score_factor([dict(rows[0], n_images=2), *rows[1:]], rows)


if __name__ == "__main__":
    unittest.main()
