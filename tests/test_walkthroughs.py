"""The examples are executable documentation, but the checker must not run them."""

import unittest
from pathlib import Path

from featureguard.engine import analyze
from featureguard.project import load_project


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class Walkthroughs(unittest.TestCase):
    def test_cross_file_quote_explains_both_ends(self):
        result = analyze(load_project(EXAMPLES / "reordered" / "quote.py"))
        self.assertEqual((result.predictions, result.verified), (1, 0))
        issue, = result.diagnostics
        self.assertEqual((issue.code, issue.at.path, issue.expected, issue.actual),
                         ("FG001", "quote.py", ("labor_hours", "machine_hours"),
                          ("machine_hours", "labor_hours")))
        locations = {point.path for _, point in issue.related}
        self.assertEqual(locations, {"quote.py", "training.py", "features.py"})

    def test_path_budget_ablation(self):
        project = load_project(EXAMPLES / "shared" / "quote.py")
        normal = analyze(project)
        self.assertEqual((normal.predictions, normal.verified, normal.diagnostics), (1, 1, ()))
        limited = analyze(project, max_paths=1)
        self.assertCountEqual([d.code for d in limited.diagnostics], ["FG003", "FG900"])
        self.assertEqual(limited.verified, 0)

    def test_real_dataset_script_without_loading_any_data(self):
        result = analyze(load_project(EXAMPLES / "diabetes" / "study.py"))
        self.assertEqual((result.predictions, result.verified), (2, 1))
        issue, = result.diagnostics
        self.assertEqual((issue.code, issue.expected, issue.actual),
                         ("FG001", ("bmi", "bp", "s5"), ("bp", "bmi", "s5")))
