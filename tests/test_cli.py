import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from featureguard.cli import main


TRAIN = """\
import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("never-opened.csv")
model = LogisticRegression()
model.fit(df[["size", "year"]].to_numpy(), y)
"""


def invoke(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        status = main([str(arg) for arg in args])
    return status, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.entry = self.root / "train.py"

    def test_unknown_schema_is_not_a_clean_prediction(self):
        self.entry.write_text(TRAIN + "model.predict(unknown())\n")
        status, out, err = invoke(self.entry, "--format", "json")
        self.assertEqual((status, err), (1, ""))
        payload = json.loads(out)
        self.assertEqual(payload["stats"], {"predictions": 1, "verified": 0})
        self.assertEqual([d["code"] for d in payload["diagnostics"]], ["FG003"])
        self.assertIsNone(payload["diagnostics"][0]["actual"])

    def test_complete_json_report_keeps_evidence_locations(self):
        self.entry.write_text(TRAIN + 'model.predict(df[["year", "size"]].to_numpy())\n')
        status, out, err = invoke("--root", self.root, "--format", "json", self.entry)
        self.assertEqual((status, err), (1, ""))
        report = json.loads(out)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["stats"], {"predictions": 1, "verified": 0})
        (problem,) = report["diagnostics"]
        self.assertEqual(problem["code"], "FG001")
        self.assertEqual(problem["expected"], ["size", "year"])
        self.assertEqual(problem["actual"], ["year", "size"])
        self.assertEqual(problem["at"]["path"], "train.py")
        self.assertEqual(problem["at"]["line"], 6)
        self.assertGreaterEqual(problem["at"]["column"], 1)
        refs = problem["related"]
        self.assertTrue(any(ref["label"] and ref["at"]["line"] == 5 for ref in refs))

    def test_clean_program_is_not_executed(self):
        sentinel = self.root / "must_not_exist"
        self.entry.write_text(
            TRAIN + 'model.predict(df[["size", "year"]].to_numpy())\n'
            + f'open({str(sentinel)!r}, "w").close()\n'
        )
        result = subprocess.run(
            [sys.executable, "-m", "featureguard", str(self.entry)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 verified", result.stdout)
        self.assertFalse(sentinel.exists())

    def test_input_error_formats(self):
        status, out, err = invoke(self.entry)
        self.assertEqual((status, out), (2, ""))
        self.assertTrue(err)
        status, out, err = invoke(self.entry, "--format", "json")
        self.assertEqual((status, err), (2, ""))
        payload = json.loads(out)
        self.assertEqual(payload["diagnostics"], [])
        self.assertTrue(payload["errors"][0]["message"])

    def test_invalid_path_budget_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as error:
            invoke("--max-paths", "0", self.entry)
        self.assertEqual(error.exception.code, 2)

    def test_path_budget_reaches_analyzer(self):
        self.entry.write_text(
            TRAIN + "if flag:\n    cols = ['size', 'year']\nelse:\n    cols = ['year', 'size']\n"
            + "model.predict(df[cols].to_numpy())\n"
        )
        for args, code in [((), "FG004"), (("--max-paths", "1"), "FG900")]:
            with self.subTest(code=code):
                status, out, err = invoke("--format", "json", *args, self.entry)
                self.assertEqual((status, err), (1, ""))
                self.assertIn(code, [d["code"] for d in json.loads(out)["diagnostics"]])

    def test_empty_prediction_coverage_is_opt_in_and_root_relative(self):
        nested = self.root / "jobs" / "serve.py"
        nested.parent.mkdir()
        nested.write_text('import joblib\nmodel = joblib.load("model.pkl")\nmodel.predict(data)\n')
        base = ("--root", self.root, "--format", "json", nested)
        status, out, _ = invoke(*base)
        self.assertEqual((status, json.loads(out)["diagnostics"]), (0, []))
        status, out, err = invoke("--require-prediction", *base)
        self.assertEqual((status, err), (1, ""))
        payload = json.loads(out)
        self.assertEqual(payload["stats"], {"predictions": 0, "verified": 0})
        (notice,) = payload["diagnostics"]
        self.assertEqual(notice["code"], "FG901")
        self.assertEqual(notice["at"], {"path": "jobs/serve.py", "line": 1, "column": 1})

    def test_require_prediction_preserves_covered_and_invalid_inputs(self):
        status, out, _ = invoke("--require-prediction", "--format", "json", self.entry)
        self.assertEqual((status, json.loads(out)["diagnostics"]), (2, []))
        self.entry.write_text(TRAIN + 'model.predict(df[["size", "year"]].to_numpy())\n')
        status, out, _ = invoke("--require-prediction", "--format", "json", self.entry)
        self.assertEqual((status, json.loads(out)["diagnostics"]), (0, []))


if __name__ == "__main__":
    unittest.main()
