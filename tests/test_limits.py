from pathlib import Path
import unittest
from unittest.mock import patch

from featureguard.engine import analyze
from featureguard.project import load_project


class LimitTests(unittest.TestCase):
    def test_work_limit_never_publishes_partial_verdicts(self):
        entry = Path(__file__).resolve().parents[1] / "examples" / "shared" / "quote.py"
        project = load_project(entry)
        with patch("featureguard.engine._MAX_TRANSFERS", 20):
            result = analyze(project)
        issue, = result.diagnostics
        self.assertEqual((issue.code, result.verified, result.predictions), ("FG900", 0, 0))
        self.assertIn("no partial verdicts", issue.message)
