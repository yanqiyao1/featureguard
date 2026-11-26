import unittest
from itertools import permutations

from tools.experiment import VALUES, run, score


class ExperimentTests(unittest.TestCase):
    def test_score_distinguishes_every_concrete_projection(self):
        self.assertEqual(len({score(cols) for cols in permutations(VALUES)}), 6)
        self.assertEqual(score(("area", "age", "rooms")), 137)
        self.assertEqual(score(("rooms", "age", "area")), 731)

    def test_all_generated_programs(self):
        report = run()
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["groups"]["permutations"], {
            "cases": 36, "expected_clean": 6, "expected_diagnostic": 30, "matched": 36,
        })
        self.assertEqual(report["groups"]["supplemental"], {
            "cases": 7, "expected_clean": 3, "expected_diagnostic": 4, "matched": 7,
        })


if __name__ == "__main__":
    unittest.main()
