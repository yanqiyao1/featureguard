"""Validate feature-order findings on generated programs; run no analyzed code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory

from featureguard.engine import analyze
from featureguard.project import load_project


HEADER = """\
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("not_accessed.csv")
y = df["target"]
flag = os.getenv("USE_LATEST")
m = LogisticRegression()
"""
# Distinct single-digit feature values and decimal-place weights make all six
# projections distinct. This is a toy positional scorer, not a trained model.
VALUES = {"area": 1, "age": 3, "rooms": 7}
WEIGHTS = (100, 10, 1)


def score(columns):
    """Concrete output of a fixed linear scorer on this one distinctive row."""
    return sum(weight * VALUES[col] for weight, col in zip(WEIGHTS, columns))


def project(columns):
    return f"df[{list(columns)!r}].to_numpy()"


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    expected: tuple[str, ...]


def order_cases():
    orders = tuple(permutations(VALUES))
    for train in orders:
        for serve in orders:
            source = f"m.fit({project(train)}, y)\nm.predict({project(serve)})\n"
            # The oracle measures a concrete output, not the checker's schema comparison.
            expected = ("FG001",) if score(train) != score(serve) else ()
            yield Case(f"{'/'.join(train)} -> {'/'.join(serve)}", HEADER + source, expected)


def extras():
    fit = f"m.fit({project(VALUES)}, y)\n"
    predict = f"m.predict({project(VALUES)})\n"
    pieces = (
        ("missing", fit + f"m.predict({project(('area', 'age'))})", ("FG002",)),
        ("extra", fit + f"m.predict({project((*VALUES, 'depth'))})", ("FG002",)),
        ("dynamic", fit + "cols = os.environ['FEATURES'].split(',')\n"
         "m.predict(df[cols].to_numpy())", ("FG003",)),
        ("paired-branch", "if flag:\n    cols = ['area', 'age']\nelse:\n"
         "    cols = ['age', 'area']\nm.fit(df[cols].to_numpy(), y)\n"
         "m.predict(df[cols].to_numpy())", ()),
        ("mixed-branch", fit + "if flag:\n    cols = ['area', 'age', 'rooms']\nelse:\n"
         "    cols = ['rooms', 'age', 'area']\nm.predict(df[cols].to_numpy())", ("FG004",)),
        ("concat", fit + "import numpy as np\nX = np.hstack([df[['area']].to_numpy(), "
         "df[['age', 'rooms']].to_numpy()])\nm.predict(X)", ()),
        ("refit-via-alias", f"m.fit({project(reversed(VALUES))}, y)\n"
         + "mirror = m\n" + fit.replace("m.fit", "mirror.fit") + predict, ()),
    )
    for name, source, expected in pieces:
        yield Case(name, HEADER + source + "\n", expected)


def run():
    groups, failures = {}, []
    with TemporaryDirectory() as tmp:
        entry = Path(tmp) / "sample.py"
        for label, cases in (("permutations", order_cases()), ("supplemental", extras())):
            totals = {"cases": 0, "expected_clean": 0, "expected_diagnostic": 0, "matched": 0}
            for case in cases:
                entry.write_text(case.source, encoding="utf-8")
                result = analyze(load_project(entry))
                codes = tuple(d.code for d in result.diagnostics)
                correct = (codes == case.expected and result.predictions == 1
                           and result.verified == int(not case.expected))
                totals["cases"] += 1
                totals["expected_diagnostic" if case.expected else "expected_clean"] += 1
                totals["matched"] += int(correct)
                if not correct:
                    failures.append({"name": case.name, "expected": case.expected, "got": codes,
                                     "predictions": result.predictions, "verified": result.verified})
            groups[label] = totals
    return {
        "numeric_example": {"trained_order": list(VALUES), "prediction_order": list(reversed(VALUES)),
                            "baseline_score": score(VALUES), "swapped_score": score(reversed(VALUES))},
        "groups": groups, "failures": failures,
    }


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2))
    raise SystemExit(int(bool(report["failures"])))
