"""Command-line rendering for FeatureGuard's read-only project checker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .domain import Location
from .engine import analyze
from .project import ProjectError, load_project
from .result import Diagnostic


def _positive(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return count


def _location(at: Location) -> dict:
    return {"path": str(at.path), "line": at.line, "column": at.column}


def _display(at: Location) -> str:
    return f"{at.path}:{at.line}:{at.column}"


def _record(d: Diagnostic) -> dict:
    return {
        "code": d.code,
        "at": _location(d.at),
        "message": d.message,
        "expected": d.expected,
        "actual": d.actual,
        "related": [{"label": label, "at": _location(at)} for label, at in d.related],
    }


def _print_diagnostic(d: Diagnostic) -> None:
    print(f"{_display(d.at)}: {d.code} {d.message}")
    for label, keys in (("expected", d.expected), ("actual", d.actual)):
        if keys is not None:
            print(f"  {label}: {json.dumps(keys, ensure_ascii=False)}")
    for label, at in d.related:
        print(f"  {label}: {_display(at)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="featureguard",
        description="Compare feature order at fit and prediction sites without running the code.",
    )
    parser.add_argument("entry", metavar="ENTRY.py", type=Path, help="Python entry file")
    parser.add_argument("--root", metavar="DIR", type=Path,
                        help="base for local imports and reported paths (default: entry directory)")
    parser.add_argument("--format", choices=("text", "json"), default="text",
                        help="output format (default: text)")
    parser.add_argument("--max-paths", type=_positive, default=32, metavar="N",
                        help="positive analysis path budget (default: 32)")
    parser.add_argument("--require-prediction", action="store_true",
                        help="report FG901 if no reachable prediction site is recognized")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    errors = []
    diagnostics = ()
    stats = {"predictions": 0, "verified": 0}
    try:
        project = load_project(args.entry, root=args.root)
        result = analyze(project, max_paths=args.max_paths)
    except (ProjectError, OSError) as exc:
        errors.append({"message": str(exc)})
    else:
        diagnostics = result.diagnostics
        stats.update(predictions=result.predictions, verified=result.verified)
        if args.require_prediction and not result.predictions:
            entry = project.modules[project.entry]
            diagnostics += (Diagnostic(
                "FG901", Location(entry.path, 1, 1),
                "No reachable prediction site on a recognized estimator; feature order was not checked.",
            ),)

    if args.format == "json":
        payload = {
            "schema_version": 1, "stats": stats,
            "diagnostics": [_record(d) for d in diagnostics], "errors": errors,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif errors:
        for error in errors:
            print(f"featureguard: {error['message']}", file=sys.stderr)
    else:
        for d in diagnostics:
            _print_diagnostic(d)
        sites = stats["predictions"]
        count = len(diagnostics)
        print(f"{sites} prediction site{'' if sites == 1 else 's'}, {stats['verified']} verified, "
              f"{count} diagnostic{'' if count == 1 else 's'}")
    return 2 if errors else int(bool(diagnostics))
