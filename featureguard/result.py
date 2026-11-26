"""Stable public result types, independent of the CLI serialization."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import Location


@dataclass(frozen=True)
class Diagnostic:
    code: str
    at: Location
    message: str
    expected: tuple[str, ...] | None = None
    actual: tuple[str, ...] | None = None
    related: tuple[tuple[str, Location], ...] = ()


@dataclass(frozen=True)
class CheckResult:
    diagnostics: tuple[Diagnostic, ...]
    predictions: int
    verified: int
