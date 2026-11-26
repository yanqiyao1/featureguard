"""Abstract values and pure transfer functions for feature-matrix construction.

Library identities come from resolved imports, never from spelling alone. The
unknown value means "no usable schema evidence", not "zero columns".
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, order=True)
class Location:
    path: str
    line: int
    column: int


@dataclass(frozen=True)
class Unknown:
    pass


UNKNOWN = Unknown()


@dataclass(frozen=True)
class Text:
    value: str


@dataclass(frozen=True)
class Columns:
    """Ordered literal column labels, not necessarily unique."""

    names: tuple[str, ...]
    is_tuple: bool = False


@dataclass(frozen=True)
class Items:
    values: tuple[Value, ...]


@dataclass(frozen=True)
class Frame:
    """A pandas DataFrame with unknown underlying column labels."""


@dataclass(frozen=True)
class Matrix:
    names: tuple[str, ...]
    origin: Location
    # DataFrames can be projected a second time by name; ndarrays cannot.
    labeled: bool = False


@dataclass(frozen=True)
class Loc:
    frame: Frame | Matrix


@dataclass(frozen=True)
class Model:
    site: Location


@dataclass(frozen=True)
class Symbol:
    name: str


Value: TypeAlias = Unknown | Text | Columns | Items | Frame | Matrix | Loc | Model | Symbol


def literal(node: ast.AST, values: tuple[Value, ...]) -> Value:
    """Interpret immediate children in evaluation order; do not evaluate code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Text(node.value)
    if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == len(values):
        if any(isinstance(item, ast.Starred) for item in node.elts):
            return UNKNOWN
        if all(isinstance(v, Text) for v in values):
            return Columns(tuple(v.value for v in values), is_tuple=isinstance(node, ast.Tuple))
        return Items(values)
    return UNKNOWN


def attribute(base: Value, name: str) -> Value:
    if isinstance(base, Symbol):
        return Symbol(f"{base.name}.{name}")
    if isinstance(base, Matrix) and base.labeled and name == "values":
        return Matrix(base.names, base.origin)
    if name == "loc" and (isinstance(base, Frame) or isinstance(base, Matrix) and base.labeled):
        return Loc(base)
    return UNKNOWN


def _full_slice(node: ast.expr) -> bool:
    return (isinstance(node, ast.Slice)
            and node.lower is None and node.upper is None and node.step is None)


def _project(frame: Frame | Matrix, names: tuple[str, ...], at: Location) -> Value:
    if isinstance(frame, Matrix):
        if not frame.labeled:
            return UNKNOWN
        # In pandas, selecting a duplicated label returns all matching columns.
        # An absent label raises; neither case justifies a one-column witness.
        known = set(frame.names)
        if len(known) != len(frame.names) or not known.issuperset(names):
            return UNKNOWN
    return Matrix(names, at, labeled=True)


def subscript(base: Value, key: Value, node: ast.Subscript, at: Location) -> Value:
    if isinstance(base, Loc):
        axes = node.slice
        if (isinstance(axes, ast.Tuple) and len(axes.elts) == 2 and _full_slice(axes.elts[0])
                and isinstance(key, Items) and len(key.values) == 2
                and isinstance(key.values[1], Columns) and not key.values[1].is_tuple):
            return _project(base.frame, key.values[1].names, at)
        return UNKNOWN
    if isinstance(base, (Frame, Matrix)) and isinstance(key, Columns):
        # df["a", "b"] is one tuple-valued label, not the list ["a", "b"].
        if not key.is_tuple:
            return _project(base, key.names, at)
    return UNKNOWN


_PANDAS_FRAMES = frozenset(f"pandas.{name}" for name in (
    "DataFrame", "read_csv", "read_table", "read_parquet", "read_feather",
    "read_json", "read_sql", "read_sql_query", "read_sql_table", "read_excel",
))
_ESTIMATOR_MODULES = {
    "linear_model": ("LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet"),
    "ensemble": ("RandomForestClassifier", "RandomForestRegressor",
                 "GradientBoostingClassifier", "GradientBoostingRegressor"),
    "svm": ("SVC", "SVR", "LinearSVC", "LinearSVR"),
    "tree": ("DecisionTreeClassifier", "DecisionTreeRegressor"),
    "neighbors": ("KNeighborsClassifier", "KNeighborsRegressor"),
}
_ESTIMATORS = frozenset(f"sklearn.{mod}.{name}"
                        for mod, names in _ESTIMATOR_MODULES.items() for name in names)


def _int_literal(node: ast.expr | None, expected: int) -> bool:
    return (isinstance(node, ast.Constant) and type(node.value) is int
            and node.value == expected)


def _pandas_frame(name: str, args: tuple[Value, ...], kwargs: dict[str, Value]) -> Value:
    if any(key in kwargs for key in ("chunksize", "iterator", "squeeze")):
        return UNKNOWN  # May produce a reader/generator/Series instead of a DataFrame.
    if name == "pandas.read_excel" and (len(args) > 1 or "sheet_name" in kwargs):
        return UNKNOWN  # None or multiple sheet names yield a dict of DataFrames.
    if name == "pandas.read_json" and "typ" in kwargs:
        if kwargs["typ"] != Text("frame"):
            return UNKNOWN
    return Frame()


def _stack(name: str, args: tuple[Value, ...], kwargs: dict[str, Value],
           node: ast.Call, at: Location) -> Value:
    params = {kw.arg: kw.value for kw in node.keywords}
    sequence_arg = "tup" if name == "numpy.hstack" else "arrays"
    allowed = {sequence_arg, "dtype", "casting"}
    matrices = args[0] if args else kwargs.get(sequence_arg)
    if args and sequence_arg in kwargs:
        return UNKNOWN
    if name == "numpy.concatenate":
        allowed.add("axis")
        if len(args) > 2 or len(args) == 2 and "axis" in kwargs:
            return UNKNOWN
        axis = node.args[1] if len(args) == 2 else params.get("axis")
        if not _int_literal(axis, 1):
            return UNKNOWN  # Default axis=0 stacks rows, not features.
    elif len(args) > 1:
        return UNKNOWN
    if not allowed.issuperset(params) or not isinstance(matrices, Items) or not matrices.values:
        return UNKNOWN
    names: list[str] = []
    for matrix in matrices.values:
        if not isinstance(matrix, Matrix):
            return UNKNOWN
        names.extend(matrix.names)
    return Matrix(tuple(names), at)


def library_call(func: Value, receiver: Value, method: str, args: tuple[Value, ...],
                 kwargs: dict[str, Value], node: ast.Call, at: Location) -> Value:
    """Trusted summaries. Return UNKNOWN for unresolved calls or argument packs."""
    if any(isinstance(a, ast.Starred) for a in node.args) or any(
        kw.arg is None for kw in node.keywords
    ):
        return UNKNOWN
    if isinstance(receiver, Matrix) and receiver.labeled and method == "to_numpy":
        if (len(args) <= 3 and kwargs.keys() <= {"dtype", "copy", "na_value"}
                and kwargs.keys().isdisjoint(("dtype", "copy", "na_value")[:len(args)])):
            return Matrix(receiver.names, receiver.origin)
    if isinstance(func, Symbol):
        if func.name in _ESTIMATORS:
            return Model(at)
        if func.name in _PANDAS_FRAMES:
            return _pandas_frame(func.name, args, kwargs)
        if func.name in {"numpy.hstack", "numpy.concatenate"}:
            return _stack(func.name, args, kwargs, node, at)
        if func.name == "numpy.asarray" and len(args) == 1 and not kwargs:
            if isinstance(args[0], Matrix):
                return Matrix(args[0].names, args[0].origin)
    return UNKNOWN
