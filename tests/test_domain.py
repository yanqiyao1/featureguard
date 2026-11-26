"""Unit tests for trusted summaries; this module never imports pandas/sklearn."""

import ast
import unittest

from featureguard.domain import (
    UNKNOWN, Columns, Frame, Items, Location, Matrix, Model, Symbol, Text,
    attribute, library_call, literal, subscript,
)


AT = Location("train.py", 8, 5)
FIRST = Location("extract.py", 3, 12)
X = Matrix(("temp", "wind"), FIRST, labeled=True)
Y = Matrix(("rain",), FIRST)


def expr(source):
    return ast.parse(source, mode="eval").body


def call(source, fn=UNKNOWN, args=(), kw=None, *, receiver=UNKNOWN, method=""):
    return library_call(fn, receiver, method, args, kw or {}, expr(source), AT)


class DomainTests(unittest.TestCase):
    def test_column_literals_distinguish_lists_from_tuple_keys(self):
        self.assertEqual(literal(expr("'wind'"), ()), Text("wind"))
        for source, is_tuple in [("['a', 'b']", False), ("('a', 'b')", True)]:
            result = literal(expr(source), (Text("a"), Text("b")))
            self.assertEqual(result, Columns(("a", "b"), is_tuple=is_tuple))
        self.assertEqual(literal(expr("[*xs]"), (Columns(("a",)),)), UNKNOWN)
        self.assertEqual(literal(expr("[]"), ()), Columns(()))
        self.assertEqual(literal(expr("[left, right]"), (X, Y)), Items((X, Y)))

    def test_frame_projection_and_alias_require_list_of_labels(self):
        base = Frame()
        self.assertEqual(subscript(base, Columns(("b", "a")), expr("df[keep]"), AT),
                         Matrix(("b", "a"), AT, labeled=True))
        for key in (Text("a"), UNKNOWN, Items((UNKNOWN,)), Columns(("a", "b"), is_tuple=True)):
            self.assertEqual(subscript(base, key, expr("df[keep]"), AT), UNKNOWN)
        self.assertEqual(subscript(UNKNOWN, Columns(("a",)), expr("df[keep]"), AT), UNKNOWN)

    def test_loc_requires_evidence_for_both_axes(self):
        loc = attribute(Frame(), "loc")
        axes = Items((UNKNOWN, Columns(("wind", "temp"))))
        self.assertEqual(subscript(loc, axes, expr("df.loc[:, cols]"), AT),
                         Matrix(("wind", "temp"), AT, labeled=True))
        for source in ("df.loc[rows, cols]", "df.loc[0:4, cols]", "df.loc[cols]",
                       "df.loc[::2, cols]"):
            self.assertEqual(subscript(loc, axes, expr(source), AT), UNKNOWN)
        self.assertEqual(subscript(loc, Items((UNKNOWN, Text("wind"))),
                                   expr("df.loc[:, 'wind']"), AT), UNKNOWN)

    def test_labeled_projection_restricts_names_and_rejects_ambiguous_duplicates(self):
        key, node = Columns(("wind",)), expr("x[keep]")
        self.assertEqual(subscript(X, key, node, AT), Matrix(("wind",), AT, labeled=True))
        self.assertEqual(subscript(X, Columns(("absent",)), node, AT), UNKNOWN)
        self.assertEqual(subscript(Y, key, node, AT), UNKNOWN)
        self.assertEqual(subscript(Matrix(("wind", "wind"), FIRST, True), key, node, AT), UNKNOWN)

    def test_resolved_imports_not_lookalike_names(self):
        mod = attribute(Symbol("sklearn"), "linear_model")
        ctor = attribute(mod, "LogisticRegression")
        self.assertEqual(call("factory()", ctor), Model(AT))
        for fn in (UNKNOWN, Symbol("user.LogisticRegression"), Symbol("sklearn.svm.Mystery")):
            self.assertEqual(call("factory()", fn), UNKNOWN)
        self.assertEqual(call("pd.read_csv(path)", Symbol("pandas.read_csv"), (UNKNOWN,)), Frame())
        self.assertEqual(call("pd.read_csv(path)", Symbol("user.read_csv"), (UNKNOWN,)), UNKNOWN)

    def test_non_frame_reader_modes(self):
        cases = [
            ("read_csv", "f(path, chunksize=10)", {"chunksize": UNKNOWN}),
            ("read_sql", "f(query, chunksize=10)", {"chunksize": UNKNOWN}),
            ("read_excel", "f(path, sheet_name=None)", {"sheet_name": UNKNOWN}),
            ("read_json", "f(path, typ='series')", {"typ": Text("series")}),
        ]
        for name, source, kwargs in cases:
            with self.subTest(fn=name):
                self.assertEqual(call(source, Symbol(f"pandas.{name}"), (UNKNOWN,), kwargs),
                                 UNKNOWN)

    def test_losing_labels_keeps_origin_not_pandas_methods(self):
        arr = Matrix(X.names, X.origin)
        self.assertEqual(attribute(X, "values"), arr)
        self.assertEqual(call("x.to_numpy()", receiver=X, method="to_numpy"), arr)
        self.assertEqual(call("np.asarray(x)", Symbol("numpy.asarray"), (X,)), arr)
        self.assertEqual(attribute(arr, "values"), UNKNOWN)
        self.assertEqual(attribute(arr, "loc"), UNKNOWN)
        self.assertEqual(call("x.to_numpy()", receiver=arr, method="to_numpy"), UNKNOWN)
        self.assertEqual(call("x.to_numpy(**opts)", receiver=X, method="to_numpy"), UNKNOWN)

    def test_horizontal_stack_preserves_order_and_evidence_location(self):
        matrices = Items((Y, X))
        expected = Matrix(("rain", "temp", "wind"), AT)
        self.assertEqual(call("f([y, x])", Symbol("numpy.hstack"), (matrices,)), expected)
        self.assertEqual(call("f([y, x], 1)", Symbol("numpy.concatenate"),
                              (matrices, UNKNOWN)), expected)
        self.assertEqual(call("f([y, x], axis=1)", Symbol("numpy.concatenate"),
                              (matrices,), {"axis": UNKNOWN}), expected)
        self.assertEqual(call("f(tup=parts)", Symbol("numpy.hstack"), (),
                              {"tup": matrices}), expected)

    def test_vertical_dynamic_or_mixed_stack_has_no_witness(self):
        mats = Items((X, Y))
        for source, args, kw in [
                ("f(parts)", (mats,), {}),
                ("f(parts, axis=0)", (mats,), {"axis": UNKNOWN}),
                ("f(parts, axis=True)", (mats,), {"axis": UNKNOWN}),
                ("f(parts, axis=dim)", (mats,), {"axis": UNKNOWN}),
                ("f(parts, axis=1, out=buf)", (mats,), {"axis": UNKNOWN, "out": UNKNOWN}),
                ("f(parts, axis=1)", (Items((X, UNKNOWN)),), {"axis": UNKNOWN}),
                ("f(*packed)", (UNKNOWN,), {})]:
            with self.subTest(source=source):
                self.assertEqual(call(source, Symbol("numpy.concatenate"), args, kw), UNKNOWN)


if __name__ == "__main__":
    unittest.main()
