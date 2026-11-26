"""Interpreter-level tests; analyzed programs are never imported or executed."""

from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
import unittest

from featureguard.engine import analyze
from featureguard.project import load_project


HEADER = """\
from sklearn.linear_model import LinearRegression
import pandas as pd
df = pd.read_csv('not opened')
m = LinearRegression()
a = df[['lon', 'lat']].to_numpy()
b = df[['lat', 'lon']].to_numpy()
"""


def check(body, modules=None, *, max_paths=32):
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        entry = root / "main.py"
        entry.write_text(HEADER + dedent(body), encoding="utf-8")
        for name, source in (modules or {}).items():
            (root / f"{name}.py").write_text(dedent(source), encoding="utf-8")
        return analyze(load_project(entry), max_paths=max_paths)


def codes(result):
    return [d.code for d in result.diagnostics]


class EngineTests(unittest.TestCase):
    def test_source_locations_and_multiset_mismatch(self):
        r = check("""
            m.fit(X=a, y=y)
            m.predict_proba(X=b)
            m.decision_function(df[['lon', 'lon']])
        """)
        self.assertEqual(codes(r), ["FG001", "FG002"])
        order, contents = r.diagnostics
        self.assertIn("position 1: expected 'lon', got 'lat'", order.message)
        self.assertEqual(order.expected, ("lon", "lat"))
        self.assertEqual(order.actual, ("lat", "lon"))
        self.assertEqual([a for a, _ in order.related], ["fit", "fit features", "prediction features"])
        self.assertEqual(contents.actual, ("lon", "lon"))
        self.assertEqual(r.predictions, 2)

    def test_distinct_factory_allocations_and_alias_refit(self):
        r = check("""
            def create():
                return LinearRegression()
            x = create()
            y = create()
            x.fit(a, labels)
            z = y.fit(b, labels)
            z.fit(a, labels)
            x.predict(a)
            y.predict(a)
        """)
        self.assertEqual((codes(r), r.verified), ([], 2))
        other = check("""
            def create():
                return LinearRegression()
            x, y = create(), create()
            x.fit(a, labels)
            y.fit(b, labels)
            x.predict(a)
            y.predict(b)
        """)
        self.assertEqual((codes(other), other.verified), ([], 2))

    def test_cross_module_positional_keyword_return_and_refit(self):
        r = check("""
            import utility as u
            from utility import train as fit_new
            other = fit_new(m, labels, data=a)
            other.predict(u.pack(df))
        """, {"utility": """
            COLUMNS = ['lat', 'lon']
            def pack(frame):
                return frame[COLUMNS].values
            def train(estimator, target, *, data):
                estimator.fit(data, target)
                return estimator
        """})
        self.assertEqual(codes(r), ["FG001"])
        self.assertEqual({p.path for _, p in r.diagnostics[0].related}, {"main.py", "utility.py"})

    def test_early_return_does_not_fall_through(self):
        r = check("""
            def prepare(flag):
                if flag:
                    m.fit(a, labels)
                    return a
                m.fit(b, labels)
                return b
            m.predict(prepare(flag))
        """)
        self.assertEqual((codes(r), r.verified), ([], 1))

    def test_repeated_predicate_and_negation_preserve_path_correlation(self):
        r = check("""
            if flag:
                m.fit(a, labels)
            else:
                m.fit(b, labels)
            if not flag:
                serving = b
            else:
                serving = a
            m.predict(serving)
        """)
        self.assertEqual((codes(r), r.verified), ([], 1))

    def test_rebinding_guard_invalidates_old_fact(self):
        r = check("""
            if flag:
                m.fit(a, labels)
            else:
                m.fit(b, labels)
            flag = another_flag
            if flag:
                serving = a
            else:
                serving = b
            m.predict(serving)
        """)
        self.assertEqual(codes(r), ["FG004"])

    def test_repeated_ternary_preserves_correlation(self):
        r = check("""
            m.fit(a if flag else b, labels)
            m.predict(b if not flag else a)
        """)
        self.assertEqual((codes(r), r.verified), ([], 1))

    def test_prediction_only_ternary_introduces_possible_mismatch(self):
        r = check("""
            m.fit(a, labels)
            m.predict(a if flag else b)
        """)
        self.assertEqual(codes(r), ["FG004"])

    def test_imported_column_list_and_module_attr_share_a_binding(self):
        r = check("""
            from config import FEATURES
            import config
            m.fit(df[FEATURES], labels)
            m.predict(df[config.FEATURES])
            FEATURES.reverse()
            m.predict(df[config.FEATURES])
        """, {"config": "FEATURES = ['lon', 'lat']"})
        self.assertEqual((codes(r), r.verified), (["FG900", "FG003"], 1))

    def test_import_time_mutation_does_not_freeze_an_obsolete_column_list(self):
        r = check("""
            from config import FEATURES
            m.fit(a, labels)
            m.predict(df[FEATURES])
        """, {"config": "FEATURES = ['lon', 'lat']\nalias = FEATURES\nalias.reverse()"})
        self.assertCountEqual(codes(r), ["FG003", "FG900"])
        self.assertEqual(r.verified, 0)

    def test_redefining_helper_does_not_replace_earlier_calls(self):
        r = check("""
            def pack():
                return a
            m.fit(pack(), labels)
            old = pack
            def pack():
                return b
            m.predict(old())
            m.predict(pack())
        """)
        self.assertEqual((codes(r), r.verified), (["FG001"], 1))

    def test_dynamic_or_absent_fit_is_not_a_known_mismatch(self):
        r = check("""
            m.predict(a)
            m.fit(df.select_dtypes(include='number'), labels)
            m.predict(a)
        """)
        self.assertEqual(codes(r), ["FG003", "FG003"])

    def test_unrecognized_method_alone_not_a_prediction_site(self):
        r = check("""
            class Other: pass
            obj = Other()
            obj.predict(42)
        """)
        self.assertEqual((r.predictions, r.verified, codes(r)), (0, 0, []))

    def test_recognized_receiver_on_only_one_path_is_not_verified(self):
        r = check("""
            m.fit(a, labels)
            if flag:
                m = external_model()
            m.predict(a)
        """)
        self.assertEqual((codes(r), r.predictions, r.verified), (["FG003", "FG900"], 1, 0))

    def test_opaque_mutation_forgets_aliases_in_callers(self):
        r = check("""
            cols = ['lon', 'lat']
            def reorder(c):
                c.reverse()
            m.fit(df[cols], labels)
            alias = cols
            reorder(alias)
            m.predict(df[cols])
        """)
        self.assertEqual(codes(r), ["FG900", "FG003"])

    def test_escape_of_model_invalidates_fit(self):
        r = check("""
            m.fit(a, labels)
            external_function(m)
            m.predict(a)
        """)
        self.assertEqual(codes(r), ["FG900", "FG003"])

    def test_recursion_is_bounded_and_not_assumed_pure(self):
        r = check("""
            def go(est):
                return go(est)
            m.fit(a, labels)
            go(m)
            m.predict(a)
        """)
        self.assertEqual(codes(r), ["FG900", "FG003"])

    def test_unsupported_control_flow_forgets_modified_matrix(self):
        r = check("""
            x = a
            m.fit(a, labels)
            for value in external_values:
                x = value
            m.predict(x)
        """)
        self.assertEqual(codes(r), ["FG900", "FG003"])

    def test_path_budget_is_reported_not_an_arbitrarily_selected_verdict(self):
        r = check("""
            if flag:
                m.fit(a, labels)
            else:
                m.fit(b, labels)
            m.predict(a)
        """, max_paths=1)
        self.assertEqual((codes(r), r.verified), (["FG900", "FG003"], 0))

    def test_rejects_nonpositive_path_budget(self):
        with self.assertRaises(ValueError):
            check("", max_paths=0)


if __name__ == "__main__":
    unittest.main()
