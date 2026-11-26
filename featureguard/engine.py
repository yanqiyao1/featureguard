"""Path-sensitive, interprocedural analysis over a deliberately small Python IR.

The IR is walked directly on Python's AST. A path owns its binding frames and
model heap; sharing an immutable abstract allocation object implements aliasing.
Unknown effects forget evidence rather than fabricating a successful check.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

from .domain import (
    UNKNOWN, Columns, Frame, Items, Loc, Location, Matrix, Model, Symbol, Text,
    Unknown, Value, attribute, library_call, literal, subscript,
)
from .project import Module, Project
from .result import CheckResult, Diagnostic

_PREDICTIONS = frozenset({"predict", "predict_proba", "decision_function"})
# A path cap alone cannot bound an acyclic graph of repeated helper calls.
_MAX_TRANSFERS = 50_000


class _WorkLimit(Exception):
    def __init__(self, at: Location):
        self.at = at


@dataclass(frozen=True)
class _Fit:
    at: Location
    matrix: Matrix | None


@dataclass
class _State:
    frames: list[dict[str, Value]]
    modules: dict[str, dict[str, Value]] = field(default_factory=dict)
    # Two calls to the same factory share a source location but not an object.
    # Retaining the object in the heap prevents recycling of its Python id.
    heap: dict[int, tuple[Model, _Fit | None]] = field(default_factory=dict)
    guards: dict[tuple[int, str], bool] = field(default_factory=dict)
    trace: tuple[tuple[str, Location], ...] = ()
    returned: Value | None = None

    def fork(self) -> _State:
        return _State([f.copy() for f in self.frames], {k: v.copy() for k, v in self.modules.items()},
                      self.heap.copy(),
                      self.guards.copy(), self.trace, self.returned)

    def bind(self, name: str, value: Value) -> None:
        level = len(self.frames) - 1
        self.frames[level][name] = value
        self.guards.pop((level, name), None)

    def havoc(self, value: Value) -> bool:
        """Opaque writes may change any alias, including nested argument packs."""
        if isinstance(value, Items):
            affected = False
            for item in value.values:
                affected |= self.havoc(item)
            return affected
        if isinstance(value, Loc):
            return self.havoc(value.frame)
        if isinstance(value, Model):
            self.heap[id(value)] = value, None
            return True
        if not isinstance(value, (Columns, Matrix, Frame)):
            return False
        for frame in (*self.frames, *self.modules.values()):
            for name, bound in frame.items():
                if _contains(bound, value):
                    frame[name] = UNKNOWN
        return True


def _contains(container: Value, value: Value) -> bool:
    if container is value:
        return True
    if isinstance(container, Items):
        return any(_contains(v, value) for v in container.values)
    return isinstance(container, Loc) and container.frame is value


def _constant(node: ast.expr) -> Value:
    if isinstance(node, (ast.List, ast.Tuple)):
        return literal(node, tuple(_constant(x) for x in node.elts))
    return literal(node, ())


def _default(node: ast.expr) -> Value:
    # Python reuses default arguments between calls. Do not mint a fresh known
    # list at each call when a previous invocation could have mutated that list.
    if any(isinstance(part, (ast.List, ast.Set, ast.Dict)) for part in ast.walk(node)):
        return UNKNOWN
    return _constant(node)


def _imports(project: Project, module: Module,
             node: ast.Import | ast.ImportFrom) -> Iterator[tuple[str, Value]]:
    if isinstance(node, ast.Import):
        for a in node.names:
            root = a.name.split(".")[0]
            yield a.asname or root, Symbol(a.name if a.asname else root)
    else:
        base = project.import_name(module, node)
        for a in node.names:
            if a.name != "*":
                yield a.asname or a.name, Symbol(f"{base}.{a.name}" if base else a.name)


@dataclass(frozen=True)
class _Obs:
    kind: str
    fit: _Fit | None
    actual: Matrix | None
    trace: tuple[tuple[str, Location], ...]


class _Engine:
    def __init__(self, project: Project, max_paths: int):
        self.project = project
        self.max_paths = max_paths
        self.steps = 0
        self.funcs: dict[str, tuple[Module, ast.FunctionDef]] = {}
        self.globals: dict[str, dict[str, Value]] = {}
        self.observed: dict[Location, list[_Obs]] = defaultdict(list)
        self.unknown_receivers: dict[Location, list[_Obs]] = defaultdict(list)
        self.notes: set[tuple[Location, str]] = set()
        self.stack: list[str] = []
        self.depth = 0  # Nested unknown branches must not explore 2**depth paths.
        self.local_roots = {name.split(".")[0] for name in project.modules}
        self.summary_models: dict[int, Model] = {}
        for mod in project.modules.values():
            env = self.globals[mod.name] = {}
            for node in mod.tree.body:
                if (mod.name != project.entry and not isinstance(node, (
                        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))):
                    self.lose_import_mutations(node, env, mod)
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    env.update(_imports(project, mod, node))
                elif isinstance(node, ast.FunctionDef):
                    key = f"{mod.name}.{node.name}"
                    explicit_key = f"{key}@{node.lineno}"
                    self.funcs[explicit_key] = mod, node
                    env[node.name] = Symbol(explicit_key) if not node.decorator_list else UNKNOWN
                elif isinstance(node, ast.Assign):
                    value = (env.get(node.value.id, UNKNOWN) if isinstance(node.value, ast.Name)
                             else _constant(node.value))
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            env[target.id] = value
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    env[node.target.id] = _constant(node.value) if node.value else UNKNOWN
                elif not isinstance(node, (ast.Expr, ast.Pass)):
                    for child in ast.walk(node):
                        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                            env[child.id] = UNKNOWN

    def resolve(self, val: Value, s: _State) -> Value:
        """Resolve local source references; imported modules are never executed."""
        seen = set()
        while isinstance(val, Symbol) and val.name not in seen:
            owner, _, name = val.name.rpartition(".")
            if owner not in s.modules or val.name in self.funcs:
                break
            seen.add(val.name)
            other = s.modules[owner].get(name)
            if other is None:
                break
            val = other
        return val

    def lookup(self, s: _State, mod: Module, name: str) -> Value:
        if name in s.frames[-1]:
            return self.resolve(s.frames[-1][name], s)
        global_env = s.frames[0] if mod.name == self.project.entry else s.modules[mod.name]
        if name in global_env:
            return self.resolve(global_env[name], s)
        if name in {"len", "print", "bool"}:
            return Symbol(f"builtins.{name}")
        return UNKNOWN

    def cutoff(self, at: Location, message: str) -> None:
        self.notes.add((at, message))

    def tick(self, node: ast.AST, module: Module) -> None:
        self.steps += 1
        if self.steps > _MAX_TRANSFERS:
            raise _WorkLimit(module.location(node))

    def lose_import_mutations(self, n: ast.AST, env: dict[str, Value], mod: Module) -> None:
        # A literal list is useful at import time only if an ignored top-level
        # call or indirect write does not subsequently receive a reference to it.
        for item in ast.walk(n):
            if isinstance(item, ast.Call) or (
                    isinstance(item, (ast.Subscript, ast.Attribute)) and isinstance(item.ctx, ast.Store)):
                for name in (e.id for e in ast.walk(item) if isinstance(e, ast.Name)):
                    bound = env.get(name)
                    if isinstance(bound, (Columns, Items)):
                        for alias, value in env.items():
                            if _contains(value, bound):
                                env[alias] = UNKNOWN
                        self.cutoff(mod.location(item), f"Import-time operation may mutate {name!r}; schema forgotten.")

    def block(self, body: list[ast.stmt], paths: list[_State], mod: Module) -> list[_State]:
        for stmt in body:
            nxt = []
            for state in paths:
                nxt.extend([state] if state.returned is not None else self.stmt(stmt, state, mod))
            paths = self.bound(nxt, mod.location(stmt))
        return paths

    def bound(self, paths: list[_State], at: Location) -> list[_State]:
        if len(paths) <= self.max_paths:
            return paths
        self.cutoff(at, "Path budget exceeded; differing bindings and fit evidence were forgotten.")
        merged = []
        # Do not send an early return back into its function's continuation.
        for returned in (False, True):
            group = [s for s in paths if (s.returned is not None) == returned]
            if not group:
                continue
            base = group[0].fork()
            base.guards.clear()
            base.trace = (("path budget join", at),)
            for i, frame in enumerate(base.frames):
                for name in set().union(*(s.frames[i] for s in group)):
                    vals = [s.frames[i].get(name, UNKNOWN) for s in group]
                    frame[name] = vals[0] if all(v is vals[0] for v in vals) else UNKNOWN
                    if all(isinstance(v, Model) for v in vals) and isinstance(frame[name], Unknown):
                        # Distinct allocations: retain estimator type, not any
                        # one allocation's fit. Subsequent predictions remain visible.
                        new = Model(at)
                        frame[name] = new
                        base.heap[id(new)] = new, None
                        self.summary_models[id(new)] = new
            for module, frame in base.modules.items():
                for name in frame:
                    if any(s.modules[module].get(name) is not frame[name] for s in group):
                        frame[name] = UNKNOWN
            for key in set().union(*(s.heap for s in group)):
                entries = [s.heap[key] for s in group if key in s.heap]
                model, fit = entries[0]
                if len(entries) != len(group) or any(f != fit for _, f in entries):
                    fit = None
                base.heap[key] = model, fit
            if returned and any(s.returned is not base.returned for s in group):
                base.returned = UNKNOWN
            merged.append(base)
        return merged

    def sequence(self, nodes: list[ast.expr], s: _State,
                 mod: Module) -> list[tuple[_State, tuple[Value, ...]]]:
        """Temporary slots keep partially evaluated arguments inside the path budget."""
        slots = [f"$arg:{id(nodes)}:{i}" for i in range(len(nodes))]
        states = [s]
        for node, slot in zip(nodes, slots):
            out = []
            for st in states:
                for path, value in self.expr(node, st, mod):
                    path.frames[-1][slot] = value
                    out.append(path)
            states = self.bound(out, mod.location(node))
        return [(st, tuple(st.frames[-1].pop(slot) for slot in slots)) for st in states]

    def condition(self, node: ast.expr, s: _State, m: Module) -> tuple[bool | None, str | None, bool]:
        if isinstance(node, ast.Constant):
            return bool(node.value), None, False
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == "__name__" and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq) and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "__main__"):
            return m.name == self.project.entry, None, False
        neg = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        subject = node.operand if neg else node
        if isinstance(subject, ast.Name):
            value = s.guards.get((len(s.frames) - 1, subject.id))
            return (not value if neg and value is not None else value), subject.id, neg
        return None, None, False

    def branches(self, node: ast.If | ast.IfExp, s: _State,
                 m: Module) -> Iterator[tuple[_State, bool | None]]:
        """Yield one guarded successor at a time, retaining the nesting budget."""
        at = m.location(node)
        for st, val in self.expr(node.test, s, m):
            truth, guard, neg = self.condition(node.test, st, m)
            if isinstance(val, Text):
                truth = bool(val.value)
            if truth is None and 2 ** self.depth > self.max_paths:
                self.cutoff(at, "Nested branch budget exceeded; branch effects are unknown.")
                self.opaque(node, st, m)
                yield st, None
                continue
            for take in (True, False):
                if truth is not None and truth != take:
                    continue
                branch = st.fork()
                if guard:
                    branch.guards[len(st.frames) - 1, guard] = take != neg
                label = "true branch" if take else "false branch"
                branch.trace = (*branch.trace[-5:], (label, at))
                self.depth += truth is None
                try:
                    yield branch, take
                finally:
                    self.depth -= truth is None

    def stmt(self, n: ast.stmt, s: _State, m: Module) -> list[_State]:
        self.tick(n, m)
        at = m.location(n)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for name, val in _imports(self.project, m, n):
                s.bind(name, val)
            if any(a.name == "*" for a in n.names):
                self.cutoff(at, "Star import is opaque; previous bindings were forgotten.")
                for name in list(s.frames[-1]):
                    s.bind(name, UNKNOWN)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{m.name}.{n.name}@{n.lineno}"
            val = Symbol(name) if name in self.funcs and not n.decorator_list else UNKNOWN
            s.bind(n.name, val)
        elif isinstance(n, ast.If):
            out = []
            for branch, take in self.branches(n, s, m):
                out.extend([branch] if take is None else
                           self.block(n.body if take else n.orelse, [branch], m))
            return self.bound(out, at)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return)):
            expr = n.value
            if expr is None:
                if isinstance(n, ast.Return):
                    s.returned = UNKNOWN
                return [s]
            out = []
            for st, val in self.expr(expr, s, m):
                targets = n.targets if isinstance(n, ast.Assign) else (
                    (n.target,) if isinstance(n, ast.AnnAssign) else ())
                for target in targets:
                    self.assign(target, val, st, m)
                if isinstance(n, ast.Return):
                    st.returned = val
                out.append(st)
            return out
        elif isinstance(n, ast.Raise):
            return []
        elif isinstance(n, (ast.Pass, ast.Assert)):
            if isinstance(n, ast.Assert):
                return [st for st, _ in self.expr(n.test, s, m)
                        if self.condition(n.test, st, m)[0] is not False]
        else:
            self.cutoff(at, f"Unsupported {type(n).__name__}; its affected bindings are unknown.")
            self.opaque(n, s, m)
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                # The next write cannot safely be treated as an ordinary local.
                for name in n.names:
                    for frame in (*s.frames, *s.modules.values()):
                        if name in frame:
                            s.havoc(frame[name])
                            frame[name] = UNKNOWN
        return [s]

    def assign(self, target: ast.expr, val: Value, s: _State, m: Module) -> None:
        if isinstance(target, ast.Name):
            s.bind(target.id, val)
        elif isinstance(target, (ast.Tuple, ast.List)):
            items = val.values if isinstance(val, Items) and len(val.values) == len(target.elts) else ()
            for i, item in enumerate(target.elts):
                self.assign(item, items[i] if items else UNKNOWN, s, m)
        else:
            self.cutoff(m.location(target), "Indirect write; receiver schema is no longer known.")
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                root = self.lookup(s, m, target.value.id)
                if isinstance(root, Symbol) and root.name in s.modules:
                    s.modules[root.name][target.attr] = UNKNOWN
                elif isinstance(root, Symbol):
                    s.bind(target.value.id, UNKNOWN)
            self.opaque(target, s, m)

    def opaque(self, node: ast.AST, s: _State, m: Module) -> None:
        for item in ast.walk(node):
            if isinstance(item, ast.Name):
                s.havoc(self.lookup(s, m, item.id))
                if not isinstance(item.ctx, ast.Load):
                    s.bind(item.id, UNKNOWN)
            elif isinstance(item, (ast.Import, ast.ImportFrom)):
                for name, _ in _imports(self.project, m, item):
                    s.bind(name, UNKNOWN)
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                s.bind(item.name, UNKNOWN)

    def expr(self, n: ast.expr, s: _State, m: Module) -> list[tuple[_State, Value]]:
        self.tick(n, m)
        at = m.location(n)
        if isinstance(n, ast.Name):
            return [(s, self.lookup(s, m, n.id))]
        if isinstance(n, ast.Constant):
            return [(s, literal(n, ()))]
        if isinstance(n, (ast.List, ast.Tuple)):
            return [(p, literal(n, vals)) for p, vals in self.sequence(n.elts, s, m)]
        if isinstance(n, ast.Attribute):
            return [(p, self.resolve(attribute(base, n.attr), p))
                    for p, base in self.expr(n.value, s, m)]
        if isinstance(n, ast.Subscript):
            return [(p, subscript(vals[0], vals[1], n, at))
                    for p, vals in self.sequence([n.value, n.slice], s, m)]
        if isinstance(n, ast.Call):
            is_attr = isinstance(n.func, ast.Attribute)
            head = n.func.value if is_attr else n.func
            method = n.func.attr if is_attr else ""
            out = []
            for p, vals in self.sequence([head, *n.args, *(k.value for k in n.keywords)], s, m):
                recv = vals[0] if is_attr else UNKNOWN
                func = self.resolve(attribute(recv, method), p) if is_attr else vals[0]
                stop = 1 + len(n.args)
                args = vals[1:stop]
                kwargs = {k.arg: v for k, v in zip(n.keywords, vals[stop:]) if k.arg}
                out.extend(self.call(n, p, m, func, recv, method, args, kwargs))
            return out
        if isinstance(n, ast.NamedExpr):
            out = self.expr(n.value, s, m)
            for p, val in out:
                self.assign(n.target, val, p, m)
            return out
        if isinstance(n, ast.IfExp):
            out = []
            for path, take in self.branches(n, s, m):
                out.extend([(path, UNKNOWN)] if take is None else
                           self.expr(n.body if take else n.orelse, path, m))
            return out
        # Eager children still have effects (e.g. helper calls inside comparisons).
        # Short-circuit expressions and comprehensions require their own CFG.
        if (isinstance(n, (ast.BoolOp, ast.ListComp, ast.DictComp, ast.SetComp,
                           ast.GeneratorExp, ast.Lambda))
                or isinstance(n, ast.Compare) and len(n.ops) > 1):
            if any(isinstance(item, (ast.Call, ast.NamedExpr)) for item in ast.walk(n)):
                self.cutoff(at, "Unsupported conditional or deferred expression; effects forgotten.")
                self.opaque(n, s, m)
            return [(s, UNKNOWN)]
        children = [x for x in ast.iter_child_nodes(n) if isinstance(x, ast.expr)]
        return [(p, UNKNOWN) for p, _ in self.sequence(children, s, m)]

    def call(self, n: ast.Call, s: _State, m: Module, func: Value, recv: Value, method: str,
             args: tuple[Value, ...], kwargs: dict[str, Value]) -> list[tuple[_State, Value]]:
        at = m.location(n)
        packed = any(isinstance(a, ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords)
        if isinstance(recv, Model):
            arg = UNKNOWN if packed else (args[0] if args else kwargs.get("X", UNKNOWN))
            matrix = arg if isinstance(arg, Matrix) else None
            if method == "fit":
                if id(recv) not in self.summary_models:
                    s.heap[id(recv)] = recv, _Fit(at, matrix)
                return [(s, recv)]
            if method in _PREDICTIONS:
                _, fit = s.heap[id(recv)]
                if not fit or not fit.matrix or not matrix:
                    kind = "unknown"
                elif fit.matrix.names == matrix.names:
                    kind = "match"
                elif Counter(fit.matrix.names) == Counter(matrix.names):
                    kind = "order"
                else:
                    kind = "contents"
                self.observed[at].append(_Obs(kind, fit, matrix, s.trace))
                return [(s, UNKNOWN)]
        if method in _PREDICTIONS:
            # A site known to be sklearn on *another* path must not be marked
            # verified based only on that path. Unrelated methods alone are ignored.
            self.unknown_receivers[at].append(_Obs("unknown", None, None, s.trace))
        if (isinstance(func, Symbol) and func.name in self.funcs and
                not self.funcs[func.name][1].decorator_list):
            return self.helper(func.name, n, s, m, args, kwargs, packed)
        local = isinstance(func, Symbol) and func.name.split(".")[0] in self.local_roots
        val = UNKNOWN if local else library_call(func, recv, method, args, kwargs, n, at)
        if isinstance(val, Model):
            s.heap[id(val)] = val, None
        nonmutating = (isinstance(func, Symbol) and func.name in
                       {"builtins.len", "builtins.print", "builtins.bool"}) or (
            isinstance(recv, (Frame, Matrix)) and method in {"to_numpy", "copy", "select_dtypes"})
        if isinstance(val, Unknown) and not nonmutating:
            affected = False
            for value in (recv, *args, *kwargs.values()):
                affected |= s.havoc(value)
            if affected:
                self.cutoff(at, "Unmodeled call may mutate a tracked argument; evidence forgotten.")
        return [(s, val)]

    def helper(self, name: str, call: ast.Call, s: _State, caller: Module, args: tuple[Value, ...],
               kwargs: dict[str, Value], packed: bool) -> list[tuple[_State, Value]]:
        mod, fn = self.funcs[name]
        params = fn.args
        positional = [p.arg for p in (*params.posonlyargs, *params.args)]
        required = positional[:len(positional) - len(params.defaults)]
        allowed = set(positional[len(params.posonlyargs):]) | {p.arg for p in params.kwonlyargs}
        bad = (packed or params.vararg is not None or params.kwarg is not None
               or len(args) > len(positional) or not allowed.issuperset(kwargs)
               or not set(positional[:len(args)]).isdisjoint(kwargs))
        env = dict(zip(positional, args))
        env.update(kwargs)
        if not bad:
            for param, default in zip(positional[len(required):], params.defaults):
                env.setdefault(param, _default(default))
            for param, default in zip(params.kwonlyargs, params.kw_defaults):
                if default is None and param.arg not in env:
                    bad = True
                else:
                    env.setdefault(param.arg, _default(default) if default else UNKNOWN)
        if bad or not set(required).issubset(env) or name in self.stack or len(self.stack) >= 16:
            reason = "recursion/depth limit" if name in self.stack or len(self.stack) >= 16 else "unsupported arguments"
            self.cutoff(caller.location(call), f"Cannot expand {name}: {reason}.")
            module_values = [v for frame in s.modules.values() for v in frame.values()]
            for value in (*args, *kwargs.values(), *s.frames[0].values(), *module_values):
                s.havoc(value)
            return [(s, UNKNOWN)]
        # Python regards an assigned name as local throughout the function.
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                env.setdefault(node.id, UNKNOWN)
        s.frames.append(env)
        s.trace = (*s.trace[-5:], ("call", caller.location(call)))
        self.stack.append(name)
        try:
            finished = self.block(fn.body, [s], mod)
        finally:
            self.stack.pop()
        out = []
        for p in finished:
            p.frames.pop()
            p.guards = {k: v for k, v in p.guards.items() if k[0] < len(p.frames)}
            out.append((p, p.returned if p.returned is not None else UNKNOWN))
            p.returned = None
        return out

    def diagnostics(self) -> CheckResult:
        found = [Diagnostic("FG900", at, message) for at, message in self.notes]
        verified = 0
        for at, obs in self.observed.items():
            obs = [*obs, *self.unknown_receivers.get(at, ())]
            kinds = {o.kind for o in obs}
            if kinds == {"match"}:
                verified += 1
                continue
            witness = next((o for o in obs if o.kind in {"order", "contents"}),
                           next((o for o in obs if o.kind == "unknown"), obs[0]))
            if kinds == {"order"}:
                code, msg = "FG001", "Prediction uses the fitted feature names in a different order."
            elif kinds <= {"order", "contents"}:
                code, msg = "FG002", "Prediction feature schema differs from fit on all modeled paths."
            elif kinds & {"order", "contents"}:
                code, msg = "FG004", "Feature mismatch on some modeled paths; see one mismatching path."
            else:
                code, msg = "FG003", "Cannot verify feature order: the estimator, fit, or input schema is unknown."
            related = list(witness.trace)
            fitted = witness.fit.matrix if witness.fit else None
            if witness.kind == "order" and fitted and witness.actual:
                for i, (expected, actual) in enumerate(zip(fitted.names, witness.actual.names), 1):
                    if expected != actual:
                        msg += f" First difference at position {i}: expected {expected!r}, got {actual!r}."
                        break
            if witness.fit:
                related.append(("fit", witness.fit.at))
            if fitted:
                related.append(("fit features", fitted.origin))
            if witness.actual:
                related.append(("prediction features", witness.actual.origin))
            found.append(Diagnostic(code, at, msg, fitted.names if fitted else None,
                                    witness.actual.names if witness.actual else None, tuple(related)))
        return CheckResult(tuple(sorted(found, key=lambda d: (d.at, d.code, d.message))),
                           len(self.observed), verified)


def analyze(project: Project, *, max_paths: int = 32) -> CheckResult:
    """Check recognized predictions reachable from the selected entry module."""
    if max_paths < 1:
        raise ValueError("max_paths must be positive")
    entry = project.modules[project.entry]
    try:
        runner = _Engine(project, max_paths)
        runner.block(entry.tree.body, [_State([{}], runner.globals)], entry)
    except (_WorkLimit, RecursionError) as exc:
        # Unvisited alternatives might disagree with already observed sites.
        # Publishing partial "verified" verdicts would hide that uncertainty.
        at = exc.at if isinstance(exc, _WorkLimit) else Location(entry.path, 1, 1)
        reason = "transfer budget" if isinstance(exc, _WorkLimit) else "interpreter nesting limit"
        issue = Diagnostic("FG900", at, f"Analysis stopped at the {reason}; no partial verdicts retained.")
        return CheckResult((issue,), 0, 0)
    return runner.diagnostics()
