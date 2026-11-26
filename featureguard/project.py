"""Load the local import graph as source, never through Python's import system."""

from __future__ import annotations

import ast
import tokenize
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .domain import Location


class ProjectError(ValueError):
    """A source or project layout problem that can be reported to the user."""


@dataclass(frozen=True)
class Module:
    name: str
    path: str  # Root-relative POSIX path, also used in diagnostics.
    tree: ast.Module
    _lines: tuple[str, ...] = field(repr=False)

    def location(self, node: ast.AST) -> Location:
        line = getattr(node, "lineno", 1)
        offset = getattr(node, "col_offset", 0)
        # AST column offsets count UTF-8 bytes even for latin-1 source files.
        column = len(self._lines[line - 1].encode("utf-8")[:offset].decode("utf-8")) + 1
        return Location(self.path, line, column)


def _absolute_import(module: Module, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    parts = module.name.split(".")
    if Path(module.path).name != "__init__.py":
        parts.pop()
    if node.level > len(parts):
        raise ProjectError(f"{module.path}:{node.lineno}: relative import outside project; check --root")
    prefix = parts[:len(parts) - node.level + 1]
    return ".".join(prefix + ([node.module] if node.module else []))


@dataclass(frozen=True)
class Project:
    entry: str
    modules: dict[str, Module]

    @staticmethod
    def import_name(module: Module, node: ast.ImportFrom) -> str:
        return _absolute_import(module, node)


def _read(path: Path, root: Path, name: str) -> Module:
    relative = path.relative_to(root).as_posix()
    try:
        with tokenize.open(path) as handle:
            source = handle.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=relative)
            # Reject scope-sensitive errors accepted by ast.parse, e.g. a
            # module-level return. Compile for validation only; never execute.
            compile(tree, relative, "exec", dont_inherit=True)
    except SyntaxError as exc:
        raise ProjectError(f"{relative}:{exc.lineno or 1}:{exc.offset or 1}: {exc.msg}") from exc
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ProjectError(f"{relative}: cannot read Python source: {exc}") from exc
    return Module(name, relative, tree, tuple(source.split("\n")))


def _module_name(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        raise ProjectError("for a package entry __init__.py, set --root to its parent package directory")
    return ".".join(parts)


def _dependencies(module: Module):
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import(module, node)
            yield base
            # 'from pkg import sub' may refer to a module or just an attribute.
            yield from (f"{base}.{alias.name}" for alias in node.names if alias.name != "*")


def load_project(entry: Path, root: Path | None = None) -> Project:
    """Read ENTRY and local statically named imports; do not touch site-packages."""
    try:
        entry = entry.resolve()
        root = root.resolve() if root is not None else entry.parent
        if not root.is_dir():
            raise ProjectError(f"not a project directory: {root}")
        if not entry.is_relative_to(root):
            raise ProjectError(f"entry is outside project root: {entry}")
        if entry.suffix != ".py" or not entry.is_file():
            raise ProjectError(f"expected an existing .py entry file: {entry}")
        entry_name = _module_name(entry.relative_to(root))
        pending = [(entry_name, entry)]
        modules: dict[str, Module] = {}
        checked: set[str] = set()
        while pending:
            name, path = pending.pop()
            if name in modules:
                continue
            module = modules[name] = _read(path, root, name)
            for dep in _dependencies(module):
                if not dep or dep in checked:
                    continue
                checked.add(dep)
                base = root.joinpath(*dep.split("."))
                for candidate in (base / "__init__.py", base.with_suffix(".py")):
                    if candidate.is_file():
                        local = candidate.resolve()
                        if not local.is_relative_to(root):
                            raise ProjectError(f"import '{dep}' resolves outside project root")
                        pending.append((dep, local))
                        break
        return Project(entry_name, modules)
    except (OSError, RuntimeError) as exc:
        raise ProjectError(f"unable to resolve project: {exc}") from exc
