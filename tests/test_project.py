import ast
import tempfile
import unittest
from pathlib import Path

from featureguard.project import ProjectError, load_project


class ProjectTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def source(self, path, data):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        return path

    def test_relative_imports_cycles_and_unused_files(self):
        entry = self.source("pkg/train.py", "from . import prepare\nfrom .sibling import helper")
        self.source("pkg/prepare.py", "from . import train")
        self.source("pkg/sibling.py", "def helper(): pass")
        self.source("broken.py", "if")
        project = load_project(entry, self.root)
        self.assertEqual(project.entry, "pkg.train")
        self.assertEqual(set(project.modules), {"pkg.train", "pkg.prepare", "pkg.sibling"})

    def test_source_coordinates_use_characters_not_bytes(self):
        path = self.source("entry.py", "名字 = 用户 + 缺失")
        mod = load_project(path).modules["entry"]
        right = mod.tree.body[0].value.right
        self.assertEqual((mod.location(right).line, mod.location(right).column), (1, 11))

    def test_python_encoding_cookie(self):
        path = self.root / "entry.py"
        path.write_bytes(b"# coding: latin-1\ncaf\xe9 = 'caf\xe9'\n")
        mod = load_project(path).modules["entry"]
        self.assertIsInstance(mod.tree.body[0], ast.Assign)

    def test_static_read_does_not_run_module_top_level(self):
        source = self.source("run.py", "import local\nimport sklearn\n")
        self.source("local.py", "raise RuntimeError('must not execute')")
        self.assertEqual(set(load_project(source).modules), {"run", "local"})

    def test_reject_bad_syntax_missing_entry_and_wrong_root(self):
        entry = self.source("run.py", "import local")
        self.source("local.py", "return 3")
        with self.assertRaisesRegex(ProjectError, "local.py:1:"):
            load_project(entry)
        with self.assertRaisesRegex(ProjectError, "existing .py"):
            load_project(self.root / "absent.py")
        with self.assertRaisesRegex(ProjectError, "outside project root"):
            load_project(entry, self.source("other/a.py", "").parent)

    def test_relative_import_cannot_escape_root(self):
        entry = self.source("main.py", "from ..outside import value")
        with self.assertRaisesRegex(ProjectError, "relative import outside"):
            load_project(entry)
