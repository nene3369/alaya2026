"""Tests for lmm.dharma.topology_ast — AST import graph analyzer."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from lmm.dharma.topology_ast import (
    _collect_python_files,
    _compute_mtime_hash,
    _parse_imports,
    build_gprec_from_codebase,
    compare_codebases,
    get_cached_gprec,
)


# ===================================================================
# Helpers
# ===================================================================


def _create_temp_package(files: dict[str, str]) -> Path:
    """Create a temporary package with given files.

    files: {relative_path: source_code}
    Returns the root directory Path.
    """
    tmpdir = tempfile.mkdtemp()
    root = Path(tmpdir) / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    for relpath, code in files.items():
        filepath = root / relpath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(code)
    return root


# ===================================================================
# _parse_imports
# ===================================================================


class TestParseImports:
    def test_import_statement(self):
        tmpdir = tempfile.mkdtemp()
        f = Path(tmpdir) / "test_mod.py"
        f.write_text("import os\nimport sys\n")
        result = _parse_imports(f)
        assert "os" in result
        assert "sys" in result

    def test_from_import(self):
        tmpdir = tempfile.mkdtemp()
        f = Path(tmpdir) / "test_mod.py"
        f.write_text("from os.path import join\nfrom sys import argv\n")
        result = _parse_imports(f)
        assert "os.path" in result
        assert "sys" in result

    def test_syntax_error_returns_empty(self):
        tmpdir = tempfile.mkdtemp()
        f = Path(tmpdir) / "bad.py"
        f.write_text("def broken(\n")
        result = _parse_imports(f)
        assert result == []

    def test_empty_file(self):
        tmpdir = tempfile.mkdtemp()
        f = Path(tmpdir) / "empty.py"
        f.write_text("")
        result = _parse_imports(f)
        assert result == []


# ===================================================================
# _collect_python_files
# ===================================================================


class TestCollectFiles:
    def test_collects_py_files(self):
        root = _create_temp_package({"a.py": "", "b.py": "", "c.txt": ""})
        files = _collect_python_files(root)
        py_names = {f.name for f in files}
        assert "a.py" in py_names
        assert "b.py" in py_names
        assert "__init__.py" in py_names
        assert "c.txt" not in py_names

    def test_excludes_vendor(self):
        root = _create_temp_package({"_vendor/v.py": ""})
        files = _collect_python_files(root, exclude_dirs=("_vendor",))
        py_names = {f.name for f in files}
        assert "v.py" not in py_names

    def test_excludes_pycache(self):
        root = _create_temp_package({"__pycache__/cached.py": ""})
        files = _collect_python_files(root, exclude_dirs=("__pycache__",))
        py_names = {f.name for f in files}
        assert "cached.py" not in py_names


# ===================================================================
# build_gprec_from_codebase
# ===================================================================


class TestBuildGprec:
    def test_simple_dependency(self):
        root = _create_temp_package(
            {
                "alpha.py": "from pkg.beta import something\n",
                "beta.py": "x = 1\n",
            }
        )
        adj, labels = build_gprec_from_codebase(root, package_prefix="pkg")
        assert len(labels) > 0
        assert adj.shape[0] == adj.shape[1] == len(labels)
        # There should be at least one edge (alpha → beta)
        assert adj.nnz > 0

    def test_no_self_loops(self):
        root = _create_temp_package(
            {
                "alpha.py": "from pkg.alpha import x\nx = 1\n",
            }
        )
        adj, labels = build_gprec_from_codebase(root, package_prefix="pkg")
        coo = adj.tocoo()
        for r, c in zip(coo.row, coo.col):
            assert int(r) != int(c), "Self-loop detected"

    def test_external_imports_ignored(self):
        root = _create_temp_package(
            {
                "alpha.py": "import numpy\nimport os\n",
            }
        )
        adj, labels = build_gprec_from_codebase(root, package_prefix="pkg")
        # No internal edges (only external imports)
        assert adj.nnz == 0

    def test_empty_directory(self):
        tmpdir = tempfile.mkdtemp()
        empty_root = Path(tmpdir) / "emptypkg"
        empty_root.mkdir()
        adj, labels = build_gprec_from_codebase(empty_root, package_prefix="emptypkg")
        assert len(labels) == 0
        assert adj.shape == (0, 0)

    def test_vendor_excluded(self):
        root = _create_temp_package(
            {
                "core.py": "x = 1\n",
                "_vendor/shim.py": "y = 2\n",
            }
        )
        adj, labels = build_gprec_from_codebase(
            root,
            package_prefix="pkg",
            exclude_dirs=("_vendor",),
        )
        label_set = set(labels)
        for label in label_set:
            assert "_vendor" not in label

    def test_real_lmm_codebase(self):
        """Smoke test: scan the real lmm/ package."""
        lmm_root = Path(__file__).resolve().parent.parent / "lmm"
        if not lmm_root.exists():
            return  # Skip if not in repo
        adj, labels = build_gprec_from_codebase(lmm_root, package_prefix="lmm")
        assert len(labels) > 5  # lmm has many modules
        assert adj.shape[0] == len(labels)
        # Should have many internal dependencies
        assert adj.nnz > 10

    def test_label_count_matches_matrix_dim(self):
        root = _create_temp_package(
            {
                "a.py": "from pkg.b import x\n",
                "b.py": "from pkg.c import y\n",
                "c.py": "z = 1\n",
            }
        )
        adj, labels = build_gprec_from_codebase(root, package_prefix="pkg")
        assert adj.shape[0] == len(labels)
        assert adj.shape[1] == len(labels)


# ===================================================================
# Cache
# ===================================================================


class TestCache:
    def test_cache_hit(self):
        root = _create_temp_package(
            {
                "mod.py": "x = 1\n",
            }
        )
        # First call
        adj1, labels1 = get_cached_gprec(root, package_prefix="pkg")
        # Second call should return same result
        adj2, labels2 = get_cached_gprec(root, package_prefix="pkg")
        assert labels1 == labels2
        assert adj1.shape == adj2.shape

    def test_cache_invalidation_on_change(self):
        root = _create_temp_package(
            {
                "mod.py": "x = 1\n",
            }
        )
        adj1, _ = get_cached_gprec(root, package_prefix="pkg")
        # Modify file (force different mtime)
        import time

        time.sleep(0.05)
        (root / "mod.py").write_text("from pkg.mod2 import y\n")
        (root / "mod2.py").write_text("y = 2\n")
        # Force different mtime hash
        os.utime(root / "mod.py")
        adj2, labels2 = get_cached_gprec(root, package_prefix="pkg")
        # Matrix dimension should change (new module)
        assert len(labels2) > len(["pkg", "pkg.mod"])

    def test_mtime_hash_deterministic(self):
        root = _create_temp_package({"a.py": "", "b.py": ""})
        files = _collect_python_files(root)
        h1 = _compute_mtime_hash(files)
        h2 = _compute_mtime_hash(files)
        assert h1 == h2


# ===================================================================
# Multi-codebase comparison
# ===================================================================


class TestCompareCodebases:
    def test_compare_two_packages(self):
        root1 = _create_temp_package(
            {
                "a.py": "from pkg.b import x\n",
                "b.py": "x = 1\n",
            }
        )
        root2 = _create_temp_package(
            {
                "c.py": "y = 1\n",
            }
        )
        results = compare_codebases([root1, root2], package_prefix="pkg")
        assert len(results) == 2
        assert results[0]["n_modules"] > 0
        assert 0.0 <= results[0]["overall_dharma"] <= 1.0
        assert 0.0 <= results[1]["overall_dharma"] <= 1.0

    def test_compare_empty_package(self):
        tmpdir = tempfile.mkdtemp()
        empty_root = Path(tmpdir) / "emptypkg"
        empty_root.mkdir()
        results = compare_codebases([empty_root], package_prefix="emptypkg")
        assert len(results) == 1
        assert results[0]["n_modules"] == 0
        assert results[0]["overall_dharma"] == 1.0

    def test_compare_result_fields(self):
        root = _create_temp_package({"a.py": "x = 1\n"})
        results = compare_codebases([root], package_prefix="pkg")
        assert len(results) == 1
        r = results[0]
        assert "name" in r
        assert "path" in r
        assert "n_modules" in r
        assert "n_edges" in r
        assert "density" in r
        assert "karma_isolation" in r
        assert "gprec_topology" in r
        assert "deleteability" in r
        assert "overall_dharma" in r
