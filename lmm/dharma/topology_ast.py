"""AST-based import graph analyzer — automatic G_prec matrix construction.

Scans Python source files under a package directory, parses import statements
via the stdlib ``ast`` module, and builds a sparse adjacency matrix where edges
represent internal module dependencies.  The matrix can be fed directly to
:class:`TopologyEvaluator` for three-pillar evaluation without manual input.

No new dependencies — uses only ``ast``, ``pathlib``, ``hashlib``, and ``os``.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Sequence

from scipy import sparse


# ===================================================================
# Internal helpers
# ===================================================================


def _collect_python_files(
    root_dir: Path,
    exclude_dirs: Sequence[str] = ("_vendor", "__pycache__"),
) -> list[Path]:
    """Recursively collect .py files, excluding specified directories."""
    files: list[Path] = []
    exclude_set = set(exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip excluded directories (in-place modification prunes walk)
        dirnames[:] = [d for d in dirnames if d not in exclude_set]
        for fname in filenames:
            if fname.endswith(".py"):
                files.append(Path(dirpath) / fname)
    files.sort()
    return files


def _file_to_module(filepath: Path, root_dir: Path, package_prefix: str) -> str:
    """Convert a file path to a dotted module name.

    Example: ``lmm/dharma/topology.py`` → ``lmm.dharma.topology``
    """
    rel = filepath.relative_to(root_dir.parent)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _build_module_map(
    files: list[Path],
    root_dir: Path,
    package_prefix: str,
) -> dict[str, int]:
    """Build a mapping from dotted module name → node index."""
    module_map: dict[str, int] = {}
    for f in files:
        mod = _file_to_module(f, root_dir, package_prefix)
        if mod and mod not in module_map:
            module_map[mod] = len(module_map)
    return module_map


def _parse_imports(filepath: Path) -> list[str]:
    """Extract import targets from a Python source file.

    Returns a list of dotted module names referenced by ``import X`` and
    ``from X import Y`` statements.  SyntaxError files return an empty list.
    """
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, ValueError):
        return []

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _compute_mtime_hash(files: list[Path]) -> str:
    """Compute a hash over file modification times for cache invalidation."""
    h = hashlib.md5()
    for f in files:
        try:
            mtime = os.path.getmtime(f)
        except OSError:
            mtime = 0.0
        h.update(f"{f}:{mtime}".encode())
    return h.hexdigest()


# ===================================================================
# Public API
# ===================================================================


def build_gprec_from_codebase(
    root_dir: str | Path,
    *,
    exclude_dirs: Sequence[str] = ("_vendor", "__pycache__"),
    package_prefix: str = "lmm",
) -> tuple[sparse.csr_matrix, list[str]]:
    """Build a G_prec adjacency matrix from Python import graph.

    Scans ``root_dir`` (e.g. ``lmm/``) for ``.py`` files, parses imports,
    and constructs a sparse matrix where ``adj[i, j] > 0`` means module *i*
    imports module *j*.  Edge weight equals the number of import references.

    Parameters
    ----------
    root_dir : path to package root (e.g. ``"lmm"``)
    exclude_dirs : directory names to skip (default: _vendor, __pycache__)
    package_prefix : only track imports starting with this prefix

    Returns
    -------
    adj : (n, n) sparse CSR adjacency matrix
    labels : list of n module names (node labels)
    """
    root = Path(root_dir).resolve()
    files = _collect_python_files(root, exclude_dirs)

    if not files:
        empty = sparse.csr_matrix(([], ([], [])), shape=(0, 0))
        return empty, []

    module_map = _build_module_map(files, root, package_prefix)
    labels = [""] * len(module_map)
    for mod, idx in module_map.items():
        labels[idx] = mod

    n = len(module_map)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for f in files:
        src_mod = _file_to_module(f, root, package_prefix)
        if src_mod not in module_map:
            continue
        src_idx = module_map[src_mod]

        imports = _parse_imports(f)
        # Count references to each target module
        target_counts: dict[int, int] = {}
        for imp in imports:
            if not imp.startswith(package_prefix):
                continue
            # Try exact match first, then parent package
            if imp in module_map:
                tgt_idx = module_map[imp]
            else:
                # Try stripping trailing component (from X.Y import Z → X.Y)
                parts = imp.split(".")
                found = False
                while parts:
                    candidate = ".".join(parts)
                    if candidate in module_map:
                        tgt_idx = module_map[candidate]
                        found = True
                        break
                    parts = parts[:-1]
                if not found:
                    continue
            if tgt_idx != src_idx:  # no self-loops
                target_counts[tgt_idx] = target_counts.get(tgt_idx, 0) + 1

        for tgt_idx, count in target_counts.items():
            rows.append(src_idx)
            cols.append(tgt_idx)
            vals.append(float(count))

    if not vals:
        empty = sparse.csr_matrix(([], ([], [])), shape=(n, n))
        return empty, labels

    adj = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return adj, labels


# ===================================================================
# Caching layer
# ===================================================================

_cache: dict[str, tuple[str, sparse.csr_matrix, list[str]]] = {}


def get_cached_gprec(
    root_dir: str | Path,
    **kwargs,
) -> tuple[sparse.csr_matrix, list[str]]:
    """Return cached G_prec matrix, rebuilding only when files change.

    Uses file modification time hashes for cache invalidation.

    Parameters
    ----------
    root_dir : path to package root
    **kwargs : forwarded to :func:`build_gprec_from_codebase`

    Returns
    -------
    adj : sparse CSR adjacency matrix
    labels : list of module names
    """
    root = Path(root_dir).resolve()
    key = str(root)
    exclude_dirs = kwargs.get("exclude_dirs", ("_vendor", "__pycache__"))
    files = _collect_python_files(root, exclude_dirs)
    current_hash = _compute_mtime_hash(files)

    if key in _cache:
        cached_hash, cached_adj, cached_labels = _cache[key]
        if cached_hash == current_hash:
            return cached_adj, cached_labels

    adj, labels = build_gprec_from_codebase(root_dir, **kwargs)
    _cache[key] = (current_hash, adj, labels)
    return adj, labels
