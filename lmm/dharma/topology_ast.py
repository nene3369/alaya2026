"""AST-based import graph analyzer — automatic G_prec matrix construction.

Scans Python source files under a package directory, parses import statements
via the stdlib ``ast`` module, and builds a sparse adjacency matrix where edges
represent internal module dependencies.  The matrix can be fed directly to
:class:`TopologyEvaluator` for three-pillar evaluation without manual input.

Also provides :class:`ChangeImpactAnalyzer` for reasoning about the blast
radius of code changes (縁起の影響分析), and :func:`compute_module_health`
for Martin's coupling metrics — enabling self-awareness and safe self-evolution.

No new dependencies — uses only ``ast``, ``pathlib``, ``hashlib``, and ``os``.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse

from lmm.rust_bridge import run_bfs_reverse


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


# ===================================================================
# Multi-codebase comparison
# ===================================================================


def compare_codebases(
    roots: list[str | Path],
    **kwargs,
) -> list[dict]:
    """Evaluate multiple codebases and return comparative topology data.

    Parameters
    ----------
    roots : list of package root paths
    **kwargs : forwarded to :func:`build_gprec_from_codebase`

    Returns
    -------
    list of dicts, each containing:
        name, n_modules, n_edges, density, topology telemetry scores
    """
    from lmm.dharma.topology import TopologyEvaluator

    evaluator = TopologyEvaluator(deleteability_method="degree")
    results: list[dict] = []

    for root_dir in roots:
        root = Path(root_dir).resolve()
        adj, labels = get_cached_gprec(root, **kwargs)
        n = adj.shape[0]
        if n == 0:
            results.append(
                {
                    "name": root.name,
                    "path": str(root),
                    "n_modules": 0,
                    "n_edges": 0,
                    "density": 0.0,
                    "karma_isolation": 1.0,
                    "gprec_topology": 1.0,
                    "deleteability": 1.0,
                    "overall_dharma": 1.0,
                }
            )
            continue

        telemetry = evaluator.evaluate(adj, node_labels=labels, include_details=True)
        max_edges = n * n - n
        density = adj.nnz / max(max_edges, 1)
        results.append(
            {
                "name": root.name,
                "path": str(root),
                "n_modules": n,
                "n_edges": adj.nnz,
                "density": density,
                "karma_isolation": telemetry.karma_isolation,
                "gprec_topology": telemetry.gprec_topology,
                "deleteability": telemetry.deleteability,
                "overall_dharma": telemetry.overall_dharma,
            }
        )

    return results


# ===================================================================
# Change Impact Analysis (縁起の影響分析)
# ===================================================================


@dataclass
class ChangeImpactReport:
    """Report of what happens when one or more modules change.

    Attributes
    ----------
    changed_module : str
        Module name (or comma-joined names for multi-module analysis).
    directly_affected : list[str]
        Modules that directly import the changed module(s).
    transitively_affected : list[str]
        All modules reachable via reverse dependency edges (BFS).
    impact_score : float
        Fraction of the codebase affected, in [0, 1].
    critical_path_length : int
        Longest reverse-dependency chain from the changed module.
    is_bottleneck : bool
        True if ``impact_score > 0.5``.
    """

    changed_module: str
    directly_affected: list[str]
    transitively_affected: list[str]
    impact_score: float
    critical_path_length: int
    is_bottleneck: bool


@dataclass
class ModuleHealthReport:
    """Health metrics for a single module (Robert C. Martin's coupling metrics).

    Attributes
    ----------
    module_name : str
        Dotted module name.
    in_degree : int
        Number of modules that import this one (dependents).
    out_degree : int
        Number of modules this one imports (dependencies).
    afferent_coupling : float
        Ca: normalized in-degree in [0, 1].
    efferent_coupling : float
        Ce: normalized out-degree in [0, 1].
    instability : float
        I = Ce / (Ca + Ce).  0 = maximally stable, 1 = maximally unstable.
    is_hub : bool
        True if in-degree is in the top quartile.
    is_authority : bool
        True if out-degree is in the top quartile.
    """

    module_name: str
    in_degree: int
    out_degree: int
    afferent_coupling: float
    efferent_coupling: float
    instability: float
    is_hub: bool
    is_authority: bool


class ChangeImpactAnalyzer:
    """Analyze the blast radius of code changes (縁起の影響分析).

    Uses the import-graph adjacency matrix to trace direct and transitive
    reverse dependencies, identifying which modules would be affected by
    a change and how deep the impact propagates.

    Parameters
    ----------
    adj : sparse CSR adjacency matrix where ``adj[i, j] > 0`` means
        module *i* imports module *j*.
    labels : list of dotted module name strings.
    """

    def __init__(
        self,
        adj: sparse.csr_matrix,
        labels: list[str],
    ):
        self._adj = adj
        self._labels = labels
        self._label_to_idx: dict[str, int] = {
            label: i for i, label in enumerate(labels)
        }
        # Transposed graph: adj_T[j, i] > 0 means module i imports j,
        # so row j of adj_T gives modules that depend on j.
        # Note: .T already returns csr_matrix in both real scipy and the shim.
        self._adj_T = adj.T
        self._n = adj.shape[0]

    def _resolve_index(self, module_name: str) -> int:
        """Resolve a module name to its index, raising ValueError if unknown."""
        if module_name not in self._label_to_idx:
            raise ValueError(
                f"Unknown module: {module_name!r}. "
                f"Known modules: {len(self._labels)}"
            )
        return self._label_to_idx[module_name]

    def _bfs_reverse(self, start_indices: list[int]) -> tuple[list[int], int]:
        """BFS on the transposed graph from start_indices.

        Returns
        -------
        reachable : list of all reachable node indices (excluding start)
        max_depth : longest path found
        """

        def _fallback():
            visited: set[int] = set(start_indices)
            queue: collections.deque[tuple[int, int]] = collections.deque()
            for idx in start_indices:
                queue.append((idx, 0))
            max_depth = 0

            while queue:
                node, depth = queue.popleft()
                row = self._adj_T.getrow(node).tocoo()
                for neighbor in row.col:
                    nb = int(neighbor)
                    if nb not in visited:
                        visited.add(nb)
                        new_depth = depth + 1
                        if new_depth > max_depth:
                            max_depth = new_depth
                        queue.append((nb, new_depth))

            reachable = sorted(visited - set(start_indices))
            return reachable, max_depth

        result = run_bfs_reverse(
            self._n, start_indices, self._adj_T, _fallback=_fallback,
        )
        return result if result is not None else _fallback()

    def analyze_change(self, module_name: str) -> ChangeImpactReport:
        """Analyze the impact of changing a single module.

        Traces reverse dependencies via BFS on the transposed graph to find
        all modules that directly or transitively depend on the changed module.

        Parameters
        ----------
        module_name : dotted module name (e.g. ``"lmm.dharma.fep"``).

        Returns
        -------
        ChangeImpactReport
        """
        idx = self._resolve_index(module_name)

        # Direct dependents: modules that import this one
        direct_row = self._adj_T.getrow(idx).tocoo()
        direct_indices = sorted(set(int(c) for c in direct_row.col))
        directly_affected = [self._labels[i] for i in direct_indices]

        # Transitive dependents via BFS
        all_reachable, max_depth = self._bfs_reverse([idx])
        transitively_affected = [self._labels[i] for i in all_reachable]

        n_total = max(self._n - 1, 1)  # exclude the changed module itself
        impact_score = len(all_reachable) / n_total if n_total > 0 else 0.0

        return ChangeImpactReport(
            changed_module=module_name,
            directly_affected=directly_affected,
            transitively_affected=transitively_affected,
            impact_score=float(np.clip(impact_score, 0.0, 1.0)),
            critical_path_length=max_depth,
            is_bottleneck=impact_score > 0.5,
        )

    def analyze_changes(self, module_names: list[str]) -> ChangeImpactReport:
        """Analyze the combined impact of changing multiple modules.

        Parameters
        ----------
        module_names : list of dotted module names.

        Returns
        -------
        ChangeImpactReport with combined blast radius.
        """
        if not module_names:
            return ChangeImpactReport(
                changed_module="",
                directly_affected=[],
                transitively_affected=[],
                impact_score=0.0,
                critical_path_length=0,
                is_bottleneck=False,
            )

        start_indices = [self._resolve_index(m) for m in module_names]
        start_set = set(start_indices)

        # Collect direct dependents for all changed modules
        direct_set: set[int] = set()
        for idx in start_indices:
            row = self._adj_T.getrow(idx).tocoo()
            for c in row.col:
                nb = int(c)
                if nb not in start_set:
                    direct_set.add(nb)

        # Transitive via combined BFS
        all_reachable, max_depth = self._bfs_reverse(start_indices)
        transitively_affected = [self._labels[i] for i in all_reachable]
        directly_affected = sorted(self._labels[i] for i in direct_set)

        n_total = max(self._n - len(start_indices), 1)
        impact_score = len(all_reachable) / n_total if n_total > 0 else 0.0

        return ChangeImpactReport(
            changed_module=", ".join(module_names),
            directly_affected=directly_affected,
            transitively_affected=transitively_affected,
            impact_score=float(np.clip(impact_score, 0.0, 1.0)),
            critical_path_length=max_depth,
            is_bottleneck=impact_score > 0.5,
        )

    def find_bottlenecks(self, threshold: float = 0.3) -> list[str]:
        """Find modules whose change would affect > *threshold* of the codebase.

        Returns module names sorted by impact_score descending.
        """
        results: list[tuple[float, str]] = []
        for i in range(self._n):
            reachable, _ = self._bfs_reverse([i])
            n_total = max(self._n - 1, 1)
            score = len(reachable) / n_total
            if score > threshold:
                results.append((score, self._labels[i]))
        results.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in results]

    def find_safe_to_modify(self, threshold: float = 0.1) -> list[str]:
        """Find modules that can be changed with minimal impact.

        Returns modules whose transitive impact is below *threshold*.
        """
        results: list[str] = []
        for i in range(self._n):
            reachable, _ = self._bfs_reverse([i])
            n_total = max(self._n - 1, 1)
            score = len(reachable) / n_total
            if score <= threshold:
                results.append(self._labels[i])
        results.sort()
        return results

    def critical_path(self, module_name: str) -> list[str]:
        """Return the longest reverse-dependency chain from the given module.

        This represents the worst-case propagation path of a bug.

        Returns
        -------
        list of module names from changed module to the furthest dependent.
        """
        idx = self._resolve_index(module_name)

        # BFS tracking parents for longest-path reconstruction
        visited: dict[int, int] = {idx: -1}  # node -> parent
        depth_map: dict[int, int] = {idx: 0}
        queue: collections.deque[int] = collections.deque([idx])
        farthest_node = idx
        max_depth = 0

        while queue:
            node = queue.popleft()
            row = self._adj_T.getrow(node).tocoo()
            for neighbor in row.col:
                nb = int(neighbor)
                if nb not in visited:
                    visited[nb] = node
                    depth_map[nb] = depth_map[node] + 1
                    if depth_map[nb] > max_depth:
                        max_depth = depth_map[nb]
                        farthest_node = nb
                    queue.append(nb)

        # Reconstruct path
        path: list[int] = []
        current = farthest_node
        while current != -1:
            path.append(current)
            current = visited[current]
        path.reverse()
        return [self._labels[i] for i in path]


# ===================================================================
# Module Health Metrics
# ===================================================================


def compute_module_health(
    adj: sparse.csr_matrix,
    labels: list[str],
) -> list[ModuleHealthReport]:
    """Compute health metrics for all modules in the import graph.

    Uses Robert C. Martin's coupling metrics:

    - Afferent coupling (Ca): who depends on me (normalized in-degree)
    - Efferent coupling (Ce): who do I depend on (normalized out-degree)
    - Instability I = Ce / (Ca + Ce): 0 = maximally stable, 1 = maximally unstable

    Parameters
    ----------
    adj : (n, n) sparse adjacency matrix where adj[i,j]>0 means i imports j.
    labels : list of module name strings.

    Returns
    -------
    list of ModuleHealthReport, one per module.
    """
    n = adj.shape[0]
    if n == 0:
        return []

    adj_abs = abs(adj)
    adj_T_abs = adj_abs.T

    # Out-degree: number of modules this one imports (row nnz in adj)
    out_degrees = np.array([adj_abs.getrow(i).nnz for i in range(n)])
    # In-degree: number of modules that import this one (row nnz in adj_T)
    in_degrees = np.array([adj_T_abs.getrow(i).nnz for i in range(n)])

    max_in = int(in_degrees.max()) if n > 0 and in_degrees.max() > 0 else 1
    max_out = int(out_degrees.max()) if n > 0 and out_degrees.max() > 0 else 1

    # Top quartile thresholds for hub/authority detection
    in_sorted = sorted(in_degrees)
    out_sorted = sorted(out_degrees)
    q3_idx = max(0, int(len(in_sorted) * 0.75) - 1)
    in_q3 = in_sorted[q3_idx] if in_sorted else 0
    out_q3 = out_sorted[q3_idx] if out_sorted else 0

    reports: list[ModuleHealthReport] = []
    for i in range(n):
        in_d = int(in_degrees[i])
        out_d = int(out_degrees[i])
        Ca = in_d / max_in
        Ce = out_d / max_out
        instability = Ce / (Ca + Ce + 1e-10)

        reports.append(ModuleHealthReport(
            module_name=labels[i],
            in_degree=in_d,
            out_degree=out_d,
            afferent_coupling=Ca,
            efferent_coupling=Ce,
            instability=float(np.clip(instability, 0.0, 1.0)),
            is_hub=in_d >= in_q3 and in_d > 0,
            is_authority=out_d >= out_q3 and out_d > 0,
        ))

    return reports


def build_and_analyze(
    root_dir: str | Path,
    *,
    exclude_dirs: Sequence[str] = ("_vendor", "__pycache__"),
    package_prefix: str = "lmm",
) -> tuple[sparse.csr_matrix, list[str], ChangeImpactAnalyzer, list[ModuleHealthReport]]:
    """Build the import graph and return analysis tools.

    Convenience function combining :func:`get_cached_gprec` with
    :class:`ChangeImpactAnalyzer` and :func:`compute_module_health`.

    Parameters
    ----------
    root_dir : path to package root (e.g. ``"lmm"``)
    exclude_dirs : directory names to skip
    package_prefix : only track imports starting with this prefix

    Returns
    -------
    adj : sparse adjacency matrix
    labels : module names
    analyzer : ChangeImpactAnalyzer ready for queries
    health : list of ModuleHealthReport for all modules
    """
    adj, labels = get_cached_gprec(
        root_dir, exclude_dirs=exclude_dirs, package_prefix=package_prefix,
    )
    analyzer = ChangeImpactAnalyzer(adj, labels)
    health = compute_module_health(adj, labels)
    return adj, labels, analyzer, health
