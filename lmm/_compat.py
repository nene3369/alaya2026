"""Runtime capability detection and sparse matrix helpers.

Probes numpy/scipy capabilities once at import time so modules can
choose fast paths vs shim-compatible fallbacks.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Basic detection: shim vs real numpy
# ---------------------------------------------------------------------------
IS_SHIM = getattr(np, "__version__", "") == "0.0.0-shim"


def _probe(fn: callable) -> bool:
    try:
        return fn()
    except (ImportError, AttributeError, TypeError, NotImplementedError,
            OverflowError, RuntimeError):
        return False


HAS_ARGPARTITION = _probe(lambda: (
    hasattr(np.argpartition(np.array([3.0, 1.0, 2.0, 5.0, 4.0]), -2), '__len__')
))
HAS_SEARCHSORTED = _probe(lambda: (
    int(np.searchsorted(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.5]))[0]) == 2
))
HAS_VIEW = not IS_SHIM and _probe(lambda: (
    np.array([1.0, 2.0], dtype=np.float64).view(np.int64).dtype == np.int64
))
HAS_INT64_ARITH = not IS_SHIM and _probe(lambda: (
    hasattr((np.array([2**60], dtype=np.int64) * np.array([3], dtype=np.int64)), 'dtype')
))
HAS_FAST_PATH = HAS_ARGPARTITION and HAS_SEARCHSORTED

# Hardware acceleration
HAS_VECTORIZED_TANH = not IS_SHIM and _probe(lambda: (
    hasattr(np.tanh(np.array([0.0, 1.0, -1.0])), '__len__')
))
HAS_SPARSE_MATMUL = not IS_SHIM and _probe(lambda: (
    abs(float((sparse.csr_matrix(([1.0], ([0], [0])), shape=(2, 2))
              @ np.array([1.0, 0.0]))[0]) - 1.0) < 1e-10
))
HAS_CUPY = _probe(lambda: (__import__('cupy').cuda.is_available()))
HAS_JAX = _probe(lambda: (__import__('jax'), True)[-1])

HAS_FEP_FAST_PATH = HAS_VECTORIZED_TANH and HAS_SPARSE_MATMUL
HAS_GPU = HAS_CUPY or HAS_JAX


# ---------------------------------------------------------------------------
# Sparse matrix helpers — shim-compatible
# ---------------------------------------------------------------------------

def sparse_matvec(J: sparse.csr_matrix, v: np.ndarray) -> np.ndarray:
    """Sparse matrix-vector product: J @ v, O(nnz)."""
    if HAS_SPARSE_MATMUL:
        return J @ v
    n = J.shape[0]
    result = np.zeros(n)
    coo = J.tocoo()
    for r, c, val in zip(coo.row, coo.col, coo.data):
        result[int(r)] = float(result[int(r)]) + float(val) * float(v[int(c)])
    return result


def sparse_dot(v1: np.ndarray, J: sparse.csr_matrix, v2: np.ndarray) -> float:
    """Quadratic form: v1^T J v2, O(nnz)."""
    if HAS_SPARSE_MATMUL:
        return float(np.dot(v1, J @ v2))
    total = 0.0
    coo = J.tocoo()
    for r, c, val in zip(coo.row, coo.col, coo.data):
        total += float(val) * float(v1[int(r)]) * float(v2[int(c)])
    return total


def sparse_row_scatter(
    J: sparse.csr_matrix, row_idx: int, target: np.ndarray, scale: float,
) -> None:
    """In-place target[cols] += scale * J[row_idx, cols], O(nnz_row)."""
    if HAS_SPARSE_MATMUL:
        start = J.indptr[row_idx]
        end = J.indptr[row_idx + 1]
        cols = J.indices[start:end]
        vals = J.data[start:end]
        target[cols] += scale * vals
    else:
        row_index = getattr(J, '_row_index_cache', None)
        if row_index is not None and row_idx in row_index:
            for c, v in row_index[row_idx]:
                target[c] = float(target[c]) + scale * v
        else:
            coo = J.tocoo()
            for r, c, v in zip(coo.row, coo.col, coo.data):
                if int(r) == row_idx:
                    target[int(c)] = float(target[int(c)]) + scale * float(v)


def sparse_getcol(
    J: sparse.csr_matrix, idx: int,
) -> list[tuple[int, float]]:
    """Get non-zero entries of column idx as [(row, value), ...]."""
    col_index = getattr(J, '_col_index_cache', None)
    if col_index is not None and idx in col_index:
        return col_index[idx]
    entries: list[tuple[int, float]] = []
    coo = J.tocoo()
    for r, c, v in zip(coo.row, coo.col, coo.data):
        if int(c) == idx:
            entries.append((int(r), float(v)))
    return entries


def sparse_col_index(
    J: sparse.csr_matrix,
) -> dict[int, list[tuple[int, float]]]:
    """Build column reverse index {col: [(row, val), ...]}, O(nnz)."""
    index: dict[int, list[tuple[int, float]]] = defaultdict(list)
    if J.nnz > 0:
        coo = J.tocoo()
        for r, c, v in zip(coo.row, coo.col, coo.data):
            index[int(c)].append((int(r), float(v)))
    result = dict(index)
    try:
        J._col_index_cache = result
    except AttributeError:
        pass
    return result


def sparse_row_index(
    J: sparse.csr_matrix,
) -> dict[int, list[tuple[int, float]]]:
    """Build row reverse index {row: [(col, val), ...]}, O(nnz)."""
    index: dict[int, list[tuple[int, float]]] = defaultdict(list)
    if J.nnz > 0:
        coo = J.tocoo()
        for r, c, v in zip(coo.row, coo.col, coo.data):
            index[int(r)].append((int(c), float(v)))
    result = dict(index)
    try:
        J._row_index_cache = result
    except AttributeError:
        pass
    return result
