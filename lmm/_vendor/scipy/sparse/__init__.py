"""Pure-Python scipy.sparse shim -- CSR, COO, issparse, triu.

Fully functional for the LMM project's usage patterns:
  csr_matrix construction from (data, (row, col)) and from dense ndarray
  .toarray(), .tocoo(), .getrow(), .sum(), .T
  .indices, .data, .nnz, .shape
  issparse(), triu()
  arithmetic: +, -, *, /  (sparse + sparse, sparse * scalar)
"""

from __future__ import annotations
import numpy as np


class csr_matrix:
    """Compressed Sparse Row matrix (pure-Python)."""

    def __init__(self, arg, shape=None, dtype=None):
        """
        Accepted forms:
          csr_matrix((data, (row, col)), shape=(M,N))
          csr_matrix(dense_ndarray)
          csr_matrix(another_csr_matrix)
        """
        if isinstance(arg, csr_matrix):
            self._rows = arg._rows
            self._cols = arg._cols
            self._vals = list(arg._vals)
            self._shape = arg._shape
            return

        if isinstance(arg, np.ndarray):
            # dense -> sparse
            if arg.ndim != 2:
                raise ValueError("csr_matrix from ndarray requires 2-D")
            r, c = arg._shape
            self._shape = (r, c)
            self._rows = []
            self._cols = []
            self._vals = []
            for i in range(r):
                for j in range(c):
                    v = arg._data[i * c + j]
                    if v != 0.0:
                        self._rows.append(i)
                        self._cols.append(j)
                        self._vals.append(float(v))
            return

        if isinstance(arg, tuple) and len(arg) == 2:
            data_part, idx_part = arg
            if isinstance(idx_part, tuple) and len(idx_part) == 2:
                row_idx, col_idx = idx_part
                # Normalize to plain lists
                if isinstance(data_part, np.ndarray):
                    data_part = list(data_part._data)
                elif isinstance(data_part, (list, tuple)):
                    data_part = [float(v) for v in data_part]
                if isinstance(row_idx, np.ndarray):
                    row_idx = [int(v) for v in row_idx._data]
                elif isinstance(row_idx, (list, tuple)):
                    row_idx = [int(v) for v in row_idx]
                if isinstance(col_idx, np.ndarray):
                    col_idx = [int(v) for v in col_idx._data]
                elif isinstance(col_idx, (list, tuple)):
                    col_idx = [int(v) for v in col_idx]

                self._rows = list(row_idx)
                self._cols = list(col_idx)
                self._vals = list(data_part)

                if shape is not None:
                    self._shape = tuple(shape)
                else:
                    mr = max(row_idx) + 1 if row_idx else 0
                    mc = max(col_idx) + 1 if col_idx else 0
                    self._shape = (mr, mc)
                return

        raise TypeError(f"Cannot create csr_matrix from {type(arg)}")

    # -- properties ---------------------------------------------------------

    @property
    def shape(self):
        return self._shape

    @property
    def nnz(self):
        return len(self._vals)

    @property
    def data(self):
        return np.array(self._vals)

    @property
    def indices(self):
        """Column indices (like real CSR .indices)."""
        return np.array([float(c) for c in self._cols], dtype=np.int64)

    @property
    def T(self):
        return csr_matrix(
            (list(self._vals), (list(self._cols), list(self._rows))),
            shape=(self._shape[1], self._shape[0]),
        )

    # -- conversion ---------------------------------------------------------

    def toarray(self):
        rows, cols = self._shape
        out = np.zeros((rows, cols))
        for r, c, v in zip(self._rows, self._cols, self._vals):
            out._data[r * cols + c] += v
        return out

    def tocoo(self):
        return _coo_matrix(self)

    def getrow(self, i):
        """Return row i as a csr_matrix (1 x N)."""
        cols_out = []
        vals_out = []
        for r, c, v in zip(self._rows, self._cols, self._vals):
            if r == i:
                cols_out.append(c)
                vals_out.append(v)
        m = csr_matrix.__new__(csr_matrix)
        m._rows = [0] * len(vals_out)
        m._cols = cols_out
        m._vals = vals_out
        m._shape = (1, self._shape[1])
        # also expose .indices and .data as arrays directly
        return m

    # -- reductions ---------------------------------------------------------

    def sum(self, axis=None):
        if axis is None:
            return sum(self._vals)
        rows, cols = self._shape
        if axis == 1:
            d = [0.0] * rows
            for r, c, v in zip(self._rows, self._cols, self._vals):
                d[r] += v
            # return as (rows, 1) dense ndarray (matches scipy behavior)
            return np.ndarray((rows, 1), np.float64, d)
        if axis == 0:
            d = [0.0] * cols
            for r, c, v in zip(self._rows, self._cols, self._vals):
                d[c] += v
            return np.ndarray((1, cols), np.float64, d)
        raise NotImplementedError

    # -- matmul (@ operator) -----------------------------------------------

    def __matmul__(self, other):
        """Sparse matrix @ dense vector/matrix."""
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                rows, cols = self._shape
                if other._shape[0] != cols:
                    raise ValueError(
                        f"matmul shape mismatch: {self._shape} @ {other._shape}"
                    )
                result = np.zeros(rows)
                for r, c, v in zip(self._rows, self._cols, self._vals):
                    result._data[r] += v * other._data[c]
                return result
            if other.ndim == 2:
                rows, cols = self._shape
                _, ncols = other._shape
                result = np.zeros((rows, ncols))
                for r, c, v in zip(self._rows, self._cols, self._vals):
                    for j in range(ncols):
                        result._data[r * ncols + j] += v * other._data[c * ncols + j]
                return result
        return NotImplemented

    # -- arithmetic ---------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, csr_matrix):
            return self._merge(other, lambda a, b: a + b)
        if isinstance(other, (int, float)):
            raise NotImplementedError("sparse + scalar not defined")
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, csr_matrix):
            return self._merge(other, lambda a, b: a - b)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return csr_matrix(
                ([v * other for v in self._vals],
                 (list(self._rows), list(self._cols))),
                shape=self._shape,
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return csr_matrix(
                ([v / other for v in self._vals],
                 (list(self._rows), list(self._cols))),
                shape=self._shape,
            )
        return NotImplemented

    def _merge(self, other, fn):
        """Element-wise merge of two sparse matrices."""
        assert self._shape == other._shape
        entries = {}  # (r, c) -> value
        for r, c, v in zip(self._rows, self._cols, self._vals):
            entries[(r, c)] = v
        for r, c, v in zip(other._rows, other._cols, other._vals):
            if (r, c) in entries:
                entries[(r, c)] = fn(entries[(r, c)], v)
            else:
                entries[(r, c)] = fn(0.0, v)
        # remove zeros
        new_rows, new_cols, new_vals = [], [], []
        for (r, c), v in entries.items():
            if v != 0.0:
                new_rows.append(r)
                new_cols.append(c)
                new_vals.append(v)
        return csr_matrix(
            (new_vals, (new_rows, new_cols)),
            shape=self._shape,
        )

    # -- element-wise operations -------------------------------------------

    def __abs__(self):
        """abs(sparse_matrix) — element-wise absolute value."""
        return csr_matrix(
            ([abs(v) for v in self._vals],
             (list(self._rows), list(self._cols))),
            shape=self._shape,
        )

    def max(self):
        """Maximum non-zero value (or 0.0 if empty)."""
        if not self._vals:
            return 0.0
        return max(self._vals)

    def min(self):
        """Minimum non-zero value (or 0.0 if empty)."""
        if not self._vals:
            return 0.0
        return min(self._vals)

    # -- repr ---------------------------------------------------------------

    def __repr__(self):
        return f"<csr_matrix shape={self._shape}, nnz={self.nnz}>"


# ---------------------------------------------------------------------------
# COO helper
# ---------------------------------------------------------------------------

class _coo_matrix:
    """Minimal COO matrix for iteration (row, col, data)."""

    def __init__(self, csr: csr_matrix):
        self.row = np.array([float(r) for r in csr._rows], dtype=np.int64)
        self.col = np.array([float(c) for c in csr._cols], dtype=np.int64)
        self.data = np.array(csr._vals)
        self.shape = csr._shape

    def __iter__(self):
        for r, c, d in zip(self.row._data, self.col._data, self.data._data):
            yield (int(r), int(c), d)


# ---------------------------------------------------------------------------
# module-level functions
# ---------------------------------------------------------------------------

class lil_matrix:
    """List of Lists sparse matrix (pure-Python shim).

    Efficient for incremental construction. Convert to csr_matrix for arithmetic.
    """

    def __init__(self, shape, dtype=None):
        if isinstance(shape, tuple) and len(shape) == 2:
            self._shape = shape
        else:
            raise ValueError("lil_matrix requires (rows, cols) shape")
        self._rows_data: dict[tuple[int, int], float] = {}

    @property
    def shape(self):
        return self._shape

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            r, c = int(key[0]), int(key[1])
            self._rows_data[(r, c)] = float(value)
        else:
            raise IndexError(f"lil_matrix requires (row, col) indexing, got {type(key)}")

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            r, c = int(key[0]), int(key[1])
            return self._rows_data.get((r, c), 0.0)
        raise IndexError("lil_matrix requires (row, col) indexing")

    @property
    def nnz(self):
        return len(self._rows_data)

    def __imul__(self, scalar):
        for key in self._rows_data:
            self._rows_data[key] *= float(scalar)
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new = lil_matrix(self._shape)
            for key, val in self._rows_data.items():
                new._rows_data[key] = val * float(other)
            return new
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def tocsr(self):
        """Convert to csr_matrix."""
        rows, cols, vals = [], [], []
        for (r, c), v in self._rows_data.items():
            if v != 0.0:
                rows.append(r)
                cols.append(c)
                vals.append(v)
        return csr_matrix(
            (vals, (rows, cols)),
            shape=self._shape,
        )

    def __repr__(self):
        return f"<lil_matrix shape={self._shape}, nnz={len(self._rows_data)}>"


def issparse(x):
    return isinstance(x, (csr_matrix, lil_matrix))


def triu(matrix, k=0):
    """Extract upper triangular entries (row < col + k offset)."""
    if isinstance(matrix, csr_matrix):
        new_rows, new_cols, new_vals = [], [], []
        for r, c, v in zip(matrix._rows, matrix._cols, matrix._vals):
            if c >= r + k:
                new_rows.append(r)
                new_cols.append(c)
                new_vals.append(v)
        return csr_matrix(
            (new_vals, (new_rows, new_cols)),
            shape=matrix._shape,
        )
    if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
        rows, cols = matrix._shape
        out = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if j >= i + k:
                    out._data[i * cols + j] = matrix._data[i * cols + j]
        return out
    raise TypeError("triu requires csr_matrix or ndarray")
