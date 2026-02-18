"""Pure-Python numpy shim -- minimal but functional.

Supports the subset of numpy used by the LMM project:
  ndarray (1-D and 2-D), zeros, ones, array, empty, full, eye, arange,
  diag, fill_diagonal, dot, exp, log, abs, clip, where, argsort,
  argmax, argmin, concatenate, isfinite, sum, histogram,
  float64, int64, uint64, inf, iinfo,
  add.at (ufunc shim), load, save,
  linalg.norm, random.default_rng / random.rand
"""

from __future__ import annotations

import math
import random as _pyrandom
import copy as _copy
import struct as _struct
import os as _os

__version__ = "0.0.0-shim"

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

class _dtype:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"dtype('{self.name}')"
    def __eq__(self, other):
        if isinstance(other, _dtype):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, type) and other.__name__ == "bool_":
            return self.name == "bool"
        return NotImplemented
    def __hash__(self):
        return hash(self.name)

float64 = _dtype("float64")
float32 = _dtype("float32")
int64 = _dtype("int64")
int32 = _dtype("int32")
uint64 = _dtype("uint64")
intp = _dtype("int64")

# bool_ must be a type for isinstance() compatibility with pytest
class bool_(int):
    """numpy.bool_ type shim."""
    pass

_bool_dtype = _dtype("bool")

inf = float("inf")

# ---------------------------------------------------------------------------
# iinfo
# ---------------------------------------------------------------------------

class _iinfo_result:
    def __init__(self, bits, signed=True):
        if signed:
            self.min = -(1 << (bits - 1))
            self.max = (1 << (bits - 1)) - 1
        else:
            self.min = 0
            self.max = (1 << bits) - 1

def iinfo(dt):
    if isinstance(dt, _dtype):
        dt = dt.name
    if dt in ("int64", int):
        return _iinfo_result(64, signed=True)
    if dt in ("int32",):
        return _iinfo_result(32, signed=True)
    if dt in ("uint64",):
        return _iinfo_result(64, signed=False)
    return _iinfo_result(64, signed=True)

# ---------------------------------------------------------------------------
# ndarray
# ---------------------------------------------------------------------------

class _scalar(float):
    """A float subclass that behaves like a numpy scalar (has .tobytes())."""
    def tobytes(self):
        return _struct.pack("d", float(self))
    def astype(self, dtype):
        return _scalar(float(self))


class ndarray:
    """Minimal ndarray backed by a flat Python list."""

    __slots__ = ("_data", "_shape", "_dtype")

    # -- construction -------------------------------------------------------

    def __init__(self, shape, dtype=None, _data=None):
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = tuple(shape)
        self._dtype = dtype or float64
        size = 1
        for s in self._shape:
            size *= s
        if _data is not None:
            self._data = _data
        else:
            self._data = [0.0] * size

    # -- properties ---------------------------------------------------------

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        s = 1
        for d in self._shape:
            s *= d
        return s

    @property
    def dtype(self):
        return self._dtype

    @property
    def T(self):
        if self.ndim == 1:
            return self.copy()
        if self.ndim == 2:
            r, c = self._shape
            out = ndarray((c, r), self._dtype)
            for i in range(r):
                for j in range(c):
                    out._data[j * r + i] = self._data[i * c + j]
            return out
        raise NotImplementedError("T only for 1-D/2-D")

    # -- helpers ------------------------------------------------------------

    def _flat_idx(self, key):
        """Convert an index tuple to flat offset."""
        if self.ndim == 1:
            if isinstance(key, tuple):
                key = key[0]
            if isinstance(key, (int, float)) and not isinstance(key, bool):
                key = int(key)
                if key < 0:
                    key += self._shape[0]
                return key
        if self.ndim == 2:
            if isinstance(key, tuple) and len(key) == 2:
                r, c = key
                if isinstance(r, (int, float)) and not isinstance(r, bool) and isinstance(c, (int, float)) and not isinstance(c, bool):
                    r, c = int(r), int(c)
                    if r < 0:
                        r += self._shape[0]
                    if c < 0:
                        c += self._shape[1]
                    return r * self._shape[1] + c
        return None  # signal fancy / slice

    # -- getitem / setitem --------------------------------------------------

    def __len__(self):
        return self._shape[0]

    def __iter__(self):
        """Iterate over the first axis."""
        if self.ndim == 1:
            for i in range(self._shape[0]):
                yield self._data[i]
        else:
            for i in range(self._shape[0]):
                yield self[i]

    def _is_bool_array(self, key):
        if isinstance(key, ndarray) and key._dtype == bool_:
            return True
        if isinstance(key, ndarray) and all(isinstance(v, bool) for v in key._data):
            return True
        if isinstance(key, list) and len(key) > 0 and isinstance(key[0], bool):
            return True
        return False

    def __getitem__(self, key):
        # --- scalar indexing ---
        fi = self._flat_idx(key)
        if fi is not None:
            v = self._data[fi]
            if isinstance(v, (int, float)) and not isinstance(v, _scalar):
                return _scalar(v)
            return v

        # --- 1-D fancy / slice ---
        if self.ndim == 1:
            if isinstance(key, slice):
                indices = range(*key.indices(self._shape[0]))
                d = [self._data[i] for i in indices]
                return _make_array_1d(d, self._dtype)
            if isinstance(key, ndarray):
                if self._is_bool_array(key):
                    d = [self._data[i] for i in range(self._shape[0]) if key._data[i]]
                    return _make_array_1d(d, self._dtype)
                else:
                    d = [self._data[int(v)] for v in key._data]
                    return _make_array_1d(d, self._dtype)
            if isinstance(key, (list, tuple)):
                if len(key) > 0 and isinstance(key[0], bool):
                    d = [self._data[i] for i, b in enumerate(key) if b]
                    return _make_array_1d(d, self._dtype)
                d = [self._data[int(v)] for v in key]
                return _make_array_1d(d, self._dtype)

        # --- 2-D indexing ---
        if self.ndim == 2:
            rows, cols = self._shape
            if isinstance(key, (int, float)) and not isinstance(key, bool):
                key = int(key)
                if key < 0:
                    key += rows
                start = key * cols
                d = self._data[start:start + cols]
                return _make_array_1d(list(d), self._dtype)
            if isinstance(key, slice):
                idx = range(*key.indices(rows))
                out_data = []
                for r in idx:
                    start = r * cols
                    out_data.extend(self._data[start:start + cols])
                return ndarray((len(idx), cols), self._dtype, out_data)
            if isinstance(key, tuple):
                rk, ck = key
                # row is ndarray/list, col is ndarray/list  (fancy)
                if isinstance(rk, (ndarray, list)) and isinstance(ck, (ndarray, list)):
                    rk_list = list(rk._data) if isinstance(rk, ndarray) else list(rk)
                    ck_list = list(ck._data) if isinstance(ck, ndarray) else list(ck)
                    d = [self._data[int(r) * cols + int(c)] for r, c in zip(rk_list, ck_list)]
                    return _make_array_1d(d, self._dtype)
                # row slice, col slice
                if isinstance(rk, slice) and isinstance(ck, slice):
                    ridx = range(*rk.indices(rows))
                    cidx = range(*ck.indices(cols))
                    out_data = []
                    for r in ridx:
                        for c in cidx:
                            out_data.append(self._data[r * cols + c])
                    return ndarray((len(ridx), len(cidx)), self._dtype, out_data)
                # int row, fancy col
                if isinstance(rk, int) and isinstance(ck, (ndarray, list, tuple)):
                    if rk < 0:
                        rk += rows
                    ck_list = list(ck._data) if isinstance(ck, ndarray) else list(ck)
                    d = [self._data[rk * cols + int(c)] for c in ck_list]
                    return _make_array_1d(d, self._dtype)
                # fancy row, int col
                if isinstance(ck, int) and isinstance(rk, (ndarray, list, tuple)):
                    if ck < 0:
                        ck += cols
                    rk_list = list(rk._data) if isinstance(rk, ndarray) else list(rk)
                    d = [self._data[int(r) * cols + ck] for r in rk_list]
                    return _make_array_1d(d, self._dtype)
                # ndarray (bool mask) row, any col
                if isinstance(rk, ndarray) and self._is_bool_array(rk):
                    row_indices = [i for i in range(rows) if rk._data[i]]
                    if isinstance(ck, int):
                        if ck < 0:
                            ck += cols
                        d = [self._data[r * cols + ck] for r in row_indices]
                        return _make_array_1d(d, self._dtype)
                    elif isinstance(ck, slice):
                        cidx = range(*ck.indices(cols))
                        out_data = []
                        for r in row_indices:
                            for c in cidx:
                                out_data.append(self._data[r * cols + c])
                        return ndarray((len(row_indices), len(cidx)), self._dtype, out_data)
                # int row, slice col
                if isinstance(rk, int) and isinstance(ck, slice):
                    if rk < 0:
                        rk += rows
                    cidx = range(*ck.indices(cols))
                    d = [self._data[rk * cols + c] for c in cidx]
                    return _make_array_1d(d, self._dtype)
                # slice row, int col
                if isinstance(ck, int) and isinstance(rk, slice):
                    if ck < 0:
                        ck += cols
                    ridx = range(*rk.indices(rows))
                    d = [self._data[r * cols + ck] for r in ridx]
                    return _make_array_1d(d, self._dtype)
            # ndarray fancy index on rows
            if isinstance(key, ndarray):
                if self._is_bool_array(key):
                    row_indices = [i for i in range(rows) if key._data[i]]
                else:
                    row_indices = [int(v) for v in key._data]
                out_data = []
                for r in row_indices:
                    start = r * cols
                    out_data.extend(self._data[start:start + cols])
                return ndarray((len(row_indices), cols), self._dtype, out_data)
            if isinstance(key, list):
                if self._is_bool_array(key):
                    row_indices = [i for i, b in enumerate(key) if b]
                else:
                    row_indices = [int(v) for v in key]
                out_data = []
                for r in row_indices:
                    start = r * cols
                    out_data.extend(self._data[start:start + cols])
                return ndarray((len(row_indices), cols), self._dtype, out_data)

        raise IndexError(f"Unsupported indexing: {type(key)} on shape {self._shape}")

    def __setitem__(self, key, value):
        fi = self._flat_idx(key)
        if fi is not None:
            self._data[fi] = _to_scalar(value)
            return

        if self.ndim == 1:
            if isinstance(key, slice):
                indices = range(*key.indices(self._shape[0]))
                if isinstance(value, ndarray):
                    for i, idx in enumerate(indices):
                        self._data[idx] = value._data[i]
                elif isinstance(value, (int, float)):
                    for idx in indices:
                        self._data[idx] = float(value)
                return
            if isinstance(key, (ndarray, list, tuple)):
                if self._is_bool_array(key):
                    idxs = [i for i in range(self._shape[0]) if (key._data[i] if isinstance(key, ndarray) else key[i])]
                else:
                    idxs = [int(v) for v in (key._data if isinstance(key, ndarray) else key)]
                if isinstance(value, ndarray):
                    for i, idx in enumerate(idxs):
                        self._data[idx] = value._data[i]
                elif isinstance(value, (int, float)):
                    for idx in idxs:
                        self._data[idx] = float(value)
                return

        if self.ndim == 2:
            rows, cols = self._shape
            if isinstance(key, tuple) and len(key) == 2:
                rk, ck = key
                # fancy indexing set: arr[ndarray, ndarray] = val
                if isinstance(rk, (ndarray, list)) and isinstance(ck, (ndarray, list)):
                    rk_list = list(rk._data) if isinstance(rk, ndarray) else list(rk)
                    ck_list = list(ck._data) if isinstance(ck, ndarray) else list(ck)
                    if isinstance(value, ndarray):
                        for i, (r, c) in enumerate(zip(rk_list, ck_list)):
                            self._data[int(r) * cols + int(c)] = value._data[i]
                    else:
                        v = _to_scalar(value)
                        for r, c in zip(rk_list, ck_list):
                            self._data[int(r) * cols + int(c)] = v
                    return
                # row int, col slice
                if isinstance(rk, int) and isinstance(ck, slice):
                    if rk < 0:
                        rk += rows
                    cidx = range(*ck.indices(cols))
                    if isinstance(value, ndarray):
                        for i, c in enumerate(cidx):
                            self._data[rk * cols + c] = value._data[i]
                    else:
                        v = _to_scalar(value)
                        for c in cidx:
                            self._data[rk * cols + c] = v
                    return

        raise IndexError(f"Unsupported setitem: {type(key)} on shape {self._shape}")

    # -- comparison (return bool array) -------------------------------------

    def __gt__(self, other):
        return self._cmp(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._cmp(other, lambda a, b: a >= b)

    def __lt__(self, other):
        return self._cmp(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._cmp(other, lambda a, b: a <= b)

    def __eq__(self, other):
        return self._cmp(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._cmp(other, lambda a, b: a != b)

    def _cmp(self, other, fn):
        if isinstance(other, ndarray):
            d = [fn(a, b) for a, b in zip(self._data, other._data)]
        else:
            ov = _to_scalar(other)
            d = [fn(a, ov) for a in self._data]
        return ndarray(self._shape, _dtype("bool"), d)

    # -- unary --------------------------------------------------------------

    def __neg__(self):
        return ndarray(self._shape, self._dtype, [-v for v in self._data])

    def __abs__(self):
        return ndarray(self._shape, self._dtype, [abs(v) for v in self._data])

    def __bool__(self):
        if self.size == 1:
            return bool(self._data[0])
        raise ValueError("Truth value of array with more than one element is ambiguous")

    def __float__(self):
        if self.size == 1:
            return float(self._data[0])
        raise TypeError("only size-1 arrays can be converted to Python scalars")

    def __int__(self):
        if self.size == 1:
            return int(self._data[0])
        raise TypeError("only size-1 arrays can be converted to Python scalars")

    # -- arithmetic ---------------------------------------------------------

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binop_r(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binop_r(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binop(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binop_r(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b if b != 0 else float('inf') if a > 0 else float('-inf') if a < 0 else float('nan'))

    def __rtruediv__(self, other):
        return self._binop_r(other, lambda a, b: b / a if a != 0 else float('inf') if b > 0 else float('-inf') if b < 0 else float('nan'))

    def __pow__(self, other):
        return self._binop(other, lambda a, b: a ** b)

    def __rpow__(self, other):
        return self._binop_r(other, lambda a, b: b ** a)

    def __mod__(self, other):
        return self._binop(other, lambda a, b: a % b)

    def _binop(self, other, fn):
        if isinstance(other, ndarray):
            # broadcast: if shapes match element-wise
            if self._shape == other._shape:
                d = [fn(a, b) for a, b in zip(self._data, other._data)]
                return ndarray(self._shape, self._dtype, d)
            # broadcast: (M,N) op (N,) or (M,N) op (1,N) or (M,N) op (M,1)
            out_shape, d = _broadcast_op(self, other, fn)
            return ndarray(out_shape, self._dtype, d)
        else:
            ov = _to_scalar(other)
            d = [fn(a, ov) for a in self._data]
            return ndarray(self._shape, self._dtype, d)

    def _binop_r(self, other, fn):
        if isinstance(other, ndarray):
            if self._shape == other._shape:
                d = [fn(a, b) for a, b in zip(self._data, other._data)]
                return ndarray(self._shape, self._dtype, d)
            out_shape, d = _broadcast_op(self, other, fn)
            return ndarray(out_shape, self._dtype, d)
        else:
            ov = _to_scalar(other)
            d = [fn(a, ov) for a in self._data]
            return ndarray(self._shape, self._dtype, d)

    # -- matmul (@ operator) -----------------------------------------------

    def __matmul__(self, other):
        return _matmul(self, other)

    def __rmatmul__(self, other):
        if not isinstance(other, ndarray):
            other = array(other)
        return _matmul(other, self)

    # -- reductions ---------------------------------------------------------

    def sum(self, axis=None):
        if axis is None:
            return sum(self._data)
        if self.ndim == 2:
            rows, cols = self._shape
            if axis == 0:
                d = [sum(self._data[r * cols + c] for r in range(rows)) for c in range(cols)]
                return _make_array_1d(d, self._dtype)
            if axis == 1:
                d = [sum(self._data[r * cols + c] for c in range(cols)) for r in range(rows)]
                return _make_array_1d(d, self._dtype)
        raise NotImplementedError(f"sum axis={axis} ndim={self.ndim}")

    def mean(self, axis=None):
        if axis is None:
            return sum(self._data) / len(self._data) if len(self._data) > 0 else 0.0
        s = self.sum(axis=axis)
        n = self._shape[axis]
        return s * (1.0 / n)

    def std(self, axis=None, ddof=0):
        if axis is None:
            m = self.mean()
            var = sum((v - m) ** 2 for v in self._data) / max(len(self._data) - ddof, 1)
            return math.sqrt(max(var, 0.0))
        raise NotImplementedError

    def max(self, axis=None):
        if axis is None:
            return max(self._data)
        if self.ndim == 2 and axis == 0:
            rows, cols = self._shape
            d = [max(self._data[r * cols + c] for r in range(rows)) for c in range(cols)]
            return _make_array_1d(d, self._dtype)
        if self.ndim == 2 and axis == 1:
            rows, cols = self._shape
            d = [max(self._data[r * cols + c] for c in range(cols)) for r in range(rows)]
            return _make_array_1d(d, self._dtype)
        raise NotImplementedError

    def min(self, axis=None):
        if axis is None:
            return min(self._data)
        if self.ndim == 2 and axis == 0:
            rows, cols = self._shape
            d = [min(self._data[r * cols + c] for r in range(rows)) for c in range(cols)]
            return _make_array_1d(d, self._dtype)
        if self.ndim == 2 and axis == 1:
            rows, cols = self._shape
            d = [min(self._data[r * cols + c] for c in range(cols)) for r in range(rows)]
            return _make_array_1d(d, self._dtype)
        raise NotImplementedError

    def any(self):
        return any(self._data)

    def all(self):
        return all(self._data)

    # -- shape manipulation -------------------------------------------------

    def copy(self):
        return ndarray(self._shape, self._dtype, list(self._data))

    def flatten(self):
        return ndarray((self.size,), self._dtype, list(self._data))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        # handle -1
        neg_idx = None
        known = 1
        for i, s in enumerate(shape):
            if s == -1:
                neg_idx = i
            else:
                known *= s
        if neg_idx is not None:
            shape = list(shape)
            shape[neg_idx] = self.size // known
            shape = tuple(shape)
        return ndarray(shape, self._dtype, list(self._data))

    def astype(self, dtype):
        if isinstance(dtype, str):
            dtype = _dtype(dtype)
        if isinstance(dtype, type) and dtype is float:
            dtype = float64
        if isinstance(dtype, type) and dtype is int:
            dtype = int64
        out = ndarray(self._shape, dtype, list(self._data))
        if dtype in (float64, float32):
            out._data = [float(v) for v in out._data]
        elif dtype in (int64, int32):
            out._data = [int(v) for v in out._data]
        elif dtype == _dtype("bool") or (isinstance(dtype, type) and dtype.__name__ == "bool_"):
            out._data = [bool(v) for v in out._data]
        return out

    def tobytes(self):
        return b"".join(_struct.pack("d", float(v)) for v in self._data)

    # -- repr ---------------------------------------------------------------

    def __repr__(self):
        if self.ndim == 1:
            inner = ", ".join(f"{v}" for v in self._data[:20])
            if len(self._data) > 20:
                inner += ", ..."
            return f"array([{inner}])"
        if self.ndim == 2:
            rows, cols = self._shape
            lines = []
            for r in range(min(rows, 10)):
                row = self._data[r * cols:(r + 1) * cols]
                lines.append("[" + ", ".join(f"{v}" for v in row[:10]) + "]")
            return "array([" + ",\n       ".join(lines) + "])"
        return f"ndarray(shape={self._shape})"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_scalar(v):
    if isinstance(v, ndarray):
        if v.size == 1:
            return v._data[0]
        raise ValueError("setting element with sequence")
    return float(v) if not isinstance(v, (bool,)) else v

def _make_array_1d(data_list, dtype=None):
    a = ndarray((len(data_list),), dtype or float64)
    a._data = list(data_list)
    return a

def _broadcast_op(a, b, fn):
    """Elementwise op with simple broadcasting rules."""
    sa, sb = a._shape, b._shape
    # (M,N) op (N,)  or  (M,N) op (1,N)
    if len(sa) == 2 and (len(sb) == 1 and sb[0] == sa[1]):
        rows, cols = sa
        d = [fn(a._data[r * cols + c], b._data[c]) for r in range(rows) for c in range(cols)]
        return sa, d
    if len(sa) == 2 and len(sb) == 2 and sb[0] == 1 and sb[1] == sa[1]:
        rows, cols = sa
        d = [fn(a._data[r * cols + c], b._data[c]) for r in range(rows) for c in range(cols)]
        return sa, d
    # (N,) op (M,N)
    if len(sa) == 1 and len(sb) == 2 and sa[0] == sb[1]:
        rows, cols = sb
        d = [fn(a._data[c], b._data[r * cols + c]) for r in range(rows) for c in range(cols)]
        return sb, d
    # (M,N) op (M,1)
    if len(sa) == 2 and len(sb) == 2 and sb[1] == 1 and sb[0] == sa[0]:
        rows, cols = sa
        d = [fn(a._data[r * cols + c], b._data[r]) for r in range(rows) for c in range(cols)]
        return sa, d
    # (M,1) op (M,N)
    if len(sa) == 2 and len(sb) == 2 and sa[1] == 1 and sa[0] == sb[0]:
        rows, cols = sb
        d = [fn(a._data[r], b._data[r * cols + c]) for r in range(rows) for c in range(cols)]
        return sb, d
    # scalar-shaped arrays
    if a.size == 1:
        av = a._data[0]
        d = [fn(av, bv) for bv in b._data]
        return sb, d
    if b.size == 1:
        bv = b._data[0]
        d = [fn(av, bv) for av in a._data]
        return sa, d
    raise ValueError(f"Cannot broadcast shapes {sa} and {sb}")


def _matmul(a, b):
    """Matrix multiplication supporting 1-D and 2-D arrays."""
    if a.ndim == 1 and b.ndim == 1:
        # dot product
        if a._shape[0] != b._shape[0]:
            raise ValueError(f"matmul shape mismatch: {a._shape} @ {b._shape}")
        return sum(x * y for x, y in zip(a._data, b._data))
    if a.ndim == 2 and b.ndim == 1:
        M, K = a._shape
        if K != b._shape[0]:
            raise ValueError(f"matmul shape mismatch: {a._shape} @ {b._shape}")
        d = [0.0] * M
        for i in range(M):
            s = 0.0
            off = i * K
            for k in range(K):
                s += a._data[off + k] * b._data[k]
            d[i] = s
        return _make_array_1d(d)
    if a.ndim == 1 and b.ndim == 2:
        K, N = b._shape
        if a._shape[0] != K:
            raise ValueError(f"matmul shape mismatch: {a._shape} @ {b._shape}")
        d = [0.0] * N
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += a._data[k] * b._data[k * N + j]
            d[j] = s
        return _make_array_1d(d)
    if a.ndim == 2 and b.ndim == 2:
        M, Ka = a._shape
        Kb, N = b._shape
        if Ka != Kb:
            raise ValueError(f"matmul shape mismatch: {a._shape} @ {b._shape}")
        K = Ka
        d = [0.0] * (M * N)
        for i in range(M):
            a_off = i * K
            out_off = i * N
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += a._data[a_off + k] * b._data[k * N + j]
                d[out_off + j] = s
        return ndarray((M, N), float64, d)
    raise NotImplementedError(f"matmul for shapes {a._shape}, {b._shape}")

# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------

def array(obj, dtype=None):
    """Create ndarray from list, list-of-lists, or scalar."""
    if isinstance(obj, ndarray):
        out = obj.copy()
        if dtype is not None:
            out = out.astype(dtype)
        return out
    if isinstance(obj, (int, float)):
        a = ndarray((1,), dtype or float64)
        a._data[0] = float(obj)
        return a
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            dt = float64
            if dtype is not None:
                dt = dtype if isinstance(dtype, _dtype) else _dtype(str(dtype))
            return ndarray((0,), dt, [])
        first = obj[0]
        if isinstance(first, (list, tuple, ndarray)):
            # 2-D
            rows = len(obj)
            cols = len(first)
            flat = []
            for row in obj:
                if isinstance(row, ndarray):
                    flat.extend(row._data)
                else:
                    flat.extend(float(v) for v in row)
            out = ndarray((rows, cols), dtype or float64, flat)
            return out
        else:
            flat = [float(v) if not isinstance(v, bool) else (1.0 if v else 0.0) for v in obj]
            # Detect boolean lists: if all elements are bool and no dtype specified, use bool_
            if dtype is None and _builtins.all(isinstance(v, bool) for v in obj):
                dt = _dtype("bool")
            else:
                dt = dtype or float64
            out = ndarray((len(flat),), dt, flat)
            if dtype is not None:
                out = out.astype(dtype)
            return out
    raise TypeError(f"Cannot create array from {type(obj)}")


def asarray(obj, dtype=None):
    if isinstance(obj, ndarray):
        if dtype is not None:
            return obj.astype(dtype)
        return obj
    return array(obj, dtype=dtype)


def zeros(shape, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)
    dt = dtype or float64
    out = ndarray(shape, dt)
    # already zeros
    return out


def ones(shape, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)
    dt = dtype or float64
    size = 1
    for s in shape:
        size *= s
    return ndarray(shape, dt, [1.0] * size)


def full(shape, fill_value, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)
    dt = dtype or float64
    size = 1
    for s in shape:
        size *= s
    fv = float(fill_value) if not isinstance(fill_value, int) else fill_value
    return ndarray(shape, dt, [fv] * size)


def empty(shape, dtype=None):
    return zeros(shape, dtype)


def eye(N, M=None, k=0):
    if M is None:
        M = N
    if isinstance(N, int) and isinstance(M, int):
        d = [0.0] * (N * M)
        for i in range(min(N, M)):
            if 0 <= i + k < M and 0 <= i < N:
                d[i * M + i + k] = 1.0
        return ndarray((N, M), float64, d)
    # eye(1, self.n, i) -- used as (1, M) matrix with a single 1
    r = int(N)
    c = int(M)
    d = [0.0] * (r * c)
    for i in range(min(r, c)):
        col = i + k
        if 0 <= col < c and 0 <= i < r:
            d[i * c + col] = 1.0
    return ndarray((r, c), float64, d)


def arange(start, stop=None, step=1, dtype=None):
    if stop is None:
        stop = start
        start = 0
    d = []
    v = start
    if step > 0:
        while v < stop:
            d.append(float(v))
            v += step
    else:
        while v > stop:
            d.append(float(v))
            v += step
    return _make_array_1d(d, dtype or float64)


def diag(arr):
    """Extract diagonal or construct diagonal matrix."""
    if isinstance(arr, ndarray):
        if arr.ndim == 1:
            n = arr._shape[0]
            d = [0.0] * (n * n)
            for i in range(n):
                d[i * n + i] = arr._data[i]
            return ndarray((n, n), float64, d)
        if arr.ndim == 2:
            n = min(arr._shape[0], arr._shape[1])
            cols = arr._shape[1]
            d = [arr._data[i * cols + i] for i in range(n)]
            return _make_array_1d(d)
    raise TypeError("diag requires ndarray")


def fill_diagonal(arr, val):
    """Fill diagonal of 2-D array in-place."""
    if arr.ndim != 2:
        raise ValueError("fill_diagonal requires 2-D array")
    n = min(arr._shape[0], arr._shape[1])
    cols = arr._shape[1]
    v = _to_scalar(val) if not isinstance(val, float) else val
    for i in range(n):
        arr._data[i * cols + i] = v


def dot(a, b):
    """Dot product or matrix multiply."""
    if not isinstance(a, ndarray):
        a = array(a)
    if not isinstance(b, ndarray):
        b = array(b)
    return _matmul(a, b)


def concatenate(arrays, axis=0):
    arrays = [a if isinstance(a, ndarray) else array(a) for a in arrays]
    if all(a.ndim == 1 for a in arrays):
        d = []
        for a in arrays:
            d.extend(a._data)
        return _make_array_1d(d)
    if all(a.ndim == 2 for a in arrays) and axis == 0:
        cols = arrays[0]._shape[1]
        d = []
        total_rows = 0
        for a in arrays:
            d.extend(a._data)
            total_rows += a._shape[0]
        return ndarray((total_rows, cols), float64, d)
    raise NotImplementedError("concatenate for given shapes/axis")


# ---------------------------------------------------------------------------
# element-wise math
# ---------------------------------------------------------------------------

def exp(x):
    if isinstance(x, ndarray):
        d = []
        for v in x._data:
            try:
                d.append(math.exp(v))
            except OverflowError:
                d.append(float('inf'))
        return ndarray(x._shape, x._dtype, d)
    try:
        return math.exp(x)
    except OverflowError:
        return float('inf')


def log(x):
    if isinstance(x, ndarray):
        d = [math.log(max(v, 1e-300)) for v in x._data]
        return ndarray(x._shape, x._dtype, d)
    return math.log(max(float(x), 1e-300))


def log2(x):
    _log2 = math.log(2)
    if isinstance(x, ndarray):
        d = [math.log(max(v, 1e-300)) / _log2 for v in x._data]
        return ndarray(x._shape, x._dtype, d)
    return math.log(max(float(x), 1e-300)) / _log2


def tanh(x):
    if isinstance(x, ndarray):
        d = [math.tanh(v) for v in x._data]
        return ndarray(x._shape, x._dtype, d)
    return math.tanh(float(x))


def abs(x):
    if isinstance(x, ndarray):
        d = [__builtins__['abs'](v) if isinstance(__builtins__, dict) else _builtin_abs(v) for v in x._data]
        return ndarray(x._shape, x._dtype, d)
    return _builtin_abs(x)

import builtins as _builtins
_builtin_abs = _builtins.abs
_builtin_max = _builtins.max
_builtin_min = _builtins.min
_builtin_sum = _builtins.sum


def abs(x):
    if isinstance(x, ndarray):
        d = [_builtin_abs(v) for v in x._data]
        return ndarray(x._shape, x._dtype, d)
    return _builtin_abs(x)


def sqrt(x):
    if isinstance(x, ndarray):
        d = [math.sqrt(_builtin_max(v, 0.0)) for v in x._data]
        return ndarray(x._shape, x._dtype, d)
    return math.sqrt(_builtin_max(float(x), 0.0))


def clip(x, a_min, a_max):
    if isinstance(x, ndarray):
        d = []
        for v in x._data:
            if a_min is not None and v < a_min:
                v = float(a_min)
            if a_max is not None and v > a_max:
                v = float(a_max)
            d.append(v)
        return ndarray(x._shape, x._dtype, d)
    v = float(x)
    if a_min is not None and v < a_min:
        v = float(a_min)
    if a_max is not None and v > a_max:
        v = float(a_max)
    return v


def where(condition, x=None, y=None):
    """np.where -- if x/y given, conditional select; else return indices."""
    if isinstance(condition, ndarray):
        if x is None and y is None:
            # return tuple of index arrays
            if condition.ndim == 1:
                idxs = [i for i, v in enumerate(condition._data) if v]
                return (_make_array_1d([float(i) for i in idxs], int64),)
            if condition.ndim == 2:
                rows, cols = condition._shape
                ridx, cidx = [], []
                for r in range(rows):
                    for c in range(cols):
                        if condition._data[r * cols + c]:
                            ridx.append(float(r))
                            cidx.append(float(c))
                return (_make_array_1d(ridx, int64), _make_array_1d(cidx, int64))
        # conditional select
        cond = condition
        if isinstance(x, ndarray) and isinstance(y, ndarray):
            d = [xv if cv else yv for cv, xv, yv in zip(cond._data, x._data, y._data)]
            return ndarray(cond._shape, x._dtype, d)
        xv = x if not isinstance(x, ndarray) else None
        yv = y if not isinstance(y, ndarray) else None
        d = []
        for i, cv in enumerate(cond._data):
            if cv:
                d.append(x._data[i] if isinstance(x, ndarray) else float(x))
            else:
                d.append(y._data[i] if isinstance(y, ndarray) else float(y))
        return ndarray(cond._shape, float64, d)
    raise TypeError("where requires ndarray condition")


def argsort(x):
    if isinstance(x, (list, tuple)):
        indices = sorted(range(len(x)), key=lambda i: x[i])
        return _make_array_1d([float(i) for i in indices], int64)
    if isinstance(x, ndarray) and x.ndim == 1:
        indices = sorted(range(len(x._data)), key=lambda i: x._data[i])
        return _make_array_1d([float(i) for i in indices], int64)
    raise NotImplementedError("argsort for non-1D")


def argpartition(x, kth):
    """np.argpartition — k 番目の要素を正しい位置に配置

    O(n) の introselect が理想だが、shim では sorted ベースで O(n log n)。
    標準 NumPy では C 実装の O(n) introselect が使われる。
    API 互換性のため提供: argpartition(a, -k)[-k:] で top-k が取得可能。
    """
    if isinstance(x, ndarray) and x.ndim == 1:
        data = x._data
    elif isinstance(x, (list, tuple)):
        data = list(x)
    else:
        raise NotImplementedError("argpartition for non-1D")
    n = len(data)
    if isinstance(kth, int):
        if kth < 0:
            kth += n
    # Full sort (shim limitation) → return indices in partitioned order
    indices = sorted(range(n), key=lambda i: data[i])
    return _make_array_1d([float(i) for i in indices], int64)


def searchsorted(a, v, side='left'):
    """np.searchsorted — ソート済み配列への二分探索

    shim では Python bisect モジュールで実装。
    標準 NumPy では C 実装の一括二分探索。
    """
    import bisect as _bisect_mod

    if isinstance(a, ndarray):
        a_list = list(a._data)
    elif isinstance(a, (list, tuple)):
        a_list = list(a)
    else:
        raise TypeError("searchsorted requires array-like")

    fn = _bisect_mod.bisect_left if side == 'left' else _bisect_mod.bisect_right

    if isinstance(v, ndarray):
        result = [fn(a_list, float(v._data[i])) for i in range(len(v._data))]
        return _make_array_1d([float(r) for r in result], int64)
    elif isinstance(v, (list, tuple)):
        result = [fn(a_list, float(vi)) for vi in v]
        return _make_array_1d([float(r) for r in result], int64)
    elif isinstance(v, (int, float)):
        return int(fn(a_list, float(v)))
    else:
        raise TypeError(f"searchsorted: unsupported v type {type(v)}")


def argmax(x, axis=None):
    if isinstance(x, ndarray) and x.ndim == 1 and axis is None:
        best = 0
        for i in range(1, len(x._data)):
            if x._data[i] > x._data[best]:
                best = i
        return best
    raise NotImplementedError


def argmin(x, axis=None):
    if isinstance(x, ndarray) and x.ndim == 1 and axis is None:
        best = 0
        for i in range(1, len(x._data)):
            if x._data[i] < x._data[best]:
                best = i
        return best
    raise NotImplementedError


def isfinite(x):
    if isinstance(x, ndarray):
        d = [math.isfinite(v) if isinstance(v, float) else True for v in x._data]
        return ndarray(x._shape, _dtype("bool"), d)
    return math.isfinite(float(x))


def cumsum(x, axis=None):
    if isinstance(x, ndarray) and x.ndim == 1:
        d = []
        s = 0.0
        for v in x._data:
            s += v
            d.append(s)
        return _make_array_1d(d)
    raise NotImplementedError("cumsum for non-1D")


def isscalar(x):
    return isinstance(x, (int, float, _scalar, bool)) and not isinstance(x, ndarray)


def sum(x, axis=None):
    if isinstance(x, ndarray):
        return x.sum(axis=axis)
    return _builtin_sum(x)


def all(x):
    if isinstance(x, ndarray):
        return x.all()
    return _builtins.all(x)


def any(x):
    if isinstance(x, ndarray):
        return x.any()
    return _builtins.any(x)


def allclose(a, b, atol=1e-8, rtol=1e-5):
    if not isinstance(a, ndarray):
        a = array(a)
    if not isinstance(b, ndarray):
        if isinstance(b, (int, float)):
            # Compare all elements to a scalar
            for av in a._data:
                if _builtin_abs(float(av) - float(b)) > atol + rtol * _builtin_abs(float(b)):
                    return False
            return True
        b = array(b)
    if a.size != b.size:
        return False
    for av, bv in zip(a._data, b._data):
        if _builtin_abs(float(av) - float(bv)) > atol + rtol * _builtin_abs(float(bv)):
            return False
    return True


def maximum(a, b):
    if isinstance(a, ndarray) and isinstance(b, ndarray):
        d = [_builtin_max(av, bv) for av, bv in zip(a._data, b._data)]
        return ndarray(a._shape, a._dtype, d)
    if isinstance(a, ndarray):
        bv = float(b)
        d = [_builtin_max(av, bv) for av in a._data]
        return ndarray(a._shape, a._dtype, d)
    raise NotImplementedError

def minimum(a, b):
    if isinstance(a, ndarray) and isinstance(b, ndarray):
        d = [_builtin_min(av, bv) for av, bv in zip(a._data, b._data)]
        return ndarray(a._shape, a._dtype, d)
    if isinstance(a, ndarray):
        bv = float(b)
        d = [_builtin_min(av, bv) for av in a._data]
        return ndarray(a._shape, a._dtype, d)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------

def histogram(data, bins=10, density=False):
    if isinstance(data, ndarray):
        vals = list(data._data)
    else:
        vals = list(data)
    if len(vals) == 0:
        counts = [0] * bins
        edges = [0.0] * (bins + 1)
        return _make_array_1d([float(c) for c in counts], int64), _make_array_1d(edges)
    mn = _builtin_min(vals)
    mx = _builtin_max(vals)
    if mn == mx:
        mx = mn + 1.0
    bin_width = (mx - mn) / bins
    counts = [0] * bins
    for v in vals:
        idx = int((v - mn) / bin_width)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1
    edges = [mn + i * bin_width for i in range(bins + 1)]
    if density:
        total = _builtin_sum(counts) * bin_width
        counts = [c / total if total > 0 else 0.0 for c in counts]
    return _make_array_1d([float(c) for c in counts], int64), _make_array_1d(edges)


# ---------------------------------------------------------------------------
# ufunc shim: add.at
# ---------------------------------------------------------------------------

class _AddUfunc:
    @staticmethod
    def at(arr, indices, values):
        """np.add.at(arr, indices, values) -- unbuffered in-place add."""
        if isinstance(indices, ndarray):
            indices = indices._data
        if isinstance(values, ndarray):
            values = values._data
        if isinstance(values, (int, float)):
            for idx in indices:
                arr._data[int(idx)] += values
        else:
            for idx, val in zip(indices, values):
                arr._data[int(idx)] += float(val)

add = _AddUfunc()


# ---------------------------------------------------------------------------
# load / save (minimal)
# ---------------------------------------------------------------------------

def load(path, mmap_mode=None):
    """Minimal np.load -- only supports .npy files written by this shim."""
    import struct
    path = str(path)
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic[:3] != b'\x93NU':
            raise ValueError("Not a npy file")
        version = struct.unpack("BB", f.read(2))
        if version[0] == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        else:
            header_len = struct.unpack("<I", f.read(4))[0]
        header = f.read(header_len).decode("latin1")
        # parse shape and dtype from header
        import re
        shape_match = re.search(r"'shape'\s*:\s*\(([^)]*)\)", header)
        shape_str = shape_match.group(1) if shape_match else ""
        shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
        if len(shape) == 0:
            shape = (0,)
        size = 1
        for s in shape:
            size *= s
        data_bytes = f.read(size * 8)
        flat = list(struct.unpack(f"<{size}d", data_bytes))
        return ndarray(shape, float64, flat)


def save(path, arr):
    """Minimal np.save for float64 arrays."""
    import struct
    path = str(path)
    if isinstance(arr, ndarray):
        shape = arr._shape
        flat = arr._data
    else:
        arr = array(arr)
        shape = arr._shape
        flat = arr._data
    header = f"{{'descr': '<f8', 'fortran_order': False, 'shape': {shape}, }}"
    # pad to multiple of 64
    while (10 + len(header) + 1) % 64 != 0:
        header += " "
    header += "\n"
    header_bytes = header.encode("latin1")
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY")
        f.write(struct.pack("BB", 1, 0))
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        for v in flat:
            f.write(struct.pack("<d", float(v)))


# ---------------------------------------------------------------------------
# random submodule
# ---------------------------------------------------------------------------

class random:
    """numpy.random namespace."""

    @staticmethod
    def default_rng(seed=None):
        return _Generator(seed)

    @staticmethod
    def RandomState(seed=None):
        return _RandomState(seed)

    @staticmethod
    def rand(*shape):
        if len(shape) == 0:
            return _pyrandom.random()
        size = 1
        for s in shape:
            size *= s
        d = [_pyrandom.random() for _ in range(size)]
        if len(shape) == 1:
            return _make_array_1d(d)
        return ndarray(shape, float64, d)

    @staticmethod
    def randn(*shape):
        if len(shape) == 0:
            return _pyrandom.gauss(0, 1)
        size = 1
        for s in shape:
            size *= s
        d = [_pyrandom.gauss(0, 1) for _ in range(size)]
        if len(shape) == 1:
            return _make_array_1d(d)
        return ndarray(shape, float64, d)


class _Generator:
    """numpy.random.Generator shim."""

    def __init__(self, seed=None):
        self._rng = _pyrandom.Random(seed)

    def integers(self, low, high=None, size=None):
        if high is None:
            high = low
            low = 0
        if size is None:
            return self._rng.randint(low, high - 1)
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        d = [float(self._rng.randint(low, high - 1)) for _ in range(total)]
        if len(size) == 1:
            return _make_array_1d(d)
        return ndarray(size, int64, d)

    def random(self, size=None):
        if size is None:
            return self._rng.random()
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        d = [self._rng.random() for _ in range(total)]
        if len(size) == 1:
            return _make_array_1d(d)
        return ndarray(size, float64, d)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return self._rng.gauss(loc, scale)
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        d = [self._rng.gauss(loc, scale) for _ in range(total)]
        if len(size) == 1:
            return _make_array_1d(d)
        return ndarray(size, float64, d)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return self._rng.uniform(low, high)
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        d = [self._rng.uniform(low, high) for _ in range(total)]
        if len(size) == 1:
            return _make_array_1d(d)
        return ndarray(size, float64, d)

    def exponential(self, scale=1.0, size=None):
        if size is None:
            return self._rng.expovariate(1.0 / scale)
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        d = [self._rng.expovariate(1.0 / scale) for _ in range(total)]
        if len(size) == 1:
            return _make_array_1d(d)
        return ndarray(size, float64, d)

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, ndarray):
            pool = list(a._data)
        elif isinstance(a, (list, tuple)):
            pool = list(a)
        elif isinstance(a, int):
            pool = list(range(a))
        else:
            pool = list(a)

        if size is None:
            if p is not None:
                # weighted choice
                r = self._rng.random()
                cumsum = 0.0
                for i, pi in enumerate(p):
                    cumsum += pi
                    if r <= cumsum:
                        return pool[i]
                return pool[-1]
            return self._rng.choice(pool)

        if isinstance(size, int):
            total = size
            out_shape = (size,)
        else:
            total = 1
            for s in size:
                total *= s
            out_shape = size

        if replace:
            d = [self._rng.choice(pool) for _ in range(total)]
        else:
            d = self._rng.sample(pool, total)

        d = [float(v) for v in d]
        if len(out_shape) == 1:
            return _make_array_1d(d)
        return ndarray(out_shape, float64, d)


class _RandomState:
    """numpy.random.RandomState shim — wraps _Generator with legacy API."""

    def __init__(self, seed=None):
        self._gen = _Generator(seed)

    def rand(self, *shape):
        if len(shape) == 0:
            return self._gen.random()
        total = 1
        for s in shape:
            total *= s
        d = [self._gen._rng.random() for _ in range(total)]
        if len(shape) == 1:
            return _make_array_1d(d)
        return ndarray(shape, float64, d)

    def randn(self, *shape):
        if len(shape) == 0:
            return self._gen._rng.gauss(0, 1)
        total = 1
        for s in shape:
            total *= s
        d = [self._gen._rng.gauss(0, 1) for _ in range(total)]
        if len(shape) == 1:
            return _make_array_1d(d)
        return ndarray(shape, float64, d)

    def randint(self, low, high=None, size=None):
        if high is None:
            high = low
            low = 0
        return self._gen.integers(low, high, size=size)

    def choice(self, a, size=None, replace=True, p=None):
        return self._gen.choice(a, size=size, replace=replace, p=p)

    def random(self, size=None):
        return self._gen.random(size=size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._gen.uniform(low, high, size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._gen.normal(loc, scale, size=size)


# ---------------------------------------------------------------------------
# linalg submodule
# ---------------------------------------------------------------------------

class linalg:

    @staticmethod
    def norm(x, axis=None, keepdims=False, ord=None):
        if not isinstance(x, ndarray):
            x = array(x)
        if axis is None:
            s = math.sqrt(_builtin_sum(v * v for v in x._data))
            if keepdims:
                return ndarray((1,) * x.ndim, float64, [s])
            return s
        if x.ndim == 2 and axis == 1:
            rows, cols = x._shape
            d = []
            for r in range(rows):
                off = r * cols
                s = math.sqrt(_builtin_sum(x._data[off + c] ** 2 for c in range(cols)))
                d.append(s)
            if keepdims:
                return ndarray((rows, 1), float64, d)
            return _make_array_1d(d)
        if x.ndim == 2 and axis == 0:
            rows, cols = x._shape
            d = []
            for c in range(cols):
                s = math.sqrt(_builtin_sum(x._data[r * cols + c] ** 2 for r in range(rows)))
                d.append(s)
            if keepdims:
                return ndarray((1, cols), float64, d)
            return _make_array_1d(d)
        if x.ndim == 1:
            s = math.sqrt(_builtin_sum(v * v for v in x._data))
            return s
        raise NotImplementedError(f"linalg.norm axis={axis} ndim={x.ndim}")


# ---------------------------------------------------------------------------
# testing submodule
# ---------------------------------------------------------------------------

class testing:
    @staticmethod
    def assert_array_equal(a, b):
        if not isinstance(a, ndarray):
            a = array(a)
        if not isinstance(b, ndarray):
            b = array(b)
        if a._shape != b._shape:
            raise AssertionError(f"Shapes differ: {a._shape} vs {b._shape}")
        for i, (av, bv) in enumerate(zip(a._data, b._data)):
            if av != bv:
                raise AssertionError(f"Arrays differ at index {i}: {av} != {bv}")
