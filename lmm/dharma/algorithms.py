"""Dharma algorithms — sparse graph construction, supermodular greedy, balancer.

Key algorithms:
  1. Sparse k-NN graph via HNSW (or brute-force fallback)
  2. Supermodular greedy warm-start for SA
  3. BodhisattvaQUBO formulation
  4. MadhyamakaBalancer (exponential gradient with Lyapunov stability)
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm._compat import HAS_ARGPARTITION
from lmm.qubo import QUBOBuilder


# ===================================================================
# 1. Sparse impact graph
# ===================================================================

def build_sparse_impact_graph(
    data: np.ndarray, k: int = 20, M: int = 16, ef_construction: int = 200,
    use_hnswlib: bool = True,
) -> sparse.csr_matrix:
    """Build sparse k-NN cosine similarity graph, O(n*k) memory."""
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError(f"data must be non-empty 2D, got shape {data.shape}")
    n, dim = data.shape
    k = min(k, n - 1)

    if use_hnswlib:
        try:
            return _build_hnsw(data, n, dim, k, M, ef_construction)
        except ImportError:
            pass
    return _build_brute_force_sparse(data, n, k)


def _build_hnsw(
    data: np.ndarray, n: int, dim: int, k: int, M: int, ef_construction: int,
) -> sparse.csr_matrix:
    import hnswlib
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, M=M, ef_construction=ef_construction)
    norms = np.clip(np.linalg.norm(data, axis=1, keepdims=True), 1e-10, None)
    normed = data / norms
    index.add_items(normed, np.arange(n))
    index.set_ef(max(k + 10, 50))
    labels, distances = index.knn_query(normed, k=k + 1)

    n_nb = labels.shape[1]
    rows = np.repeat(np.arange(n), n_nb)
    cols = labels.ravel().astype(int)
    sims = np.maximum(0.0, 1.0 - distances.ravel())
    mask = ((rows != cols) * (sims > 1e-6)) > 0
    graph = sparse.csr_matrix(
        (sims[mask], (rows[mask], cols[mask])), shape=(n, n), dtype=np.float64,
    )
    return (graph + graph.T) / 2.0


def _build_brute_force_sparse(data: np.ndarray, n: int, k: int) -> sparse.csr_matrix:
    if n > 50_000:
        import warnings
        warnings.warn(
            f"Brute-force k-NN with n={n:,} is O(n^2). "
            f"Install hnswlib for O(n*log(n)): pip install hnswlib"
        )
    norms = np.clip(np.linalg.norm(data, axis=1, keepdims=True), 1e-10, None)
    normed = data / norms
    norm_T = normed.T
    _dot_works = True
    try:
        _test = np.dot(normed[0], norm_T)
        if not np.isfinite(_test).all() or abs(float(_test[0]) - 1.0) > 0.1:
            _dot_works = False
    except (TypeError, ValueError):
        _dot_works = False

    max_nnz = n * k
    out_r = np.empty(max_nnz, dtype=int)
    out_c = np.empty(max_nnz, dtype=int)
    out_v = np.empty(max_nnz)
    ptr = 0

    for i in range(n):
        if _dot_works:
            row = np.dot(normed[i], norm_T)
        else:
            row = np.array([float(np.dot(normed[i], normed[j])) for j in range(n)])
        row[i] = -np.inf
        if HAS_ARGPARTITION:
            top = np.argpartition(row, -k)[-k:]
        else:
            top = np.argsort(row)[-k:]
        sims = row[top]
        pos = sims > 0.0
        if pos.any():
            gi = top[pos]
            gv = sims[pos]
            m = len(gi)
            out_r[ptr:ptr + m] = i
            out_c[ptr:ptr + m] = gi
            out_v[ptr:ptr + m] = gv
            ptr += m

    if ptr == 0:
        for i in range(n):
            if _dot_works:
                row = np.dot(normed[i], norm_T)
            else:
                row = np.array([float(np.dot(normed[i], normed[j])) for j in range(n)])
            row[i] = -np.inf
            if HAS_ARGPARTITION:
                top = np.argpartition(row, -k)[-k:]
            else:
                top = np.argsort(row)[-k:]
            top = np.array([int(x) for x in top])
            not_self = top != i
            if not_self.any():
                good = top[not_self]
                m = len(good)
                out_r[ptr:ptr + m] = i
                out_c[ptr:ptr + m] = good
                out_v[ptr:ptr + m] = 1.0 / k
                ptr += m

    if ptr > 0:
        graph = sparse.csr_matrix(
            (out_v[:ptr], (out_r[:ptr], out_c[:ptr])), shape=(n, n), dtype=np.float64,
        )
    else:
        graph = sparse.csr_matrix((n, n), dtype=np.float64)
    return (graph + graph.T) / 2.0


# ===================================================================
# 1b. Query-conditioned and subgraph extraction
# ===================================================================

def query_condition_graph(
    J_static: sparse.csr_matrix, relevance_scores: np.ndarray,
) -> sparse.csr_matrix:
    """Dynamic graph: J_dynamic[i,j] = J_static[i,j] * rel[i] * rel[j]."""
    n = J_static.shape[0]
    rel = np.asarray(relevance_scores).flatten()
    if len(rel) < n:
        rel = np.concatenate([rel, np.zeros(n - len(rel))])
    elif len(rel) > n:
        rel = rel[:n]
    coo = J_static.tocoo()
    if len(coo.data) == 0:
        return sparse.csr_matrix(J_static.shape)
    new_data = coo.data * rel[coo.row] * rel[coo.col]
    return sparse.csr_matrix((new_data, (coo.row, coo.col)), shape=J_static.shape)


def extract_subgraph(
    J_full: sparse.csr_matrix, indices: np.ndarray,
) -> sparse.csr_matrix:
    """Extract induced subgraph for given indices."""
    idx = np.array([int(i) for i in indices])
    n_sub = len(idx)
    if n_sub == 0:
        return sparse.csr_matrix((0, 0))
    N = J_full.shape[0]
    local_map = np.full(N, -1, dtype=int)
    local_map[idx] = np.arange(n_sub)
    coo = J_full.tocoo()
    rl = local_map[coo.row]
    cl = local_map[coo.col]
    mask = ((rl >= 0) * (cl >= 0)) > 0
    return sparse.csr_matrix(
        (coo.data[mask], (rl[mask], cl[mask])), shape=(n_sub, n_sub),
    )


# ===================================================================
# 2. Supermodular greedy warm-start
# ===================================================================

def vectorized_greedy_initialize(
    surprises: np.ndarray, impact_graph: sparse.csr_matrix | np.ndarray,
    k: int, alpha: float = 1.0, beta: float = 0.5,
) -> np.ndarray:
    """Supermodular greedy: gains increase as sangha grows."""
    n = len(surprises)
    k = min(k, n)
    gains = alpha * surprises.copy()
    x = np.zeros(n, dtype=float)
    mask = np.ones(n, dtype=bool)
    is_sp = sparse.issparse(impact_graph)

    for _ in range(k):
        masked_gains = np.where(mask, gains, -np.inf)
        best = int(np.argmax(masked_gains))
        if masked_gains[best] == -np.inf:
            break
        mask[best] = False
        x[best] = 1.0

        if is_sp:
            row = impact_graph.getrow(best)
            nb, w = row.indices, row.data
        else:
            nb = np.where(impact_graph[best] > 1e-6)[0]
            w = impact_graph[best, nb]

        if len(nb) > 0:
            nb = np.array([int(v) for v in nb])
            w = np.array([float(v) for v in w])
            active = mask[nb]
            if active.any():
                np.add.at(gains, nb[active], 2.0 * beta * w[active])

    return x


# ===================================================================
# 3. BodhisattvaQUBO
# ===================================================================

class BodhisattvaQUBO:
    """QUBO formulation with Prajna + Karuna + Sila terms."""

    def __init__(self, n: int):
        self.builder = QUBOBuilder(n)
        self.n = n

    def add_prajna_term(self, surprises: np.ndarray, alpha: float = 1.0) -> None:
        self.builder.add_surprise_objective(surprises, alpha=alpha)

    def add_karuna_term(self, impact_graph: np.ndarray, beta: float = 0.5) -> None:
        n = self.n
        for i in range(n):
            self.builder.add_linear(i, beta * impact_graph[i].sum())
            for j in range(i + 1, n):
                w = impact_graph[i, j]
                if abs(w) > 1e-10:
                    self.builder.add_quadratic(i, j, -beta * w)

    def add_karuna_term_sparse(
        self, impact_graph: sparse.csr_matrix, beta: float = 0.5,
    ) -> None:
        row_sums = np.asarray(impact_graph.sum(axis=1)).flatten()
        self.builder._diag += beta * row_sums
        coo = sparse.triu(impact_graph, k=1).tocoo()
        mask = np.abs(coo.data) > 1e-10
        rows, cols, vals = coo.row[mask], coo.col[mask], coo.data[mask]
        if len(vals) > 0:
            scaled = -beta * vals / 2.0
            r_list = [int(r) for r in rows]
            c_list = [int(c) for c in cols]
            s_list = [float(v) for v in scaled]
            self.builder._offdiag_rows.extend(r_list)
            self.builder._offdiag_cols.extend(c_list)
            self.builder._offdiag_vals.extend(s_list)
            self.builder._offdiag_rows.extend(c_list)
            self.builder._offdiag_cols.extend(r_list)
            self.builder._offdiag_vals.extend(s_list)
            self.builder._invalidate_cache()

    def add_sila_term(self, k: int, gamma: float = 10.0) -> None:
        self.builder.add_cardinality_constraint(k, gamma=gamma)

    def get_builder(self) -> QUBOBuilder:
        return self.builder


# ===================================================================
# 4. MadhyamakaBalancer
# ===================================================================

class MadhyamakaBalancer:
    """Auto-balance alpha/beta via exponential gradient with Lyapunov stability."""

    def __init__(self, target_cv: float = 0.5, learning_rate: float = 0.1):
        self.target_cv = target_cv
        self.learning_rate = learning_rate

    def balance(
        self, surprises: np.ndarray, current_alpha: float, current_beta: float,
    ) -> tuple[float, float]:
        mean, std = surprises.mean(), surprises.std()
        cv = std / mean if mean > 1e-10 else 0.0
        error = cv - self.target_cv
        return (max(0.01, current_alpha - self.learning_rate * error),
                max(0.01, current_beta + self.learning_rate * error))

    def balance_exponential(
        self, surprises: np.ndarray, current_alpha: float, current_beta: float,
    ) -> tuple[float, float]:
        mean, std = surprises.mean(), surprises.std()
        cv = std / mean if mean > 1e-10 else 0.0
        error = np.clip(cv - self.target_cv, -5.0, 5.0)
        new_alpha = current_alpha * np.exp(-self.learning_rate * error)
        new_beta = current_beta * np.exp(self.learning_rate * error)
        return (float(np.clip(new_alpha, 0.01, 100.0)),
                float(np.clip(new_beta, 0.01, 100.0)))
