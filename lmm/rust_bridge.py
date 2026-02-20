"""lmm.rust_bridge — Rust-accelerated inner-loop bridge.

Drop-in accelerator wrappers for the hottest loops in LMM:

  * :func:`run_sa_ising_loop`       — Ising SA main loop
  * :func:`run_fep_analog_loop`     — FEP KCL ODE numerical integration
  * :func:`run_submodular_greedy`   — Submodular lazy greedy (CELF)
  * :func:`run_ngram_vectors`       — N-gram CRC32 hash vectorization
  * :func:`run_bfs_components`      — BFS connected components
  * :func:`run_bfs_reverse`         — BFS reverse traversal

All functions follow the same contract:

  * If ``lmm_rust_core`` is installed they delegate to the compiled Rust
    extension (zero-copy CSR arrays, SmallRng, LLVM-vectorised loops).
  * Otherwise they invoke the ``_fallback`` callable you supply, which
    should be a no-argument lambda wrapping the pure-Python implementation.

Install the Rust extension (requires maturin ≥ 1.4)::

    cd lmm_rust_core
    maturin develop --release        # editable install into current venv
    # or: pip install .              # wheel install

Usage in lmm/solvers.py (Ising SA)::

    from lmm.rust_bridge import run_sa_ising_loop, has_rust_core

    best_s = run_sa_ising_loop(
        n, n_iterations, temp_start, temp_end, gamma,
        diag, s, local_field, energy, csr,
        _fallback=lambda: _python_sa_loop(...)
    )

Usage in lmm/dharma/fep.py (FEP analog)::

    from lmm.rust_bridge import run_fep_analog_loop

    V, steps, ph = run_fep_analog_loop(
        v_init, V_s, n, G_prec_base, tau_leak, dt, max_steps,
        nirvana_threshold, j_scale, J_dynamic,
        _fallback=lambda: _solve_analog_shim(...)
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from scipy.sparse import csr_matrix

# ── Try to import the compiled Rust extension ─────────────────────────────
try:
    import lmm_rust_core as _rc  # type: ignore[import]
    _HAS_RUST = True
except ImportError:
    _rc = None  # type: ignore[assignment]
    _HAS_RUST = False


def has_rust_core() -> bool:
    """Return ``True`` if the Rust extension ``lmm_rust_core`` is available."""
    return _HAS_RUST


# ── Internal helpers ──────────────────────────────────────────────────────

def _csr_parts(
    csr: "csr_matrix",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (data, indices, indptr) as contiguous float64 / int32 arrays.

    Returns empty arrays (no-op CSR) when the matrix has no stored values,
    so Rust can detect ``has_j = csr_d.is_empty()`` cleanly.
    """
    n = csr.shape[0]
    if csr.nnz == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.zeros(n + 1, dtype=np.int32),
        )
    return (
        np.array(csr.data,    dtype=np.float64),
        np.array(csr.indices, dtype=np.int32),
        np.array(csr.indptr,  dtype=np.int32),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Ising SA wrapper
# ═══════════════════════════════════════════════════════════════════════════

def run_sa_ising_loop(
    n: int,
    n_iterations: int,
    temp_start: float,
    temp_end: float,
    gamma: float,
    diag: np.ndarray,
    s: np.ndarray,
    local_field: np.ndarray,
    energy: float,
    csr: "csr_matrix",
    *,
    _fallback: Callable[[], np.ndarray] | None = None,
) -> np.ndarray:
    """Run the Ising SA inner loop, preferring the Rust implementation.

    All pre-computation (initial spins, local field, energy) must be done
    on the Python side before calling this function; it only runs the MC
    loop and returns ``best_s`` (Ising spins ±1, shape ``(n,)``).

    Parameters
    ----------
    n            : problem size
    n_iterations : total Monte-Carlo steps
    temp_start   : initial temperature
    temp_end     : final temperature
    gamma        : cardinality-constraint penalty weight
    diag         : (n,) QUBO diagonal
    s            : (n,) initial Ising spins (±1)
    local_field  : (n,) precomputed local field  h + 2·J·s
    energy       : scalar initial energy
    csr          : off-diagonal QUBO as ``scipy.sparse.csr_matrix``
    _fallback    : zero-argument callable that runs the pure-Python loop.
                   Called when Rust is not available.

    Returns
    -------
    best_s : (n,) Ising-spin configuration at minimum energy seen.
    """
    if _HAS_RUST:
        try:
            csr_d, csr_i, csr_p = _csr_parts(csr)
            return _rc.solve_sa_ising_rust(
                int(n),
                int(n_iterations),
                float(temp_start),
                float(temp_end),
                float(gamma),
                np.asarray(diag,        dtype=np.float64),
                np.asarray(s,           dtype=np.float64),
                np.asarray(local_field, dtype=np.float64),
                float(energy),
                csr_d, csr_i, csr_p,
            )
        except Exception:
            pass  # Fall through to Python fallback

    if _fallback is not None:
        return _fallback()

    raise ImportError(
        "lmm_rust_core is not installed and no _fallback was provided.\n"
        "Build it with:  cd lmm_rust_core && maturin develop --release"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2.  FEP KCL Analog ODE wrapper
# ═══════════════════════════════════════════════════════════════════════════

def run_fep_analog_loop(
    v_init: np.ndarray,
    v_s: np.ndarray,
    n: int,
    g_prec_base: float,
    tau_leak: float,
    dt: float,
    max_steps: int,
    nirvana_threshold: float,
    j_scale: float,
    csr: "csr_matrix",
    *,
    _fallback: Callable[[], tuple[np.ndarray, int, list[float]]] | None = None,
) -> tuple[np.ndarray, int, list[float]]:
    """Run the FEP KCL Analog ODE inner loop, preferring Rust.

    Simulates the analog circuit KCL ODE with adaptive timestep and Polyak
    momentum until dissipation power falls below ``nirvana_threshold``.

    Parameters
    ----------
    v_init            : (n,) initial membrane potentials
    v_s               : (n,) sensory input (query relevance scores)
    n                 : circuit size
    g_prec_base       : precision conductance G
    tau_leak          : leak time constant τ
    dt                : initial time step
    max_steps         : maximum simulation steps
    nirvana_threshold : stop when P_err or kinetic energy < this
    j_scale           : impedance-matching scale  1/(1+max_row_sum)
    csr               : J_dynamic as ``scipy.sparse.csr_matrix``
    _fallback         : zero-argument callable that runs the pure-Python loop.

    Returns
    -------
    V_final       : (n,) final membrane potentials
    steps_done    : actual number of steps executed
    power_history : dissipation power recorded at each step
    """
    if _HAS_RUST:
        try:
            csr_d, csr_i, csr_p = _csr_parts(csr)
            v_out, steps_done, power_history = _rc.solve_fep_analog_rust(
                np.asarray(v_init, dtype=np.float64),
                np.asarray(v_s,    dtype=np.float64),
                int(n),
                float(g_prec_base),
                float(tau_leak),
                float(dt),
                int(max_steps),
                float(nirvana_threshold),
                float(j_scale),
                csr_d, csr_i, csr_p,
            )
            return v_out, steps_done, power_history
        except Exception:
            pass  # Fall through to Python fallback

    if _fallback is not None:
        return _fallback()

    raise ImportError(
        "lmm_rust_core is not installed and no _fallback was provided.\n"
        "Build it with:  cd lmm_rust_core && maturin develop --release"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Submodular Lazy Greedy wrapper
# ═══════════════════════════════════════════════════════════════════════════

def run_submodular_greedy(
    surprises: np.ndarray,
    csr: "csr_matrix",
    k: int,
    alpha: float,
    beta: float,
    *,
    _fallback: Callable | None = None,
):
    """Run submodular lazy greedy (CELF), preferring Rust."""
    if _HAS_RUST:
        try:
            csr_obj = csr.tocsr() if hasattr(csr, "tocsr") else csr
            csr_d, csr_i, csr_p = _csr_parts(csr_obj)
            sel, obj, gains = _rc.solve_submodular_greedy_rust(
                np.asarray(surprises, dtype=np.float64),
                csr_d, csr_i, csr_p,
                int(k), float(alpha), float(beta),
            )
            return sel.astype(int), obj, gains
        except Exception:
            pass
    if _fallback:
        return _fallback()
    raise ImportError("No rust core and no fallback.")


# ═══════════════════════════════════════════════════════════════════════════
#  4.  N-gram Hash Vectorization wrapper
# ═══════════════════════════════════════════════════════════════════════════

def run_ngram_vectors(
    texts: list[str],
    max_features: int,
    *,
    _fallback: Callable | None = None,
):
    """Run n-gram CRC32 hash vectorization, preferring Rust."""
    if _HAS_RUST:
        try:
            return _rc.ngram_vectors_rust(texts, max_features)
        except Exception:
            pass
    return _fallback() if _fallback else None


# ═══════════════════════════════════════════════════════════════════════════
#  5.  BFS Connected Components wrapper
# ═══════════════════════════════════════════════════════════════════════════

def run_bfs_components(
    n: int,
    csr: "csr_matrix",
    *,
    _fallback: Callable | None = None,
):
    """Run BFS connected-component detection, preferring Rust."""
    if _HAS_RUST:
        try:
            csr_obj = csr.tocsr() if hasattr(csr, "tocsr") else csr
            _, csr_i, csr_p = _csr_parts(csr_obj)
            return _rc.bfs_components_rust(n, csr_i, csr_p)
        except Exception:
            pass
    return _fallback() if _fallback else None


# ═══════════════════════════════════════════════════════════════════════════
#  6.  BFS Reverse Traversal wrapper
# ═══════════════════════════════════════════════════════════════════════════

def run_bfs_reverse(
    n: int,
    start_indices: list[int],
    csr_T: "csr_matrix",
    *,
    _fallback: Callable | None = None,
):
    """Run BFS reverse traversal on transposed graph, preferring Rust."""
    if _HAS_RUST:
        try:
            # SciPy .T returns csc_matrix; must convert to CSR for correct traversal
            csr_obj = csr_T.tocsr() if hasattr(csr_T, "tocsr") else csr_T
            _, csr_i, csr_p = _csr_parts(csr_obj)
            return _rc.bfs_reverse_rust(n, start_indices, csr_i, csr_p)
        except Exception:
            pass
    return _fallback() if _fallback else None
