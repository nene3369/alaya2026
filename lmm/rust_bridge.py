"""lmm.rust_bridge — Rust-accelerated inner-loop bridge.

Drop-in accelerator wrappers for the two hottest loops in LMM:

  * :func:`run_sa_ising_loop`   — Ising SA main loop
  * :func:`run_fep_analog_loop` — FEP KCL ODE numerical integration

Both functions follow the same contract:

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
        np.ascontiguousarray(csr.data,   dtype=np.float64),
        np.ascontiguousarray(csr.indices, dtype=np.int32),
        np.ascontiguousarray(csr.indptr,  dtype=np.int32),
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
        csr_d, csr_i, csr_p = _csr_parts(csr)
        return _rc.solve_sa_ising_rust(
            int(n),
            int(n_iterations),
            float(temp_start),
            float(temp_end),
            float(gamma),
            np.ascontiguousarray(diag,        dtype=np.float64),
            np.ascontiguousarray(s,           dtype=np.float64),
            np.ascontiguousarray(local_field, dtype=np.float64),
            float(energy),
            csr_d, csr_i, csr_p,
        )

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
        csr_d, csr_i, csr_p = _csr_parts(csr)
        v_out, steps_done, power_history = _rc.solve_fep_analog_rust(
            np.ascontiguousarray(v_init, dtype=np.float64),
            np.ascontiguousarray(v_s,    dtype=np.float64),
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

    if _fallback is not None:
        return _fallback()

    raise ImportError(
        "lmm_rust_core is not installed and no _fallback was provided.\n"
        "Build it with:  cd lmm_rust_core && maturin develop --release"
    )
