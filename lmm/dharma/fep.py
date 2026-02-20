"""FEP KCL ODE solver — Free Energy Principle as Kirchhoff's Current Law.

Maps the Free Energy Principle to analog circuit simulation:
  - V_mu[i]: membrane potential (belief state)
  - g(V) = tanh(V): generative model nonlinear prediction
  - g'(V) = 1 - tanh^2(V): Jacobian (precision gate)

KCL ODE:
  C * dV/dt = -V/R_leak + g'(V) * G_prec * error

Three backends auto-selected:
  1. GPU (CuPy/CUDA)
  2. CPU BLAS (NumPy/SciPy)
  3. Shim fallback (Python loops)
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from lmm._compat import HAS_FEP_FAST_PATH, HAS_CUPY, sparse_matvec as _default_sparse_matvec
from lmm.rust_bridge import run_fep_analog_loop, has_rust_core


def solve_fep_kcl(
    h: np.ndarray,
    J: sparse.csr_matrix,
    k: int,
    n: int,
    *,
    sila_gamma: float = 0.0,
    G_prec: float = 5.0,
    tau_leak: float = 1.0,
    dt: float = 0.08,
    max_steps: int = 300,
    nirvana_threshold: float = 1e-4,
    initial_state: np.ndarray | None = None,
    sparse_matvec=None,
) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
    """FEP KCL analog circuit solver with auto backend selection.

    Parameters
    ----------
    h : (n,) linear energy terms
    J : (n, n) quadratic interaction matrix (sparse)
    k : target selection count
    n : number of variables
    sila_gamma : implicit cardinality weight
    G_prec : precision conductance
    tau_leak : leak time constant
    dt : time step
    max_steps : maximum simulation steps
    nirvana_threshold : dissipation power stop criterion
    initial_state : (n,) initial voltage (None -> zeros)
    sparse_matvec : sparse matrix-vector product function (shim)

    Returns
    -------
    V_mu : (n,) final membrane potentials
    x : (n,) QUBO variables [0, 1]
    steps_used : number of steps used
    power_history : dissipation power per step
    """
    if HAS_CUPY and sparse_matvec is None:
        return _solve_gpu(
            h, J, k, n,
            sila_gamma=sila_gamma, G_prec=G_prec, tau_leak=tau_leak,
            dt=dt, max_steps=max_steps, nirvana_threshold=nirvana_threshold,
            initial_state=initial_state,
        )

    if HAS_FEP_FAST_PATH and sparse_matvec is None:
        return _solve_vectorized(
            h, J, k, n,
            sila_gamma=sila_gamma, G_prec=G_prec, tau_leak=tau_leak,
            dt=dt, max_steps=max_steps, nirvana_threshold=nirvana_threshold,
            initial_state=initial_state,
        )

    return _solve_shim(
        h, J, k, n,
        sila_gamma=sila_gamma, G_prec=G_prec, tau_leak=tau_leak,
        dt=dt, max_steps=max_steps, nirvana_threshold=nirvana_threshold,
        initial_state=initial_state, sparse_matvec=sparse_matvec,
    )


# =========================================================================
# Backend 1: Shim fallback (Python loops)
# =========================================================================

def _solve_shim(
    h, J, k, n, *, sila_gamma, G_prec, tau_leak, dt,
    max_steps, nirvana_threshold, initial_state, sparse_matvec,
):
    """Shim-compatible FEP ODE with adaptive timestep and momentum."""
    V_mu = initial_state.copy() if initial_state is not None else np.zeros(n)

    if sparse_matvec is None:
        sparse_matvec = _default_sparse_matvec

    has_J = J.nnz > 0
    power_history: list[float] = []
    steps_done = 0

    inv_tau = 1.0 / tau_leak
    dt_min = dt * 0.1
    dt_max = dt * 10.0
    prev_P = float("inf")

    V_prev = V_mu.copy()
    momentum = 0.05

    def _tanh_vec(v):
        return np.array([float(math.tanh(float(x))) for x in v])

    last_x = None

    for step in range(max_steps):
        g_V = _tanh_vec(V_mu)
        x = (g_V + 1.0) * 0.5
        jacobian = 1.0 - g_V * g_V

        grad = h.copy()
        if has_J:
            grad = grad + 2.0 * sparse_matvec(J, x)
        if sila_gamma > 0:
            grad = grad + 2.0 * sila_gamma * (float(np.sum(x)) - x)

        error = -grad
        dV_dt = -V_mu * inv_tau + jacobian * (G_prec * error)

        V_new = V_mu + dV_dt * dt + momentum * (V_mu - V_prev)
        V_prev = V_mu
        V_mu = V_new
        last_x = x

        P_err = float(G_prec * np.dot(error, error))
        power_history.append(P_err)
        steps_done = step + 1

        if P_err < nirvana_threshold:
            break

        if float(np.dot(dV_dt, dV_dt)) < nirvana_threshold:
            break

        if P_err < prev_P * 0.95:
            dt = min(dt * 1.2, dt_max)
            momentum = min(momentum + 0.05, 0.6)
        elif P_err > prev_P:
            dt = max(dt * 0.7, dt_min)
            momentum = max(momentum * 0.5, 0.0)
        prev_P = P_err

    x_final = last_x if last_x is not None else (_tanh_vec(V_mu) + 1.0) * 0.5
    return V_mu, x_final, steps_done, power_history


# =========================================================================
# Backend 2: CPU BLAS vectorized (NumPy/SciPy)
# =========================================================================

def _solve_vectorized(
    h, J, k, n, *, sila_gamma, G_prec, tau_leak, dt,
    max_steps, nirvana_threshold, initial_state,
):
    """CPU BLAS fast path: np.tanh + scipy.sparse @."""
    V_mu = initial_state.copy() if initial_state is not None else np.zeros(n)
    has_J = J.nnz > 0
    inv_tau = 1.0 / tau_leak
    dt_min = dt * 0.1
    dt_max = dt * 10.0
    prev_P = float("inf")
    power_history: list[float] = []
    steps_done = 0

    V_prev = V_mu.copy()
    momentum = 0.05
    last_x = None

    for step in range(max_steps):
        g_V = np.tanh(V_mu)
        x = (g_V + 1.0) * 0.5
        jacobian = 1.0 - g_V * g_V

        grad = h + (2.0 * (J @ x) if has_J else 0.0)
        if sila_gamma > 0:
            grad = grad + 2.0 * sila_gamma * (float(x.sum()) - x)

        error = -grad
        dV_dt = -V_mu * inv_tau + jacobian * (G_prec * error)

        V_new = V_mu + dV_dt * dt + momentum * (V_mu - V_prev)
        V_prev = V_mu
        V_mu = V_new
        last_x = x

        P_err = float(G_prec * np.dot(error, error))
        power_history.append(P_err)
        steps_done = step + 1

        if P_err < nirvana_threshold:
            break

        if float(np.dot(dV_dt, dV_dt)) < nirvana_threshold:
            break

        if P_err < prev_P * 0.95:
            dt = min(dt * 1.2, dt_max)
            momentum = min(momentum + 0.05, 0.6)
        elif P_err > prev_P:
            dt = max(dt * 0.7, dt_min)
            momentum = max(momentum * 0.5, 0.0)
        prev_P = P_err

    x_final = last_x if last_x is not None else (np.tanh(V_mu) + 1.0) * 0.5
    return V_mu, x_final, steps_done, power_history


# =========================================================================
# Backend 3: GPU (CuPy/CUDA)
# =========================================================================

def _solve_gpu(
    h, J, k, n, *, sila_gamma, G_prec, tau_leak, dt,
    max_steps, nirvana_threshold, initial_state,
):
    """GPU fast path: CuPy full CUDA execution."""
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    h_gpu = cp.asarray(h)
    J_gpu = cp_sparse.csr_matrix(J)
    V_mu = cp.asarray(initial_state) if initial_state is not None else cp.zeros(n)

    has_J = J.nnz > 0
    inv_tau = 1.0 / tau_leak
    power_history: list[float] = []
    steps_done = 0

    dt_min = dt * 0.1
    dt_max = dt * 10.0
    prev_P = float("inf")

    V_prev = V_mu.copy()
    momentum = 0.0

    for step in range(max_steps):
        g_V = cp.tanh(V_mu)
        x = (g_V + 1.0) * 0.5
        jacobian = 1.0 - g_V * g_V

        grad = h_gpu + (2.0 * (J_gpu @ x) if has_J else 0.0)
        if sila_gamma > 0:
            grad = grad + 2.0 * sila_gamma * (x.sum() - x)

        error = -grad
        dV_dt = -V_mu * inv_tau + jacobian * (G_prec * error)

        V_new = V_mu + dV_dt * dt + momentum * (V_mu - V_prev)
        V_prev = V_mu
        V_mu = V_new

        P_err = float(G_prec * cp.dot(error, error))
        power_history.append(P_err)
        steps_done = step + 1

        if P_err < nirvana_threshold:
            break

        if float(cp.dot(dV_dt, dV_dt)) < nirvana_threshold:
            break

        if P_err < prev_P * 0.95:
            dt = min(dt * 1.2, dt_max)
            momentum = min(momentum + 0.05, 0.6)
        elif P_err > prev_P:
            dt = max(dt * 0.7, dt_min)
            momentum = max(momentum * 0.5, 0.0)
        prev_P = P_err

    V_mu_host = cp.asnumpy(V_mu)
    x_final = (np.tanh(V_mu_host) + 1.0) * 0.5
    return V_mu_host, x_final, steps_done, power_history


# =========================================================================
# Analog FEP solver — direct physics simulation
# =========================================================================

def solve_fep_kcl_analog(
    V_s: np.ndarray,
    J_dynamic: sparse.csr_matrix,
    n: int,
    *,
    G_prec_base: float = 10.0,
    tau_leak: float = 1.0,
    dt: float = 0.01,
    max_steps: int = 1000,
    nirvana_threshold: float = 1e-5,
    initial_state: np.ndarray | None = None,
    sparse_matvec=None,
) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
    """Analog FEP KCL solver — direct physics simulation.

    Bypasses global QUBO energy synthesis. Simulates local KCL ODE
    directly from sensory input V_s and dynamic interdependence J_dynamic.

    Physics:
      g(V) = tanh(V)                         generative model
      g'(V) = 1 - tanh^2(V)                  Jacobian (precision gate)
      error = V_s + J_scale * J * g(V) - g(V)  prediction error
      dV/dt = -V/tau + g'(V) * G * error      KCL ODE
      P_err = G * ||error||^2                  dissipation power

    Parameters
    ----------
    V_s : (n,) sensory input (query relevance scores)
    J_dynamic : (n, n) dynamic interdependence network
    n : number of nodes
    G_prec_base : precision conductance
    tau_leak : leak time constant
    dt : time step
    max_steps : maximum simulation steps
    nirvana_threshold : stop criterion
    initial_state : (n,) initial voltage (None -> zeros)
    sparse_matvec : sparse matrix-vector product function (shim)

    Returns
    -------
    V_mu : (n,) final membrane potentials
    x_final : (n,) activation [0, 1]
    steps_used : steps used
    power_history : dissipation power per step
    """
    # Auto impedance matching: normalize J coupling strength
    if J_dynamic.nnz > 0:
        row_sums = np.asarray(J_dynamic.sum(axis=1)).flatten()
        max_row_sum = max(abs(float(x)) for x in row_sums)
        J_scale = 1.0 / (1.0 + max_row_sum) if max_row_sum > 0 else 1.0
    else:
        J_scale = 1.0

    # Rust path: fastest backend when lmm_rust_core is available
    if has_rust_core() and sparse_matvec is None:
        v_init = initial_state.copy() if initial_state is not None else np.zeros(n)

        def _python_fallback():
            r = _solve_analog_vectorized(
                V_s, J_dynamic, n,
                G_prec_base=G_prec_base, tau_leak=tau_leak, dt=dt,
                max_steps=max_steps, nirvana_threshold=nirvana_threshold,
                initial_state=initial_state, J_scale=J_scale,
            )
            # run_fep_analog_loop expects (V_final, steps, power_history)
            return r[0], r[2], r[3]

        V_out, steps, ph = run_fep_analog_loop(
            v_init, V_s, n, G_prec_base, tau_leak, dt, max_steps,
            nirvana_threshold, J_scale, J_dynamic,
            _fallback=_python_fallback,
        )
        x_final = (np.tanh(V_out) + 1.0) * 0.5
        return V_out, x_final, steps, ph

    if HAS_CUPY and sparse_matvec is None:
        return _solve_analog_gpu(
            V_s, J_dynamic, n,
            G_prec_base=G_prec_base, tau_leak=tau_leak, dt=dt,
            max_steps=max_steps, nirvana_threshold=nirvana_threshold,
            initial_state=initial_state, J_scale=J_scale,
        )

    if HAS_FEP_FAST_PATH and sparse_matvec is None:
        return _solve_analog_vectorized(
            V_s, J_dynamic, n,
            G_prec_base=G_prec_base, tau_leak=tau_leak, dt=dt,
            max_steps=max_steps, nirvana_threshold=nirvana_threshold,
            initial_state=initial_state, J_scale=J_scale,
        )

    return _solve_analog_shim(
        V_s, J_dynamic, n,
        G_prec_base=G_prec_base, tau_leak=tau_leak, dt=dt,
        max_steps=max_steps, nirvana_threshold=nirvana_threshold,
        initial_state=initial_state, sparse_matvec=sparse_matvec,
        J_scale=J_scale,
    )


# =========================================================================
# Analog Backend 1: Shim fallback
# =========================================================================

def _solve_analog_shim(
    V_s, J_dynamic, n, *, G_prec_base, tau_leak, dt,
    max_steps, nirvana_threshold, initial_state, sparse_matvec,
    J_scale,
):
    """Shim-compatible analog FEP ODE with adaptive timestep."""
    V = initial_state.copy() if initial_state is not None else np.zeros(n)

    if sparse_matvec is None:
        sparse_matvec = _default_sparse_matvec

    has_J = J_dynamic.nnz > 0
    inv_tau = 1.0 / tau_leak
    dt_min = dt * 0.1
    dt_max = dt * 10.0
    prev_P = float("inf")
    power_history: list[float] = []
    steps_done = 0

    V_prev = V.copy()
    momentum = 0.0

    def _tanh_vec(v):
        return np.array([float(math.tanh(float(x))) for x in v])

    last_x = None

    for step in range(max_steps):
        g_V = _tanh_vec(V)
        jacobian = 1.0 - g_V * g_V

        if has_J:
            error = V_s + J_scale * sparse_matvec(J_dynamic, g_V) - g_V
        else:
            error = V_s - g_V

        dV_dt = -V * inv_tau + jacobian * (G_prec_base * error)

        V_new = V + dV_dt * dt + momentum * (V - V_prev)
        V_prev = V
        V = V_new
        last_x = (g_V + 1.0) * 0.5

        P_err = float(G_prec_base * np.dot(error, error))
        power_history.append(P_err)
        steps_done = step + 1

        if P_err < nirvana_threshold:
            break

        if float(np.dot(dV_dt, dV_dt)) < nirvana_threshold:
            break

        if P_err < prev_P * 0.95:
            dt = min(dt * 1.2, dt_max)
            momentum = min(momentum + 0.05, 0.6)
        elif P_err > prev_P:
            dt = max(dt * 0.7, dt_min)
            momentum = max(momentum * 0.5, 0.0)
        prev_P = P_err

    x_final = last_x if last_x is not None else (_tanh_vec(V) + 1.0) * 0.5
    return V, x_final, steps_done, power_history


# =========================================================================
# Analog Backend 2: CPU BLAS vectorized
# =========================================================================

def _solve_analog_vectorized(
    V_s, J_dynamic, n, *, G_prec_base, tau_leak, dt,
    max_steps, nirvana_threshold, initial_state, J_scale,
):
    """CPU BLAS fast path for analog FEP ODE."""
    V = initial_state.copy() if initial_state is not None else np.zeros(n)
    has_J = J_dynamic.nnz > 0
    inv_tau = 1.0 / tau_leak
    dt_min = dt * 0.1
    dt_max = dt * 10.0
    prev_P = float("inf")
    power_history: list[float] = []
    steps_done = 0

    V_prev = V.copy()
    momentum = 0.0
    last_x = None

    for step in range(max_steps):
        g_V = np.tanh(V)
        jacobian = 1.0 - g_V * g_V

        error = V_s + (J_scale * (J_dynamic @ g_V) if has_J else 0.0) - g_V

        dV_dt = -V * inv_tau + jacobian * (G_prec_base * error)

        V_new = V + dV_dt * dt + momentum * (V - V_prev)
        V_prev = V
        V = V_new
        last_x = (g_V + 1.0) * 0.5

        P_err = float(G_prec_base * np.dot(error, error))
        power_history.append(P_err)
        steps_done = step + 1

        if P_err < nirvana_threshold:
            break

        if float(np.dot(dV_dt, dV_dt)) < nirvana_threshold:
            break

        if P_err < prev_P * 0.95:
            dt = min(dt * 1.2, dt_max)
            momentum = min(momentum + 0.05, 0.6)
        elif P_err > prev_P:
            dt = max(dt * 0.7, dt_min)
            momentum = max(momentum * 0.5, 0.0)
        prev_P = P_err

    x_final = last_x if last_x is not None else (np.tanh(V) + 1.0) * 0.5
    return V, x_final, steps_done, power_history


# =========================================================================
# Analog Backend 3: GPU (CuPy/CUDA)
# =========================================================================

def _solve_analog_gpu(
    V_s, J_dynamic, n, *, G_prec_base, tau_leak, dt,
    max_steps, nirvana_threshold, initial_state, J_scale,
):
    """GPU fast path for analog FEP ODE."""
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    V_s_gpu = cp.asarray(V_s)
    J_gpu = cp_sparse.csr_matrix(J_dynamic)
    V = cp.asarray(initial_state) if initial_state is not None else cp.zeros(n)

    has_J = J_dynamic.nnz > 0
    inv_tau = 1.0 / tau_leak
    dt_min = dt * 0.1
    dt_max = dt * 10.0
    prev_P = float("inf")
    power_history: list[float] = []
    steps_done = 0

    V_prev = V.copy()
    momentum = 0.0

    for step in range(max_steps):
        g_V = cp.tanh(V)
        jacobian = 1.0 - g_V * g_V

        error = V_s_gpu + (J_scale * (J_gpu @ g_V) if has_J else 0.0) - g_V
        dV_dt = -V * inv_tau + jacobian * (G_prec_base * error)

        V_new = V + dV_dt * dt + momentum * (V - V_prev)
        V_prev = V
        V = V_new

        P_err = float(G_prec_base * cp.dot(error, error))
        power_history.append(P_err)
        steps_done = step + 1

        if P_err < nirvana_threshold:
            break

        if float(cp.dot(dV_dt, dV_dt)) < nirvana_threshold:
            break

        if P_err < prev_P * 0.95:
            dt = min(dt * 1.2, dt_max)
            momentum = min(momentum + 0.05, 0.6)
        elif P_err > prev_P:
            dt = max(dt * 0.7, dt_min)
            momentum = max(momentum * 0.5, 0.0)
        prev_P = P_err

    V_host = cp.asnumpy(V)
    x_final = (np.tanh(V_host) + 1.0) * 0.5
    return V_host, x_final, steps_done, power_history
