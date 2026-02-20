//! # lmm_rust_core
//!
//! High-performance Rust extension for LMM's two hottest inner loops:
//!
//! 1. **`solve_sa_ising_rust`** — Ising-form Simulated Annealing (SA)
//!    - Geometric annealing schedule
//!    - O(1) energy-delta + Metropolis acceptance
//!    - Dense local-field update (LLVM auto-vectorised)
//!    - Sparse row scatter via CSR (zero-copy from Python)
//!    - SmallRng (xoshiro128++) initialised once before the loop
//!
//! 2. **`solve_fep_analog_rust`** — Free-Energy-Principle KCL ODE
//!    - tanh activation + precision Jacobian
//!    - Inline CSR SpMV (J @ g_V)
//!    - Euler step with Polyak momentum
//!    - Adaptive timestep (dt grows/shrinks by dissipation power ratio)
//!    - Two stopping criteria: power threshold and kinetic-energy threshold
//!
//! Build with:
//!   cd lmm_rust_core && maturin develop --release
//!
//! All NumPy arrays are received as `PyReadonlyArray1<f64>` (zero-copy);
//! results are returned as freshly allocated `PyArray1<f64>`.

#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ═══════════════════════════════════════════════════════════════════════════
//  1.  Ising Simulated Annealing
// ═══════════════════════════════════════════════════════════════════════════

/// Ising-form SA main loop — zero-copy CSR, SmallRng hot path.
///
/// # Arguments (match Python call-site in `lmm/rust_bridge.py`)
/// * `n`               – number of Ising spins
/// * `n_iterations`    – total Monte-Carlo steps
/// * `temp_start`      – initial temperature
/// * `temp_end`        – final temperature (geometric schedule)
/// * `gamma`           – cardinality-constraint penalty weight
/// * `diag`            – (n,)   QUBO diagonal  [zero-copy]
/// * `s_in`            – (n,)   initial Ising spins ±1
/// * `local_field_in`  – (n,)   precomputed local field  h + 2·J·s
/// * `energy_init`     – scalar initial energy
/// * `csr_data`        – (nnz,) off-diagonal QUBO values  [zero-copy]
/// * `csr_indices`     – (nnz,) CSR column indices  [zero-copy]
/// * `csr_indptr`      – (n+1,) CSR row pointers    [zero-copy]
///
/// # Returns
/// `best_s` – (n,) Ising-spin configuration with the lowest energy seen.
#[pyfunction]
#[pyo3(signature = (
    n, n_iterations, temp_start, temp_end, gamma,
    diag, s_in, local_field_in, energy_init,
    csr_data, csr_indices, csr_indptr
))]
fn solve_sa_ising_rust<'py>(
    py: Python<'py>,
    n: usize,
    n_iterations: usize,
    temp_start: f64,
    temp_end: f64,
    gamma: f64,
    diag: PyReadonlyArray1<'py, f64>,
    s_in: PyReadonlyArray1<'py, f64>,
    local_field_in: PyReadonlyArray1<'py, f64>,
    energy_init: f64,
    csr_data: PyReadonlyArray1<'py, f64>,
    csr_indices: PyReadonlyArray1<'py, i32>,
    csr_indptr: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // ── Zero-copy read-only slices ─────────────────────────────────────────
    let diag  = diag.as_slice()?;
    let csr_d = csr_data.as_slice()?;
    let csr_i = csr_indices.as_slice()?;
    let csr_p = csr_indptr.as_slice()?;

    // ── Mutable working copies (two O(n) allocations only) ────────────────
    let mut s:    Vec<f64> = s_in.as_slice()?.to_vec();
    let mut lf:   Vec<f64> = local_field_in.as_slice()?.to_vec();
    let mut best_s: Vec<f64> = s.clone();

    // ── RNG: xoshiro128++ — seeded from OS entropy, initialised once ───────
    let mut rng = SmallRng::from_entropy();

    let mut energy      = energy_init;
    let mut best_energy = energy;

    let temp_ratio  = temp_end / temp_start;
    let n_iter_f    = n_iterations as f64;

    for step in 0..n_iterations {
        // Geometric annealing schedule: T(t) = T0 · r^(t/N)
        let temp = temp_start * temp_ratio.powf(step as f64 / n_iter_f);

        let idx = rng.gen_range(0..n);

        // SAFETY: idx < n; all slices have been validated to length n (or n+1).
        let (s_old, lf_val) = unsafe {
            (*s.get_unchecked(idx), *lf.get_unchecked(idx))
        };

        // O(1) energy delta from single-spin flip
        let delta_e = -2.0 * s_old * lf_val;

        // Metropolis acceptance criterion
        let accept = delta_e < 0.0 || {
            let t = if temp > 1e-15 { temp } else { 1e-15 };
            rng.gen::<f64>() < (-delta_e / t).exp()
        };

        if accept {
            let s_new = -s_old; // flip
            unsafe { *s.get_unchecked_mut(idx) = s_new };
            energy += delta_e;

            // ── Dense local-field update: lf[j] += γ·s_new  (∀j) ──────────
            // Iterator-based loop: bounds-check free, LLVM vectorises to AVX2.
            let gamma_s = gamma * s_new;
            for v in lf.iter_mut() {
                *v += gamma_s;
            }

            // Self-correction at the flipped node (cancels the γ term, adds diag)
            // Δlf[idx] = (diag[idx] - γ)·s_new  (net: diag[idx]·s_new)
            unsafe {
                *lf.get_unchecked_mut(idx) +=
                    (*diag.get_unchecked(idx) - gamma) * s_new;
            }

            // ── Sparse row scatter: lf[c] += J[idx, c]·s_new ──────────────
            // Reads the CSR row for `idx` (symmetric off-diagonal part).
            let row_start = unsafe { *csr_p.get_unchecked(idx) }     as usize;
            let row_end   = unsafe { *csr_p.get_unchecked(idx + 1) } as usize;
            for ptr in row_start..row_end {
                unsafe {
                    let c   = *csr_i.get_unchecked(ptr) as usize;
                    let val = *csr_d.get_unchecked(ptr);
                    *lf.get_unchecked_mut(c) += val * s_new;
                }
            }

            // Track global minimum
            if energy < best_energy {
                best_energy = energy;
                best_s.copy_from_slice(&s);
            }
        }
    }

    // Return best_s as a new (owned) NumPy array — zero extra copy.
    Ok(best_s.into_pyarray(py))
}

// ═══════════════════════════════════════════════════════════════════════════
//  2.  FEP KCL Analog ODE Solver
// ═══════════════════════════════════════════════════════════════════════════

/// KCL ODE solver for the analog Free-Energy-Principle circuit.
///
/// Physics model:
/// ```text
///   g(V)    = tanh(V)                             generative model
///   J(V)    = 1 - g(V)²                           precision Jacobian
///   J_g_V   = J_dynamic @ g(V)                    SpMV (CSR, inline)
///   error   = V_s + j_scale·J_g_V − g(V)          prediction error
///   dV/dt   = −V/τ + J(V)·G·error                 KCL dynamics
///   V_new   = V + dV/dt·dt + momentum·(V − V_prev) Euler + Polyak
///   P_err   = G·‖error‖²                           dissipation power
/// ```
///
/// # Arguments
/// * `v_init`             – (n,) initial membrane potentials
/// * `v_s`                – (n,) sensory input (zero-copy)
/// * `n`                  – circuit size
/// * `g_prec_base`        – precision conductance G
/// * `tau_leak`           – leak time constant τ
/// * `dt`                 – initial time step
/// * `max_steps`          – upper bound on simulation steps
/// * `nirvana_threshold`  – stop when P_err or kinetic energy < this
/// * `j_scale`            – impedance-matching scale for J coupling
/// * `csr_data/indices/indptr` – J_dynamic in CSR (zero-copy)
///
/// # Returns
/// `(V_final, steps_done, power_history)`
#[pyfunction]
#[pyo3(signature = (
    v_init, v_s, n, g_prec_base, tau_leak, dt, max_steps,
    nirvana_threshold, j_scale,
    csr_data, csr_indices, csr_indptr
))]
fn solve_fep_analog_rust<'py>(
    py: Python<'py>,
    v_init: PyReadonlyArray1<'py, f64>,
    v_s: PyReadonlyArray1<'py, f64>,
    n: usize,
    g_prec_base: f64,
    tau_leak: f64,
    dt: f64,
    max_steps: usize,
    nirvana_threshold: f64,
    j_scale: f64,
    csr_data: PyReadonlyArray1<'py, f64>,
    csr_indices: PyReadonlyArray1<'py, i32>,
    csr_indptr: PyReadonlyArray1<'py, i32>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, Vec<f64>)> {
    // ── Zero-copy read-only slices ─────────────────────────────────────────
    let v_s_sl = v_s.as_slice()?;
    let csr_d  = csr_data.as_slice()?;
    let csr_i  = csr_indices.as_slice()?;
    let csr_p  = csr_indptr.as_slice()?;
    let has_j  = !csr_d.is_empty();

    // ── Mutable state vectors ──────────────────────────────────────────────
    let mut v:      Vec<f64> = v_init.as_slice()?.to_vec();
    let mut v_prev: Vec<f64> = v.clone();
    // v_new is a swap buffer — never returned to Python directly.
    let mut v_new:  Vec<f64> = vec![0.0; n];

    // ── Scratch buffers — single allocation, reused every step ────────────
    let mut g_v:      Vec<f64> = vec![0.0; n];
    let mut jacobian: Vec<f64> = vec![0.0; n];
    let mut j_g_v:    Vec<f64> = vec![0.0; n];
    let mut error:    Vec<f64> = vec![0.0; n];
    let mut dv_dt:    Vec<f64> = vec![0.0; n];

    let inv_tau = 1.0 / tau_leak;

    let mut dt_cur = dt;
    let dt_min     = dt * 0.1;
    let dt_max     = dt * 10.0;

    let mut prev_p        = f64::INFINITY;
    let mut power_history = Vec::<f64>::with_capacity(max_steps.min(2048));
    let mut steps_done    = 0usize;
    let mut momentum      = 0.0f64;

    for step in 0..max_steps {
        // ── 1. Activation g(V) = tanh(V),  Jacobian J = 1 − g² ───────────
        for i in 0..n {
            let gv    = v[i].tanh();
            g_v[i]    = gv;
            jacobian[i] = 1.0 - gv * gv;
        }

        // ── 2. SpMV: J_g_V[i] = Σ_c  J[i,c] · g_V[c]  (inline CSR) ──────
        if has_j {
            for i in 0..n {
                // csr_p is length n+1; csr_p[i+1] is valid for i < n.
                let start = csr_p[i]     as usize;
                let end   = csr_p[i + 1] as usize;
                let mut acc = 0.0f64;
                // SAFETY: ptr in [start, end) ⊂ [0, nnz); c = csr_i[ptr] < n.
                unsafe {
                    for ptr in start..end {
                        let c   = *csr_i.get_unchecked(ptr) as usize;
                        let val = *csr_d.get_unchecked(ptr);
                        acc += val * *g_v.get_unchecked(c);
                    }
                }
                j_g_v[i] = acc;
            }
        } else {
            // No coupling matrix — J_g_V is identically zero.
            j_g_v.iter_mut().for_each(|x| *x = 0.0);
        }

        // ── 3. Error, KCL dynamics, accumulate norms ──────────────────────
        let mut p_err_acc = 0.0f64;
        let mut dv_norm2  = 0.0f64;
        for i in 0..n {
            let e  = v_s_sl[i] + j_scale * j_g_v[i] - g_v[i];
            let dv = -v[i] * inv_tau + jacobian[i] * (g_prec_base * e);
            error[i] = e;
            dv_dt[i] = dv;
            p_err_acc += e  * e;
            dv_norm2  += dv * dv;
        }
        let p_err = g_prec_base * p_err_acc;

        // ── 4. Euler step + Polyak momentum ───────────────────────────────
        for i in 0..n {
            v_new[i] = v[i]
                + dv_dt[i] * dt_cur
                + momentum * (v[i] - v_prev[i]);
        }
        // Swap without re-allocation:
        //   v_prev ← v (old V becomes previous)
        //   v      ← v_new (newly computed V)
        //   v_new  ← garbage (will be overwritten next iteration)
        std::mem::swap(&mut v, &mut v_new);      // v = v_new_computed, v_new = v_old
        std::mem::swap(&mut v_prev, &mut v_new); // v_prev = v_old,     v_new = v_prev_old

        power_history.push(p_err);
        steps_done = step + 1;

        // ── 5. Stopping criteria ──────────────────────────────────────────
        if p_err < nirvana_threshold
            || dv_norm2 * dt_cur * dt_cur < nirvana_threshold
        {
            break;
        }

        // ── 6. Adaptive timestep & momentum (Nesterov-style schedule) ─────
        if p_err < prev_p * 0.95 {
            dt_cur   = (dt_cur * 1.2).min(dt_max);
            momentum = (momentum + 0.05).min(0.6);
        } else if p_err > prev_p {
            dt_cur   = (dt_cur * 0.7).max(dt_min);
            momentum = (momentum * 0.5).max(0.0);
        }
        prev_p = p_err;
    }

    // `v` now holds the final membrane potentials — transfer ownership to Python.
    Ok((v.into_pyarray(py), steps_done, power_history))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Module registration
// ═══════════════════════════════════════════════════════════════════════════

#[pymodule]
fn lmm_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_sa_ising_rust, m)?)?;
    m.add_function(wrap_pyfunction!(solve_fep_analog_rust, m)?)?;
    Ok(())
}
