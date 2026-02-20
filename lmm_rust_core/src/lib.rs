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

use std::collections::{BinaryHeap, VecDeque};
use std::cmp::Ordering;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use crc32fast::Hasher;
use ordered_float::OrderedFloat;

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
    s_in, local_field_in, energy_init,
    csr_data, csr_indices, csr_indptr
))]
fn solve_sa_ising_rust<'py>(
    py: Python<'py>,
    n: usize,
    n_iterations: usize,
    temp_start: f64,
    temp_end: f64,
    gamma: f64,
    s_in: PyReadonlyArray1<'py, f64>,
    local_field_in: PyReadonlyArray1<'py, f64>,
    energy_init: f64,
    csr_data: PyReadonlyArray1<'py, f64>,
    csr_indices: PyReadonlyArray1<'py, i32>,
    csr_indptr: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Early return for empty problem — avoids gen_range(0..0) panic.
    if n == 0 {
        return Ok(PyArray1::zeros_bound(py, [0], false));
    }

    // ── Extract slices and validate dimensions ───────────────────────────────
    let csr_d = csr_data.as_slice()?;
    let csr_i = csr_indices.as_slice()?;
    let csr_p = csr_indptr.as_slice()?;
    let s_sl  = s_in.as_slice()?;
    let lf_sl = local_field_in.as_slice()?;

    if s_sl.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("s_in length mismatch: expected {}, got {}", n, s_sl.len())
        ));
    }
    if lf_sl.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("local_field_in length mismatch: expected {}, got {}", n, lf_sl.len())
        ));
    }
    // Validate indptr unconditionally — required for safe scatter even when nnz==0.
    if csr_p.len() != n + 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("csr_indptr length mismatch: expected {}, got {}", n + 1, csr_p.len())
        ));
    }

    // ── Copy to owned Vecs for GIL release ──────────────────────────────────
    let csr_d_owned: Vec<f64> = csr_d.to_vec();
    let csr_i_owned: Vec<i32> = csr_i.to_vec();
    let csr_p_owned: Vec<i32> = csr_p.to_vec();
    let mut s: Vec<f64> = s_sl.to_vec();
    let mut lf: Vec<f64> = lf_sl.to_vec();
    let mut best_s: Vec<f64> = s.clone();

    // ── RNG: xoshiro128++ — seeded from OS entropy, initialised once ────────
    let mut rng = SmallRng::from_entropy();

    // ── Release GIL for the entire computation ───────────────────────────────
    let best_s = py.allow_threads(move || {
        let mut energy      = energy_init;
        let mut best_energy = energy;

        let temp_ratio  = temp_end / temp_start;
        let n_iter_f    = n_iterations as f64;

        for step in 0..n_iterations {
            // Geometric annealing schedule: T(t) = T0 · r^(t/N)
            let temp = temp_start * temp_ratio.powf(step as f64 / n_iter_f);

            let idx = rng.gen_range(0..n);

            // Safe indexed access — idx < n, slices validated to length n.
            let s_old  = s[idx];
            let lf_val = lf[idx];

            // O(1) energy delta from single-spin flip
            let delta_e = -2.0 * s_old * lf_val;

            // Metropolis acceptance criterion
            let accept = delta_e < 0.0 || {
                let t = if temp > 1e-15 { temp } else { 1e-15 };
                rng.gen::<f64>() < (-delta_e / t).exp()
            };

            if accept {
                let s_new = -s_old; // flip
                s[idx] = s_new;
                energy += delta_e;

                // ── Dense local-field update: lf[j] += γ·s_new  (∀j) ──────
                // Iterator-based loop: LLVM vectorises to AVX2.
                let gamma_s = gamma * s_new;
                for v in lf.iter_mut() {
                    *v += gamma_s;
                }

                // Self-correction at the flipped node: cancel the γ broadcast
                // (no self-interaction — Ising model has no diagonal coupling)
                lf[idx] -= gamma * s_new;

                // ── Sparse row scatter: lf[c] += J[idx, c]·s_new ──────────
                // Clamp row bounds to actual array lengths to guard against
                // malformed indptr values from the Python side.
                let row_start = (csr_p_owned[idx]     as usize)
                    .min(csr_d_owned.len()).min(csr_i_owned.len());
                let row_end   = (csr_p_owned[idx + 1] as usize)
                    .min(csr_d_owned.len()).min(csr_i_owned.len())
                    .max(row_start);
                for ptr in row_start..row_end {
                    let c   = csr_i_owned[ptr] as usize;
                    let val = csr_d_owned[ptr];
                    if c < n {
                        lf[c] += val * s_new;
                    }
                }

                // Track global minimum
                if energy < best_energy {
                    best_energy = energy;
                    best_s.copy_from_slice(&s);
                }
            }
        }

        best_s
    });

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
    // ── Extract slices and validate dimensions (Fix 3: boundary checks) ─────
    let v_s_sl = v_s.as_slice()?;
    let v_init_sl = v_init.as_slice()?;
    let csr_d  = csr_data.as_slice()?;
    let csr_i  = csr_indices.as_slice()?;
    let csr_p  = csr_indptr.as_slice()?;
    let has_j  = !csr_d.is_empty();

    if v_init_sl.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("v_init length mismatch: expected {}, got {}", n, v_init_sl.len())
        ));
    }
    if v_s_sl.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("v_s length mismatch: expected {}, got {}", n, v_s_sl.len())
        ));
    }
    if has_j && csr_p.len() != n + 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("csr_indptr length mismatch: expected {}, got {}", n + 1, csr_p.len())
        ));
    }

    // ── Copy to owned Vecs for GIL release (Fix 2) ─────────────────────────
    let v_s_owned: Vec<f64> = v_s_sl.to_vec();
    let csr_d_owned: Vec<f64> = csr_d.to_vec();
    let csr_i_owned: Vec<i32> = csr_i.to_vec();
    let csr_p_owned: Vec<i32> = csr_p.to_vec();
    let mut v: Vec<f64> = v_init_sl.to_vec();
    let mut v_prev: Vec<f64> = v.clone();
    let mut v_new: Vec<f64> = vec![0.0; n];

    // ── Release GIL for the entire computation (Fix 2) ─────────────────────
    let (v_final, steps_done, power_history) = py.allow_threads(move || {
        // ── Scratch buffers — single allocation, reused every step ──────────
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
            // ── 1. Activation g(V) = tanh(V),  Jacobian J = 1 − g² ─────────
            for i in 0..n {
                let gv    = v[i].tanh();
                g_v[i]    = gv;
                jacobian[i] = 1.0 - gv * gv;
            }

            // ── 2. SpMV: J_g_V[i] = Σ_c  J[i,c] · g_V[c]  (inline CSR) ────
            if has_j {
                for i in 0..n {
                    let start = csr_p_owned[i]     as usize;
                    let end   = csr_p_owned[i + 1] as usize;
                    let mut acc = 0.0f64;
                    // Clamp bounds to guard against malformed indptr from Python.
                    let safe_start = start.min(csr_i_owned.len()).min(csr_d_owned.len());
                    let safe_end   = end.min(csr_i_owned.len()).min(csr_d_owned.len())
                        .max(safe_start);
                    for ptr in safe_start..safe_end {
                        let c   = csr_i_owned[ptr] as usize;
                        let val = csr_d_owned[ptr];
                        if c < n {
                            acc += val * g_v[c];
                        }
                    }
                    j_g_v[i] = acc;
                }
            } else {
                j_g_v.iter_mut().for_each(|x| *x = 0.0);
            }

            // ── 3. Error, KCL dynamics, accumulate norms ────────────────────
            let mut p_err_acc = 0.0f64;
            let mut dv_norm2  = 0.0f64;
            for i in 0..n {
                let e  = v_s_owned[i] + j_scale * j_g_v[i] - g_v[i];
                let dv = -v[i] * inv_tau + jacobian[i] * (g_prec_base * e);
                error[i] = e;
                dv_dt[i] = dv;
                p_err_acc += e  * e;
                dv_norm2  += dv * dv;
            }
            let p_err = g_prec_base * p_err_acc;

            // ── 4. Euler step + Polyak momentum ─────────────────────────────
            for i in 0..n {
                v_new[i] = v[i]
                    + dv_dt[i] * dt_cur
                    + momentum * (v[i] - v_prev[i]);
            }
            std::mem::swap(&mut v, &mut v_new);
            std::mem::swap(&mut v_prev, &mut v_new);

            power_history.push(p_err);
            steps_done = step + 1;

            // ── 5. Stopping criteria ────────────────────────────────────────
            if p_err < nirvana_threshold
                || dv_norm2 < nirvana_threshold
            {
                break;
            }

            // ── 6. Adaptive timestep & momentum (Nesterov-style schedule) ───
            if p_err < prev_p * 0.95 {
                dt_cur   = (dt_cur * 1.2).min(dt_max);
                momentum = (momentum + 0.05).min(0.6);
            } else if p_err > prev_p {
                dt_cur   = (dt_cur * 0.7).max(dt_min);
                momentum = (momentum * 0.5).max(0.0);
            }
            prev_p = p_err;
        }

        (v, steps_done, power_history)
    });

    Ok((v_final.into_pyarray(py), steps_done, power_history))
}

// ═══════════════════════════════════════════════════════════════════════════
//  3.  Submodular Lazy Greedy (CELF)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct CelfNode {
    id: usize,
    gain: OrderedFloat<f64>,
    iteration: usize,
}
impl PartialEq for CelfNode { fn eq(&self, other: &Self) -> bool { self.gain == other.gain } }
impl Eq for CelfNode {}
impl PartialOrd for CelfNode { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }
impl Ord for CelfNode { fn cmp(&self, other: &Self) -> Ordering { self.gain.cmp(&other.gain) } }

#[pyfunction]
#[pyo3(signature = (surprises, csr_data, csr_indices, csr_indptr, k, alpha, beta))]
fn solve_submodular_greedy_rust<'py>(
    py: Python<'py>,
    surprises: PyReadonlyArray1<'py, f64>,
    csr_data: PyReadonlyArray1<'py, f64>,
    csr_indices: PyReadonlyArray1<'py, i32>,
    csr_indptr: PyReadonlyArray1<'py, i32>,
    k: usize,
    alpha: f64,
    beta: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, f64, Bound<'py, PyArray1<f64>>)> {
    let surp = surprises.as_slice()?;
    let data = csr_data.as_slice()?;
    let indices = csr_indices.as_slice()?;
    let indptr = csr_indptr.as_slice()?;
    let n = surp.len();
    let actual_k = k.min(n);
    let has_graph = !data.is_empty();

    if has_graph && indptr.len() < n + 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("csr_indptr length mismatch"));
    }

    // Copy to owned Vecs for GIL release
    let surp_owned: Vec<f64> = surp.to_vec();
    let data_owned: Vec<f64> = data.to_vec();
    let indices_owned: Vec<i32> = indices.to_vec();
    let indptr_owned: Vec<i32> = indptr.to_vec();

    let (sel, obj, gains) = py.allow_threads(move || {
        let mut row_sums = vec![0.0; n];
        if has_graph {
            for i in 0..n {
                let start = (indptr_owned[i] as usize).min(data_owned.len());
                let end = (indptr_owned[i + 1] as usize).min(data_owned.len()).max(start);
                for j in start..end {
                    row_sums[i] += data_owned[j];
                }
            }
        }

        let mut pq = BinaryHeap::with_capacity(n);
        for i in 0..n {
            let mut gain = alpha * surp_owned[i] + beta * row_sums[i];
            if gain.is_nan() { gain = 0.0; }
            pq.push(CelfNode { id: i, gain: OrderedFloat(gain), iteration: 0 });
        }

        let mut selected = Vec::with_capacity(actual_k);
        let mut selected_mask = vec![false; n];
        let mut gains_history = Vec::with_capacity(actual_k);
        let mut total_value = 0.0;
        let mut current_iter = 0;

        while selected.len() < actual_k && !pq.is_empty() {
            let mut top = pq.pop().unwrap();
            if top.gain.into_inner() == std::f64::NEG_INFINITY { break; }
            if top.gain.into_inner() <= 0.0 && !selected.is_empty() { break; }

            if top.iteration == current_iter {
                selected.push(top.id as i64);
                selected_mask[top.id] = true;
                total_value += top.gain.into_inner();
                gains_history.push(top.gain.into_inner());
                current_iter += 1;
            } else {
                let mut new_gain = alpha * surp_owned[top.id] + beta * row_sums[top.id];
                if has_graph {
                    let start = (indptr_owned[top.id] as usize).min(data_owned.len());
                    let end = (indptr_owned[top.id + 1] as usize).min(data_owned.len()).max(start);
                    for ptr in start..end {
                        if ptr < indices_owned.len() {
                            let neighbor = indices_owned[ptr] as usize;
                            if neighbor < n && selected_mask[neighbor] {
                                new_gain -= 2.0 * beta * data_owned[ptr];
                            }
                        }
                    }
                }
                if new_gain.is_nan() { new_gain = 0.0; }
                // Clip rounding errors: marginal gain is monotone non-increasing.
                let safe_gain = new_gain.min(top.gain.into_inner());
                top.gain = OrderedFloat(safe_gain);
                top.iteration = current_iter;
                pq.push(top);
            }
        }

        (selected, total_value, gains_history)
    });

    Ok((sel.into_pyarray(py), obj, gains.into_pyarray(py)))
}

// ═══════════════════════════════════════════════════════════════════════════
//  4.  N-gram Hash Vectorization (Rayon Multi-threading)
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
fn ngram_vectors_rust<'py>(
    py: Python<'py>,
    texts: Vec<String>,
    max_features: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = texts.len();
    if n == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, max_features)).into_pyarray(py));
    }
    if max_features == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("max_features must be > 0"));
    }

    let mut out_flat = vec![0.0; n * max_features];

    py.allow_threads(|| {
        out_flat
            .par_chunks_mut(max_features)
            .zip(texts.par_iter())
            .for_each(|(row, text)| {
                let lower = text.to_lowercase();
                let bytes = lower.as_bytes();
                let len = bytes.len();
                let mut counts = vec![0.0; max_features];

                for &ng in &[3usize, 4, 5] {
                    if len >= ng {
                        for j in 0..=(len - ng) {
                            let mut hasher = Hasher::new();
                            hasher.update(&bytes[j..j + ng]);
                            let hash = hasher.finalize() as usize;
                            counts[hash % max_features] += 1.0;
                        }
                    }
                }

                let mut norm_sq = 0.0;
                for c in &counts {
                    norm_sq += c * c;
                }
                let norm = norm_sq.sqrt().max(1e-10);
                for i in 0..max_features {
                    row[i] = counts[i] / norm;
                }
            });
    });

    let arr2 = ndarray::Array2::from_shape_vec((n, max_features), out_flat).unwrap();
    Ok(arr2.into_pyarray(py))
}

// ═══════════════════════════════════════════════════════════════════════════
//  5.  Graph Topology Analysis (BFS)
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
fn bfs_components_rust(
    n: usize,
    csr_indices: PyReadonlyArray1<i32>,
    csr_indptr: PyReadonlyArray1<i32>,
) -> PyResult<Vec<Vec<i32>>> {
    let i_arr = csr_indices.as_slice()?;
    let p_arr = csr_indptr.as_slice()?;

    if p_arr.len() < n + 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("csr_indptr length mismatch"));
    }

    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(node) = queue.pop_front() {
            comp.push(node as i32);
            let s = (p_arr[node] as usize).min(i_arr.len());
            let e = (p_arr[node + 1] as usize).min(i_arr.len()).max(s);
            for ptr in s..e {
                let nb = i_arr[ptr] as usize;
                if nb < n && !visited[nb] {
                    visited[nb] = true;
                    queue.push_back(nb);
                }
            }
        }
        components.push(comp);
    }

    Ok(components)
}

#[pyfunction]
fn bfs_reverse_rust(
    n: usize,
    start_indices: Vec<usize>,
    csr_indices: PyReadonlyArray1<i32>,
    csr_indptr: PyReadonlyArray1<i32>,
) -> PyResult<(Vec<i32>, usize)> {
    let i_arr = csr_indices.as_slice()?;
    let p_arr = csr_indptr.as_slice()?;

    if p_arr.len() < n + 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("csr_indptr length mismatch"));
    }

    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    let mut max_depth = 0;

    for &idx in &start_indices {
        if idx < n {
            visited[idx] = true;
            queue.push_back((idx, 0));
        }
    }

    while let Some((node, depth)) = queue.pop_front() {
        let s = (p_arr[node] as usize).min(i_arr.len());
        let e = (p_arr[node + 1] as usize).min(i_arr.len()).max(s);
        for ptr in s..e {
            let nb = i_arr[ptr] as usize;
            if nb < n && !visited[nb] {
                visited[nb] = true;
                if depth + 1 > max_depth {
                    max_depth = depth + 1;
                }
                queue.push_back((nb, depth + 1));
            }
        }
    }

    let mut reachable = Vec::new();
    for j in 0..n {
        if visited[j] && !start_indices.contains(&j) {
            reachable.push(j as i32);
        }
    }

    Ok((reachable, max_depth))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Module registration
// ═══════════════════════════════════════════════════════════════════════════

#[pymodule]
fn lmm_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_sa_ising_rust, m)?)?;
    m.add_function(wrap_pyfunction!(solve_fep_analog_rust, m)?)?;
    m.add_function(wrap_pyfunction!(solve_submodular_greedy_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_vectors_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bfs_components_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bfs_reverse_rust, m)?)?;
    Ok(())
}
