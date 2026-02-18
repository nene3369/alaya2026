"""UniversalDharmaEngine — 3-layer Dharma-Algebra architecture.

Layer 1 (Philosophy): User-defined energy terms via DharmaEnergyTerm
Layer 2 (Validation): Mathematical purification of matrices
Layer 3 (Solver):     Auto-routing based on detected properties
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from lmm._compat import sparse_matvec, sparse_dot, sparse_col_index, sparse_row_index
from lmm.dharma.energy import DharmaEnergyTerm, SilaTerm
from lmm.dharma.fep import solve_fep_kcl, solve_fep_kcl_analog


@dataclass
class EngineResult:
    selected_indices: np.ndarray
    energy: float
    solver_used: str
    properties_detected: set
    terms_summary: list
    h_total: np.ndarray = field(repr=False)
    J_total: sparse.csr_matrix | None = field(default=None, repr=False)


class UniversalDharmaEngine:
    """Synthesise energy terms and auto-route to optimal solver."""

    def __init__(
        self, n_variables: int, *,
        sa_iterations: int = 5000, sa_temp_start: float = 10.0,
        sa_temp_end: float = 0.01, solver: str | None = None,
    ):
        self.n = n_variables
        self.terms: list[DharmaEnergyTerm] = []
        self.sa_iterations = sa_iterations
        self.sa_temp_start = sa_temp_start
        self.sa_temp_end = sa_temp_end
        self._solver_override = solver

    def add(self, term: DharmaEnergyTerm) -> UniversalDharmaEngine:
        if not isinstance(term, DharmaEnergyTerm):
            raise TypeError(f"Expected DharmaEnergyTerm, got {type(term).__name__}")
        self.terms.append(term)
        return self

    def clear(self) -> UniversalDharmaEngine:
        self.terms.clear()
        return self

    # -- validation layer ------------------------------------------------

    def _purify_matrix(self, J: sparse.csr_matrix, term_name: str) -> sparse.csr_matrix:
        coo = J.tocoo()
        rows = [int(r) for r in coo.row]
        cols = [int(c) for c in coo.col]
        vals = [float(v) for v in coo.data]

        bad_count = 0
        entries: dict[tuple[int, int], float] = {}
        for r, c, v in zip(rows, cols, vals):
            if not math.isfinite(v):
                bad_count += 1
                continue
            if r == c or v == 0.0:
                continue
            entries[(r, c)] = entries.get((r, c), 0.0) + v

        if bad_count > 0:
            warnings.warn(f"[purify] {term_name}: {bad_count} NaN/Inf entries zeroed.")

        max_asym = 0.0
        checked: set[tuple[int, int]] = set()
        for r, c in entries:
            if (r, c) in checked or (c, r) in checked:
                continue
            diff = abs(entries.get((r, c), 0.0) - entries.get((c, r), 0.0))
            if diff > max_asym:
                max_asym = diff
            checked.add((r, c))

        if max_asym > 1e-10:
            warnings.warn(f"[purify] {term_name}: asymmetric matrix detected, symmetrising.")
            sym: dict[tuple[int, int], float] = {}
            for (r, c), v in entries.items():
                key = (min(r, c), max(r, c))
                sym[key] = sym.get(key, 0.0) + v
            entries = {}
            for (r, c), v in sym.items():
                avg = v / 2.0
                if avg != 0.0:
                    entries[(r, c)] = avg
                    entries[(c, r)] = avg

        fr, fc, fv = [], [], []
        for (r, c), v in entries.items():
            if v != 0.0:
                fr.append(r)
                fc.append(c)
                fv.append(v)
        return sparse.csr_matrix((fv, (fr, fc)), shape=J.shape)

    # -- synthesis -------------------------------------------------------

    def _synthesize(self, k: int):
        total_h = np.zeros(self.n)
        total_J = sparse.csr_matrix(([], ([], [])), shape=(self.n, self.n))
        properties: set[str] = set()
        sila_gamma = 0.0

        for term in self.terms:
            h, J = term.build(self.n)
            if len(h) != self.n:
                raise ValueError(f"{term.name}: h length {len(h)}, expected {self.n}")
            total_h = total_h + h
            if J is not None:
                J_p = self._purify_matrix(J, term.name)
                if J_p.shape != (self.n, self.n):
                    raise ValueError(f"{term.name}: J shape {J_p.shape}, expected ({self.n}, {self.n})")
                total_J = total_J + J_p
            if isinstance(term, SilaTerm):
                sila_gamma += term.sila_gamma
            properties.add(term.math_property)

        return total_h, total_J, properties, sila_gamma

    # -- main entry point ------------------------------------------------

    def synthesize_and_solve(self, k: int) -> EngineResult:
        if not self.terms:
            raise ValueError("No energy terms added. Use engine.add() first.")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        k = min(k, self.n)
        total_h, total_J, properties, sila_gamma = self._synthesize(k)
        has_quad = total_J.nnz > 0 or sila_gamma > 0

        if self._solver_override == "fep_analog":
            sel, e, sn = self._solve_fep_analog(total_h, total_J, k, sila_gamma=sila_gamma)
        elif self._solver_override == "fep" and has_quad:
            sel, e, sn = self._solve_fep(total_h, total_J, k, sila_gamma=sila_gamma)
        elif not has_quad:
            sel, e, sn = self._solve_topk(total_h, k)
        elif "frustrated" in properties or ("submodular" in properties and "supermodular" in properties):
            sel, e, sn = self._solve_ising_sa(total_h, total_J, k, sila_gamma=sila_gamma)
        elif "supermodular" in properties and "submodular" not in properties:
            sel, e, sn = self._solve_supermodular_warmstart(total_h, total_J, k, sila_gamma=sila_gamma)
        elif "submodular" in properties and "supermodular" not in properties:
            sel, e, sn = self._solve_submodular_greedy(total_h, total_J, k, sila_gamma=sila_gamma)
        else:
            sel, e, sn = self._solve_ising_sa(total_h, total_J, k, sila_gamma=sila_gamma)

        summary = [f"{t.name} (w={t.weight}, {t.math_property})" for t in self.terms]
        return EngineResult(
            selected_indices=sel, energy=e, solver_used=sn,
            properties_detected=properties, terms_summary=summary,
            h_total=total_h, J_total=total_J if has_quad else None,
        )

    # -- solvers ---------------------------------------------------------

    def _solve_topk(self, h, k):
        indices = np.argsort(h)[:k]
        energy = float(np.sum(h[indices]))
        return np.array(sorted(int(i) for i in indices)), energy, "topk_sort"

    def _solve_submodular_greedy(self, h, J, k, sila_gamma=0.0):
        mask = np.ones(self.n, dtype=bool)
        marginal = h.copy() + np.asarray(J.sum(axis=1)).flatten()
        if sila_gamma > 0:
            marginal += sila_gamma * (self.n - 1)
        INF = float('inf')
        masked = marginal.copy()
        selected: list[int] = []

        for step in range(k):
            best = int(np.argmin(masked))
            if float(masked[best]) >= INF:
                break
            selected.append(best)
            mask[best] = False
            masked[best] = INF
            if step == k - 1:
                break
            if J.nnz > 0:
                row_coo = J.getrow(best).tocoo()
                for idx, val in zip(row_coo.col, row_coo.data):
                    idx = int(idx)
                    if mask[idx]:
                        marginal[idx] = float(marginal[idx]) + 2.0 * float(val)
                        masked[idx] = float(masked[idx]) + 2.0 * float(val)

        sel = np.array(selected)
        return sel, self._eval(h, J, sila_gamma, sel), "submodular_greedy"

    def _solve_supermodular_warmstart(self, h, J, k, sila_gamma=0.0):
        gains = h.copy()
        mask = np.ones(self.n, dtype=bool)
        x_init = np.zeros(self.n)
        INF = float('inf')
        masked = gains.copy()

        for step in range(k):
            best = int(np.argmin(masked))
            if float(masked[best]) >= INF:
                break
            mask[best] = False
            masked[best] = INF
            x_init[best] = 1.0
            if step == k - 1:
                break
            if J.nnz > 0:
                row_coo = J.getrow(best).tocoo()
                for idx, val in zip(row_coo.col, row_coo.data):
                    idx = int(idx)
                    if mask[idx]:
                        gains[idx] = float(gains[idx]) + 2.0 * float(val)
                        masked[idx] = float(masked[idx]) + 2.0 * float(val)

        sel, e, _ = self._solve_ising_sa(h, J, k, sila_gamma=sila_gamma, initial_state=x_init)
        return sel, e, "supermodular_warmstart_ising_sa"

    def _solve_ising_sa(self, h, J, k, sila_gamma=0.0, initial_state=None):
        rng = np.random.default_rng()
        n = self.n
        rs = np.asarray(J.sum(axis=1)).flatten() if J.nnz > 0 else np.zeros(n)
        h_ising = 3.0 * h / 4.0 + rs / 8.0 + sila_gamma * (n - 1) / 8.0

        s = (2.0 * initial_state - 1.0).copy() if initial_state is not None else rng.choice([-1.0, 1.0], size=n)
        sum_s = float(np.sum(s))
        Js = sparse_matvec(J, s) if J.nnz > 0 else np.zeros(n)
        local_field = h_ising + Js / 2.0 - sila_gamma / 2.0 * s
        offset = sila_gamma / 2.0 * sum_s

        energy = (sparse_dot(s, J, s) / 4.0 if J.nnz > 0 else 0.0) + sila_gamma / 4.0 * (sum_s * sum_s - n) + float(np.dot(h_ising, s))
        best_s, best_energy = s.copy(), energy

        col_idx = sparse_col_index(J)
        if J.nnz > 0:
            sparse_row_index(J)

        for step in range(self.sa_iterations):
            temp = self.sa_temp_start * (self.sa_temp_end / self.sa_temp_start) ** (step / self.sa_iterations)
            idx = int(rng.integers(0, n))
            delta_e = -2.0 * float(s[idx]) * (float(local_field[idx]) + offset)

            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-15)):
                s[idx] *= -1.0
                energy += delta_e
                ns = float(s[idx])
                sum_s += 2.0 * ns
                if idx in col_idx:
                    for r, v in col_idx[idx]:
                        local_field[r] = float(local_field[r]) + v * ns
                if sila_gamma > 0:
                    offset += sila_gamma * ns
                    local_field[idx] = float(local_field[idx]) - sila_gamma * ns
                if energy < best_energy:
                    best_energy, best_s = energy, s.copy()

        best_x = (best_s + 1.0) / 2.0
        if k is not None:
            best_x = self._project_sparse(best_x, h, J, sila_gamma, k)
        selected = np.where(best_x > 0.5)[0][:k]
        return selected, self._eval(h, J, sila_gamma, selected), "ising_sa"

    def _solve_fep(self, h, J, k, sila_gamma=0.0, initial_state=None):
        V_mu, x_final, steps, _ = solve_fep_kcl(
            h=h, J=J, k=k, n=self.n, sila_gamma=sila_gamma,
            initial_state=initial_state, sparse_matvec=sparse_matvec,
        )
        best_x = self._project_sparse(x_final, h, J, sila_gamma, k)
        selected = np.where(best_x > 0.5)[0][:k]
        return selected, self._eval(h, J, sila_gamma, selected), "fep_kcl"

    def _solve_fep_analog(self, h, J, k, sila_gamma=0.0):
        V_s = -h.copy()
        if sila_gamma > 0:
            V_s -= sila_gamma * (self.n - 2 * k)
        J_dyn = J * (-1.0) if J.nnz > 0 else J
        density = J.nnz / max(self.n * self.n, 1)
        G_prec = max(5.0, 10.0 * (1.0 - density))

        V_mu, x_final, steps, _ = solve_fep_kcl_analog(
            V_s=V_s, J_dynamic=J_dyn, n=self.n, G_prec_base=G_prec,
        )
        top_k = np.argsort(-x_final)[:k]
        selected = np.array(sorted(int(i) for i in top_k))
        return selected, self._eval(h, J, sila_gamma, selected), "fep_kcl_analog"

    # -- helpers ---------------------------------------------------------

    def _project_sparse(self, x, h, J, sg, k):
        n_sel = int((x > 0.5).sum())
        if n_sel == k:
            return x
        Jx = sparse_matvec(J, x) if J.nnz > 0 else np.zeros(self.n)
        sx = float(np.sum(x))

        if n_sel > k:
            sel = np.where(x > 0.5)[0]
            scores = [(float(h[int(i)]) + 2.0 * (float(Jx[int(i)]) + sg * (sx - float(x[int(i)]))), int(i)) for i in sel]
            scores.sort()
            x_new = np.zeros(self.n)
            for _, i in scores[:k]:
                x_new[i] = 1.0
            return x_new
        else:
            unsel = np.where(x <= 0.5)[0]
            if len(unsel) == 0:
                return x
            scores = [(float(h[int(i)]) + 2.0 * (float(Jx[int(i)]) + sg * (sx - float(x[int(i)]))), int(i)) for i in unsel]
            scores.sort()
            x_new = x.copy()
            for _, i in scores[:k - n_sel]:
                x_new[i] = 1.0
            return x_new

    def _eval(self, h, J, sg, selected):
        k = len(selected)
        if k == 0:
            return 0.0
        energy = sum(float(h[int(i)]) for i in selected)
        if J.nnz > 0 and k > 0:
            ss = set(int(i) for i in selected)
            for i in selected:
                row_coo = J.getrow(int(i)).tocoo()
                for c, v in zip(row_coo.col, row_coo.data):
                    if c in ss:
                        energy += v
        if sg > 0:
            energy += sg * (k * k - k)
        return energy

    def solve_fep_kcl_analog(
        self, V_s, J_dynamic, k_final, G_prec_base=10.0,
        tau_leak=1.0, dt=0.01, max_steps=1000, nirvana_threshold=1e-5,
    ) -> EngineResult:
        """Direct analog FEP solver bypassing QUBO synthesis."""
        k = min(k_final, self.n)
        V_mu, x_final, steps, power = solve_fep_kcl_analog(
            V_s=V_s, J_dynamic=J_dynamic, n=self.n,
            G_prec_base=G_prec_base, tau_leak=tau_leak, dt=dt,
            max_steps=max_steps, nirvana_threshold=nirvana_threshold,
        )
        top_k = np.argsort(-x_final)[:k]
        selected = np.array(sorted(int(i) for i in top_k))
        x_bin = np.zeros(self.n)
        for i in selected:
            x_bin[int(i)] = 1.0
        energy = float(-np.dot(V_s, x_bin))
        if J_dynamic.nnz > 0:
            energy -= sparse_dot(x_bin, J_dynamic, x_bin)
        return EngineResult(
            selected_indices=selected, energy=energy, solver_used="fep_kcl_analog",
            properties_detected={"analog_fep"},
            terms_summary=[f"V_s (n={self.n})", f"J_dynamic (nnz={J_dynamic.nnz})",
                           f"steps={steps}, P_final={power[-1]:.2e}"],
            h_total=-V_s, J_total=J_dynamic,
        )
