"""DharmaLMM — high-level integrated API for Digital Dharma optimization.

End-to-end pipeline combining:
  1. Sparse graph construction (Indra's net)
  2. Supermodular greedy warm-start (Sangha)
  3. Solver execution (SA / Ising SA / engine)
  4. Madhyamaka balance (exponential gradient)
  5. Dharma interpretation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from lmm.dharma.algorithms import (
    BodhisattvaQUBO,
    MadhyamakaBalancer,
    build_sparse_impact_graph,
    vectorized_greedy_initialize,
)
from lmm.dharma.concepts import DharmaWeights
from lmm.dharma.energy import (
    DharmaEnergyTerm,
    KarunaTerm,
    PrajnaTerm,
    SilaTerm,
)
from lmm.dharma.engine import UniversalDharmaEngine
from lmm.dharma.interpreter import DharmaInterpretation, DharmaInterpreter
from lmm.dharma.sangha import CouncilResult, SanghaOrchestrator
from lmm.solvers import ClassicalQUBOSolver, SubmodularSelector
from lmm.surprise import SurpriseCalculator


class SanghaRejectedError(RuntimeError):
    """Raised when the Sangha council rejects a select_dharma request."""


@dataclass
class DharmaResult:
    """DharmaLMM result."""

    selected_indices: np.ndarray
    surprise_values: np.ndarray
    energy: float
    weights: DharmaWeights
    interpretation: DharmaInterpretation | None
    method: str
    council: CouncilResult | None = None


class DharmaLMM:
    """Digital Dharma optimization pipeline.

    Usage::

        model = DharmaLMM(k=15, solver_mode="submodular")
        model.fit(reference_data)
        result = model.select_dharma(candidates)

    solver_mode:
      - "submodular": greedy O(n*k), (1-1/e) guarantee [recommended]
      - "ising_sa": Ising SA + projection
      - "sa": classical SA + projection
      - "greedy": QUBO greedy (no guarantee)
      - "engine": Dharma-Algebra engine with auto-routing
    """

    def __init__(
        self,
        k: int = 10,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 10.0,
        surprise_method: str = "entropy",
        solver_mode: str = "submodular",
        use_sparse_graph: bool = True,
        sparse_k: int = 20,
        use_greedy_warmstart: bool = True,
        use_ising_sa: bool = True,
        use_exponential_balance: bool = True,
        n_balance_iterations: int = 3,
        sa_iterations: int = 5000,
        sa_temp_start: float = 10.0,
        sa_temp_end: float = 0.01,
        use_sangha: bool = True,
        alaya_memory: Any = None,
    ):
        self.k = k
        self.weights = DharmaWeights(alpha=alpha, beta=beta, gamma=gamma)
        self.surprise_method = surprise_method
        self.solver_mode = solver_mode
        self.use_sparse_graph = use_sparse_graph
        self.sparse_k = sparse_k
        self.use_greedy_warmstart = use_greedy_warmstart
        self.use_ising_sa = use_ising_sa or solver_mode == "ising_sa"
        self.use_exponential_balance = use_exponential_balance
        self.n_balance_iterations = n_balance_iterations
        self.sa_iterations = sa_iterations
        self.sa_temp_start = sa_temp_start
        self.sa_temp_end = sa_temp_end

        self._calculator = SurpriseCalculator(method=surprise_method)
        self._balancer = MadhyamakaBalancer()
        self._interpreter = DharmaInterpreter()
        self._submodular = SubmodularSelector(alpha=alpha, beta=beta)
        self._sangha = SanghaOrchestrator(alaya_memory=alaya_memory) if use_sangha else None

    def fit(self, reference_data: np.ndarray) -> DharmaLMM:
        """Learn reference distribution (prior)."""
        self._calculator.fit(reference_data)
        return self

    def create_engine(
        self,
        n_variables: int,
        extra_terms: list[DharmaEnergyTerm] | None = None,
    ) -> UniversalDharmaEngine:
        """Create Dharma-Algebra engine with default terms."""
        return UniversalDharmaEngine(
            n_variables,
            sa_iterations=self.sa_iterations,
            sa_temp_start=self.sa_temp_start,
            sa_temp_end=self.sa_temp_end,
        )

    def select_dharma(
        self,
        candidates: np.ndarray,
        interpret: bool = True,
        extra_terms: list[DharmaEnergyTerm] | None = None,
        query: str | None = None,
    ) -> DharmaResult:
        """Select K items from candidates using Dharma optimization."""
        n = len(candidates)
        k = min(self.k, n)

        # ── Step 2: Sangha council (自動起動) ──────────────────────────────
        council: CouncilResult | None = None
        if self._sangha is not None:
            _query = query or f"select_dharma: k={k} n={n} solver={self.solver_mode}"
            council = self._sangha.hold_council_sync({
                "query": _query,
                "issue_id": f"dharma-{self.solver_mode}",
                "fep_state": 0.5,
            })
            if council.final == "REJECTED":
                raise SanghaRejectedError(council.reason)
        # ──────────────────────────────────────────────────────────────────

        surprises = self._calculator.compute(candidates)

        _needs_full_graph = self.solver_mode not in ("submodular",)
        impact_graph = None
        normalized_data = None

        if self.solver_mode == "submodular" and candidates.ndim >= 2:
            norms = np.linalg.norm(candidates, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-10, None)
            normalized_data = candidates / norms
        elif _needs_full_graph and self.use_sparse_graph and candidates.ndim >= 2:
            impact_graph = build_sparse_impact_graph(
                candidates, k=min(self.sparse_k, n - 1), use_hnswlib=True,
            )
        elif _needs_full_graph and candidates.ndim >= 2:
            norms = np.linalg.norm(candidates, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-10, None)
            normalized = candidates / norms
            impact_graph = np.clip(normalized @ normalized.T, 0, None)
            np.fill_diagonal(impact_graph, 0)

        alpha, beta = self.weights.alpha, self.weights.beta
        if self.n_balance_iterations > 0:
            for _ in range(self.n_balance_iterations):
                if self.use_exponential_balance:
                    alpha, beta = self._balancer.balance_exponential(
                        surprises, alpha, beta,
                    )
                else:
                    alpha, beta = self._balancer.balance(
                        surprises, alpha, beta,
                    )

        if self.solver_mode == "engine":
            engine = self.create_engine(n)
            engine.add(PrajnaTerm(surprises, weight=alpha))
            if impact_graph is not None:
                engine.add(KarunaTerm(impact_graph, weight=beta))
            engine.add(SilaTerm(k=k, weight=max(
                self.weights.gamma,
                2.0 * alpha * (float(surprises.max()) if len(surprises) > 0 else 1.0),
            )))
            if extra_terms:
                for term in extra_terms:
                    engine.add(term)
            result = engine.synthesize_and_solve(k=k)
            selected = result.selected_indices
            energy = result.energy
            method = f"engine:{result.solver_used}"

        elif self.solver_mode == "submodular":
            self._submodular.alpha = alpha
            self._submodular.beta = beta
            if normalized_data is not None:
                sub_result = self._submodular.select_lazy(
                    surprises, normalized_data, k=k,
                    sparse_k=min(self.sparse_k, n - 1),
                )
            else:
                sub_result = self._submodular.select(surprises, impact_graph, k=k)
            selected = sub_result.selected_indices
            energy = -sub_result.objective_value
            method = "submodular_greedy"
        else:
            max_surprise = float(surprises.max()) if len(surprises) > 0 else 1.0
            auto_gamma = max(self.weights.gamma, 2.0 * alpha * max_surprise)

            qubo = BodhisattvaQUBO(n)
            qubo.add_prajna_term(surprises, alpha=alpha)
            qubo.add_sila_term(k, gamma=auto_gamma)

            if impact_graph is not None:
                if sparse.issparse(impact_graph):
                    qubo.add_karuna_term_sparse(impact_graph, beta=beta)
                else:
                    qubo.add_karuna_term(impact_graph, beta=beta)

            initial_state = None
            if self.use_greedy_warmstart and impact_graph is not None:
                initial_state = vectorized_greedy_initialize(
                    surprises, impact_graph, k=k, alpha=alpha, beta=beta,
                )

            solver = ClassicalQUBOSolver(qubo.get_builder())

            if self.solver_mode == "ising_sa" or self.use_ising_sa:
                x = solver.solve_sa_ising(
                    initial_state=initial_state,
                    n_iterations=self.sa_iterations,
                    temp_start=self.sa_temp_start,
                    temp_end=self.sa_temp_end, k=k,
                )
                method = "ising_sa"
            elif self.solver_mode == "greedy":
                x = solver.solve_greedy(k=k)
                method = "qubo_greedy"
            else:
                x = solver.solve_sa(
                    initial_state=initial_state,
                    n_iterations=self.sa_iterations,
                    temp_start=self.sa_temp_start,
                    temp_end=self.sa_temp_end, k=k,
                )
                method = "sa"

            selected = np.where(x > 0.5)[0][:k]
            energy = qubo.get_builder().evaluate(x)

        interpretation = None
        if interpret:
            if impact_graph is not None:
                graph_dense = (
                    impact_graph.toarray() if sparse.issparse(impact_graph)
                    else impact_graph
                )
            elif normalized_data is not None:
                graph_dense = np.clip(normalized_data @ normalized_data.T, 0, None)
                np.fill_diagonal(graph_dense, 0)
            else:
                graph_dense = None
            interpretation = self._interpreter.interpret(
                selected, surprises, graph_dense, k,
            )

        return DharmaResult(
            selected_indices=selected,
            surprise_values=surprises[selected] if len(selected) > 0 else np.array([]),
            energy=energy,
            weights=DharmaWeights(alpha=alpha, beta=beta, gamma=self.weights.gamma),
            interpretation=interpretation,
            method=method,
            council=council,
        )

    def select_from_scores(
        self,
        surprises: np.ndarray,
        impact_graph: np.ndarray | sparse.csr_matrix | None = None,
    ) -> DharmaResult:
        """Select from pre-computed surprise scores."""
        n = len(surprises)
        k = min(self.k, n)

        alpha, beta = self.weights.alpha, self.weights.beta
        for _ in range(self.n_balance_iterations):
            if self.use_exponential_balance:
                alpha, beta = self._balancer.balance_exponential(
                    surprises, alpha, beta,
                )
            else:
                alpha, beta = self._balancer.balance(surprises, alpha, beta)

        max_surprise = float(surprises.max()) if len(surprises) > 0 else 1.0
        auto_gamma = max(self.weights.gamma, 2.0 * alpha * max_surprise)

        qubo = BodhisattvaQUBO(n)
        qubo.add_prajna_term(surprises, alpha=alpha)
        qubo.add_sila_term(k, gamma=auto_gamma)

        if impact_graph is not None:
            if sparse.issparse(impact_graph):
                qubo.add_karuna_term_sparse(impact_graph, beta=beta)
            else:
                qubo.add_karuna_term(impact_graph, beta=beta)

        initial_state = None
        if self.use_greedy_warmstart and impact_graph is not None:
            initial_state = vectorized_greedy_initialize(
                surprises, impact_graph, k=k, alpha=alpha, beta=beta,
            )

        solver = ClassicalQUBOSolver(qubo.get_builder())

        if self.use_ising_sa:
            x = solver.solve_sa_ising(
                initial_state=initial_state,
                n_iterations=self.sa_iterations, k=k,
            )
            method = "ising_sa"
        else:
            x = solver.solve_sa(
                initial_state=initial_state,
                n_iterations=self.sa_iterations, k=k,
            )
            method = "sa"

        selected = np.where(x > 0.5)[0][:k]
        energy = qubo.get_builder().evaluate(x)

        return DharmaResult(
            selected_indices=selected,
            surprise_values=surprises[selected] if len(selected) > 0 else np.array([]),
            energy=energy,
            weights=DharmaWeights(alpha=alpha, beta=beta, gamma=self.weights.gamma),
            interpretation=None,
            method=method,
        )
