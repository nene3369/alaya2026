"""DharmaReranker — 3-stage cascade RAG reranker.

Cascade architecture:
  1. Shravakayana: Vector DB returns Top-N (external)
  2. Pratyekabuddha: Subgraph extraction + dynamic masking
  3. Bodhisattva: UniversalDharmaEngine selects Top-K
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from lmm.dharma.algorithms import (
    build_sparse_impact_graph,
    extract_subgraph,
    query_condition_graph,
)
from lmm.dharma.energy import (
    DharmaEnergyTerm,
    KarunaTerm,
    MettaTerm,
    QueryDukkhaTerm,
    SilaTerm,
)
from lmm.dharma.engine import UniversalDharmaEngine


@dataclass
class RerankerResult:
    """DharmaReranker output."""

    selected_indices: np.ndarray
    original_indices: np.ndarray | None
    energy: float
    solver_used: str
    relevance_scores: np.ndarray


class DharmaReranker:
    """3-stage cascade RAG reranker."""

    def __init__(
        self,
        k: int = 10,
        *,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 10.0,
        query_weight: float = 0.7,
        corpus_weight: float = 0.3,
        diversity_weight: float = 0.0,
        J_static: sparse.csr_matrix | None = None,
        sparse_k: int = 20,
        sa_iterations: int = 2000,
        solver_mode: str = "auto",
    ):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.query_weight = query_weight
        self.corpus_weight = corpus_weight
        self.diversity_weight = diversity_weight
        self.J_static = J_static
        self.sparse_k = sparse_k
        self.sa_iterations = sa_iterations
        self.solver_mode = solver_mode

    def rerank(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        *,
        candidate_indices: np.ndarray | None = None,
        extra_terms: list[DharmaEnergyTerm] | None = None,
    ) -> RerankerResult:
        """Rerank candidates and return optimal Top-K."""
        n = len(candidate_embeddings)
        k = min(self.k, n)

        if k >= n:
            rel = self._compute_relevance(query_embedding, candidate_embeddings)
            return RerankerResult(
                selected_indices=np.arange(n),
                original_indices=candidate_indices,
                energy=0.0, solver_used="passthrough",
                relevance_scores=rel,
            )

        relevance = self._compute_relevance(query_embedding, candidate_embeddings)

        if self.J_static is not None and candidate_indices is not None:
            J_sub = extract_subgraph(self.J_static, candidate_indices)
            J_dynamic = query_condition_graph(J_sub, relevance)
        elif self.J_static is not None:
            J_dynamic = query_condition_graph(self.J_static, relevance)
        else:
            graph_k = min(self.sparse_k, n - 1)
            J_local = build_sparse_impact_graph(
                candidate_embeddings, k=graph_k, use_hnswlib=True,
            )
            J_dynamic = query_condition_graph(J_local, relevance)

        if self.solver_mode == "fep_analog":
            engine = UniversalDharmaEngine(n)
            result = engine.solve_fep_kcl_analog(
                V_s=relevance, J_dynamic=J_dynamic, k_final=k,
            )
        else:
            engine = UniversalDharmaEngine(n, sa_iterations=self.sa_iterations)

            engine.add(QueryDukkhaTerm(
                query_embedding, candidate_embeddings,
                query_weight=self.query_weight,
                corpus_weight=self.corpus_weight,
                weight=self.alpha,
            ))

            if J_dynamic.nnz > 0:
                engine.add(KarunaTerm(J_dynamic, weight=self.beta))

            if self.diversity_weight > 0 and J_dynamic.nnz > 0:
                engine.add(MettaTerm(J_dynamic, weight=self.diversity_weight))

            engine.add(SilaTerm(k=k, weight=self.gamma))

            if extra_terms:
                for term in extra_terms:
                    engine.add(term)

            result = engine.synthesize_and_solve(k=k)

        original_idx = None
        if candidate_indices is not None:
            original_idx = candidate_indices[result.selected_indices]

        return RerankerResult(
            selected_indices=result.selected_indices,
            original_indices=original_idx,
            energy=result.energy,
            solver_used=result.solver_used,
            relevance_scores=relevance[result.selected_indices],
        )

    def fit_offline(
        self,
        corpus_embeddings: np.ndarray,
        *,
        graph_k: int | None = None,
    ) -> DharmaReranker:
        """Offline: build static interdependence graph."""
        self._corpus_embeddings = corpus_embeddings
        gk = graph_k or self.sparse_k
        gk = min(gk, len(corpus_embeddings) - 1)
        self.J_static = build_sparse_impact_graph(
            corpus_embeddings, k=gk, use_hnswlib=True,
        )
        return self

    def rerank_online(
        self,
        query_embedding: np.ndarray,
        fetched_indices: np.ndarray,
        *,
        extra_terms: list[DharmaEnergyTerm] | None = None,
    ) -> RerankerResult:
        """Online: rerank Vector DB Top-N to optimal Top-K."""
        if not hasattr(self, "_corpus_embeddings") or self._corpus_embeddings is None:
            raise RuntimeError("Call fit_offline() first")

        idx = np.array([int(i) for i in fetched_indices])
        candidate_embeddings = self._corpus_embeddings[idx]

        return self.rerank(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_indices=idx,
            extra_terms=extra_terms,
        )

    @staticmethod
    def _entropic_concentration(relevance: np.ndarray) -> float:
        """Information-geometric concentration (normalized negative entropy)."""
        r = np.asarray(relevance).flatten()
        n = len(r)
        if n <= 1:
            return 0.0

        total = float(r.sum())
        if total < 1e-10:
            return 0.0

        p = r / total
        log_n = float(np.log(float(n)))
        mask = p > 1e-15
        entropy = -float(np.sum(p[mask] * np.log(p[mask])))

        return max(0.0, min(1.0, 1.0 - entropy / log_n))

    @staticmethod
    def _diagnose_intent(relevance: np.ndarray) -> dict:
        """Diagnose query intent from score distribution."""
        if len(relevance) == 0:
            return {"concentration": 0.0, "entropy": 0.0, "intent": "exploratory"}

        c = DharmaReranker._entropic_concentration(relevance)

        if c > 2.0 / 3.0:
            intent = "focused"
        elif c > 1.0 / 3.0:
            intent = "balanced"
        else:
            intent = "exploratory"

        return {"concentration": c, "entropy": 1.0 - c, "intent": intent}

    @staticmethod
    def _compute_relevance(
        query: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """Cosine similarity between query and candidates, mapped to [0, 1]."""
        q = np.asarray(query).flatten()
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-10:
            return np.zeros(len(candidates))

        c_norms = np.linalg.norm(candidates, axis=1)
        c_norms = np.clip(c_norms, 1e-10, None)

        dots = candidates @ q
        scores = dots / (c_norms * q_norm)
        return np.clip(scores, 0.0, 1.0)


@dataclass
class RouterDiagnosis:
    """IntentAwareRouter diagnosis."""

    intent: str
    concentration: float
    entropy: float
    beta_used: float
    metta_used: float


class IntentAwareRouter:
    """Madhyamaka auto-tuning router.

    Analyzes score distribution to auto-adjust Karuna/Metta blend.
    """

    def __init__(
        self,
        k: int = 10,
        *,
        alpha: float = 1.0,
        beta_range: tuple[float, float] = (0.1, 0.8),
        metta_range: tuple[float, float] = (0.0, 1.0),
        gamma: float = 10.0,
        query_weight: float = 0.7,
        corpus_weight: float = 0.3,
        sparse_k: int = 20,
        sa_iterations: int = 2000,
        solver_mode: str = "auto",
    ):
        self.k = k
        self.alpha = alpha
        self.beta_min, self.beta_max = beta_range
        self.metta_min, self.metta_max = metta_range
        self.gamma = gamma
        self.query_weight = query_weight
        self.corpus_weight = corpus_weight
        self.sparse_k = sparse_k
        self.sa_iterations = sa_iterations
        self.solver_mode = solver_mode

        self._corpus_embeddings: np.ndarray | None = None
        self._J_static: sparse.csr_matrix | None = None

    def fit_offline(
        self, corpus_embeddings: np.ndarray, *, graph_k: int | None = None,
    ) -> IntentAwareRouter:
        """Offline: build static graph."""
        self._corpus_embeddings = corpus_embeddings
        gk = graph_k or self.sparse_k
        gk = min(gk, len(corpus_embeddings) - 1)
        self._J_static = build_sparse_impact_graph(
            corpus_embeddings, k=gk, use_hnswlib=True,
        )
        return self

    def route(
        self,
        query_embedding: np.ndarray,
        fetched_indices: np.ndarray,
        *,
        extra_terms: list[DharmaEnergyTerm] | None = None,
    ) -> tuple[RerankerResult, RouterDiagnosis]:
        """Auto-diagnose intent and rerank with optimal blend."""
        if self._corpus_embeddings is None:
            raise RuntimeError("Call fit_offline() first")

        idx = np.array([int(i) for i in fetched_indices])
        candidate_embeddings = self._corpus_embeddings[idx]

        relevance = DharmaReranker._compute_relevance(
            query_embedding, candidate_embeddings,
        )
        diag = DharmaReranker._diagnose_intent(relevance)
        c = diag["concentration"]

        beta = self.beta_min + (self.beta_max - self.beta_min) * c
        metta = self.metta_max - (self.metta_max - self.metta_min) * c

        reranker = DharmaReranker(
            k=self.k, alpha=self.alpha, beta=beta, gamma=self.gamma,
            query_weight=self.query_weight, corpus_weight=self.corpus_weight,
            diversity_weight=metta, J_static=self._J_static,
            sparse_k=self.sparse_k, sa_iterations=self.sa_iterations,
            solver_mode=self.solver_mode,
        )

        result = reranker.rerank(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_indices=idx,
            extra_terms=extra_terms,
        )

        diagnosis = RouterDiagnosis(
            intent=diag["intent"], concentration=c,
            entropy=diag["entropy"],
            beta_used=beta, metta_used=metta,
        )

        return result, diagnosis
