"""OutputReranker — LLM output surprise-diversity reranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.dharma.algorithms import build_sparse_impact_graph
from lmm.llm.embeddings import ngram_vectors as _ngram_vectors
from lmm.solvers import SubmodularSelector
from lmm.surprise import SurpriseCalculator


@dataclass
class RerankResult:
    """Reranking result."""

    indices: np.ndarray
    texts: list[str]
    surprises: np.ndarray
    diversity_score: float


class OutputReranker:
    """LLM output reranker — surprise x diversity selection."""

    def __init__(
        self,
        k: int = 1,
        alpha: float = 1.0,
        beta: float = 0.5,
        surprise_method: str = "entropy",
    ):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.surprise_method = surprise_method

    def rerank(
        self,
        candidates: list[str],
        reference_texts: list[str] | None = None,
        scores: np.ndarray | None = None,
        vectors: np.ndarray | None = None,
    ) -> RerankResult:
        """Rerank candidates by surprise and diversity."""
        n = len(candidates)
        k = min(self.k, n)

        if n == 0:
            return RerankResult(
                indices=np.array([], dtype=int), texts=[],
                surprises=np.array([], dtype=float), diversity_score=0.0,
            )

        cand_vecs = np.asarray(vectors) if vectors is not None else _ngram_vectors(candidates)
        calc = SurpriseCalculator(method=self.surprise_method)

        if reference_texts is not None:
            calc.fit(_ngram_vectors(reference_texts))
        else:
            calc.fit(cand_vecs)

        surprises = calc.compute(cand_vecs)
        combined = surprises.copy()

        if scores is not None:
            scores = np.asarray(scores)
            s_range = surprises.max() - surprises.min()
            sc_range = scores.max() - scores.min()
            if s_range > 1e-10 and sc_range > 1e-10:
                combined = (surprises - surprises.min()) / s_range + (scores - scores.min()) / sc_range

        if k >= n:
            order = np.argsort(-combined)
            return RerankResult(
                indices=order, texts=[candidates[int(i)] for i in order],
                surprises=surprises[order], diversity_score=float(combined.sum()),
            )

        graph_k = min(20, n - 1)
        impact_graph = build_sparse_impact_graph(cand_vecs, k=graph_k, use_hnswlib=False)
        selector = SubmodularSelector(alpha=self.alpha, beta=self.beta)
        result = selector.select(combined, impact_graph, k=k)
        idx = result.selected_indices

        return RerankResult(
            indices=idx, texts=[candidates[int(i)] for i in idx],
            surprises=surprises[idx], diversity_score=float(result.objective_value),
        )

    select = rerank
