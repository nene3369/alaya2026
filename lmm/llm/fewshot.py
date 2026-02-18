"""FewShotSelector — information-theoretic few-shot example selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.dharma.algorithms import build_sparse_impact_graph
from lmm.llm.embeddings import cosine_similarity as _cosine_similarity
from lmm.llm.embeddings import ngram_vectors as _ngram_vectors
from lmm.solvers import SubmodularSelector


@dataclass
class FewShotResult:
    """Few-shot selection result."""

    indices: np.ndarray
    examples: list[dict]
    scores: np.ndarray


class FewShotSelector:
    """Information-theoretic few-shot example selection.

    Uses submodular optimization for relevance + diversity.
    """

    def __init__(self, k: int = 3, alpha: float = 1.0, beta: float = 0.5):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.examples_: list[dict] = []
        self.vectors_: np.ndarray | None = None

    def fit(
        self, examples: list[dict], vectors: np.ndarray | None = None,
    ) -> FewShotSelector:
        """Register example pool."""
        self.examples_ = list(examples)
        if vectors is not None:
            self.vectors_ = np.asarray(vectors)
        else:
            texts = [self._example_to_text(ex) for ex in examples]
            self.vectors_ = _ngram_vectors(texts)
        return self

    def select(self, query: str | np.ndarray) -> FewShotResult:
        """Select k best few-shot examples for query."""
        if self.vectors_ is None or not self.examples_:
            raise RuntimeError("Call fit() first")

        n = len(self.examples_)
        k = min(self.k, n)

        if isinstance(query, str):
            query_vec = _ngram_vectors([query])[0]
        else:
            query_vec = np.asarray(query)

        relevance = np.clip(_cosine_similarity(self.vectors_, query_vec), 0.0, None)

        if k >= n:
            order = np.argsort(-relevance)
            return FewShotResult(
                indices=order,
                examples=[self.examples_[int(i)] for i in order],
                scores=relevance[order],
            )

        graph_k = min(20, n - 1)
        impact_graph = build_sparse_impact_graph(
            self.vectors_, k=graph_k, use_hnswlib=False,
        )
        selector = SubmodularSelector(alpha=self.alpha, beta=self.beta)
        result = selector.select(relevance, impact_graph, k=k)
        idx = result.selected_indices

        return FewShotResult(
            indices=idx,
            examples=[self.examples_[int(i)] for i in idx],
            scores=relevance[idx],
        )

    @staticmethod
    def _example_to_text(example: dict) -> str:
        parts = []
        for key in ("input", "question", "text", "prompt"):
            if key in example:
                parts.append(str(example[key]))
        for key in ("output", "answer", "response", "completion"):
            if key in example:
                parts.append(str(example[key]))
        if not parts:
            parts = [str(v) for v in example.values()]
        return " ".join(parts)
