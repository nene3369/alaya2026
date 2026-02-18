"""LangChain integration — DharmaRetriever, DharmaDocumentCompressor, DharmaExampleSelector.

Diversity-aware components implementing LangChain base classes.
Uses submodular optimization for (1-1/e) approximation guarantee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lmm.dharma.algorithms import build_sparse_impact_graph
from lmm.dharma.reranker import DharmaReranker
from lmm.llm.embeddings import cosine_similarity as _cosine_similarity
from lmm.llm.embeddings import ngram_vectors as _ngram_vectors
from lmm.solvers import SubmodularSelector

if TYPE_CHECKING:
    pass

_LANGCHAIN_AVAILABLE = False
_BaseRetriever: type | None = None

try:
    from langchain_core.retrievers import BaseRetriever as _LCBaseRetriever
    _BaseRetriever = _LCBaseRetriever
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


def _check_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required. Install: pip install langchain-core"
        )


# ---------------------------------------------------------------------------
# Core reranking logic (no LangChain dependency)
# ---------------------------------------------------------------------------

def rerank_by_diversity(
    relevance_scores: np.ndarray,
    embeddings_matrix: np.ndarray,
    k: int = 10,
    alpha: float = 1.0,
    beta: float = 0.5,
    sparse_k: int = 20,
) -> np.ndarray:
    """Diversity-aware reranking using submodular optimization."""
    n = len(relevance_scores)
    k = min(k, n)
    if k >= n:
        return np.arange(n)
    graph_k = min(sparse_k, n - 1)
    impact_graph = build_sparse_impact_graph(
        embeddings_matrix, k=graph_k, use_hnswlib=True,
    )
    selector = SubmodularSelector(alpha=alpha, beta=beta)
    return selector.select(relevance_scores, impact_graph, k=k).selected_indices


def rerank_by_diversity_adaptive(
    relevance_scores: np.ndarray,
    embeddings_matrix: np.ndarray,
    query_embedding: np.ndarray,
    k: int = 10,
    alpha: float = 1.0,
    beta_range: tuple[float, float] = (0.1, 0.8),
    sparse_k: int = 20,
) -> np.ndarray:
    """Adaptive diversity-aware reranking."""
    n = len(relevance_scores)
    k = min(k, n)
    if k >= n:
        return np.arange(n)

    # Entropic concentration -> adaptive beta
    r = np.asarray(relevance_scores).flatten()
    total = float(r.sum())
    if total > 1e-10 and n > 1:
        p = r / total
        mask = p > 1e-15
        entropy = -float(np.sum(p[mask] * np.log(p[mask])))
        c = max(0.0, min(1.0, 1.0 - entropy / float(np.log(float(n)))))
    else:
        c = 0.0

    beta = beta_range[0] + (beta_range[1] - beta_range[0]) * (1.0 - c)
    return rerank_by_diversity(
        relevance_scores, embeddings_matrix, k=k, alpha=alpha, beta=beta,
        sparse_k=sparse_k,
    )


def rerank_by_dharma_engine(
    relevance_scores: np.ndarray,
    embeddings_matrix: np.ndarray,
    query_embedding: np.ndarray,
    k: int = 10,
    solver_mode: str = "auto",
) -> np.ndarray:
    """Full Dharma engine reranking."""
    n = len(relevance_scores)
    k = min(k, n)
    if k >= n:
        return np.arange(n)

    reranker = DharmaReranker(k=k, solver_mode=solver_mode)
    result = reranker.rerank(
        query_embedding=query_embedding,
        candidate_embeddings=embeddings_matrix,
    )
    return result.selected_indices


# ---------------------------------------------------------------------------
# LangChain components
# ---------------------------------------------------------------------------

class DharmaDocumentCompressor:
    """LangChain BaseDocumentCompressor implementation.

    Works with ContextualCompressionRetriever for diversity-aware retrieval.
    """

    def __init__(
        self,
        k: int = 10,
        alpha: float = 1.0,
        beta: float = 0.5,
        sparse_k: int = 20,
        solver_mode: str = "submodular",
        embeddings: Any = None,
    ):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.sparse_k = sparse_k
        self.solver_mode = solver_mode
        self.embeddings = embeddings

    def compress_documents(
        self,
        documents: list,
        query: str,
        callbacks: Any = None,
    ) -> list:
        """Compress (rerank) documents for diversity + relevance."""
        if not documents:
            return []

        n = len(documents)
        k = min(self.k, n)
        if k >= n:
            return documents

        # Vectorize
        if self.embeddings is not None:
            texts = [doc.page_content for doc in documents]
            doc_vecs = np.array(self.embeddings.embed_documents(texts))
            query_vec = np.array(self.embeddings.embed_query(query))
        else:
            texts = [doc.page_content for doc in documents]
            doc_vecs = _ngram_vectors(texts)
            query_vec = _ngram_vectors([query])[0]

        # Relevance scores
        relevance = np.clip(_cosine_similarity(doc_vecs, query_vec), 0.0, None)

        # Rerank
        if self.solver_mode == "fep_analog":
            indices = rerank_by_dharma_engine(
                relevance, doc_vecs, query_vec, k=k, solver_mode="fep_analog",
            )
        else:
            indices = rerank_by_diversity(
                relevance, doc_vecs, k=k,
                alpha=self.alpha, beta=self.beta, sparse_k=self.sparse_k,
            )

        return [documents[int(i)] for i in indices]


class DharmaExampleSelector:
    """LangChain BaseExampleSelector implementation."""

    def __init__(
        self,
        examples: list[dict] | None = None,
        k: int = 3,
        alpha: float = 1.0,
        beta: float = 0.5,
        input_keys: list[str] | None = None,
    ):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.input_keys = input_keys or ["input"]
        self.examples: list[dict] = []
        self._vectors: np.ndarray | None = None

        if examples:
            for ex in examples:
                self.add_example(ex)

    def add_example(self, example: dict) -> None:
        """Add an example to the pool."""
        self.examples.append(example)
        self._vectors = None  # invalidate cache

    def _ensure_vectors(self) -> np.ndarray:
        if self._vectors is None:
            texts = [self._to_text(ex) for ex in self.examples]
            self._vectors = _ngram_vectors(texts)
        return self._vectors

    def select_examples(self, input_variables: dict) -> list[dict]:
        """Select k examples most relevant to input."""
        if not self.examples:
            return []

        vectors = self._ensure_vectors()
        n = len(self.examples)
        k = min(self.k, n)

        query_text = " ".join(str(input_variables.get(key, "")) for key in self.input_keys)
        query_vec = _ngram_vectors([query_text])[0]
        relevance = np.clip(_cosine_similarity(vectors, query_vec), 0.0, None)

        if k >= n:
            order = np.argsort(-relevance)
            return [self.examples[int(i)] for i in order]

        indices = rerank_by_diversity(relevance, vectors, k=k, alpha=self.alpha, beta=self.beta)
        return [self.examples[int(i)] for i in indices]

    def _to_text(self, example: dict) -> str:
        parts = []
        for key in self.input_keys:
            if key in example:
                parts.append(str(example[key]))
        if not parts:
            parts = [str(v) for v in example.values()]
        return " ".join(parts)


class DharmaRetriever:
    """LangChain retriever wrapper with diversity-aware reranking."""

    def __init__(
        self,
        base_retriever: Any = None,
        k: int = 10,
        alpha: float = 1.0,
        beta: float = 0.5,
        embeddings: Any = None,
        solver_mode: str = "submodular",
    ):
        self.base_retriever = base_retriever
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.embeddings = embeddings
        self.solver_mode = solver_mode

    def invoke(self, query: str, **kwargs) -> list:
        """Retrieve and rerank documents."""
        if self.base_retriever is None:
            raise RuntimeError("base_retriever is required")

        docs = self.base_retriever.invoke(query, **kwargs)
        if len(docs) <= self.k:
            return docs

        compressor = DharmaDocumentCompressor(
            k=self.k, alpha=self.alpha, beta=self.beta,
            embeddings=self.embeddings, solver_mode=self.solver_mode,
        )
        return compressor.compress_documents(docs, query)

    def get_relevant_documents(self, query: str, **kwargs) -> list:
        """Legacy API compatibility."""
        return self.invoke(query, **kwargs)
