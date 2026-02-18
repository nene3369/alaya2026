"""LlamaIndex integration — DharmaNodePostprocessor.

Diversity-aware node postprocessor implementing LlamaIndex BaseNodePostprocessor.
Uses submodular optimization for (1-1/e) approximation guarantee, or
FEP analog neuromorphic KCL ODE solver for deterministic convergence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lmm.dharma.algorithms import build_sparse_impact_graph
from lmm.dharma.reranker import DharmaReranker, IntentAwareRouter
from lmm.llm.embeddings import cosine_similarity as _cosine_similarity
from lmm.llm.embeddings import ngram_vectors as _ngram_vectors
from lmm.solvers import SubmodularSelector

if TYPE_CHECKING:
    pass

_LLAMAINDEX_AVAILABLE = False

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor as _LIPostprocessor  # noqa: F401
    from llama_index.core.schema import NodeWithScore, QueryBundle  # noqa: F401

    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    pass


def _check_llamaindex() -> None:
    if not _LLAMAINDEX_AVAILABLE:
        raise ImportError(
            "llama-index-core is required for DharmaNodePostprocessor. "
            "Install it with: pip install llama-index-core"
        )


# ---------------------------------------------------------------------------
# Text extraction helper
# ---------------------------------------------------------------------------

def _extract_texts(nodes: list[Any]) -> list[str]:
    """Extract text from node-like objects."""
    texts = []
    for node in nodes:
        if hasattr(node, "get_text"):
            texts.append(node.get_text())
        elif hasattr(node, "text"):
            texts.append(node.text)
        elif hasattr(node, "node") and hasattr(node.node, "get_text"):
            texts.append(node.node.get_text())
        else:
            texts.append(str(node))
    return texts


def _extract_scores(nodes: list[Any]) -> np.ndarray:
    """Extract scores from nodes, defaulting to 1.0."""
    scores = []
    for node in nodes:
        if hasattr(node, "score") and node.score is not None:
            scores.append(float(node.score))
        else:
            scores.append(1.0)
    return np.array(scores)


# ---------------------------------------------------------------------------
# Core reranking logic (no LlamaIndex dependency)
# ---------------------------------------------------------------------------

def rerank_nodes_by_diversity(
    nodes: list[Any],
    query_str: str | None = None,
    top_k: int = 10,
    alpha: float = 1.0,
    beta: float = 0.5,
    sparse_k: int = 20,
) -> list[Any]:
    """Diversity-aware node reranking using submodular optimization.

    Works with any node-like objects that have ``get_text()`` or ``text``.
    """
    n = len(nodes)
    top_k = min(top_k, n)
    if top_k >= n:
        return list(nodes)

    texts = _extract_texts(nodes)
    vecs = _ngram_vectors(texts)

    if query_str is not None:
        query_vec = _ngram_vectors([query_str])[0]
        relevance = np.clip(_cosine_similarity(vecs, query_vec), 0.0, None)
    else:
        relevance = _extract_scores(nodes)

    graph_k = min(sparse_k, n - 1)
    impact_graph = build_sparse_impact_graph(vecs, k=graph_k, use_hnswlib=True)

    selector = SubmodularSelector(alpha=alpha, beta=beta)
    result = selector.select(relevance, impact_graph, k=top_k)

    return [nodes[i] for i in result.selected_indices]


def rerank_nodes_by_diversity_adaptive(
    nodes: list[Any],
    query_str: str | None = None,
    k_max: int = 50,
    alpha: float = 1.0,
    beta: float = 0.5,
    sparse_k: int = 20,
) -> list[Any]:
    """Adaptive diversity-aware node reranking with auto K selection.

    Uses marginal gain 1/e decay to determine cutoff.
    """
    n = len(nodes)
    k_max = min(k_max, n)
    if k_max >= n:
        return list(nodes)

    texts = _extract_texts(nodes)
    vecs = _ngram_vectors(texts)

    if query_str is not None:
        query_vec = _ngram_vectors([query_str])[0]
        relevance = np.clip(_cosine_similarity(vecs, query_vec), 0.0, None)
    else:
        relevance = _extract_scores(nodes)

    graph_k = min(sparse_k, n - 1)
    impact_graph = build_sparse_impact_graph(vecs, k=graph_k, use_hnswlib=True)

    selector = SubmodularSelector(alpha=alpha, beta=beta)
    result = selector.select_adaptive(relevance, impact_graph, k_max=k_max)

    return [nodes[i] for i in result.selected_indices]


def rerank_nodes_by_dharma_engine(
    nodes: list[Any],
    query_str: str | None = None,
    top_k: int = 10,
    *,
    alpha: float = 1.0,
    beta_range: tuple[float, float] = (0.1, 0.8),
    metta_range: tuple[float, float] = (0.0, 1.0),
    gamma: float = 10.0,
    sparse_k: int = 20,
    solver_mode: str = "fep_analog",
) -> list[Any]:
    """Full Dharma pipeline: IntentAwareRouter + UniversalDharmaEngine."""
    n = len(nodes)
    top_k = min(top_k, n)
    if top_k >= n:
        return list(nodes)

    texts = _extract_texts(nodes)
    vecs = _ngram_vectors(texts)

    graph_k = min(sparse_k, n - 1)
    J_static = build_sparse_impact_graph(vecs, k=graph_k, use_hnswlib=True)

    if query_str is not None:
        query_vec = _ngram_vectors([query_str])[0]
        router = IntentAwareRouter(
            k=top_k,
            alpha=alpha,
            beta_range=beta_range,
            metta_range=metta_range,
            gamma=gamma,
            sparse_k=sparse_k,
            solver_mode=solver_mode,
        )
        router._corpus_embeddings = vecs
        router._J_static = J_static

        fetched_indices = np.arange(n)
        result, _diagnosis = router.route(query_vec, fetched_indices)
        selected_idx = result.selected_indices
    else:
        reranker = DharmaReranker(
            k=top_k,
            alpha=alpha,
            beta=sum(beta_range) / 2,
            gamma=gamma,
            diversity_weight=sum(metta_range) / 2,
            J_static=J_static,
            sparse_k=sparse_k,
            solver_mode=solver_mode,
        )
        centroid = vecs.mean(axis=0) if n > 0 else np.zeros(vecs.shape[1])
        result = reranker.rerank(
            query_embedding=centroid,
            candidate_embeddings=vecs,
        )
        selected_idx = result.selected_indices

    return [nodes[int(i)] for i in selected_idx]


# ---------------------------------------------------------------------------
# DharmaNodePostprocessor (LlamaIndex integration)
# ---------------------------------------------------------------------------

def _build_dharma_postprocessor_class() -> type:
    """Dynamically build DharmaNodePostprocessor with BaseNodePostprocessor parent."""
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

    class _DharmaNodePostprocessorImpl(BaseNodePostprocessor):
        """Diversity-aware node postprocessor.

        Pipelines:
          1. ``solver_mode="submodular"`` (default): (1-1/e) approximation
          2. ``solver_mode="fep_analog"``: KCL ODE neuromorphic solver
        """

        top_k: int = 10
        alpha: float = 1.0
        beta: float = 0.5
        solver_mode: str = "submodular"

        def _postprocess_nodes(
            self,
            nodes: list[NodeWithScore],
            query_bundle: QueryBundle | None = None,
        ) -> list[NodeWithScore]:
            if len(nodes) <= self.top_k:
                return nodes

            texts = [n.node.get_text() for n in nodes]
            vecs = _ngram_vectors(texts)
            n_nodes = len(nodes)

            if query_bundle is not None and query_bundle.query_str:
                query_vec = _ngram_vectors([query_bundle.query_str])[0]
                relevance = np.clip(_cosine_similarity(vecs, query_vec), 0.0, None)
            else:
                relevance = np.array([
                    n.score if n.score is not None else 1.0 for n in nodes
                ])
                query_vec = None

            if self.solver_mode == "fep_analog" and query_vec is not None:
                graph_k = min(20, n_nodes - 1)
                J_static = build_sparse_impact_graph(
                    vecs, k=graph_k, use_hnswlib=True,
                )
                router = IntentAwareRouter(
                    k=self.top_k,
                    alpha=self.alpha,
                    sparse_k=graph_k,
                    solver_mode="fep_analog",
                )
                router._corpus_embeddings = vecs
                router._J_static = J_static

                result, _diagnosis = router.route(query_vec, np.arange(n_nodes))
                selected_indices = result.selected_indices
                method = "fep_kcl_analog"
            else:
                graph_k = min(20, n_nodes - 1)
                impact_graph = build_sparse_impact_graph(
                    vecs, k=graph_k, use_hnswlib=True,
                )
                selector = SubmodularSelector(alpha=self.alpha, beta=self.beta)
                result = selector.select(relevance, impact_graph, k=self.top_k)
                selected_indices = result.selected_indices
                method = "submodular_greedy"

            reranked = []
            for rank, idx in enumerate(selected_indices):
                node_with_score = nodes[idx]
                node_with_score.node.metadata = dict(node_with_score.node.metadata or {})
                node_with_score.node.metadata["dharma_rank"] = rank
                node_with_score.node.metadata["dharma_relevance"] = float(relevance[idx])
                node_with_score.node.metadata["dharma_method"] = method
                reranked.append(node_with_score)

            return reranked

    return _DharmaNodePostprocessorImpl


_DharmaNodePostprocessorCls: type | None = None


def get_dharma_postprocessor_class() -> type:
    """Get the DharmaNodePostprocessor class (requires llama-index-core)."""
    global _DharmaNodePostprocessorCls
    _check_llamaindex()
    if _DharmaNodePostprocessorCls is None:
        _DharmaNodePostprocessorCls = _build_dharma_postprocessor_class()
    return _DharmaNodePostprocessorCls


def DharmaNodePostprocessor(**kwargs: Any) -> Any:
    """Create a DharmaNodePostprocessor instance.

    Convenience constructor that handles the lazy import.
    """
    cls = get_dharma_postprocessor_class()
    return cls(**kwargs)
