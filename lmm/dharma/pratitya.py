"""Pratītyasamutpāda RAG — Dependent Origination retrieval-augmented generation.

Retrieval that combines vector similarity with causal graph structure from
``PatthanaEngine``, so that causally relevant documents are boosted over
merely semantically similar ones.

Scoring pipeline (three stages):
1. **Prajna** — dot-product vector similarity
2. **Karma boost** — +0.5 for causal ancestors in the Patthana graph
3. **Avidya filter** — -0.3 for documents with strongly negative karma
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from lmm.dharma.patthana import PatthanaEngine


@dataclass
class Bija:
    """A seed document: content, embedding, and karmic valence.

    Attributes:
        id: Unique identifier.
        content: Human-readable text content.
        embedding: Dense vector representation.
        karma_val: Karmic valence in [-1.0, 1.0].  Negative values indicate
            historically harmful / low-quality documents.
    """

    id: str
    content: str
    embedding: np.ndarray
    karma_val: float = 0.0


class PratityaRAG:
    """Dependent-Origination RAG: semantic similarity enriched by causal context.

    Parameters:
        patthana: Shared ``PatthanaEngine`` instance that carries the causal web.
    """

    def __init__(self, patthana: PatthanaEngine) -> None:
        self.patthana = patthana
        self.vector_store: Dict[str, Bija] = {}

    def add(self, bija: Bija) -> None:
        """Add a ``Bija`` document to the vector store."""
        self.vector_store[bija.id] = bija

    def retrieve(
        self,
        query_vec: np.ndarray,
        current_context_id: str,
        top_k: int = 5,
    ) -> List[Bija]:
        """Retrieve top-k Bijas using causal + semantic scoring.

        Parameters:
            query_vec: Query embedding (should be normalised for best results).
            current_context_id: Node ID in the Patthana graph representing the
                current context.  Causal ancestors of this node receive a boost.
            top_k: Number of documents to return.

        Returns:
            List of ``Bija`` objects ordered by descending composite score.
        """
        scores: Dict[str, float] = {}

        # Stage 1 — Prajna: vector similarity
        for bid, bija in self.vector_store.items():
            scores[bid] = float(np.dot(query_vec, bija.embedding))

        # Stage 2 — Karma: boost causal ancestors of the current context
        if current_context_id:
            for anc_id in self.patthana.graph.predecessors(current_context_id):
                if anc_id in scores:
                    scores[anc_id] += 0.5

        # Stage 3 — Avidya filter: penalise strongly negative karma
        for bid, bija in self.vector_store.items():
            if bija.karma_val < -0.5:
                scores[bid] -= 0.3

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.vector_store[sid] for sid, _ in sorted_ids if sid in self.vector_store]
