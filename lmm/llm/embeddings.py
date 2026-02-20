"""Embedding adapters — ngram / SentenceTransformer / custom.

Provides unified embedding interface with auto-detection.
Includes inlined ngram_vectors from the former _vectorize.py.
"""

from __future__ import annotations

import zlib
from typing import Callable

import numpy as np

from lmm.rust_bridge import run_ngram_vectors


# ---------------------------------------------------------------------------
# Inline vectorization (was _vectorize.py)
# ---------------------------------------------------------------------------

def ngram_vectors(texts: list[str], max_features: int = 500) -> np.ndarray:
    """Character n-gram (3-5) hashing with deterministic CRC32."""

    def _fallback():
        n = len(texts)
        vectors = np.zeros((n, max_features), dtype=np.float64)
        _crc32 = zlib.crc32

        for i, text in enumerate(texts):
            raw = text.lower().encode("utf-8")
            mv = memoryview(raw)
            text_len = len(mv)

            indices: list[int] = []
            for ng in (3, 4, 5):
                end = text_len - ng + 1
                for j in range(end):
                    indices.append(_crc32(mv[j:j + ng]) % max_features)

            if indices:
                buckets = [0] * max_features
                for idx in indices:
                    buckets[idx] += 1
                for k in range(max_features):
                    if buckets[k]:
                        vectors[i, k] = float(buckets[k])

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        return vectors / norms

    result = run_ngram_vectors(texts, max_features, _fallback=_fallback)
    return result if result is not None else _fallback()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity: (n, d) vs (d,) -> (n,)."""
    dots = a @ b
    a_norms = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b)
    denom = np.clip(a_norms * b_norm, 1e-10, None)
    return dots / denom


# ---------------------------------------------------------------------------
# SentenceTransformer lazy detection
# ---------------------------------------------------------------------------

_HAS_SENTENCE_TRANSFORMERS: bool | None = None
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _probe_sentence_transformers() -> bool:
    global _HAS_SENTENCE_TRANSFORMERS
    if _HAS_SENTENCE_TRANSFORMERS is None:
        try:
            import sentence_transformers  # noqa: F401
            _HAS_SENTENCE_TRANSFORMERS = True
        except ImportError:
            _HAS_SENTENCE_TRANSFORMERS = False
    return _HAS_SENTENCE_TRANSFORMERS


class EmbeddingAdapter:
    """Unified embedding adapter.

    Backends: "auto", "ngram", "sentence_transformer", "custom"
    """

    def __init__(
        self,
        backend: str = "auto",
        *,
        model_name: str = _DEFAULT_MODEL_NAME,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        max_features: int = 500,
    ):
        self.backend = backend
        self.model_name = model_name
        self.embed_fn = embed_fn
        self.max_features = max_features
        self._st_model = None

        if backend == "custom" and embed_fn is None:
            raise ValueError("backend='custom' requires embed_fn")

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if _probe_sentence_transformers():
            return "sentence_transformer"
        return "ngram"

    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
        return self._st_model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts to (n, d) matrix (L2-normalized)."""
        if not texts:
            return np.zeros((0, self.max_features))

        backend = self._resolve_backend()

        if backend == "sentence_transformer":
            model = self._get_st_model()
            vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float64)
        elif backend == "custom":
            vectors = self.embed_fn(texts)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / np.clip(norms, 1e-10, None)
        else:
            return ngram_vectors(texts, max_features=self.max_features)

    @property
    def is_semantic(self) -> bool:
        return self._resolve_backend() in ("sentence_transformer", "custom")


def auto_embed(texts: list[str], **kwargs) -> np.ndarray:
    """Auto-embed using best available backend."""
    return EmbeddingAdapter(backend="auto", **kwargs).embed(texts)
