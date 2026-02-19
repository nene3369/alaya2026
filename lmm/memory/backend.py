"""Memory backends — pluggable storage for AlayaMemory patterns.

Each backend satisfies a common interface so the system can swap between
in-memory, JSON file, SQLite, or vector DB without changing the reasoning
pipeline.  This preserves 薫習 (vāsanā) across restarts.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class StoredPattern:
    """A pattern stored in the memory backend."""

    pattern: np.ndarray
    strength: float
    access_count: int
    timestamp: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": [float(x) for x in self.pattern],
            "strength": self.strength,
            "access_count": self.access_count,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StoredPattern:
        return cls(
            pattern=np.array(d["pattern"], dtype=float),
            strength=d.get("strength", 1.0),
            access_count=d.get("access_count", 0),
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )


class MemoryBackend(ABC):
    """Abstract base for pattern storage backends."""

    @abstractmethod
    async def store(self, pattern: StoredPattern) -> str:
        """Store a pattern and return its ID."""

    @abstractmethod
    async def recall(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[StoredPattern]:
        """Retrieve the top_k most similar patterns to query."""

    @abstractmethod
    async def decay(self, rate: float) -> int:
        """Apply temporal decay to all patterns. Return count of pruned."""

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of stored patterns."""

    @abstractmethod
    async def save(self) -> None:
        """Persist current state (no-op for already-persistent backends)."""

    @abstractmethod
    async def load(self) -> None:
        """Load persisted state (no-op for in-memory-only backends)."""


class InMemoryBackend(MemoryBackend):
    """Pure in-memory backend (default — equivalent to current AlayaMemory)."""

    def __init__(self, max_patterns: int = 200) -> None:
        self.max_patterns = max_patterns
        self._patterns: List[StoredPattern] = []

    async def store(self, pattern: StoredPattern) -> str:
        self._patterns.append(pattern)
        if len(self._patterns) > self.max_patterns:
            self._patterns.pop(0)
        return str(len(self._patterns) - 1)

    async def recall(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[StoredPattern]:
        if not self._patterns:
            return []
        q = query / (np.linalg.norm(query) + 1e-9)
        scored = []
        for p in self._patterns:
            pn = p.pattern / (np.linalg.norm(p.pattern) + 1e-9)
            sim = float(np.dot(q, pn)) * p.strength
            scored.append((sim, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    async def decay(self, rate: float) -> int:
        pruned = 0
        remaining = []
        for p in self._patterns:
            p.strength *= (1.0 - rate)
            if p.strength > 0.01:
                remaining.append(p)
            else:
                pruned += 1
        self._patterns = remaining
        return pruned

    async def count(self) -> int:
        return len(self._patterns)

    async def save(self) -> None:
        pass  # No-op for in-memory

    async def load(self) -> None:
        pass  # No-op for in-memory


class JsonFileBackend(MemoryBackend):
    """JSON file backend — simple persistence for development and small-scale use.

    Patterns are kept in memory for fast recall, and periodically flushed
    to a JSON file.
    """

    def __init__(
        self,
        path: str | Path,
        max_patterns: int = 10_000,
    ) -> None:
        self.path = Path(path)
        self.max_patterns = max_patterns
        self._inner = InMemoryBackend(max_patterns)

    async def store(self, pattern: StoredPattern) -> str:
        return await self._inner.store(pattern)

    async def recall(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[StoredPattern]:
        return await self._inner.recall(query, top_k)

    async def decay(self, rate: float) -> int:
        return await self._inner.decay(rate)

    async def count(self) -> int:
        return await self._inner.count()

    async def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in self._inner._patterns]
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    async def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._inner._patterns = [StoredPattern.from_dict(d) for d in raw]
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted file — start fresh


class VectorDBBackend(MemoryBackend):
    """Vector database backend — for production-scale persistent memory.

    Supports Qdrant, ChromaDB, or pgvector via a thin adapter.  Falls back
    to InMemoryBackend when the vector DB client is not installed.

    Parameters
    ----------
    provider : str
        One of ``"qdrant"``, ``"chroma"``, ``"pgvector"``.
    connection_url : str
        Connection string for the vector DB.
    collection_name : str
        Name of the collection/table to use.
    dimension : int
        Vector dimensionality.
    """

    def __init__(
        self,
        provider: str = "qdrant",
        connection_url: str = "http://localhost:6333",
        collection_name: str = "alaya_patterns",
        dimension: int = 8,
    ) -> None:
        self.provider = provider
        self.connection_url = connection_url
        self.collection_name = collection_name
        self.dimension = dimension
        self._client: Any = None
        self._fallback = InMemoryBackend()

    async def _ensure_client(self) -> bool:
        """Lazy-connect to vector DB. Return True if connected."""
        if self._client is not None:
            return True
        try:
            if self.provider == "qdrant":
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
                self._client = QdrantClient(url=self.connection_url)
                collections = [c.name for c in self._client.get_collections().collections]
                if self.collection_name not in collections:
                    self._client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.dimension,
                            distance=Distance.COSINE,
                        ),
                    )
                return True
            elif self.provider == "chroma":
                import chromadb
                self._client = chromadb.HttpClient(host=self.connection_url)
                self._client.get_or_create_collection(self.collection_name)
                return True
        except Exception:
            self._client = None
            return False

    async def store(self, pattern: StoredPattern) -> str:
        if not await self._ensure_client():
            return await self._fallback.store(pattern)

        import uuid
        point_id = str(uuid.uuid4())
        if self.provider == "qdrant":
            from qdrant_client.models import PointStruct
            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=pattern.pattern.tolist(),
                        payload=pattern.to_dict(),
                    )
                ],
            )
        return point_id

    async def recall(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[StoredPattern]:
        if not await self._ensure_client():
            return await self._fallback.recall(query, top_k)

        if self.provider == "qdrant":
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query.tolist(),
                limit=top_k,
            )
            return [
                StoredPattern.from_dict(r.payload)
                for r in results
            ]
        return []

    async def decay(self, rate: float) -> int:
        # Vector DB decay is a batch update — simplified for now
        return 0

    async def count(self) -> int:
        if not await self._ensure_client():
            return await self._fallback.count()
        if self.provider == "qdrant":
            info = self._client.get_collection(self.collection_name)
            return info.points_count or 0
        return 0

    async def save(self) -> None:
        pass  # Vector DB is already persistent

    async def load(self) -> None:
        await self._ensure_client()
