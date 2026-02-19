"""Persistent Memory (蔵識・Alaya) — Scalable long-term memory backends.

Abstracts the storage layer so that AlayaMemory can persist patterns to
a Vector DB (Qdrant, pgvector, ChromaDB) or fall back to JSON/SQLite,
ensuring 薫習 (accumulated experience) survives restarts.
"""

from lmm.memory.backend import MemoryBackend, InMemoryBackend, JsonFileBackend
from lmm.memory.persistent import PersistentAlayaMemory

__all__ = [
    "MemoryBackend",
    "InMemoryBackend",
    "JsonFileBackend",
    "PersistentAlayaMemory",
]
