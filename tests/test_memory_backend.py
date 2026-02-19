"""Tests for lmm.memory — Persistent AlayaMemory backends (蔵識)."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np

from lmm.memory.backend import (
    InMemoryBackend,
    JsonFileBackend,
    StoredPattern,
)
from lmm.memory.persistent import PersistentAlayaMemory


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests: StoredPattern
# ---------------------------------------------------------------------------

class TestStoredPattern:
    def test_round_trip(self):
        sp = StoredPattern(
            pattern=np.array([1.0, 2.0, 3.0]),
            strength=0.8,
            access_count=5,
            timestamp=1000.0,
            metadata={"mode": "active"},
        )
        d = sp.to_dict()
        sp2 = StoredPattern.from_dict(d)
        for i in range(3):
            assert abs(float(sp2.pattern[i]) - float(sp.pattern[i])) < 1e-6
        assert sp2.strength == sp.strength
        assert sp2.access_count == sp.access_count


# ---------------------------------------------------------------------------
# Tests: InMemoryBackend
# ---------------------------------------------------------------------------

class TestInMemoryBackend:
    def test_store_and_recall(self):
        async def _test():
            backend = InMemoryBackend(max_patterns=100)
            sp = StoredPattern(
                pattern=np.array([1.0, 0.0, 0.0]),
                strength=1.0,
                access_count=0,
                timestamp=0.0,
                metadata={},
            )
            await backend.store(sp)
            assert await backend.count() == 1

            results = await backend.recall(np.array([1.0, 0.0, 0.0]), top_k=5)
            assert len(results) == 1
            for i in range(3):
                assert abs(float(results[0].pattern[i]) - float(sp.pattern[i])) < 1e-6
        _run(_test())

    def test_fifo_eviction(self):
        async def _test():
            backend = InMemoryBackend(max_patterns=3)
            for i in range(5):
                sp = StoredPattern(
                    pattern=np.array([float(i), 0.0, 0.0]),
                    strength=1.0,
                    access_count=0,
                    timestamp=float(i),
                    metadata={},
                )
                await backend.store(sp)
            assert await backend.count() == 3
        _run(_test())

    def test_decay(self):
        async def _test():
            backend = InMemoryBackend()
            sp = StoredPattern(
                pattern=np.array([1.0, 0.0]),
                strength=0.05,
                access_count=0,
                timestamp=0.0,
                metadata={},
            )
            await backend.store(sp)
            pruned = await backend.decay(0.9)
            assert pruned == 1
            assert await backend.count() == 0
        _run(_test())


# ---------------------------------------------------------------------------
# Tests: JsonFileBackend
# ---------------------------------------------------------------------------

class TestJsonFileBackend:
    def test_save_and_load(self):
        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "alaya_memory.json"
                backend = JsonFileBackend(path)

                sp = StoredPattern(
                    pattern=np.array([1.0, 2.0, 3.0]),
                    strength=0.9,
                    access_count=3,
                    timestamp=1000.0,
                    metadata={},
                )
                await backend.store(sp)
                await backend.save()

                backend2 = JsonFileBackend(path)
                await backend2.load()
                assert await backend2.count() == 1

                results = await backend2.recall(np.array([1.0, 2.0, 3.0]), top_k=5)
                assert len(results) == 1
        _run(_test())


# ---------------------------------------------------------------------------
# Tests: PersistentAlayaMemory
# ---------------------------------------------------------------------------

class TestPersistentAlayaMemory:
    def test_store_mirrors_to_backend(self):
        backend = InMemoryBackend()
        mem = PersistentAlayaMemory(
            n_variables=4,
            backend=backend,
            auto_save_interval=100,
        )
        mem.store(np.array([1.0, 0.5, -0.3, 0.8]))
        assert mem.n_patterns == 1

    def test_recall_works(self):
        mem = PersistentAlayaMemory(n_variables=4)
        mem.store(np.array([1.0, 0.0, 0.0, 0.0]))
        mem.store(np.array([0.0, 1.0, 0.0, 0.0]))
        result = mem.recall(np.array([1.0, 0.1, 0.0, 0.0]))
        assert result is not None
        assert len(result) == 4

    def test_warm_from_backend(self):
        async def _test():
            backend = InMemoryBackend()
            sp = StoredPattern(
                pattern=np.array([0.5, 0.5, 0.5, 0.5]),
                strength=0.9,
                access_count=2,
                timestamp=0.0,
                metadata={},
            )
            await backend.store(sp)

            mem = PersistentAlayaMemory(n_variables=4, backend=backend)
            loaded = await mem.warm_from_backend()
            assert loaded == 1
            assert mem.n_patterns == 1
        _run(_test())
