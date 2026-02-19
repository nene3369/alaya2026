"""PersistentAlayaMemory — drop-in replacement for AlayaMemory with backend storage.

Wraps the original AlayaMemory's Modern Hopfield recall logic while
delegating pattern persistence to a MemoryBackend.  This ensures 薫習
(vāsanā — accumulated experience seeds) survive process restarts.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

import numpy as np
from scipy import sparse

from lmm.reasoning.alaya import AlayaMemory, MemoryTrace
from lmm.memory.backend import (
    InMemoryBackend,
    MemoryBackend,
    StoredPattern,
)


class PersistentAlayaMemory(AlayaMemory):
    """AlayaMemory with pluggable persistent backend.

    Inherits the Modern Hopfield recall mechanism from AlayaMemory while
    adding backend-backed persistence.  All ``store()`` calls are mirrored
    to the backend; ``recall()`` still uses the fast in-memory softmax
    attention path but can be warmed from the backend on startup.

    Parameters
    ----------
    backend : MemoryBackend
        Persistent storage backend.
    auto_save_interval : int
        Trigger ``backend.save()`` every N stores.
    **kwargs
        Passed through to ``AlayaMemory.__init__``.
    """

    def __init__(
        self,
        n_variables: int,
        *,
        backend: MemoryBackend | None = None,
        auto_save_interval: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_variables, **kwargs)
        self.backend = backend or InMemoryBackend()
        self._auto_save_interval = auto_save_interval
        self._stores_since_save = 0

    def store(self, pattern: np.ndarray) -> None:
        """Store pattern in both in-memory Hopfield and persistent backend."""
        super().store(pattern)

        # Mirror to backend (fire-and-forget via event loop if available)
        stored = StoredPattern(
            pattern=np.asarray(pattern).flatten()[:self.n].copy(),
            strength=1.0,
            access_count=0,
            timestamp=time.time(),
            metadata={},
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.backend.store(stored))
        except RuntimeError:
            # No event loop — synchronous fallback
            asyncio.run(self.backend.store(stored))

        self._stores_since_save += 1
        if self._stores_since_save >= self._auto_save_interval:
            self._stores_since_save = 0
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.backend.save())
            except RuntimeError:
                asyncio.run(self.backend.save())

    async def warm_from_backend(self, limit: int = 200) -> int:
        """Load patterns from backend into in-memory Hopfield network.

        Called once at startup to restore 薫習 from persistent storage.
        Returns the number of patterns loaded.
        """
        await self.backend.load()
        count = await self.backend.count()
        if count == 0:
            return 0

        # Recall all patterns (using a zero vector to get everything)
        query = np.zeros(self.n)
        patterns = await self.backend.recall(query, top_k=limit)

        loaded = 0
        for sp in patterns:
            if len(sp.pattern) == self.n:
                # Bypass the backend mirror to avoid duplicates
                AlayaMemory.store(self, sp.pattern)
                # Restore strength
                if self._patterns:
                    self._patterns[-1].strength = sp.strength
                    self._patterns[-1].access_count = sp.access_count
                loaded += 1

        return loaded

    async def save(self) -> None:
        """Explicitly save all patterns to backend."""
        await self.backend.save()

    async def backend_count(self) -> int:
        """Return the number of patterns in the persistent backend."""
        return await self.backend.count()
