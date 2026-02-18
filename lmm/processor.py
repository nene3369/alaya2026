"""SmartProcessor — priority-tiered processing with caching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class ProcessedItem:
    index: int
    score: float
    result: object
    tier: str  # "critical" | "normal" | "background"


@dataclass
class ProcessingReport:
    items: list[ProcessedItem]
    n_critical: int
    n_normal: int
    n_background: int
    cache_hits: int

    @property
    def total(self) -> int:
        return len(self.items)

    def critical_items(self) -> list[ProcessedItem]:
        return [item for item in self.items if item.tier == "critical"]


class SmartProcessor:
    """Process selected items with priority tiering and caching."""

    def __init__(
        self,
        process_fn: Callable[[int, float], object] | None = None,
        critical_threshold: float = 0.8,
        batch_size: int = 16,
    ):
        self._process_fn = process_fn or self._default_process
        self._critical_threshold = critical_threshold
        self._batch_size = batch_size
        self._cache: dict[int, ProcessedItem] = {}

    def process(self, indices: np.ndarray, scores: np.ndarray) -> ProcessingReport:
        tiers = self._classify(scores)
        tier_order = {"critical": 0, "normal": 1, "background": 2}
        order = np.argsort([tier_order[t] for t in tiers])

        items: list[ProcessedItem] = []
        cache_hits = 0

        for idx in order:
            idx = int(idx)
            global_idx = int(indices[idx])
            score = float(scores[idx])
            tier = tiers[idx]

            if global_idx in self._cache:
                items.append(self._cache[global_idx])
                cache_hits += 1
                continue

            result = self._process_fn(global_idx, score)
            item = ProcessedItem(index=global_idx, score=score, result=result, tier=tier)
            items.append(item)
            self._cache[global_idx] = item

        n_crit = sum(1 for t in tiers if t == "critical")
        n_norm = sum(1 for t in tiers if t == "normal")
        n_bg = sum(1 for t in tiers if t == "background")
        return ProcessingReport(
            items=items, n_critical=n_crit, n_normal=n_norm,
            n_background=n_bg, cache_hits=cache_hits,
        )

    def process_batch(
        self, indices: np.ndarray, scores: np.ndarray,
        batch_fn: Callable[[np.ndarray, np.ndarray], list[object]] | None = None,
    ) -> ProcessingReport:
        tiers = self._classify(scores)
        items: list[ProcessedItem] = []
        cache_hits = 0

        for tier_name in ("critical", "normal", "background"):
            mask = np.array([t == tier_name for t in tiers])
            if not mask.any():
                continue
            tier_indices = indices[mask]
            tier_scores = scores[mask]

            for start in range(0, len(tier_indices), self._batch_size):
                end = min(start + self._batch_size, len(tier_indices))
                batch_idx = tier_indices[start:end]
                batch_scores = tier_scores[start:end]

                uncached_mask = np.array(
                    [int(i) not in self._cache for i in batch_idx],
                )
                if uncached_mask.any() and batch_fn is not None:
                    uc_idx = batch_idx[uncached_mask]
                    uc_scores = batch_scores[uncached_mask]
                    results = batch_fn(uc_idx, uc_scores)
                    for gi, sc, res in zip(uc_idx, uc_scores, results):
                        item = ProcessedItem(
                            index=int(gi), score=float(sc),
                            result=res, tier=tier_name,
                        )
                        self._cache[int(gi)] = item

                idx_pos = {int(v): pos for pos, v in enumerate(batch_idx)}
                for gi, sc in zip(batch_idx, batch_scores):
                    gi = int(gi)
                    if gi in self._cache:
                        items.append(self._cache[gi])
                        if not uncached_mask[idx_pos[gi]]:
                            cache_hits += 1
                    else:
                        result = self._process_fn(gi, float(sc))
                        item = ProcessedItem(
                            index=gi, score=float(sc), result=result, tier=tier_name,
                        )
                        items.append(item)
                        self._cache[gi] = item

        n_crit = sum(1 for t in tiers if t == "critical")
        n_norm = sum(1 for t in tiers if t == "normal")
        n_bg = sum(1 for t in tiers if t == "background")
        return ProcessingReport(
            items=items, n_critical=n_crit, n_normal=n_norm,
            n_background=n_bg, cache_hits=cache_hits,
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    def _classify(self, scores: np.ndarray) -> list[str]:
        if len(scores) == 0:
            return []
        max_s, min_s = scores.max(), scores.min()
        rng = max_s - min_s
        if rng < 1e-10:
            return ["normal"] * len(scores)
        normalised = (scores - min_s) / rng
        tiers: list[str] = []
        for s in normalised:
            if s >= self._critical_threshold:
                tiers.append("critical")
            elif s >= self._critical_threshold * 0.5:
                tiers.append("normal")
            else:
                tiers.append("background")
        return tiers

    @staticmethod
    def _default_process(index: int, score: float) -> dict:
        return {"index": index, "score": score, "processed": True}
