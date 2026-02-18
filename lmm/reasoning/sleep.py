"""SleepConsolidation — NREM/REM memory distillation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmm.reasoning.alaya import AlayaMemory


@dataclass
class SleepReport:
    """Report from a sleep consolidation cycle."""

    n_consolidated: int
    n_pruned: int
    replay_cycles: int
    pre_strength: float
    post_strength: float


class SleepConsolidation:
    """Sleep-inspired memory consolidation.

    Two phases:
      1. NREM: Replay and strengthen important patterns
      2. REM: Creative recombination and pruning
    """

    def __init__(
        self,
        memory: AlayaMemory,
        *,
        nrem_replay_cycles: int = 5,
        rem_noise_scale: float = 0.1,
        pruning_threshold: float = 0.05,
    ):
        self.memory = memory
        self.nrem_replay_cycles = nrem_replay_cycles
        self.rem_noise_scale = rem_noise_scale
        self.pruning_threshold = pruning_threshold
        self._rng = np.random.default_rng(42)

    def consolidate(self) -> SleepReport:
        """Run one full sleep cycle (NREM + REM)."""
        pre_strength = self.memory.total_strength

        # NREM: replay stored patterns to strengthen connections
        for _ in range(self.nrem_replay_cycles):
            for trace in self.memory._patterns:
                if trace.strength > self.pruning_threshold:
                    # Replay with reduced learning rate
                    self.memory.store(trace.pattern)
                    trace.access_count += 1

        # REM: creative recombination + noise injection
        if self.memory.n_patterns >= 2:
            patterns = self.memory._patterns
            n_recombine = min(3, len(patterns) // 2)
            for _ in range(n_recombine):
                idxs = self._rng.choice(len(patterns), size=2, replace=False)
                idx1, idx2 = int(idxs[0]), int(idxs[1])
                p1 = patterns[idx1].pattern
                p2 = patterns[idx2].pattern
                # Blend patterns with noise
                blended = 0.5 * p1 + 0.5 * p2
                noise = self._rng.normal(0, self.rem_noise_scale, size=len(blended))
                blended = np.clip(blended + noise, 0, 1)
                self.memory.store(blended)

        # Pruning: remove weak patterns
        n_pruned = 0
        remaining = []
        for trace in self.memory._patterns:
            if trace.strength < self.pruning_threshold:
                n_pruned += 1
            else:
                remaining.append(trace)
        self.memory._patterns = remaining

        # Apply global decay
        self.memory.decay()

        return SleepReport(
            n_consolidated=self.memory.n_patterns,
            n_pruned=n_pruned,
            replay_cycles=self.nrem_replay_cycles,
            pre_strength=pre_strength,
            post_strength=self.memory.total_strength,
        )
