"""DharmaInterpreter — interpret QUBO solutions in Buddhist philosophy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DharmaInterpretation:
    """Dharma interpretation result."""

    selected_indices: np.ndarray
    prajna_score: float
    karuna_score: float
    sila_violation: float
    madhyamaka_cv: float
    sangha_strength: float
    narrative: str


class DharmaInterpreter:
    """Interpret optimization results through Buddhist philosophical lens."""

    def interpret(
        self,
        selected: np.ndarray,
        surprises: np.ndarray,
        impact_graph: np.ndarray | None = None,
        k: int = 10,
    ) -> DharmaInterpretation:
        """Interpret selection results."""
        sel_surprises = surprises[selected]

        prajna = float(sel_surprises.sum())

        mean = float(sel_surprises.mean()) if len(sel_surprises) > 0 else 0.0
        std = float(sel_surprises.std()) if len(sel_surprises) > 0 else 0.0
        cv = std / mean if mean > 1e-10 else 0.0

        sila = abs(len(selected) - k)

        karuna = 0.0
        sangha = 0.0
        if impact_graph is not None and len(selected) > 0:
            karuna, sangha = self._compute_karuna_sangha(
                selected, impact_graph, len(surprises),
            )

        narrative = self._generate_narrative(
            prajna, karuna, sila, cv, sangha, len(selected), k,
        )

        return DharmaInterpretation(
            selected_indices=selected,
            prajna_score=prajna,
            karuna_score=karuna,
            sila_violation=float(sila),
            madhyamaka_cv=cv,
            sangha_strength=sangha,
            narrative=narrative,
        )

    def _compute_karuna_sangha(
        self,
        selected: np.ndarray,
        impact_graph: np.ndarray,
        n: int,
    ) -> tuple[float, float]:
        """Compute karuna (compassion) and sangha (synergy) scores."""
        sel_set = set(int(i) for i in selected)
        karuna = 0.0
        sangha = 0.0

        for i in selected:
            i = int(i)
            if i >= impact_graph.shape[0] or impact_graph.ndim != 2:
                continue
            for j in range(min(n, impact_graph.shape[1])):
                w = float(impact_graph[i, j])
                if j not in sel_set:
                    karuna += w
                else:
                    sangha += w

        return karuna, sangha

    def _generate_narrative(
        self,
        prajna: float,
        karuna: float,
        sila: float,
        cv: float,
        sangha: float,
        n_selected: int,
        k: int,
    ) -> str:
        parts = []

        parts.append(f"Prajna (wisdom): {prajna:.2f}")
        if prajna > 0:
            parts.append("  Candidates with deep insight were selected")

        if karuna > 0:
            parts.append(f"Karuna (compassion): {karuna:.2f}")
            parts.append("  Selected items positively influence surroundings")

        if sangha > 0:
            parts.append(f"Sangha (community): {sangha:.2f}")
            parts.append("  Selected items harmonize with synergistic effect")

        parts.append(f"Madhyamaka (CV): {cv:.3f} (target: 0.500)")
        if abs(cv - 0.5) < 0.1:
            parts.append("  Edge of chaos reached — middle way realized")
        elif cv < 0.5:
            parts.append("  Somewhat uniform — increase wisdom diversity")
        else:
            parts.append("  Somewhat skewed — seek compassion balance")

        if sila == 0:
            parts.append(f"Sila: perfect compliance ({n_selected}/{k})")
        else:
            parts.append(f"Sila: deviation {sila:.0f} ({n_selected}/{k})")

        return "\n".join(parts)
