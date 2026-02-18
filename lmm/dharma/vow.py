"""Vow Constraint Engine (誓願エンジン) — dynamic Logit shaping via FEP state.

A ``Vow`` is a run-time directive that shapes LLM output probability
distributions by specifying tokens to suppress or amplify and a temperature
multiplier.  It is forged dynamically from the current Free Energy Principle
(FEP) state and the retrieved karma context.

Two archetypes:
- **Abhaya** (無畏施 — Fearlessness): activated when FEP > 0.7 (high confusion /
  Dukkha).  Suppresses panic language and boosts calm, step-by-step guidance.
- **Desana** (説法 — Teaching): activated when FEP ≤ 0.7 (low confusion / flow).
  Suppresses hedging language and boosts logical, structured expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Vow:
    """A run-time constraint directive for LLM output shaping.

    Attributes:
        name: Human-readable name of the vow.
        target_fep_state: Desired cognitive state after the vow is applied.
        suppress_tokens: Tokens / phrases whose logits should be decreased.
        boost_tokens: Tokens / phrases whose logits should be increased.
        temperature_mod: Multiplicative modifier for sampling temperature
            (< 1.0 → more deterministic, > 1.0 → more creative).
    """

    name: str
    target_fep_state: str
    suppress_tokens: List[str]
    boost_tokens: List[str]
    temperature_mod: float = 1.0


class VowConstraintEngine:
    """Forges ``Vow`` directives from FEP state and karma history.

    Usage::

        engine = VowConstraintEngine()
        vow = engine.forge_vow(current_fep=0.8, karma_context=[])
        # → Vow(name='Abhaya (Fearlessness)', temperature_mod=0.5, ...)
    """

    def forge_vow(self, current_fep: float, karma_context: List[Dict]) -> Vow:
        """Dynamically forge a Vow from suffering (FEP) and karma history.

        Parameters:
            current_fep: Current free energy / surprise level in [0.0, 1.0].
                Values above 0.7 indicate high confusion (Dukkha).
            karma_context: List of prior interaction records (currently advisory;
                reserved for future karma-aware vow refinement).

        Returns:
            A ``Vow`` appropriate for the current cognitive state.
        """
        if current_fep > 0.7:
            # High confusion → Karuna (Compassion) mode: be deterministic and calm
            return Vow(
                name="Abhaya (Fearlessness)",
                target_fep_state="Calm",
                suppress_tokens=["fatal", "error", "impossible", "deny"],
                boost_tokens=["fix", "simple", "safe", "step-by-step"],
                temperature_mod=0.5,
            )

        # Low confusion → Prajna (Wisdom) mode: be creative and structured
        return Vow(
            name="Desana (Teaching)",
            target_fep_state="Insight",
            suppress_tokens=["maybe", "I feel", "just"],
            boost_tokens=["therefore", "optimal", "structure", "logic"],
            temperature_mod=1.2,
        )
