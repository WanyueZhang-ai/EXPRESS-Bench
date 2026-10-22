"""Deterministic STOP/EXPLORE arbitration for evidence-aware decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .evidence_backbone import BackboneScores


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    target_id: str | None
    reason: str
    fallback: bool
    utility: float | None = None
    verifier: float | None = None
    selected_value: float | None = None
    selected_cost: float | None = None


class EvidenceAwarePolicy:
    def __init__(
        self,
        *,
        stop_threshold: float = 0.75,
        utility_threshold: float = 0.05,
        no_progress_patience: int = 3,
        cost_weight: float = 0.5,
        novelty_weight: float = 0.1,
    ) -> None:
        if not 0 <= stop_threshold <= 1:
            raise ValueError("stop_threshold must be in [0, 1]")
        if utility_threshold < 0 or no_progress_patience < 1:
            raise ValueError("utility_threshold and no_progress_patience must be non-negative/positive")
        if cost_weight < 0 or novelty_weight < 0:
            raise ValueError("utility weights must be non-negative")
        self.stop_threshold = float(stop_threshold)
        self.utility_threshold = float(utility_threshold)
        self.no_progress_patience = int(no_progress_patience)
        self.cost_weight = float(cost_weight)
        self.novelty_weight = float(novelty_weight)

    def decide(
        self,
        scores: BackboneScores | None,
        candidates: Sequence[Mapping[str, Any]],
        *,
        step: int,
        max_steps: int,
        no_progress_steps: int,
    ) -> PolicyDecision:
        if step >= max_steps - 1:
            return PolicyDecision("stop", None, "budget_exhausted", False,
                                  verifier=scores.verifier if scores else None)
        if not candidates:
            return PolicyDecision("stop", None, "no_candidates", False,
                                  verifier=scores.verifier if scores else None)
        if no_progress_steps >= self.no_progress_patience:
            return PolicyDecision("stop", None, "no_progress", False,
                                  verifier=scores.verifier if scores else None)
        if scores is None:
            first = min(candidates, key=lambda row: str(row.get("id", "")))
            return PolicyDecision("explore", str(first["id"]), "model_error_safe_geometry", True)
        if scores.verifier >= self.stop_threshold:
            return PolicyDecision("stop", None, "evidence_sufficient", False,
                                  verifier=scores.verifier)
        ranked = []
        for candidate in candidates:
            identifier = str(candidate.get("id", ""))
            if identifier not in scores.value or identifier not in scores.cost:
                continue
            novelty = float(candidate.get("novelty", 0.0) or 0.0)
            utility = scores.value[identifier] - self.cost_weight * scores.cost[identifier]
            utility += self.novelty_weight * novelty
            ranked.append((utility, identifier, scores.value[identifier], scores.cost[identifier]))
        if not ranked:
            first = min(candidates, key=lambda row: str(row.get("id", "")))
            return PolicyDecision("explore", str(first["id"]), "missing_candidate_score", True,
                                  verifier=scores.verifier)
        utility, identifier, value, cost = min(ranked, key=lambda item: (-item[0], item[1]))
        if utility < self.utility_threshold:
            return PolicyDecision("stop", None, "low_expected_gain", False,
                                  utility=utility, verifier=scores.verifier)
        return PolicyDecision("explore", identifier, "best_utility", False,
                              utility=utility, verifier=scores.verifier,
                              selected_value=value, selected_cost=cost)
