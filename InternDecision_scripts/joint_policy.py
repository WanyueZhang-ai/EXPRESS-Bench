"""Single-request Intern-Decision policy with per-candidate auxiliary fields."""
from __future__ import annotations

import copy
import math
import time

from .decision_router import InternDecisionRouter, RouterDecision
from .evidence_adapters import request_with_questions


def joint_questions(state, action_mode="planner_candidates", direct_actions=()):
    if action_mode == "direct_actions":
        candidates = {name: name for name in direct_actions if name != "stop"}
    else:
        candidates = {item["id"]: str(item) for kind in ("goe", "fbe")
                      for item in state["map"].get(kind + "_candidates", [])}
    if len(candidates) > 7:
        raise ValueError("joint policy supports at most 7 movement candidates (16 decision fields)")
    candidates["stop"] = "Stop exploring and pass collected visual evidence to the answer VLM."
    questions = {
        "action": {"type": "choice", "instructions": "Select the executable action with best expected grounded answer quality minus exploration cost.", "criteria": candidates},
        "sufficiency": {"type": "noul", "instructions": "Does observed evidence resolve every requirement of the question, including counting coverage and conflicting observations?"},
    }
    for index, candidate in enumerate(key for key in candidates if key != "stop"):
        questions[f"gain_{index}"] = {
            "type": "score", "instructions": f"Expected new question-relevant evidence from {candidate}. Ignore duplicate observations.",
            "criteria": {"0": "No new evidence", "0.5": "Partial new evidence", "1": "Likely resolves remaining evidence"},
        }
        questions[f"cost_{index}"] = {
            "type": "score", "instructions": f"Expected fraction of remaining exploration budget consumed by {candidate}.",
            "criteria": {"0": "Negligible", "0.5": "Moderate", "1": "Most of remaining budget"},
        }
    return questions


class JointDecisionRouter(InternDecisionRouter):
    def _questions(self, state):
        return joint_questions(state, self.config.action_mode, self.config.direct_actions)

    def decide(self, state, *, baseline_action, baseline_target_id):
        questions = self._questions(state)
        started = time.perf_counter()
        try:
            response = self.engine.predict(request_with_questions(state, questions))
            answer = self._answer(response, "action")
            selected = self._valid_choice(answer, questions["action"]["criteria"])
            probability = self._probability(answer or {}, selected)
            if selected is None or probability is None or probability < self.config.min_choice_confidence:
                raise ValueError("invalid action distribution")
            sufficiency = float(self._answer(response, "sufficiency")["noul"])
            if not math.isfinite(sufficiency) or not 0 <= sufficiency <= 1:
                raise ValueError("invalid sufficiency probability")
        except Exception:
            return self._fallback(state=state, action=baseline_action, target_id=baseline_target_id,
                                  reason="invalid_joint_response")
        action, target = selected, None
        if selected.startswith(("goe_", "fbe_")):
            action, target = selected.split("_", 1)[0], selected
        return RouterDecision(action=action, target_id=target, fallback=False,
                              stop_probability=sufficiency, action_probability=probability,
                              latency_ms=(time.perf_counter() - started) * 1000,
                              usage=self._usage(response), state=copy.deepcopy(dict(state)),
                              response=copy.deepcopy(response))
