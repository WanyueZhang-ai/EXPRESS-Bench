"""Run Jev routing on structured online steps or deterministic fixtures.

The fixture path is offline and never calls TypeSafe.  The ``--live`` path
uses the exact same ``DecisionRouter.decide`` call that the Habitat integration
uses, so its returned action can be applied to the next simulator step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .decision_router import DecisionRouter, RouterConfig, RouterDecision, SCHEMA_VERSION
from .jev_client import JevClient, JevError, JevResponse


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _eac_values(value: Any) -> Optional[tuple]:
    if not isinstance(value, str):
        return None
    value = value.replace("Your mark:", "").strip()
    try:
        grounding, accuracy = value.split(",", 1)
        return float(grounding), int(float(accuracy))
    except (ValueError, TypeError):
        return None


def express_metrics(results: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    c_values: List[float] = []
    c_star_values: List[float] = []
    path_values: List[float] = []
    geodesic_values: List[float] = []
    target_values: List[float] = []
    for result in results:
        path_len = result.get("path_len")
        if _finite(path_len) and _finite(result.get("geodesic_distance")):
            eac = _eac_values(result.get("EAC"))
            if eac is not None:
                grounding, accuracy = eac
                c_values.append(grounding * accuracy)
                c_star_values.append(float(accuracy))
                path_values.append(float(path_len))
                geodesic_values.append(float(result["geodesic_distance"]))
        if _finite(result.get("goal_dis")):
            target_values.append(float(result["goal_dis"]))

    def scaled_mean(values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return sum(100.0 * max(0.0, min(5.0, value)) / 5.0 for value in values) / len(values)

    c_avg = scaled_mean(c_values)
    c_star_avg = scaled_mean(c_star_values)
    e_path: Optional[float] = None
    if c_values and geodesic_values:
        terms = []
        for value, path_len, geodesic in zip(c_values, path_values, geodesic_values):
            denom = max(path_len, geodesic)
            weight = geodesic / denom if denom > 0 else 0.0
            terms.append(100.0 * max(0.0, min(5.0, value)) / 5.0 * weight)
        e_path = sum(terms) / len(terms)
    return {
        "C_avg": c_avg,
        "C_star_avg": c_star_avg,
        "E_path": e_path,
        "d_T_avg": sum(target_values) / len(target_values) if target_values else None,
    }


def _metric_for_record(record: Mapping[str, Any], metric: str) -> Optional[float]:
    if isinstance(record.get("express_result"), Mapping):
        return express_metrics([record["express_result"]]).get(metric)
    value = record.get(metric)
    return float(value) if _finite(value) else None


def paired_bootstrap_ci(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap right-minus-left differences paired by episode id."""
    left_by_id = {record.get("episode_id"): record for record in left}
    right_by_id = {record.get("episode_id"): record for record in right}
    ids = sorted(set(left_by_id) & set(right_by_id), key=str)
    if not ids:
        raise ValueError("conditions have no paired episode ids")
    differences = []
    for episode_id in ids:
        left_value = _metric_for_record(left_by_id[episode_id], metric)
        right_value = _metric_for_record(right_by_id[episode_id], metric)
        if left_value is not None and right_value is not None:
            differences.append(float(right_value) - float(left_value))
    if not differences:
        raise ValueError("paired episodes have no finite {} values".format(metric))
    rng = random.Random(seed)
    samples = []
    for _ in range(max(1, int(n_bootstrap))):
        samples.append(
            sum(rng.choice(differences) for _ in differences) / len(differences)
        )
    samples.sort()

    def percentile(fraction: float) -> float:
        position = (len(samples) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(samples) - 1)
        weight = position - low
        return samples[low] * (1.0 - weight) + samples[high] * weight

    return {
        "metric": metric,
        "n_pairs": len(differences),
        "mean_difference": sum(differences) / len(differences),
        "low": percentile(0.025),
        "high": percentile(0.975),
        "seed": seed,
        "n_bootstrap": max(1, int(n_bootstrap)),
    }


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    episode_ids = sorted({record.get("episode_id") for record in records if record.get("episode_id") is not None})
    latencies = [float(record["latency_ms"]) for record in records if _finite(record.get("latency_ms"))]
    fallback_count = sum(bool(record.get("fallback")) for record in records)
    input_tokens = [int(record["usage"]["input_tokens"]) for record in records if isinstance(record.get("usage"), Mapping) and _finite(record["usage"].get("input_tokens"))]
    output_tokens = [int(record["usage"]["output_tokens"]) for record in records if isinstance(record.get("usage"), Mapping) and _finite(record["usage"].get("output_tokens"))]
    result_rows = [record["express_result"] for record in records if isinstance(record.get("express_result"), Mapping)]
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_count": len(episode_ids),
        "episode_ids": episode_ids,
        "decision_count": len(records),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(records) if records else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "total_input_tokens": sum(input_tokens) if input_tokens else 0,
        "total_output_tokens": sum(output_tokens) if output_tokens else 0,
        "action_counts": {
            action: sum(record.get("action") == action for record in records)
            for action in ("stop", "goe", "fbe")
        },
        "strata_counts": {},
    }
    for record in records:
        key = str(record.get("question_type", "unknown"))
        summary["strata_counts"][key] = summary["strata_counts"].get(key, 0) + 1
    summary.update(express_metrics(result_rows))
    return summary


def decision_record(
    input_record: Mapping[str, Any], decision: RouterDecision, *, router_mode: str
) -> Dict[str, Any]:
    usage = decision.usage if isinstance(decision.usage, Mapping) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": input_record.get("episode_id"),
        "step": input_record.get("step"),
        "seed": input_record.get("seed"),
        "question_type": input_record.get("question_type"),
        "router_mode": router_mode,
        "semantic_source": "habitat_current_fov" if router_mode == "oracle_semantic" else "vlm_tsdf",
        "action": decision.action,
        "target_id": decision.target_id,
        "candidate_ids": {
            "goe": [item.get("id") for item in input_record.get("goe_candidates", [])],
            "fbe": [item.get("id") for item in input_record.get("fbe_candidates", [])],
        },
        "fallback": decision.fallback,
        "fallback_reason": decision.fallback_reason,
        "status": decision.status,
        "stop_probability": decision.stop_probability,
        "mode_probability": decision.mode_probability,
        "target_probability": decision.target_probability,
        "response_model": decision.response_model,
        "latency_ms": decision.latency_ms,
        "usage": dict(usage),
        "request_id": decision.request_id,
        "express_result": input_record.get("express_result"),
    }


class OfflineJevClient:
    """Deterministic stand-in used only for schema and execution smoke tests."""

    def evaluate(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> JevResponse:
        modes = questions.get("mode", {}).get("criteria", {})
        targets = questions.get("target", {}).get("criteria", {})
        mode = "fbe" if "fbe" in modes else ("goe" if "goe" in modes else "fbe")
        target = next((name for name in targets if name.startswith(mode + "_")), None)
        answers: Dict[str, Dict[str, Any]] = {
            "stop": {"type": "noul", "noul": 0.0}
        }
        if modes:
            answers["mode"] = {
                "type": "choice",
                "choice": mode,
                "confidence": 1.0,
                "probabilities": {name: 1.0 if name == mode else 0.0 for name in modes},
            }
        if targets:
            answers["target"] = {
                "type": "choice",
                "choice": target,
                "confidence": 1.0,
                "probabilities": {name: 1.0 if name == target else 0.0 for name in targets},
            }
        return JevResponse("offline-fixture", answers, {}, 0.0, None)


class BudgetedClient:
    def __init__(self, client: Any, max_calls: int) -> None:
        self.client = client
        self.max_calls = max_calls
        self.calls = 0

    def evaluate(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> JevResponse:
        if self.calls >= self.max_calls:
            raise JevError("jev_call_budget_exceeded")
        self.calls += 1
        return self.client.evaluate(state, questions)


def run_records(
    records: Iterable[Mapping[str, Any]],
    *,
    live: bool = False,
    router_mode: str = "fair",
    stop_threshold: float = 0.75,
    max_calls: int = 100000,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if live:
        client: Any = JevClient(api_key=api_key, max_retries=2, timeout=20.0)
    else:
        client = OfflineJevClient()
    client = BudgetedClient(client, max_calls)
    router = DecisionRouter(
        client,
        RouterConfig(stop_threshold=stop_threshold, evidence_mode=router_mode),
    )
    output: List[Dict[str, Any]] = []
    for record in records:
        decision = router.decide(
            question=str(record["question"]),
            question_type=str(record.get("question_type", "unknown")),
            history=record.get("history", {}),
            perception=record.get("perception", {}),
            goe_candidates=record.get("goe_candidates", []),
            fbe_candidates=record.get("fbe_candidates", []),
            baseline_action=str(record.get("baseline_action", "fbe")),
            baseline_target_id=record.get("baseline_target_id"),
            evidence_mode=router_mode,
            oracle_semantic=record.get("oracle_semantic"),
        )
        output.append(decision_record(record, decision, router_mode=router_mode))
    return output


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if isinstance(value, Mapping):
        value = value.get("records", [value])
    if not isinstance(value, list):
        raise ValueError("input must be a JSON list or JSONL file")
    return value


def _write_outputs(records: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = aggregate_records(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            writer.writerow([key, value])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("fixtures") / "sample_state.json")
    parser.add_argument("--output-dir", type=Path, default=Path("jev_replay_result"))
    parser.add_argument("--live", action="store_true", help="call the real Jev API")
    parser.add_argument("--mode", choices=("fair", "oracle_semantic"), default="fair")
    parser.add_argument("--stop-threshold", type=float, default=0.75)
    parser.add_argument("--max-calls", type=int, default=100000)
    args = parser.parse_args(argv)
    records = _load_records(args.input)
    output = run_records(
        records,
        live=args.live,
        router_mode=args.mode,
        stop_threshold=args.stop_threshold,
        max_calls=args.max_calls,
    )
    _write_outputs(output, args.output_dir)
    print(json.dumps(aggregate_records(output), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
