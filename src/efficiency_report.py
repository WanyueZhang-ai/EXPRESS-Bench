"""Deterministic resume-aware efficiency summaries and an offline HTML report."""
from __future__ import annotations

import csv
import html
import json
import math
import logging
from pathlib import Path
import statistics
from collections import Counter, defaultdict

from src.efficiency import SCHEMA_VERSION, TOKEN_FIELDS, TRACE_FILES


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                if not line.endswith("\n"):
                    logging.warning("Ignoring incomplete final efficiency row at %s:%s", path, number)
                    break
                raise ValueError(f"Invalid efficiency JSON at {path}:{number}") from error
            if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported efficiency record at {path}:{number}")
            records.append(record)
    return records


def distribution(values):
    values = sorted(float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
    if not values:
        return {key: 0 if key == "count" else None for key in ("count", "mean", "p50", "p90", "p95", "max", "min", "std")}
    def percentile(fraction):
        position = (len(values) - 1) * fraction
        left = int(position)
        return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left)
    return {"count": len(values), "mean": statistics.mean(values), "p50": percentile(.5), "p90": percentile(.9),
            "p95": percentile(.95), "max": values[-1], "min": values[0], "std": statistics.pstdev(values)}


def _tokens(records):
    result = {}
    for key in TOKEN_FIELDS:
        values = [record.get("tokens", {}).get(key) for record in records]
        known = [value for value in values if isinstance(value, int) and not isinstance(value, bool) and value >= 0]
        result[key] = {"total": sum(known) if len(known) == len(values) else None,
                       "known_subtotal": sum(known), "known_calls": len(known), "missing_calls": len(values) - len(known)}
    return result


def _aggregate(records):
    result = {"count": len(records), "status_counts": dict(Counter(record.get("status", "unknown") for record in records)),
              "wall_ms": distribution(record.get("wall_ms") for record in records), "tokens": _tokens(records)}
    for field in ("input_to_image_encoded_ms", "image_encoded_to_output_ms", "input_to_first_token_ms",
                  "generation_tokens_per_second", "decode_tokens_per_second", "task_completion_ms", "evaluation_ms"):
        result[field] = distribution(record.get(field) for record in records)
    stages = defaultdict(list)
    categories = defaultdict(float)
    for record in records:
        for name, value in record.get("stages", {}).items():
            stages[name].append(value.get("wall_ms"))
        for category, value in record.get("category_wall_ms", {}).items():
            categories[category] += value
    result["stage_wall_ms"] = {name: distribution(values) for name, values in sorted(stages.items())}
    result["category_wall_ms"] = dict(sorted(categories.items()))
    result["peak_allocated_bytes"] = distribution(record.get("memory", {}).get("peak_allocated_bytes") for record in records)
    fallback = [row for row in records if "decision_fallback" in row]
    result["fallback_count"] = sum(bool(row["decision_fallback"]) for row in fallback)
    result["fallback_rate"] = result["fallback_count"] / len(fallback) if fallback else None
    result["retry_count"] = sum(row.get("retry_count", 0) for row in records)
    result["simulator_transition_count"] = sum(row.get("simulator_transition_count", 0) for row in records)
    result["collision_count"] = sum(bool(row.get("collision")) for row in records)
    result["candidate_count"] = distribution(row.get("candidate_count") for row in records)
    result["decision_fields"] = distribution(row.get("decision_fields") for row in records)
    return result


def _selected_attempts(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode.get("condition"), episode.get("episode_id"))].append(episode)
    selected = []
    for rows in groups.values():
        complete = [row for row in rows if row.get("status") == "ok"]
        selected.append(max(complete or rows, key=lambda row: (row.get("finished_at_unix", 0), row["record_id"])))
    return selected


def summarize(records):
    episodes = _selected_attempts(records["episode"])
    attempts = {row["attempt_id"] for row in episodes}
    # Calls made during model initialization may have no episode. Preserve them
    # in diagnostics, but never attribute them to a question or a simulator step.
    calls = [row for row in records["call"] if row.get("attempt_id") in attempts]
    steps = [row for row in records["step"] if row.get("attempt_id") in attempts]
    groups = defaultdict(list)
    type_groups = defaultdict(list)
    for row in calls:
        key = tuple(row.get(key) for key in ("condition", "mode", "model", "checkpoint", "purpose", "input_modality", "output_kind", "status"))
        groups[key].append(row)
        type_groups[(row.get("condition"), row.get("question_type"))].append(row)
    group_records = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        labels = dict(zip(("condition", "mode", "model", "checkpoint", "purpose", "input_modality", "output_kind", "status"), key))
        group_records.append({**labels, "all_calls": _aggregate(rows), "steady_state_calls": _aggregate([row for row in rows if not row.get("warmup", False)])})
    attempted = {row.get("attempt_id") for kind in ("call", "step", "episode") for row in records[kind] if row.get("attempt_id")}
    excluded = attempted - attempts
    quality = defaultdict(list)
    for row in episodes:
        quality[row.get("condition")].append({"EAC": row.get("eac"), "path_len": row.get("path_length_m"),
                                              "goal_dis": row.get("goal_distance_m"), "geodesic_distance": row.get("geodesic_distance")})
    from Jev_scripts.run_jev_replay import express_metrics
    return {
        "schema_version": SCHEMA_VERSION,
        "units": {"time": "milliseconds", "memory": "bytes"},
        "selection_policy": "latest successful attempt per condition/episode; otherwise latest failed attempt",
        "notes": ["Nested stage durations are inclusive and cannot be added; category_wall_ms is exclusive.",
                  "GPU timeline durations and wall durations are separate; detailed synchronization/streaming adds overhead.",
                  "Token totals are null when coverage is incomplete; known_subtotal is explicitly partial.",
                  "Warmup is labelled per worker run/model/purpose/modality and never executes extra model calls.",
                  "Episode wall time includes initialization/evaluation and trace writes between steps; step wall time excludes writing its own trace."],
        "calls": _aggregate(calls),
        "steady_state_calls": _aggregate([row for row in calls if not row.get("warmup", False)]),
        "steps": _aggregate(steps), "episodes": _aggregate(episodes),
        "model_groups": group_records,
        "effectiveness_by_condition": {condition: express_metrics(rows) for condition, rows in quality.items()},
        "question_type_groups": [{"condition": key[0], "question_type": key[1], **_aggregate(rows)} for key, rows in sorted(type_groups.items(), key=lambda item: str(item[0]))],
        "diagnostics": {"raw_calls": len(records["call"]), "raw_steps": len(records["step"]), "raw_episodes": len(records["episode"]),
                        "excluded_or_incomplete_attempts": len(excluded), "failed_episode_attempts": sum(row.get("status") != "ok" for row in records["episode"]),
                        "excluded_calls": len(records["call"]) - len(calls),
                        "excluded_call_statistics": _aggregate([row for row in records["call"] if row.get("attempt_id") not in attempts]),
                        "encoding_status_counts": dict(Counter(row.get("image_encoding_status") for row in calls)),
                        "profiling_io_errors": [error for run in records["run"] for error in run.get("profiling_io_errors", [])]},
        "episode_records": [{key: row.get(key) for key in ("condition", "episode_id", "attempt_id", "question_type", "status", "wall_ms", "task_completion_ms", "evaluation_ms", "path_length_m", "goal_distance_m", "geodesic_distance", "answer_status", "eac")} for row in episodes],
        "runs": records["run"],
    }


def _html_report(summary):
    def fmt(value):
        return "—" if value is None else f"{value:,.2f}"
    rows = []
    for group in summary["model_groups"]:
        stats = group["all_calls"]
        labels = [group.get(key) for key in ("condition", "mode", "model", "purpose", "input_modality", "status")]
        values = [stats["count"], *[fmt(stats["wall_ms"][key]) for key in ("mean", "p50", "p95")],
                  fmt(stats["input_to_image_encoded_ms"]["mean"]), fmt(stats["image_encoded_to_output_ms"]["mean"]),
                  stats["tokens"]["effective_input_tokens"]["total"], stats["tokens"]["generated_tokens"]["total"]]
        rows.append("<tr>" + "".join("<td>" + html.escape("—" if value is None else str(value)) + "</td>" for value in labels + values) + "</tr>")
    step = summary["steps"]
    total = sum(step["category_wall_ms"].values())
    bars = []
    for name, value in sorted(step["category_wall_ms"].items(), key=lambda pair: -pair[1]):
        percentage = 100 * value / total if total else 0
        bars.append(f'<div class="barrow"><span>{html.escape(name)}</span><div class="track"><div class="bar" style="width:{percentage:.3f}%"></div></div><b>{percentage:.1f}%</b><span>{value / 1000:,.2f} s</span></div>')
    # JSON embedded in HTML must not be able to terminate the script element.
    payload = json.dumps(summary, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXPRESS-Bench · Efficiency</title><style>
body{font:15px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#172337;margin:0;padding:36px}main{max-width:1500px;margin:auto}h1{margin-bottom:0}h2{font-size:20px;margin-top:32px}.muted{color:#56677b}.cards{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.card{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:18px 24px;min-width:160px}.card b{display:block;font-size:27px}section{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:22px;margin-bottom:18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:13px}td,th{padding:10px 13px;text-align:left;border-bottom:1px solid #e3e9f0}th{cursor:pointer;background:#f4f7fb}input{font:inherit;padding:8px 12px;min-width:280px;border:1px solid #b7c6d8;border-radius:6px}.barrow{display:grid;grid-template-columns:170px 1fr 70px 110px;gap:14px;align-items:center;margin:12px 0}.track{background:#edf1f8;height:19px;border-radius:4px}.bar{background:#4279d4;height:100%;border-radius:4px}button{padding:9px 15px;border:0;border-radius:6px;background:#255da9;color:white;cursor:pointer}li{margin:8px 0}@media(max-width:700px){body{padding:15px}.barrow{grid-template-columns:100px 1fr 55px}.barrow>span:last-child{display:none}}</style>
<main><h1>EXPRESS-Bench · 效率分析</h1><p class="muted">逐调用、逐探索 step、逐 episode。时间单位：ms；文件可离线打开。</p>
<div class="cards">""" + "".join(f'<div class="card">{label}<b>{value}</b></div>' for label, value in [
        ("Episodes", summary["episodes"]["count"]), ("Exploration steps", step["count"]), ("Model calls", summary["calls"]["count"]),
        ("Step P50 · ms", fmt(step["wall_ms"]["p50"])), ("Step P95 · ms", fmt(step["wall_ms"]["p95"]))]) + """</div>
<h2>Step 时间构成</h2><section>""" + ("".join(bars) or "暂无 step 数据。") + """</section><h2>模型调用</h2><section><p><input id="filter" aria-label="筛选模型调用" placeholder="筛选模型、用途、输入模式…"> <button id="download">下载完整 JSON</button></p><div class="scroll"><table id="calls"><thead><tr>""" + "".join("<th>" + label + "</th>" for label in ("Condition", "Mode", "Model", "Purpose", "Input", "Status", "Calls", "Mean", "P50", "P95", "Input → encoded", "Encoded → output", "Input tokens", "Generated tokens")) + """</tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div><p class="muted">编码时间只统计可观察到的调用；— 表示不可用或 token 覆盖不完整。点击表头排序。</p></section>
<h2>统计口径与完整性</h2><section><ul>""" + "".join("<li>" + html.escape(note) + "</li>" for note in summary["notes"]) + "</ul><p>失败 episode 尝试：" + str(summary["diagnostics"]["failed_episode_attempts"]) + "；被替代或未完成的尝试：" + str(summary["diagnostics"]["excluded_or_incomplete_attempts"]) + """。原始数据保留在 JSONL，历史失败开销见 JSON diagnostics。</p></section></main>
<script type="application/json" id="data">""" + payload + """</script><script>
const table=document.querySelector('#calls'),body=table.tBodies[0];document.querySelector('#filter').addEventListener('input',e=>{for(const row of body.rows)row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase());});
for(const [i,th] of [...table.tHead.rows[0].cells].entries()){let asc=true;th.addEventListener('click',()=>{const rows=[...body.rows];rows.sort((a,b)=>{let x=a.cells[i].textContent.replaceAll(',',''),y=b.cells[i].textContent.replaceAll(',','');return (asc?1:-1)*(x!==''&&y!==''&&Number.isFinite(+x)&&Number.isFinite(+y)?+x-+y:x.localeCompare(y));});for(const row of rows)body.append(row);asc=!asc;});}
document.querySelector('#download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([document.querySelector('#data').textContent],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='efficiency_summary.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});</script></html>"""


def build_report(input_dirs, output_dir, *, merge_traces=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {kind: [] for kind in TRACE_FILES}
    seen = {}
    for directory in input_dirs:
        for kind, filename in TRACE_FILES.items():
            for row in read_jsonl(Path(directory) / filename):
                key = (kind, row["record_id"])
                if key in seen:
                    if seen[key] != row:
                        raise ValueError(f"Conflicting efficiency record: {key}")
                    continue
                seen[key] = row
                records[kind].append(row)
    for rows in records.values():
        rows.sort(key=lambda row: (row.get("started_at_unix", 0), row["record_id"]))
    if merge_traces:
        for kind, filename in TRACE_FILES.items():
            temporary = output_dir / (filename + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in records[kind]:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            temporary.replace(output_dir / filename)
    summary = summarize(records)
    (output_dir / "efficiency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    labels = ["condition", "mode", "model", "checkpoint", "purpose", "input_modality", "output_kind", "status"]
    fields = labels + ["population", "count", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "max_ms"]
    measurements = ("input_to_image_encoded_ms", "image_encoded_to_output_ms", "input_to_first_token_ms", "generation_tokens_per_second", "decode_tokens_per_second")
    fields.extend(field + "_mean" for field in measurements)
    for token in TOKEN_FIELDS:
        fields.extend([token, token + "_known_subtotal", token + "_missing_calls"])
    with (output_dir / "efficiency_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in summary["model_groups"]:
            for population in ("all_calls", "steady_state_calls"):
                stats = group[population]
                row = {key: group[key] for key in labels}
                row.update(population=population, count=stats["count"])
                row.update({key + "_ms": stats["wall_ms"][key] for key in ("mean", "p50", "p90", "p95", "max")})
                row.update({field + "_mean": stats[field]["mean"] for field in measurements})
                for token in TOKEN_FIELDS:
                    row[token] = stats["tokens"][token]["total"]
                    row[token + "_known_subtotal"] = stats["tokens"][token]["known_subtotal"]
                    row[token + "_missing_calls"] = stats["tokens"][token]["missing_calls"]
                writer.writerow(row)
    (output_dir / "efficiency_report.html").write_text(_html_report(summary), encoding="utf-8")
    return summary
