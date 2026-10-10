#!/usr/bin/env python3
"""Build a self-contained EXPRESS-Bench experiment report.

The reader for result.pkl is intentionally implemented with pickletools instead of
pickle.load: the report only needs scalar fields and should never execute a global
object embedded in a serialized trajectory.
"""
from __future__ import annotations

import base64
import json
import math
import pickletools
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "express_results_report.html"


class _Mark: pass
MARK = _Mark()


class Opaque:
    def __init__(self, label: str, value: Any = None):
        self.label, self.value = label, value
    def __repr__(self):
        return f"Opaque({self.label})"


def safe_pickle_read(path: Path) -> Any:
    """Interpret the data/container subset of pickle opcodes without globals."""
    data = path.read_bytes()
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    for op, arg, _pos in pickletools.genops(data):
        n = op.name
        if n in {"PROTO", "FRAME", "STOP", "NEWLINE", "POP_MARK"}:
            if n == "STOP":
                return stack[-1] if stack else None
            if n == "POP_MARK":
                while stack and stack.pop() is not MARK:
                    pass
            continue
        if n == "MARK": stack.append(MARK); continue
        if n in {"NONE", "NEWTRUE", "NEWFALSE"}:
            stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[n]); continue
        if n in {"BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "BINFLOAT"}:
            stack.append(arg); continue
        if n in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "STRING"}:
            stack.append(arg); continue
        if n in {"SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "BYTEARRAY8"}:
            stack.append(Opaque("bytes", len(arg) if hasattr(arg, "__len__") else None)); continue
        if n == "EMPTY_DICT": stack.append({}); continue
        if n == "EMPTY_LIST": stack.append([]); continue
        if n == "EMPTY_TUPLE": stack.append(()); continue
        if n == "EMPTY_SET": stack.append(set()); continue
        if n == "TUPLE":
            idx = max(i for i, v in enumerate(stack) if v is MARK)
            vals = stack[idx + 1:]; del stack[idx:]
            stack.append(tuple(vals)); continue
        if n in {"TUPLE1", "TUPLE2", "TUPLE3"}:
            k = int(n[-1]); vals = stack[-k:]; del stack[-k:]; stack.append(tuple(vals)); continue
        if n == "APPEND":
            val = stack.pop();
            if stack and isinstance(stack[-1], list): stack[-1].append(val)
            else: stack.append([val])
            continue
        if n == "APPENDS":
            idx = max(i for i, v in enumerate(stack) if v is MARK)
            vals = stack[idx + 1:]; target = stack[idx - 1] if idx else None
            del stack[idx:]
            if isinstance(target, list): target.extend(vals)
            continue
        if n == "SETITEM":
            val, key = stack.pop(), stack.pop()
            if stack and isinstance(stack[-1], dict): stack[-1][key] = val
            continue
        if n == "SETITEMS":
            idx = max(i for i, v in enumerate(stack) if v is MARK)
            vals = stack[idx + 1:]; target = stack[idx - 1] if idx else None
            del stack[idx:]
            if isinstance(target, dict):
                for i in range(0, len(vals) - 1, 2): target[vals[i]] = vals[i + 1]
            continue
        if n == "ADDITEMS":
            idx = max(i for i, v in enumerate(stack) if v is MARK)
            vals = stack[idx + 1:]; target = stack[idx - 1] if idx else None
            del stack[idx:]
            if isinstance(target, set): target.update(vals)
            continue
        if n in {"BINGET", "LONG_BINGET"}: stack.append(memo.get(int(arg))); continue
        if n in {"BINPUT", "LONG_BINPUT"}: memo[int(arg)] = stack[-1]; continue
        if n == "MEMOIZE": memo[len(memo)] = stack[-1]; continue
        if n == "DUP": stack.append(stack[-1]); continue
        if n == "POP": stack.pop(); continue
        if n in {"GLOBAL", "STACK_GLOBAL", "EXT1", "EXT2", "EXT4"}:
            if n == "STACK_GLOBAL" and len(stack) >= 2:
                name, module = stack.pop(), stack.pop(); stack.append(Opaque(f"{module}.{name}"))
            else: stack.append(Opaque(str(arg)))
            continue
        if n in {"NEWOBJ", "NEWOBJ_EX", "REDUCE"}:
            if n == "NEWOBJ_EX" and len(stack) >= 3: kwargs, args, fn = stack.pop(), stack.pop(), stack.pop()
            elif len(stack) >= 2: args, fn = stack.pop(), stack.pop()
            else: args, fn = (), None
            stack.append(Opaque(getattr(fn, "label", "object"), args)); continue
        if n == "BUILD":
            state = stack.pop() if stack else None
            if stack and isinstance(stack[-1], dict) and isinstance(state, dict): stack[-1].update(state)
            continue
        if n in {"PERSID", "BINPERSID", "NEXT_BUFFER", "READONLY_BUFFER"}:
            stack.append(Opaque(n)); continue
        # Unknown opcodes are kept as opaque values so a new numpy/magnum opcode
        # cannot make the report reader execute arbitrary code.
        stack.append(Opaque(n, arg))
    return stack[-1] if stack else None


def num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else None


def parse_eac(x: Any) -> tuple[float | None, float | None]:
    if not isinstance(x, str): return None, None
    try:
        a, b = x.replace("Your mark:", "").strip().split(",")[:2]
        return float(a), float(b)
    except Exception: return None, None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c, cs, ep, dt = [], [], [], []
    for r in rows:
        grd, acc = parse_eac(r.get("EAC"))
        pl, gd, goal = num(r.get("path_len")), num(r.get("geodesic_distance")), num(r.get("goal_dis"))
        if grd is not None and acc is not None:
            c.append(max(0, min(5, grd * acc)))
            cs.append(max(0, min(5, acc)))
            if pl is not None and gd is not None and max(pl, gd) > 0:
                ep.append(max(0, min(5, grd * acc)) / 5 * gd / max(pl, gd) * 100)
        if goal is not None: dt.append(goal)
    def avg(xs): return round(statistics.fmean(xs), 4) if xs else None
    nav = [r for r in rows if num(r.get("path_len")) is not None]
    ans5 = [r for r in rows if (parse_eac(r.get("EAC"))[1] or 0) >= 5]
    return {
        "n": len(rows), "metric_n": len(c), "path_n": len(nav), "goal_n": len(dt),
        "C": round(avg(c) / 5 * 100, 4) if c else None,
        "C_star": round(avg(cs) / 5 * 100, 4) if cs else None,
        "E_path": round(avg(ep), 4) if ep else None,
        "d_T": round(avg(dt), 4) if dt else None,
        "answer_success": round(len(ans5) / len(rows) * 100, 4) if rows else None,
        "nav_success": round(len(nav) / len(rows) * 100, 4) if rows else None,
    }


def read_baseline() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    # The original Qwen reproduction writes one durable result per question
    # under experiment_result_qwen/<question_ind>/result.pkl. Do not mix the
    # JEV ablations into this baseline.
    for p in sorted((ROOT / "experiment_result" / "experiment_result_qwen").glob("**/result.pkl"), key=lambda x: int(x.parent.name) if x.parent.name.isdigit() else 10**9):
        x = safe_pickle_read(p)
        if not isinstance(x, dict): continue
        r = {k: x.get(k) for k in ["question_ind", "type", "question", "answer", "gen_answer", "EAC", "geodesic_distance", "cnt_step", "goal_dis", "path_len", "scene"]}
        if not isinstance(r.get("question_ind"), int): r["question_ind"] = int(p.parent.name)
        rows.append(r)
    return rows, metrics(rows)


def read_episode_pickles(base: Path) -> dict[int, dict[str, Any]]:
    out = {}
    for p in base.glob("**/result.pkl"):
        x = safe_pickle_read(p)
        if not isinstance(x, dict): continue
        q = x.get("question_ind")
        if isinstance(q, int): out[q] = x
        elif p.parent.name.isdigit(): out[int(p.parent.name)] = x
    return out


def decision_stats(files: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique = {}
    for f in files:
        if not f.exists(): continue
        for line in f.read_text(errors="replace").splitlines():
            try: x = json.loads(line)
            except Exception: continue
            key = (x.get("episode_id"), x.get("step"), x.get("router_mode"))
            unique[key] = x
    rows = list(unique.values())
    act = Counter(x.get("action", "unknown") for x in rows)
    status = Counter(x.get("status", "unknown") for x in rows)
    fb = Counter(x.get("fallback_reason", "none") for x in rows if x.get("fallback"))
    lat = [num(x.get("latency_ms")) for x in rows if num(x.get("latency_ms")) is not None]
    inp = [num((x.get("usage") or {}).get("input_tokens")) for x in rows if num((x.get("usage") or {}).get("input_tokens")) is not None]
    out = [num((x.get("usage") or {}).get("output_tokens")) for x in rows if num((x.get("usage") or {}).get("output_tokens")) is not None]
    episodes = sorted(set(x.get("episode_id") for x in rows if isinstance(x.get("episode_id"), int)))
    return rows, {"records": len(rows), "episodes": len(episodes), "actions": dict(act), "status": dict(status), "fallback_reasons": dict(fb),
                  "fallback_rate": round(sum(bool(x.get("fallback")) for x in rows) / len(rows) * 100, 4) if rows else None,
                  "mean_latency_ms": round(statistics.fmean(lat), 3) if lat else None,
                  "p50_latency_ms": round(statistics.median(lat), 3) if lat else None,
                  "mean_input_tokens": round(statistics.fmean(inp), 2) if inp else None,
                  "mean_output_tokens": round(statistics.fmean(out), 2) if out else None}


def progress_table() -> list[dict[str, Any]]:
    rows=[]
    shards = ROOT / "experiment_result_jev_8gpu" / "shards"
    manifest = {}
    try:
        manifest = {int(x["worker_id"]): x for x in json.loads((shards.parent / "manifest.json").read_text()).get("workers", [])}
    except Exception:
        pass
    for p in sorted(shards.glob("worker_*/progress.json")):
        try: x=json.loads(p.read_text())
        except Exception: x={}
        wid=int(re.search(r"worker_(\d+)", p.parent.name).group(1))
        # worker_00 progress.json records the resumed suffix; manifest.json
        # stores the original shard size. Keep both so stale metadata is visible.
        expected=int(manifest.get(wid, {}).get("count", x.get("total_questions", 0)) or 0)
        progress_total=int(x.get("total_questions", 0) or 0); done=int(x.get("completed_questions", 0) or 0)
        actual=len(list(p.parent.glob("*/result.pkl")))
        completed=max(done,actual)
        rows.append({"worker":wid,"gpu":x.get("gpu_id",wid),"total":expected,"progress_total":progress_total,"completed":completed,"files":actual,"pending":max(0,expected-completed),"status":x.get("status"),"current_question":x.get("current_question"),"current_step":x.get("current_step"),"metadata_mismatch":progress_total != expected})
    return rows


def question_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    for t, rs in sorted((k,list(v)) for k,v in __import__('itertools').groupby(sorted(rows,key=lambda r:r.get('type') or 'unknown'), key=lambda r:r.get('type') or 'unknown')):
        m=metrics(rs); out.append({"type":t,**m})
    return out


def safe_json(x):
    if isinstance(x, dict): return {str(k):safe_json(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [safe_json(v) for v in x]
    if isinstance(x, Opaque): return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
    return x


def svg_table() -> str:
    # Reproduction of the user-provided Table 2, embedded as SVG so the HTML is standalone.
    rows=[("Human Agent","-","83.99","-","-"),("DeepSeek-V3","-","59.15","-","-"),("GPT4","-","58.96","-","-"),("LLaMA-3-8b","-","57.25","-","-"),("DeepSeek-V3 w/ GPT-4o-mini","-","62.60","-","-"),("GPT4 w/ GPT-4o-mini","-","62.56","-","-"),("LLaMA-3-8b w/ GPT-4o-mini","-","59.95","-","-"),("DeepSeek-V3 w/ LLaVA-v1.5-7b","-","60.63","-","-"),("GPT4 w/ LLaVA-v1.5-7b","-","59.53","-","-"),("LLaMA-3-8b w/ LLaVA-v1.5-7b","-","58.59","-","-"),("GPT-4o-mini","-","58.37","-","-"),("LLaVA-v1.5-7b","-","57.66","-","-"),("RE","36.95","62.75","12.06","7.26"),("FBE","38.60","60.24","14.55","6.64"),("GOE","38.54","63.34","12.74","6.46"),("Fine-EQA (Ours)","40.55","63.95","16.22","6.43")]
    y=52; out=['<svg viewBox="0 0 980 690" role="img" aria-label="Recreated Table 2">','<rect width="980" height="690" fill="#fff"/>','<text x="490" y="34" text-anchor="middle" font-size="24" font-family="Georgia" font-weight="700">Table 2. Performance comparison on EXPRESS-Bench.</text>','<text x="540" y="64" text-anchor="middle" font-size="17" font-family="Georgia">C↑</text><text x="650" y="64" text-anchor="middle" font-size="17" font-family="Georgia">C*↑</text><text x="760" y="64" text-anchor="middle" font-size="17" font-family="Georgia">E_path↑</text><text x="870" y="64" text-anchor="middle" font-size="17" font-family="Georgia">d_T↓</text>']
    for i,(name,a,b,c,d) in enumerate(rows):
        yy=y+i*34
        if i==15: out.append(f'<rect x="38" y="{yy-23}" width="900" height="30" rx="3" fill="#d7f8da"/>')
        out += [f'<text x="60" y="{yy}" font-size="16" font-family="Georgia" font-weight="{700 if i in (0,12,13,14,15) else 400}">{name}</text>',f'<text x="540" y="{yy}" text-anchor="middle" font-size="16" font-family="Georgia">{a}</text>',f'<text x="650" y="{yy}" text-anchor="middle" font-size="16" font-family="Georgia">{b}</text>',f'<text x="760" y="{yy}" text-anchor="middle" font-size="16" font-family="Georgia">{c}</text>',f'<text x="870" y="{yy}" text-anchor="middle" font-size="16" font-family="Georgia">{d}</text>']
    out += ['<line x1="40" y1="72" x2="940" y2="72" stroke="#333"/><line x1="40" y1="606" x2="940" y2="606" stroke="#333"/>','</svg>']
    return ''.join(out)



def _episode_values(x: dict[str, Any]) -> dict[str, Any]:
    grd, acc = parse_eac(x.get("EAC"))
    pl, gd, goal = num(x.get("path_len")), num(x.get("geodesic_distance")), num(x.get("goal_dis"))
    c = max(0, min(5, grd * acc)) if grd is not None and acc is not None else None
    cs = max(0, min(5, acc)) if acc is not None else None
    ep = c / 5 * gd / max(pl, gd) * 100 if c is not None and pl is not None and gd is not None and max(pl, gd) > 0 else None
    return {"c": c, "cs": cs, "ep": ep, "dt": goal, "eac": x.get("EAC"),
            "answer": x.get("gen_answer"), "steps": x.get("cnt_step"), "path": pl}


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 6) if xs else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    vx, vy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy), 6) if vx and vy else None


def paired_stats(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    lv = {i: _episode_values(left[i]) for i in ids}
    rv = {i: _episode_values(right[i]) for i in ids}
    out: dict[str, Any] = {"n": len(ids), "left_only": len(set(left) - set(right)), "right_only": len(set(right) - set(left))}
    for key in ("c", "cs", "ep", "dt", "path", "steps"):
        pairs = [(lv[i][key], rv[i][key]) for i in ids if lv[i][key] is not None and rv[i][key] is not None]
        if not pairs:
            continue
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        ds = [y - x for x, y in pairs]
        out[key] = {"n": len(pairs), "left_mean": _mean(xs), "right_mean": _mean(ys),
                    "delta_right_minus_left": _mean(ds), "median_delta": round(statistics.median(ds), 6),
                    "right_better": sum(d > 1e-9 for d in ds), "left_better": sum(d < -1e-9 for d in ds),
                    "ties": sum(abs(d) <= 1e-9 for d in ds), "pearson": _pearson(xs, ys)}
    for key in ("eac", "answer"):
        out[key + "_exact_pct"] = round(sum(lv[i][key] == rv[i][key] for i in ids) / len(ids) * 100, 4) if ids else None
    out["same_steps_pct"] = round(sum(lv[i]["steps"] == rv[i]["steps"] for i in ids) / len(ids) * 100, 4) if ids else None
    out["largest_c_deltas"] = sorted(
        [{"id": i, "left": lv[i]["c"], "right": rv[i]["c"],
          "delta": rv[i]["c"] - lv[i]["c"] if lv[i]["c"] is not None and rv[i]["c"] is not None else None,
          "left_eac": lv[i]["eac"], "right_eac": rv[i]["eac"]} for i in ids],
        key=lambda x: abs(x["delta"] or 0), reverse=True)[:8]
    return out


def read_trace(files: list[Path]) -> dict[int, dict[int, dict[str, Any]]]:
    out: dict[int, dict[int, dict[str, Any]]] = {}
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text(errors="replace").splitlines():
            try:
                x = json.loads(line)
            except Exception:
                continue
            q, step = x.get("episode_id"), x.get("step")
            if isinstance(q, int) and isinstance(step, int):
                out.setdefault(q, {})[step] = x
    return out


def trace_consistency(left_files: list[Path], right_files: list[Path], valid_ids: set[int]) -> dict[str, Any]:
    left, right = read_trace(left_files), read_trace(right_files)
    ids = sorted(valid_ids & set(left) & set(right))
    pairs = [(left[q][s], right[q][s]) for q in ids for s in sorted(set(left[q]) & set(right[q]))]
    out = {"trace_episodes_left": len(left), "trace_episodes_right": len(right),
           "durable_overlap_episodes": len(ids), "left_trace_only": len(set(left) - valid_ids),
           "right_trace_only": len(set(right) - valid_ids), "paired_steps": len(pairs)}
    for field in ("action", "target_id", "fallback", "fallback_reason", "status"):
        out[field + "_step_exact_pct"] = round(sum(a.get(field) == b.get(field) for a, b in pairs) / len(pairs) * 100, 4) if pairs else None
        exact_eps = []
        for q in ids:
            ss = sorted(set(left[q]) & set(right[q]))
            exact_eps.append(bool(ss) and set(left[q]) == set(right[q]) and all(left[q][s].get(field) == right[q][s].get(field) for s in ss))
        out[field + "_episode_exact_pct"] = round(sum(exact_eps) / len(exact_eps) * 100, 4) if exact_eps else None
    return out


def inventory(name: str, expected: int = 2044) -> dict[str, Any]:
    base = ROOT / "experiment_result" / name
    paths = list(base.glob("**/result.pkl"))
    ids = []
    for p in paths:
        x = safe_pickle_read(p)
        if isinstance(x, dict) and isinstance(x.get("question_ind"), int):
            ids.append(int(x["question_ind"]))
    counts = Counter(ids)
    duplicates = sorted(k for k, v in counts.items() if v > 1)
    missing = sorted(set(range(expected)) - set(ids))
    return {"result_files": len(paths), "valid_episode_files": len(ids), "unique_question_ids": len(counts),
            "duplicates": duplicates[:30], "duplicate_count": len(duplicates), "missing_count": len(missing),
            "missing_preview": missing[:30], "complete": len(counts) == expected and not duplicates}



def inventory_from_paths(name: str, records: dict[int, dict[str, Any]], expected: int = 2044) -> dict[str, Any]:
    base = ROOT / "experiment_result" / name
    paths = list(base.glob("**/result.pkl"))
    aggregate_paths = list(base.glob("results.pkl")) + list(base.glob("*/results.pkl")) + list(base.glob("shards/worker_*/results.pkl"))
    # A worker aggregate is named results.pkl (plural), while durable episode
    # files are result.pkl. Keep the two counts separate.
    aggregate_paths = sorted(set(aggregate_paths))
    numeric_ids = [int(p.parent.name) for p in paths if p.parent.name.isdigit()]
    counts = Counter(numeric_ids)
    duplicates = sorted(k for k, v in counts.items() if v > 1)
    missing = sorted(set(range(expected)) - set(records))
    return {"result_files": len(paths) + len(aggregate_paths), "aggregate_files": len(aggregate_paths),
            "valid_episode_files": len(numeric_ids), "unique_question_ids": len(counts),
            "duplicates": duplicates[:30], "duplicate_count": len(duplicates),
            "missing_count": len(missing), "missing_preview": missing[:30],
            "complete": len(records) == expected and not duplicates and not missing}

def trace_files(name: str) -> list[Path]:
    p = ROOT / "experiment_result" / name
    root = p / "jev_decisions.jsonl"
    return [root] if root.exists() else sorted((p / "shards").glob("worker_*/jev_decisions.jsonl"))

def build():
    total = 2044
    qmeta = {int(x.get("episode_id")): x for x in json.loads((ROOT / "data" / "express-bench.json").read_text())}

    qwen_p = read_episode_pickles(ROOT / "experiment_result" / "experiment_result_qwen")
    fair_p = read_episode_pickles(ROOT / "experiment_result" / "experiment_result_jev")
    fair8_p = read_episode_pickles(ROOT / "experiment_result" / "experiment_result_jev_8gpu")
    oracle_p = read_episode_pickles(ROOT / "experiment_result" / "experiment_result_jev_oracle")
    oracle8_p = read_episode_pickles(ROOT / "experiment_result" / "experiment_result_jev_oracle_8gpu")

    def enrich(pickles):
        rows = []
        for q, x in sorted(pickles.items()):
            if not isinstance(x, dict):
                continue
            qx = qmeta.get(int(q), {})
            r = {k: x.get(k) for k in ["question_ind", "type", "question", "answer", "gen_answer", "EAC", "geodesic_distance", "cnt_step", "goal_dis", "path_len", "scene"]}
            r["question_ind"] = int(q)
            r["type"] = r.get("type") or qx.get("type")
            r["question"] = r.get("question") or qx.get("question")
            rows.append(r)
        return rows

    qwen_rows, fair_rows, fair8_rows = enrich(qwen_p), enrich(fair_p), enrich(fair8_p)
    oracle_rows, oracle8_rows = enrich(oracle_p), enrich(oracle8_p)
    qm, fm, f8m = metrics(qwen_rows), metrics(fair_rows), metrics(fair8_rows)
    om, o8m = metrics(oracle_rows), metrics(oracle8_rows)

    decision_paths = {name: trace_files(name) for name in ["experiment_result_jev", "experiment_result_jev_8gpu", "experiment_result_jev_oracle", "experiment_result_jev_oracle_8gpu"]}
    dec_rows, dec_stats = {}, {}
    for name, files in decision_paths.items():
        dec_rows[name], dec_stats[name] = decision_stats(files)

    progress = progress_table()
    inventories = {
        name: inventory_from_paths(name, rows, total)
        for name, rows in [("experiment_result_qwen", qwen_p), ("experiment_result_jev", fair_p),
                           ("experiment_result_jev_8gpu", fair8_p), ("experiment_result_jev_oracle", oracle_p),
                           ("experiment_result_jev_oracle_8gpu", oracle8_p)]
    }

    fair_overlap = paired_stats(fair_p, fair8_p)
    oracle_overlap = paired_stats(oracle_p, oracle8_p)
    fair_oracle = paired_stats(fair8_p, oracle8_p)
    fair_trace = trace_consistency(decision_paths["experiment_result_jev"], decision_paths["experiment_result_jev_8gpu"], set(fair_p))
    oracle_trace = trace_consistency(decision_paths["experiment_result_jev_oracle"], decision_paths["experiment_result_jev_oracle_8gpu"], set(oracle_p))
    valid_oracle8 = {q for q, steps in read_trace(decision_paths["experiment_result_jev_oracle_8gpu"]).items()
                     if not any(x.get("fallback_reason") == "jev_http_402" for x in steps.values())}
    oracle8_contamination = {
        "episodes_with_http_402": len(set(x.get("episode_id") for x in dec_rows["experiment_result_jev_oracle_8gpu"] if x.get("fallback_reason") == "jev_http_402")),
        "records_with_http_402": sum(x.get("fallback_reason") == "jev_http_402" for x in dec_rows["experiment_result_jev_oracle_8gpu"]),
        "valid_no_402_episode_count": len(valid_oracle8),
        "valid_no_402_fraction_pct": round(len(valid_oracle8) / total * 100, 4),
    }

    comparison = [
        {"name": "论文 Fine-EQA", "C": 40.55, "C_star": 63.95, "E_path": 16.22, "d_T": 6.43, "source": "Table 2 / arXiv:2503.11117"},
        {"name": "论文 GOE", "C": 38.54, "C_star": 63.34, "E_path": 12.74, "d_T": 6.46, "source": "Table 2 / arXiv:2503.11117"},
        {"name": "论文 FBE", "C": 38.60, "C_star": 60.24, "E_path": 14.55, "d_T": 6.64, "source": "Table 2 / arXiv:2503.11117"},
        {"name": "论文 RE", "C": 36.95, "C_star": 62.75, "E_path": 12.06, "d_T": 7.26, "source": "Table 2 / arXiv:2503.11117"},
        {"name": "本地 Qwen3-VL-8B 复现", **qm, "source": "experiment_result_qwen / 2044题"},
        {"name": "JEV fair 单卡（42题）", **fm, "source": "experiment_result_jev"},
        {"name": "JEV fair 8GPU（2044题）", **f8m, "source": "experiment_result_jev_8gpu"},
        {"name": "JEV oracle 单卡（38题）", **om, "source": "experiment_result_jev_oracle"},
        {"name": "JEV oracle 8GPU（2044题，含402错误）", **o8m, "source": "experiment_result_jev_oracle_8gpu"},
    ]

    data = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(ROOT), "total_questions": total,
        "baseline": {"metrics": qm, "rows": qwen_rows, "by_type": question_types(qwen_rows), "inventory": inventories["experiment_result_qwen"]},
        "jev_fair": {"episodes": len(fair_p), "metrics": fm, "by_type": question_types(fair_rows), "decision_stats": dec_stats["experiment_result_jev"], "decisions": dec_rows["experiment_result_jev"][:1200], "inventory": inventories["experiment_result_jev"]},
        "jev_8gpu": {"episodes": len(fair8_p), "completed_questions": len(fair8_p), "completion_pct": round(len(fair8_p) / total * 100, 3), "metrics": f8m, "by_type": question_types(fair8_rows), "decision_stats": dec_stats["experiment_result_jev_8gpu"], "decisions": dec_rows["experiment_result_jev_8gpu"][:1800], "progress": progress, "inventory": inventories["experiment_result_jev_8gpu"]},
        "jev_oracle": {"episodes": len(oracle_p), "metrics": om, "by_type": question_types(oracle_rows), "decision_stats": dec_stats["experiment_result_jev_oracle"], "decisions": dec_rows["experiment_result_jev_oracle"][:1200], "inventory": inventories["experiment_result_jev_oracle"]},
        "jev_oracle_8gpu": {"episodes": len(oracle8_p), "metrics": o8m, "by_type": question_types(oracle8_rows), "decision_stats": dec_stats["experiment_result_jev_oracle_8gpu"], "decisions": dec_rows["experiment_result_jev_oracle_8gpu"][:1800], "inventory": inventories["experiment_result_jev_oracle_8gpu"], "contamination": oracle8_contamination},
        "comparison": comparison,
        "paired": {"fair_single_vs_8gpu": fair_overlap, "oracle_single_vs_8gpu": oracle_overlap, "fair_8gpu_vs_oracle_8gpu": fair_oracle,
                    "fair_trace": fair_trace, "oracle_trace": oracle_trace},
        "inventory": inventories,
    }
    # Keep the baseline sample-level data compact and useful for the explorer.
    data["baseline"]["rows"] = [{k: r.get(k) for k in ["question_ind", "type", "question", "EAC", "cnt_step", "goal_dis", "path_len", "geodesic_distance", "gen_answer", "scene"]} for r in qwen_rows]
    payload = json.dumps(safe_json(data), ensure_ascii=False, separators=(",", ":"))
    html = HTML.replace("__DATA__", payload).replace("__TABLE_SVG__", svg_table())
    OUT.write_text(html, encoding="utf-8")
    validation = {"html": str(OUT), "bytes": OUT.stat().st_size, "qwen_rows": len(qwen_p), "fair_episodes": len(fair_p), "fair8_episodes": len(fair8_p), "oracle_episodes": len(oracle_p), "oracle8_episodes": len(oracle8_p), "metrics": qm, "fair8_metrics": f8m, "oracle8_metrics": o8m}
    (ROOT / "report_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EXPRESS-Bench 实验结果分析</title>
<style>
:root{--bg:#0b1020;--panel:#121a30;--panel2:#182440;--ink:#ecf3ff;--muted:#98a9c6;--line:#2a3a60;--accent:#55d6a5;--blue:#70a1ff;--amber:#f3bd69;--red:#f07b8e}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#0b1020 0%,#101a31 55%,#111b2b 100%);color:var(--ink);font:14px/1.55 Inter,system-ui,-apple-system,Segoe UI,sans-serif}a{color:#9bc0ff}header{padding:34px 6vw 26px;border-bottom:1px solid var(--line);position:sticky;top:0;background:rgba(11,16,32,.94);backdrop-filter:blur(10px);z-index:4}h1{margin:0;font-size:31px;letter-spacing:-.03em}h2{margin:0 0 15px;font-size:22px}h3{margin:0 0 8px;font-size:16px}.sub{color:var(--muted);margin:5px 0 0}.nav{display:flex;gap:8px;margin-top:20px;flex-wrap:wrap}.nav button,.btn{border:1px solid var(--line);background:#172441;color:var(--ink);padding:8px 13px;border-radius:9px;cursor:pointer}.nav button.active,.btn.active{background:var(--accent);color:#0b1020;border-color:var(--accent);font-weight:700}main{max-width:1440px;margin:0 auto;padding:28px 6vw 80px}.page{display:none}.page.active{display:block}.grid{display:grid;gap:14px}.cards{grid-template-columns:repeat(auto-fit,minmax(175px,1fr))}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:17px}.metric .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.metric .value{font-size:29px;font-weight:750;margin:7px 0 2px}.metric .note{color:var(--muted);font-size:12px}.two{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}.three{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}.section{margin:26px 0}.callout{border-left:3px solid var(--accent);padding:13px 16px;background:#11253a;border-radius:8px;color:#d5e6ff}.warn{border-left-color:var(--amber);background:#302719}.bad{border-left-color:var(--red);background:#301d29}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}.tbl{width:100%;border-collapse:collapse;min-width:760px}.tbl th,.tbl td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.tbl th{color:#a9bbd9;background:#15213a;position:sticky;top:0}.tbl td.num,.tbl th.num{text-align:right;font-variant-numeric:tabular-nums}.bar{height:10px;background:#243454;border-radius:99px;overflow:hidden}.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--blue));border-radius:inherit}.bar.red>i{background:linear-gradient(90deg,var(--amber),var(--red))}.pill{display:inline-block;padding:2px 7px;border-radius:99px;background:#203357;color:#bcd4ff;font-size:12px}.pill.good{background:#123d39;color:#8ef0c7}.pill.warn{background:#49341d;color:#ffd28e}.filters{display:flex;gap:9px;flex-wrap:wrap;margin:12px 0}.filters input,.filters select{background:#101a31;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px}.mini{font-size:12px;color:var(--muted)}pre{white-space:pre-wrap;word-break:break-word;color:#c6d5ee}.svgbox{background:#fff;padding:12px;border-radius:12px;overflow:auto}.svgbox svg{min-width:800px;width:100%;height:auto}.legend{display:flex;gap:13px;flex-wrap:wrap;color:var(--muted);font-size:12px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}.dot.g{background:var(--accent)}.dot.b{background:var(--blue)}.dot.a{background:var(--amber)}details{background:#111d35;border:1px solid var(--line);border-radius:9px;padding:9px 12px;margin:8px 0}summary{cursor:pointer;color:#c7d8f7}.sticky-note{position:fixed;right:18px;bottom:18px;background:#162744;border:1px solid var(--line);padding:10px 13px;border-radius:10px;color:#bcd0ee;font-size:12px;max-width:320px;z-index:5}
</style></head><body>
<header><h1>EXPRESS-Bench 实验结果分析</h1><p class="sub">Baseline · JEV fair · JEV 8GPU · JEV oracle | 生成时间：<span id="generated"></span></p><div class="nav"><button class="active" data-page="overview">总览仪表盘</button><button data-page="paper">论文式分析</button><button data-page="consistency">一致性与 Oracle</button><button data-page="explorer">轨迹探索器</button><button data-page="ops">运行与分片</button></div></header>
<main>
<section id="overview" class="page active"><div class="section"><h2>一页结论</h2><div class="grid cards" id="metricCards"></div></div><div class="grid two"><div class="card"><h3>当前状态</h3><div id="stateCallout"></div></div><div class="card"><h3>指标口径</h3><p class="mini">C = clip(grd × acc, 0, 5) / 5 × 100；C* = clip(acc, 0, 5) / 5 × 100。E_path 在同时有 path_len 和 geodesic_distance 的样本上计算；d_T 只使用有限 goal_dis。未完成分片不计入性能均值。</p></div></div><div class="section"><h2>结论摘要</h2><div id="conclusion" class="grid two"></div></div><div class="section"><h2>论文基线与当前 baseline</h2><div id="comparison"></div></div><div class="section"><h2>问题类型分层</h2><div id="byType"></div></div></section>
<section id="paper" class="page"><div class="section"><h2>论文 Table 2（用户提供图表的可缩放重绘）</h2><p class="sub">下图以 SVG 内嵌，数值按附图重绘；当前 baseline 使用本地实验结果，二者受运行条件和完成度影响，不应直接当作论文复现。</p><div class="svgbox">__TABLE_SVG__</div></div><div class="section"><h2>可比性与解释</h2><div class="grid two"><div class="callout"><b>C* 是答案/一致性维度。</b><br>它反映 EAC 的 acc 分量，不能单独证明探索路径有效。</div><div class="callout"><b>E_path 同时惩罚低一致性和低路径效率。</b><br>长路径会压低结果；分母是有有效 path_len/geodesic 的样本。</div><div class="callout warn"><b>Oracle 是 privileged upper bound。</b><br>它使用当前 Habitat semantic FOV，应用于能力上界分析，不是公平主比较。</div><div class="callout warn"><b>Oracle 8GPU 的 API 失败会改变解释。</b><br>它有大量 HTTP 402 fallback；在修复服务配额前，不能把 oracle 结果当作干净的感知上界。</div></div></div><div class="section"><h2>结果表</h2><div id="paperTable"></div></div></section>
<section id="consistency" class="page"><div class="section"><h2>单卡 vs 8GPU：样本级一致性</h2><p class="sub">只在两次运行都存在 durable result.pkl 的同一 question_ind 上配对；fair 为 42 题，oracle 为 38 题。数值是右侧 8GPU 减左侧单卡。</p><div id="pairedTables"></div></div><div class="section"><h2>决策轨迹一致性</h2><div class="grid two"><div class="card" id="fairTrace"></div><div class="card" id="oracleTrace"></div></div></div><div class="section"><h2>Oracle 解释</h2><div class="grid two"><div class="callout"><b>Oracle 输入范围</b><br>JEV 只额外接收当前 Habitat semantic sensor FOV 的类别标签；不是全场景语义图，也不包含答案、目标位姿、参考轨迹或 geodesic distance。</div><div class="callout warn" id="oracleWarning"></div></div></div><div class="section"><h2>问题类型：全量 8GPU</h2><div id="allTypeTables"></div></div><div class="section"><h2>结果完整性审计</h2><div id="inventoryTable"></div></div><div class="section"><h2>来源与口径</h2><p>论文：<a href="https://arxiv.org/abs/2503.11117">Beyond the Destination: A Novel Benchmark for Exploration-Aware Embodied Question Answering</a>；本地 oracle 约束见 <code>Jev_scripts/README.md</code> 与 <code>Jev_scripts/tsdf_candidates.py</code>。报告中的 C、C*、E_path、d_T 按仓库 <code>evaluation.py</code> 的实现重算。</p></div></section>
<section id="explorer" class="page"><div class="section"><h2>样本/轨迹探索器</h2><p class="sub">baseline 的 2,044 个问题可筛选；JEV 路由轨迹以决策日志展示。点击样本可查看 EAC、路径、目标距离和场景。</p><div class="filters"><input id="qSearch" placeholder="搜索问题文本…"><select id="qType"><option value="">全部类型</option></select><select id="qState"><option value="">全部状态</option><option value="success">答案 acc=5</option><option value="nonzero">C>0</option><option value="missing">缺失路径/指标</option></select><button class="btn" id="clearFilter">清除</button></div><div id="explorerTable"></div></div><div class="section"><h2>轨迹分布</h2><div class="grid three" id="traceStats"></div></div></section>
<section id="ops" class="page"><div class="section"><h2>JEV 调用与 GPU 分片</h2><div class="grid three"><div class="card"><h3>Fair 单卡</h3><div id="fairOps"></div></div><div class="card"><h3>Fair 8GPU</h3><div id="fair8Ops"></div></div><div class="card"><h3>Oracle 8GPU</h3><div id="oracle8Ops"></div></div></div></div><div class="section"><h2>8GPU 进度</h2><div id="gpuTable"></div></div><div class="section"><h2>动作、fallback 与延迟</h2><div id="opsTable"></div></div><div class="section"><h2>读取限制</h2><div class="callout bad">JEV 的每题 result.pkl 同时保存 EAC、路径和决策轨迹；本报告仍将 JSONL 路由日志与 episode 结果分开审计。oracle 8GPU 的 402 错误会触发 fallback，不能当作纯粹的 oracle 上界。</div></div></section>
</main><div class="sticky-note">数据来自本地 EXPRESS-Bench 目录；报告无外部 CDN 依赖，可离线打开。</div>
<script>const DATA=__DATA__;document.getElementById('generated').textContent=DATA.generated_at;
const $=s=>document.querySelector(s), fmt=(v,d=2)=>v==null?'—':Number(v).toFixed(d), esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.nav button').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));$('#'+b.dataset.page).classList.add('active')});
function metricCard(label,value,note){return `<div class="card metric"><div class="label">${label}</div><div class="value">${value}</div><div class="note">${note}</div></div>`}
const bm=DATA.baseline.metrics;$('#conclusion').innerHTML=`<div class="callout"><b>Qwen 复现</b><br>2,044 题完整运行，C ${fmt(bm.C)}、C* ${fmt(bm.C_star)}、E_path ${fmt(bm.E_path)}、d_T ${fmt(bm.d_T)}。这些数值高于论文 Fine-EQA 的 40.55/63.95/16.22/6.43，但本地 Qwen 同时替代了原论文中的 GPT 接口与 EAC 评估器，应该作为“模型替换后的复现”，不能直接归因于方法改进。</div><div class="callout"><b>JEV 全量结果</b><br>fair 8GPU 的 C/C*/E_path/d_T 为 ${fmt(DATA.jev_8gpu.metrics.C)}/${fmt(DATA.jev_8gpu.metrics.C_star)}/${fmt(DATA.jev_8gpu.metrics.E_path)}/${fmt(DATA.jev_8gpu.metrics.d_T)}；与 Qwen baseline 相比，主要下降在一致性与探索效率，而不是路径是否能执行（path 样本覆盖仍为 100%）。</div><div class="callout warn"><b>单卡与 8GPU</b><br>重叠 fair 42 题的生成答案完全相同率 ${fmt(DATA.paired.fair_single_vs_8gpu.answer_exact_pct,1)}%，EAC 字符串完全相同率 ${fmt(DATA.paired.fair_single_vs_8gpu.eac_exact_pct,1)}%。答案文本较稳定，但 EAC/路径存在明显运行间波动。</div><div class="callout bad"><b>Oracle 判读</b><br>oracle 8GPU 的 ${DATA.jev_oracle_8gpu.contamination.records_with_http_402} 条 HTTP 402 使 ${DATA.jev_oracle_8gpu.contamination.episodes_with_http_402} 题进入服务错误 fallback；当前不能用它证明“oracle 不如 RGB”，应先重跑无 402 的 oracle。</div>`;$('#metricCards').innerHTML=metricCard('样本覆盖',`${bm.n}/${DATA.total_questions}`,`baseline 已完成 ${fmt(bm.n/DATA.total_questions*100,1)}%`)+metricCard('C',fmt(bm.C),`有效 EAC ${bm.metric_n} 题`)+metricCard('C*',fmt(bm.C_star),`答案 acc=5：${fmt(bm.answer_success,1)}%`)+metricCard('E_path',fmt(bm.E_path),`路径样本 ${bm.path_n} 题`)+metricCard('d_T',fmt(bm.d_T),`goal 样本 ${bm.goal_n} 题`)+metricCard('JEV 8GPU',`${DATA.jev_8gpu.completed_questions}/${DATA.total_questions}`,`完成度 ${fmt(DATA.jev_8gpu.completion_pct,1)}%`);
$('#stateCallout').innerHTML=`<p><span class="pill good">Qwen 完整</span> ${bm.n}/${DATA.total_questions} 题；JEV fair 8GPU ${DATA.jev_8gpu.episodes}/${DATA.total_questions}；oracle 8GPU ${DATA.jev_oracle_8gpu.episodes}/${DATA.total_questions}。</p><p><span class="pill warn">单卡仅小样本</span> fair ${DATA.jev_fair.episodes} 题，oracle ${DATA.jev_oracle.episodes} 题；8GPU 与单卡只能在重叠 question_ind 上做配对。</p><p><span class="pill warn">oracle 8GPU 受污染</span> ${DATA.jev_oracle_8gpu.contamination.records_with_http_402} 条 402，涉及 ${DATA.jev_oracle_8gpu.contamination.episodes_with_http_402} 题。</p>`;
function compareTable(){let rows=DATA.comparison;return `<div class="table-wrap"><table class="tbl"><thead><tr><th>条件</th><th class="num">C</th><th class="num">C*</th><th class="num">E_path</th><th class="num">d_T</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.name)} <span class="mini">${esc(r.source||'')}</span></td><td class="num">${fmt(r.C)}</td><td class="num">${fmt(r.C_star)}</td><td class="num">${fmt(r.E_path)}</td><td class="num">${fmt(r.d_T)}</td></tr>`).join('')}</tbody></table></div>`}$('#comparison').innerHTML=compareTable();$('#paperTable').innerHTML=compareTable();
$('#byType').innerHTML=`<div class="table-wrap"><table class="tbl"><thead><tr><th>类型</th><th class="num">题数</th><th class="num">C</th><th class="num">C*</th><th class="num">E_path</th><th class="num">d_T</th><th class="num">答案 acc=5</th></tr></thead><tbody>${DATA.baseline.by_type.map(r=>`<tr><td><span class="pill">${esc(r.type)}</span></td><td class="num">${r.n}</td><td class="num">${fmt(r.C)}</td><td class="num">${fmt(r.C_star)}</td><td class="num">${fmt(r.E_path)}</td><td class="num">${fmt(r.d_T)}</td><td class="num">${fmt(r.answer_success,1)}%</td></tr>`).join('')}</tbody></table></div>`;
function opBox(s){return `<p>决策记录 <b>${s.records}</b>，涉及题目 <b>${s.episodes}</b></p><p>平均延迟 <b>${fmt(s.mean_latency_ms)} ms</b>（P50 ${fmt(s.p50_latency_ms)} ms）</p><p>平均 token：输入 <b>${fmt(s.mean_input_tokens,0)}</b> / 输出 <b>${fmt(s.mean_output_tokens,0)}</b></p><p>fallback <b>${fmt(s.fallback_rate,1)}%</b></p><details><summary>动作分布</summary><pre>${esc(JSON.stringify(s.actions,null,2))}</pre></details><details><summary>fallback 原因</summary><pre>${esc(JSON.stringify(s.fallback_reasons,null,2))}</pre></details>`}$('#fairOps').innerHTML=opBox(DATA.jev_fair.decision_stats);$('#fair8Ops').innerHTML=opBox(DATA.jev_8gpu.decision_stats);$('#oracle8Ops').innerHTML=opBox(DATA.jev_oracle_8gpu.decision_stats);
function pairedTable(name,p){let row=(k,label)=>p[k]?`<tr><td>${label}</td><td>${fmt(p[k].left_mean)}</td><td>${fmt(p[k].right_mean)}</td><td>${fmt(p[k].delta_right_minus_left)}</td><td>${p[k].right_better}/${p[k].left_better}/${p[k].ties}</td><td>${fmt(p[k].pearson,3)}</td></tr>`:'';return `<h3>${name}（n=${p.n}，单卡独有 ${p.left_only}，8GPU 独有 ${p.right_only}）</h3><div class="table-wrap"><table class="tbl"><thead><tr><th>指标</th><th>单卡均值</th><th>8GPU均值</th><th>差值</th><th>8GPU优/单卡优/平</th><th>Pearson</th></tr></thead><tbody>${row('c','C raw (0–5)')}${row('cs','C* raw (0–5)')}${row('ep','E_path')}${row('dt','d_T')}${row('path','path_len')}${row('steps','cnt_step')}<tr><td>EAC 字符串完全相同</td><td colspan="5">${fmt(p.eac_exact_pct,1)}%</td></tr><tr><td>生成答案完全相同</td><td colspan="5">${fmt(p.answer_exact_pct,1)}%</td></tr><tr><td>步数完全相同</td><td colspan="5">${fmt(p.same_steps_pct,1)}%</td></tr></tbody></table></div>`}
$('#pairedTables').innerHTML=pairedTable('Fair',DATA.paired.fair_single_vs_8gpu)+pairedTable('Oracle',DATA.paired.oracle_single_vs_8gpu)+pairedTable('Fair 8GPU vs Oracle 8GPU',DATA.paired.fair_8gpu_vs_oracle_8gpu);
function traceBox(title,t){return `<h3>${title}</h3><p>可配对题目 ${t.durable_overlap_episodes}；可配对步 ${t.paired_steps}；单卡日志孤立题 ${t.left_trace_only}。</p><p>按步完全一致：action ${fmt(t.action_step_exact_pct,1)}%，fallback ${fmt(t.fallback_step_exact_pct,1)}%，target_id ${fmt(t.target_id_step_exact_pct,1)}%。按题完全一致：action ${fmt(t.action_episode_exact_pct,1)}%，fallback ${fmt(t.fallback_episode_exact_pct,1)}%。</p>`}$('#fairTrace').innerHTML=traceBox('Fair trace',DATA.paired.fair_trace);$('#oracleTrace').innerHTML=traceBox('Oracle trace',DATA.paired.oracle_trace);$('#oracleWarning').innerHTML=`<b>Oracle 8GPU 不能直接作上界。</b><br>日志中有 ${DATA.jev_oracle_8gpu.contamination.records_with_http_402} 条 HTTP 402，涉及 ${DATA.jev_oracle_8gpu.contamination.episodes_with_http_402}/${DATA.total_questions} 题；这些步全部进入 fallback。先修复配额/服务错误，再重跑 oracle 才能判断 RGB 解析瓶颈。`;
function typeTable(title,rows){return `<h3>${title}</h3><div class="table-wrap"><table class="tbl"><thead><tr><th>类型</th><th class="num">题数</th><th class="num">C</th><th class="num">C*</th><th class="num">E_path</th><th class="num">d_T</th><th class="num">acc=5</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.type)}</td><td class="num">${r.n}</td><td class="num">${fmt(r.C)}</td><td class="num">${fmt(r.C_star)}</td><td class="num">${fmt(r.E_path)}</td><td class="num">${fmt(r.d_T)}</td><td class="num">${fmt(r.answer_success,1)}%</td></tr>`).join('')}</tbody></table></div>`}$('#allTypeTables').innerHTML=typeTable('Qwen baseline',DATA.baseline.by_type)+typeTable('JEV fair 8GPU',DATA.jev_8gpu.by_type)+typeTable('JEV oracle 8GPU（含 402 fallback）',DATA.jev_oracle_8gpu.by_type);
let invRows=Object.entries(DATA.inventory).map(([n,x])=>`<tr><td>${n}</td><td>${x.result_files}</td><td>${x.aggregate_files}</td><td>${x.unique_question_ids}</td><td>${x.duplicate_count}</td><td>${x.missing_count}</td><td>${x.complete?'完整':'需检查'}</td></tr>`).join('');$('#inventoryTable').innerHTML=`<div class="table-wrap"><table class="tbl"><thead><tr><th>目录</th><th>result.pkl 文件</th><th>聚合文件</th><th>唯一题目</th><th>重复</th><th>缺失</th><th>结论</th></tr></thead><tbody>${invRows}</tbody></table></div>`;

$('#gpuTable').innerHTML=`<div class="table-wrap"><table class="tbl"><thead><tr><th>Worker/GPU</th><th>状态</th><th>完成</th><th>目标</th><th>进度</th><th>当前题/步</th></tr></thead><tbody>${DATA.jev_8gpu.progress.map(r=>{let pct=r.total?r.completed/r.total*100:0;return `<tr><td>W${String(r.worker).padStart(2,'0')} / GPU ${r.gpu}</td><td>${r.status==='completed'?'<span class="pill good">completed</span>':'<span class="pill warn">running</span>'}</td><td>${r.completed}（文件 ${r.files}）</td><td>${r.total}</td><td style="min-width:180px"><div class="bar"><i style="width:${pct}%"></i></div><span class="mini">${fmt(pct,1)}%</span></td><td>${r.current_question??'—'} / ${r.current_step??'—'}</td></tr>`}).join('')}</tbody></table></div>`;
$('#opsTable').innerHTML=`<div class="table-wrap"><table class="tbl"><thead><tr><th>条件</th><th>动作</th><th>状态</th><th>fallback 原因</th><th>记录</th><th>均延迟 ms</th></tr></thead><tbody>${[['fair 单卡',DATA.jev_fair.decision_stats],['fair 8GPU',DATA.jev_8gpu.decision_stats],['oracle 单卡',DATA.jev_oracle.decision_stats],['oracle 8GPU',DATA.jev_oracle_8gpu.decision_stats]].map(([n,s])=>`<tr><td>${n}</td><td><pre>${esc(JSON.stringify(s.actions))}</pre></td><td><pre>${esc(JSON.stringify(s.status))}</pre></td><td><pre>${esc(JSON.stringify(s.fallback_reasons))}</pre></td><td>${s.records}</td><td>${fmt(s.mean_latency_ms)}</td></tr>`).join('')}</tbody></table></div>`;
let rows=DATA.baseline.rows;const types=[...new Set(rows.map(r=>r.type||'unknown'))].sort();$('#qType').innerHTML+=[...types].map(t=>`<option>${esc(t)}</option>`).join('');function renderExplorer(){let q=($('#qSearch').value||'').toLowerCase(),t=$('#qType').value,st=$('#qState').value;let rs=rows.filter(r=>(!q||`${r.question||''} ${r.scene||''}`.toLowerCase().includes(q))&&(!t||(r.type||'unknown')===t)&&(!st||(st==='success'&&String(r.EAC||'').endsWith(',5'))||(st==='nonzero'&&parseFloat(r.EAC||'0')>0)||(st==='missing'&&(!r.path_len||!r.goal_dis))));let show=rs.slice(0,300);$('#explorerTable').innerHTML=`<p class="mini">显示 ${show.length}/${rs.length}（最多 300 行）</p><div class="table-wrap"><table class="tbl"><thead><tr><th>ID</th><th>类型/问题</th><th class="num">EAC</th><th class="num">步数</th><th class="num">path_len</th><th class="num">goal_dis</th><th>答案</th></tr></thead><tbody>${show.map(r=>`<tr><td>${r.question_ind}</td><td><span class="pill">${esc(r.type||'unknown')}</span><br>${esc(r.question)}</td><td class="num">${esc(r.EAC||'—')}</td><td class="num">${r.cnt_step??'—'}</td><td class="num">${fmt(r.path_len)}</td><td class="num">${fmt(r.goal_dis)}</td><td>${esc(r.gen_answer||'—')}</td></tr>`).join('')}</tbody></table></div>`;$('#traceStats').innerHTML=statsCards(rs)}function statsCards(rs){let vals=rs.map(r=>Number(r.cnt_step)).filter(Number.isFinite),g=rs.map(r=>Number(r.goal_dis)).filter(Number.isFinite),p=rs.map(r=>Number(r.path_len)).filter(Number.isFinite);let av=a=>a.length?(a.reduce((x,y)=>x+y,0)/a.length).toFixed(2):'—';return metricCard('样本',rs.length,'筛选后')+metricCard('平均步数',av(vals),'cnt_step')+metricCard('平均 goal distance',av(g),'goal_dis')+metricCard('平均 path_len',av(p),'path_len')}['qSearch','qType','qState'].forEach(id=>$('#'+id).addEventListener('input',renderExplorer));$('#clearFilter').onclick=()=>{$('#qSearch').value='';$('#qType').value='';$('#qState').value='';renderExplorer()};renderExplorer();
</script></body></html>'''


if __name__ == "__main__": build()
