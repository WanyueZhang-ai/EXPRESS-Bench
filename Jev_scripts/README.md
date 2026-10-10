# Jev-full-router for EXPRESS-Bench

This directory contains the TypeSafe Jev decision layer for EXPRESS-Bench.
Jev receives JSON state and typed questions. RGB images remain with the local
VLM; Jev does not receive image bytes.

## Conditions

- **fair**: question, history, local VLM outputs, and TSDF/planner candidates.
- **oracle_semantic**: the same state and candidate generation plus semantic
  labels visible in the *current Habitat semantic sensor FOV*. It is a
  privileged upper bound. A full-scene semantic graph, answer, goal pose,
  reference trajectory, or geodesic distance is rejected.

Both conditions use the same candidate IDs and executor. Jev can return
`stop`, `goe`, or `fbe`; a target is accepted only when its ID is currently
feasible and has the selected mode prefix. Invalid, stale, low-confidence, or
failed API responses fall back to the exact baseline action and are recorded.

## Install and API contract

The scripts use only the Python standard library for the API call. No
TypeSafe SDK is required. Set the key without putting it in a config file:

```bash
export TYPESAFE_API_KEY='...'
```

The client sends:

```text
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY
Content-Type: application/json
```

with `{"state": ..., "model": "jev-latest", "questions": ...}`. It uses a
20 second timeout and two bounded exponential retries for transport errors and
HTTP 408/425/429/5xx. A per-run call budget can stop additional requests; those
steps use the baseline action.

## Offline smoke test

The default fixture never calls the network:

```bash
python -m Jev_scripts.run_jev_replay \
  --input Jev_scripts/fixtures/sample_state.json \
  --output-dir /tmp/jev_fair_smoke
```

For the oracle schema:

```bash
python -m Jev_scripts.run_jev_replay \
  --input Jev_scripts/fixtures/sample_oracle_state.json \
  --mode oracle_semantic \
  --output-dir /tmp/jev_oracle_smoke
```

Each output directory contains:

- `decisions.jsonl`: one secret-free decision trace per step;
- `summary.json` and `summary.csv`: action/fallback/latency/token counts,
  question-type strata, and the EXPRESS-Bench metrics
  `C_avg`, `C_star_avg`, `E_path`, and `d_T_avg` when episode results are
  supplied.

## Live Jev smoke test

Use the same command with `--live` after setting the key:

```bash
python -m Jev_scripts.run_jev_replay \
  --live \
  --input Jev_scripts/fixtures/sample_state.json \
  --max-calls 1 \
  --output-dir /tmp/jev_live_smoke
```

The request state is not printed and the API key is never written to traces.
A real EXPRESS-Bench run should call `DecisionRouter.decide(...)` after each
current-step perception/map update and pass `decision.action` and
`decision.target_id` to the existing GOE/FBE executor. The action is therefore
part of the online trajectory, not a post-hoc label.

## Experiment protocol

Run the baseline and Jev conditions with identical split, seed, checkpoints,
prompts, sensor resolution, planner parameters, and step budget. Keep the
stop threshold fixed for the main comparison; use `0.60`, `0.75`, and `0.90`
only as a separate threshold ablation. Report the four existing metrics plus
answer step, stop rate, GOE/FBE share, Jev calls, input/output tokens, mean
latency, and fallback rate. Pair by `episode_id` and stratify by question type
and target region. Record the oracle condition as privileged.

For paired confidence intervals in a comparison script, import
`paired_bootstrap_ci` and pass the two JSONL record lists plus one of
`C_avg`, `C_star_avg`, `E_path`, or `d_T_avg`. The result is right-minus-left
with a deterministic seed and percentile interval.

## Online EXPRESS-Bench runs

Make separate config copies so baseline, fair, and oracle outputs are never
mixed:

```bash
cp fine_eqa.yaml fine_eqa_jev.yaml
# In fine_eqa_jev.yaml set:
#   output_dir: ./experiment_result_jev_fair
#   jev.enabled: true
#   jev.mode: fair
python main.py -cf fine_eqa_jev.yaml

cp fine_eqa.yaml fine_eqa_jev_oracle.yaml
# In fine_eqa_jev_oracle.yaml set:
#   output_dir: ./experiment_result_jev_oracle
#   jev.enabled: true
#   jev.mode: oracle_semantic
python main.py -cf fine_eqa_jev_oracle.yaml
```

The oracle run requires the annotated HM3D scene dataset and semantic assets.
If the current semantic sensor observation is missing, the run stops with an
explicit error instead of silently changing to fair mode.

## Multi-GPU parallel execution

`run_parallel.py` uses one independent process per GPU. Each process loads its
own VLM and Habitat simulator, so this is data parallelism; it does not split a
single model across GPUs. The launcher keeps each worker's original
`question_ind`, writes `shards/worker_XX/`, and merges only after every worker
has produced a complete `results.pkl`.

Use one, four, or eight workers by changing `--num-gpus`:

```bash
# 1 GPU (default physical device 0)
python Jev_scripts/run_parallel.py \
  --cfg-file fine_eqa_jev.yaml \
  --num-gpus 1 \
  --output-dir experiment_result_jev_fair_1gpu

# 4 GPUs, explicitly selected physical devices
python Jev_scripts/run_parallel.py \
  --cfg-file fine_eqa_jev.yaml \
  --num-gpus 4 --gpus 0,2,5,7 \
  --output-dir experiment_result_jev_fair_4gpu

# 8 GPUs (devices 0 through 7)
python Jev_scripts/run_parallel.py \
  --cfg-file fine_eqa_jev.yaml \
  --num-gpus 8 \
  --output-dir experiment_result_jev_fair_8gpu
```

The same launcher works for `jev.mode: oracle_semantic`. Use a separate output
directory for fair and oracle conditions. Add `--dry-run` to create the shard
manifests and inspect the exact worker commands without starting Habitat. Add
`--overwrite` only when intentionally restarting a previous generated output
directory. The merged directory contains `results.pkl`,
`jev_decisions.jsonl`, `parallel_summary.json`, and the per-worker logs.

### 查看并行进度

并行启动器会每隔 5 秒打印一行汇总进度，例如：

```text
[parallel] questions 512/2044 | w0:gpu0 running 64/256 q=64 step=7 | ...
```

可用 `--progress-interval 2` 调整刷新周期。每个 worker 目录还会持续更新
`progress.json`，其中包含 `status`（`queued`、`starting`、`loading_model`、
`running`、`completed`、`failed`）、已完成问题数、当前问题索引和当前仿真步数。
模型加载期间会显示 `loading_model`，因此长时间没有问题完成时也能区分是在加载模型还是正在处理问题。

每个 worker 的详细 VLM/Habitat 日志在：

```bash
tail -f experiment_result_jev_fair_8gpu/shards/worker_00/worker_stdout.log
```

### 中断后补跑与断点续跑

完成当前仍在运行的 worker 后，可以对同一个输出目录使用 `--resume`：

```bash
python Jev_scripts/run_parallel.py \
  --cfg-file fine_eqa_jev.yaml \
  --num-gpus 8 \
  --output-dir experiment_result_jev_8gpu \
  --resume
```

`--resume` 会检查每个 worker 下的单题目录 `<question_ind>/result.pkl`（以及已有的
`results.pkl`），只把缺失索引写入 `pending_indices.json`。已经完成的 worker 会直接跳过，
没有完成的 worker 只处理剩余问题；合并时会同时读取历史单题结果和本次补跑结果。Jev trace
在补跑时以追加方式保存，不会清空之前的决策记录。

普通再次启动不会自动跳过已有结果；需要显式写 `--resume`。不要在原任务的其他 worker
仍运行时对同一个输出目录执行补跑，避免两个进程同时写同一 shard。
