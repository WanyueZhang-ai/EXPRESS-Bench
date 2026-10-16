# Efficiency analysis

Profiling is disabled by default. It covers Intern-Decision, the Prismatic
perception scorer/generator, Qwen3-VL, and the optional Jev API, as well as
Habitat observations/actions, TSDF, planning, visualization and storage.
It never runs an extra inference or changes the decision prompt/options.

## Run

For the complete Chinese server workflow (download/patch, 10-question validation,
monitoring, resume and report download), see
[server-efficiency-quickstart.zh-CN.md](server-efficiency-quickstart.zh-CN.md).

Use the normal dataset, model and Habitat environment. For a first check, use
`--sample-size 10 --sample-seed 42` or an existing `--indices-file`. All launchers
support these mutually exclusive selection methods. Exact original IDs are saved
in `selected_indices.json`, with provenance in `question_selection.json`.
Parallel sharding/resume retains these IDs and rejects changed subsets.
Omitting both selection options runs the full dataset.

```bash
# Detailed image encoding boundaries, generation timing and token accounting
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_decision_image.yaml \
  --output-dir experiment_result_efficiency \
  --efficiency-mode detailed

# Baseline: lightweight end-to-end measurements
python main.py --cfg-file fine_eqa.yaml \
  --output-dir experiment_result_baseline_efficiency --efficiency

# Independent workers, automatic trace/report merge, same resume protocol
python Jev_scripts/run_parallel.py \
  --cfg-file InternDecision_scripts/config_decision_image.yaml \
  --num-gpus 4 --gpus 0,1,2,3 \
  --output-dir experiment_result_parallel_efficiency \
  --efficiency-mode detailed
# Add --resume to the SAME parallel command after the workers have stopped.
```

The equivalent configuration is available in all seven experiment YAMLs:

```yaml
efficiency:
  enabled: true
  mode: detailed        # light or detailed
  warmup_calls: 1       # label first N calls per worker/model/purpose/modality
  report: true
```

`--efficiency` enables the configured mode, defaulting to `light`.
`--efficiency-mode` enables profiling and overrides the mode. The parallel
launcher forwards an explicit mode to every worker. Configuration alone also
works. Model prompts, sampling settings, seeds and episode budgets are unchanged.

## Timing definitions

Each model call starts when its wrapper accepts the prepared request/image
reference and ends when the parsed decision, score or answer returns. It
includes prompt construction, reading/processing images, tokenization, device
transfer, inference and output conversion. Evidence construction and router
validation are additional step spans. Model loading is recorded separately;
Qwen's lazy first load remains part of that step/episode's actual total.

| Field/stage | Meaning |
| --- | --- |
| `wall_ms` | Actual elapsed host time to return from the call/step/episode |
| `input.processor`, `input.image_preprocess`, `input.tokenize` | Processor/tokenizer work, distinct from neural image encoding |
| `input.prepare` | Checkpoint adapter's `encode()` preparation, when exposed |
| `vision.encode` | Real visual module forward, including its internal merger when integrated |
| `vision.project` | Separate feature projector, when exposed (Prismatic) |
| `input_to_image_encoded_ms` | Call entry through completion of the visual features used by the LLM |
| `image_encoded_to_output_ms` | That completion boundary through the returned decision/answer |
| `generation` | Generation invocation including multimodal prefill and decoding |
| `input_to_first_token_ms` | Call entry through receipt of the first generated token; detailed HF streaming only |
| `generation_tokens_per_second` | Actual generated token count divided by `generation` wall seconds, including prefill |
| `decode_tokens_per_second` | Tokens after the first divided by time after the first token; only when observed |
| `gpu_timeline_ms` | CUDA Event elapsed timeline on the configured device; separate from host time |

`stages` have **inclusive** wall times. Do not add a call's total, its forward
and its encoder durations: they overlap. `category_wall_ms` uses **exclusive**
times and partitions the enclosing wall duration. A child's own category takes
precedence (for example, drawing inside a planner belongs to visualization).
Uninstrumented work, Python bookkeeping and logging remain in `other`.

An exploration step is one `cnt_step` cycle. `simulator_transition_count`
counts actual `simulator.step()` calls; `observation_count` counts explicit
observation requests. A cycle can contain zero transitions or three rotations.
Direct-action `continue`, answer `break`, black frames and exceptions are
finalized. Loop-external fallback answers and EAC scoring belong to the episode.

`task_completion_ms` measures episode entry through availability of the final
answer. `evaluation_ms` measures EAC scoring. Episode `wall_ms` also includes
scene setup, final distance checks and output writing. Run traces include shared
model startup, all episodes and cleanup. A step excludes writing its own
efficiency row; time spent writing child rows is included in its enclosing wall
time. `profiling_write_ms` reports measured trace-write overhead for the run.

### Light and detailed modes

**Light** keeps ordinary wall timings and reads real token IDs/feature shapes.
It adds no explicit CUDA timing synchronization and no token streamer. Neural
encoding boundaries and TTFT are `null`. Prismatic's string-only `generate()`
does not expose exact generated IDs in light mode, so its generated count is
`null`; the benchmark's usual `get_loss()` calls are non-generative and count 0.

**Detailed** attaches temporary forward hooks to the loaded modules, uses CUDA
Events and synchronizes phase boundaries on the configured device. Supported
paths include Qwen `visual`/`language_model` and Prismatic
`vision_backbone`/`projector`/`llm_backbone`; wrappers such as a checkpoint engine's
`backend.model` are discovered without changing checkpoint files. The actual
paths are written to `observed_module_paths`. Encoding ends after a separate
projector when present, otherwise after the visual module's integrated output.
Repeated encoding calls are counted; the split uses the last completed boundary.
For generative HF paths, a streamer observes actual tokens. Synchronization,
hooks and streaming (including token copies to CPU) add overhead: compare runs
using the same measurement mode. Use light mode for production-like wall time.

No percentage estimate substitutes for an unobservable boundary. An unsupported
checkpoint adapter records `image_encoding_status: unavailable`, a reason and
`null` durations. Text-only requests record `not_applicable`; light image calls
record `not_instrumented_in_light_mode`. Validate the observed paths on the
installed checkpoint before using detailed numbers in a paper. The supplied
experiment launchers keep each worker's models on its selected single GPU.

## Token accounting

- `text_input_tokens`: active text **and non-image special** IDs, including
  system prompts, history, candidates and structured decision markers.
- `image_input_tokens`: expanded image positions in HF input IDs, or actual
  projected feature sequence positions for Prismatic. Parallel DINO/SigLIP
  towers are not counted twice; raw patches are not substituted for LLM positions.
- `effective_input_tokens`: active multimodal sequence length. Padding is excluded.
- `generated_tokens`: new output IDs including EOS/other special output IDs.
  Decisions and candidate scoring are non-generative and record 0. Their
  `decision_fields` and `candidate_count` are separate work descriptors.

Backend `reported_usage`/`reported_timing` are retained separately. For remote
Jev, token numbers are provider-reported and transport/retry waits are separate
stages. No server-internal encoding timings can be observed. Each count has a
`token_sources` entry. Unknown counts are `null`, never guessed or replaced with
0. Step totals are `null` unless **all** child calls provide that count;
`known_token_subtotals` and coverage fields show partial information explicitly.

## Outputs and reports

Each worker writes these files into its normal output directory:

- `efficiency_calls.jsonl`: one row per real model/API invocation.
- `efficiency_steps.jsonl`: one row per exploration cycle, including tokens,
  simulator transitions, actions/fallbacks and process allocator memory peaks.
- `efficiency_episodes.jsonl`: one row per question attempt, including answer
  time, EAC time, path length, terminal distance and EAC result.
- `efficiency_runs.jsonl`: run identity, status, startup stages, Git revision,
  package versions, selected configuration, checkpoint paths and prompt hashes.
- `efficiency_summary.json`, `efficiency_summary.csv`: distributions, all-call
  and steady-state populations, token coverage, fallback/retry counts and groups.
- `efficiency_report.html`: self-contained offline report, searchable/sortable
  model table, step cost shares, P50/P95, integrity notes and a JSON download.

Memory is PyTorch **process allocator** memory on the worker GPU, including
all resident models. Peaks are reset at step entry. This is neither the memory
of one isolated model nor total GPU memory (Habitat/OpenGL and other processes
are outside this allocator). Unavailable CPU memory measurements are omitted.

Model groups distinguish checkpoint, purpose, modality, profiling mode and
status; question-type groups are also supplied. EAC calls are labelled
`eac_evaluation`; they do not contaminate final-answer groups. Quality metrics
use the repository's existing EXPRESS aggregation. `episode_records` allow
joining task effectiveness, time and token cost by condition/question.

Warmup only labels existing calls: no extra inference, random sampling or state
transition is executed. All calls remain in raw logs and all-call statistics;
steady-state statistics exclude the configured first N calls for each
worker-run/model/purpose/modality. Failed calls remain in their status groups.

Rebuild a report without loading model weights, Habitat, Torch or NumPy:

```bash
python tools/build_efficiency_report.py experiment_result_efficiency \
  --output-dir experiment_result_efficiency

python tools/build_efficiency_report.py \
  experiment_result_parallel_efficiency/shards/worker_* \
  --output-dir experiment_result_parallel_efficiency --merge-traces
```

Inputs to a merged report should be shards of one experiment. Raw records carry
`run_id`, `record_id` and `attempt_id`. Exact repeated record IDs are deduplicated;
conflicting copies fail explicitly. On resume, the report selects the latest
successful attempt per condition/question, or the latest failed attempt if none
succeeded. Older and unfinished attempts remain in raw logs and their cost is
reported under `diagnostics.excluded_call_statistics`. A killed worker's
unterminated final JSON row is ignored when reading; before appending, its bytes
are saved as `*.truncated-*` and only that damaged tail is removed. Invalid
complete records fail validation. The parallel launcher also writes a partial
report before raising for failed workers. Raw efficiency traces do not contain
question text, image contents, HTTP headers or API keys.

## Validation

```bash
python -m pytest -q tests/test_efficiency*.py
```

Tests cover exclusive accounting, token ID counts, actual temporary Torch
hooks, exceptions, checkpoint preparation restoration, retry costs, resume and
report integrity, actual parallel subprocess merging and failed-worker recovery,
plus execution of the real `main()` body with synthetic scene/model boundaries.
These CPU tests do not measure actual checkpoint/GPU
performance. In the real environment, run a small identical index set with
profiling disabled, light and detailed; verify decisions/results, observed
module paths, token coverage and expected phase completion before a full run.
