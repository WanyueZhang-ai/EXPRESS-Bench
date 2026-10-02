# Intern-Decision-4B Decision Experiments Design

## Goal

Add reproducible local Intern-Decision-4B decision experiments to EXPRESS-Bench
without changing the benchmark's existing episode scoring contract. The
implementation must support three evidence conditions and an optional action
generation condition:

1. existing VLM/TSDF state plus Intern-Decision (`vlm_state`),
2. current RGB image plus Intern-Decision (`decision_image`),
3. Habitat object summaries plus Intern-Decision (`decision_objects`), and
4. direct atomic Habitat actions selected by Intern-Decision (`direct_actions`).

The final answer generator is configurable: `qwen_vl` remains the default and
`intern_decision` is available for an ablation.

## Scope and compatibility

The existing Habitat loop, TSDF planner, GOE/FBE executor, result pickle format,
and `evaluation.py` metrics remain compatible. Jev is not required for the new
conditions. New code lives in `InternDecision_scripts/`; existing `Jev_scripts/`
continues to work unchanged.

## Architecture

`intern_decision_engine.py` lazily imports the model's local `inference.py` and
reuses one `DecisionEngine` instance. It accepts checkpoint, device, dtype,
temperature, media root, maximum input length, attention implementation, and an
optional inference module path from configuration. The loader resolves
`inference.py` with `importlib.util.spec_from_file_location` relative to the
checkpoint, so execution does not depend on CWD or PYTHONPATH, and reports clear errors when model files or dependencies are
unavailable. It never attempts network access.

`decision_router.py` compiles the shared question into Intern-Decision's
structured schema. Planner mode asks `stop` (`noul`), `mode` (`choice` with
`goe`/`fbe` only when that candidate list is non-empty), and `target` (`choice`
over stable current candidate IDs). Direct mode asks `stop` and `action`
(`choice` over the fixed atomic action whitelist). It maps
`answers[field].noul`, `choice`, `probabilities`, and `confidence`, and rejects
malformed, non-finite, out-of-range, stale, or wrong-prefix values. Invalid
responses fall back to the baseline action and are recorded. Fake-engine tests
implement this exact request/response contract.

`evidence_adapters.py` creates JSON-safe state for the three evidence modes.
`evidence_mode` is one of `vlm_state`, `decision_image`, or
`decision_objects`. Planner candidates are truncated to `max_candidates` before
prompt construction, normalized in deterministic planner order, and assigned
stable `goe_0`/`fbe_0` IDs. The same IDs are used for the prompt, target-to-index
mapping, executor call, and trace. Candidate criteria include only safe planner
features already materialized at the step (mode, region/category, direction,
counts, point availability, and bounded relative features); IDs map separately to
executor indices, and geodesic/goal/reference fields are rejected. If no candidates exist, the corresponding
question is omitted and the router uses the exact baseline route.

The image mode passes the same-timestep current RGB frame as one absolute,
readable path in `images`. Object mode uses only object ids present in the current semantic sensor frame;
it never serializes the full scene object list. Each visible object contains a
stable local id, category, and optional camera/agent-relative center and extent
in Habitat meters. Hidden/offscreen objects, target metadata, and world-global
scene graphs are excluded. A recursive allowlist is used for object and history
fields, and forbidden privileged fields such as gold answers, goal poses,
reference trajectories, geodesic distances, and raw goal metadata are rejected
before serialization. Smoke fixtures include hidden and offscreen objects and
assert they are absent. Candidate metadata is
sanitized before evidence and trace output. Every request records the simulator
step and decision id so RGB, object, and planner snapshots cannot be mixed.

`run_intern_decision.py` is the online runner. It reuses the existing question
loop and executor, selects an evidence mode and action mode from configuration,
writes per-step secret-free JSONL traces, and preserves per-episode `result.pkl`
files. The new hook is inserted at the same point as the existing Jev hook: after
planner state/candidates are materialized and before the existing
`target_index`/GOE/FBE executor branch. Existing Jev configuration and path are
left unchanged.

`run_intern_decision_smoke.py` uses a fake engine and fixture state to test
request compilation, image plumbing, object serialization, action routing, and
fallback without Habitat or model weights.

## Data flow

In planner mode, the current observation updates the existing planner and its
GOE/FBE candidates. The adapter combines those candidates with the question,
history, and one of the three evidence payloads. Intern-Decision scores all
fields in one forward pass. A stop decision invokes the configured answer
backend; otherwise the selected feasible target is passed to the existing
GOE/FBE executor.

In direct mode, the decision model receives the current RGB/object evidence,
question, short action history, and a fixed whitelist of atomic actions:
`move_forward`, `turn_left`, `turn_right`, `look_left`, `look_right`, and
`stop`. Direct mode may use `decision_image` or `decision_objects`; `vlm_state`
is reserved for planner-only ablations. The executor maps one choice to one
Habitat transition: `move_forward` invokes the existing forward action,
`turn_left`/`turn_right` rotate by the configured camera increment, and
`look_left`/`look_right` rotate without translation. The default whitelist is
`move_forward`, `turn_left`, `turn_right`, and `stop`; look actions are optional
and allowed only when explicitly present in the configured action space. Turns use
Habitat's configured turn angle and forward uses its configured stride. The runner
validates every configured action against `agent.agent_config.action_space` and
fails configuration validation rather than silently no-oping. A transition's
collision signal is read from the existing control/action return value and
recorded. `stop` invokes the answer
backend before any environment transition. Invalid action names do not step the simulator and immediately use the configured
safe rotation. A collision after `move_forward` records the collision and uses a
deterministic turn on the next decision cycle; it never issues a duplicate
forward transition in the same cycle and never switches to planner mode. Every
transition and fallback is logged. When `max_direct_actions` or `max_steps` is
reached, the runner forces `stop`, invokes the answer backend, and writes the
normal result; fallback transitions count toward both limits.

## Configuration

Each run selects:

```yaml
intern_decision:
  enabled: true
  checkpoint: /ssd/wyzhang/all_models/Intern-Decision-4B
  device: cuda:0
  dtype: bfloat16
  evidence_mode: vlm_state  # vlm_state, decision_image, decision_objects
  action_mode: planner_candidates  # planner_candidates or direct_actions
  answer_backend: qwen_vl  # qwen_vl or intern_decision
  answer_checkpoint: /ssd/wyzhang/all_models/Intern-Decision-4B
  answer_options: {}  # finite, non-gold options required for Intern answers
  temperature: 1.99241824
  max_length: 8192
  attn_implementation: sdpa
  max_candidates: 8
  stop_threshold: 0.75
  min_choice_confidence: 0.0
  max_direct_actions: 1
  direct_action_whitelist: [move_forward, turn_left, turn_right, stop]
  safe_direct_action: turn_right
  max_steps: 0  # 0 means use the benchmark-derived planner budget
  deterministic: true
  seed: 42
```

Intern-Decision is deterministic: it scores logits and does not sample; temperature is
probability calibration, not a decoder temperature. Qwen answer generation keeps
its existing greedy setting. The configured seed is recorded for simulator and
trace reproducibility.

The default stop threshold applies to the `stop` answer's `noul` probability.
`min_choice_confidence` applies independently to selected `mode`, `target`, or
direct `action`. Ties use the model wrapper's deterministic lexical argmax.
Planner mode falls back to the exact baseline GOE/FBE action and target; direct
mode falls back to the configured safe rotation and remains image/object-only. All fallback
reasons are enumerated in the trace.

Every condition has a separate output directory. The trace records the schema
version, question index, simulator step, decision id, evidence mode, action mode,
answer backend, checkpoint/model name, selected action/target, fallback reason,
latency, token usage, sanitized candidate ids, and deterministic runtime
settings, without model secrets or gold labels. Error strings are normalized and
never include request contents or credentials.

## Final answer contract

`qwen_vl` uses the existing answer prompt and image path. `intern_decision` uses the same `DecisionEngine.predict(request)` API (there is
no free-form `generate` or `gen_answer` method). Because EXPRESS-Bench answers
are often open-ended, this backend is enabled only when configuration supplies a
finite `answer_options` vocabulary or a per-question-type schema (for example,
`noul` for yes/no state questions). These options are static configuration and
never derived from the gold answer. The answer request includes the same-timestep
question and evidence state used by the selected decision adapter, plus the
current RGB path in `images` for `decision_image`; it never includes gold or
future fields. The adapter submits one text `choice`/`noul` question, maps
`answers.answer.choice` or `answers.answer.noul` to the existing `gen_answer`
string, and otherwise records `answer_fallback=baseline_vlm` and uses Qwen. If both Intern and Qwen answer generation fail, the run writes an
empty `gen_answer`, `answer_status=error`, and the exception class in the result
without aborting trace serialization. Answer model loading is shared with the
decision engine when the checkpoint is identical.

## Error handling and fairness

Model load, input validation, image read, inference, or response-schema errors
never silently switch evidence modes. They produce an explicit fallback trace.
The Intern model is loaded once per process with the configured local checkpoint,
device, dtype, temperature, maximum length, media root, and attention
implementation; network access is never attempted. Planner and direct runs use
the same seed, question indices, camera settings, observation counts, action
increments, step budget, answer scoring, and output format. No extra simulator
transition is inserted for decision capture. Smoke and integration fixtures assert
that planner/direct result.pkl files contain the existing `question_ind`, `path`,
`gen_answer`, `EAC`, `cnt_step`, `goal_dis`, and `path_len` keys with compatible
types for both answer backends. `direct_actions` is an additional
autonomy ablation rather than a replacement for the fair planner comparison.

## Verification

The smoke test runs first and must pass without GPU, Habitat, or network. It
covers all evidence modes, malformed output/fallback, image path lifecycle,
object redaction, direct action mapping, candidate feature redaction, and answer
normalization using a fake engine. It asserts direct mode remains image/object
only and does not transition into planner mode on collision. A single-index integration command then verifies model loading, one
same-timestep Habitat step, trace output, and result serialization compatible
with `evaluation.py`. Existing Jev and project tests are run to ensure the new
package does not regress the baseline.
