# InternDecision_scripts

These scripts add a local [Intern-Decision-4B](https://github.com/InternLM/Intern-Decision)
decision backend to EXPRESS-Bench. The checkpoint is loaded from
`/ssd/wyzhang/all_models/Intern-Decision-4B` by default and must contain the
model's `inference.py`, `config.json`, tokenizer, processor, and weights.
Install the versions shipped with that checkpoint in the environment used to
run the model (the current file requests `torch==2.9.1` and
`transformers==5.14.1`):

```bash
python -m pip install -r /ssd/wyzhang/all_models/Intern-Decision-4B/requirements.txt
```

Use Python 3.12+ for the model environment, as required by the checkpoint. All requests are local and offline. If the environment has an older
Transformers build, the launcher reports the missing `Qwen3_5` class and the
same requirements path instead of silently using another model.

## Offline smoke test

This does not load Habitat, Transformers, CUDA, or model weights:

```bash
cd /ssd/wyzhang/EXPRESS-Bench
python -m InternDecision_scripts.run_intern_decision_smoke
```

## Three planner experiments

Each command keeps the existing Habitat episode loop and result format. The
dataset currently contains 2044 questions. The commands below process the
complete dataset; do not add `--indices-file` unless you intentionally want a
subset.

### Full dataset

Use a separate output directory for each condition:

```bash
# 1. Existing VLM/TSDF perception -> Intern-Decision planner
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_vlm_state.yaml

# 2. Intern-Decision reads the current RGB frame directly -> planner
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_decision_image.yaml

# 3. Intern-Decision reads current semantic-sensor-visible objects -> planner
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_decision_objects.yaml
```

### One-question check

Use this only to validate the model and scene assets before a full run:

```bash
printf '[0]\n' > /tmp/express_one.json

# VLM/TSDF perception -> Intern-Decision planner
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_vlm_state.yaml \
  --indices-file /tmp/express_one.json

# Intern-Decision reads the current RGB frame directly -> planner
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_decision_image.yaml \
  --indices-file /tmp/express_one.json

# Intern-Decision reads current semantic-sensor-visible objects -> planner
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_decision_objects.yaml \
  --indices-file /tmp/express_one.json
```

Override the checkpoint or GPU without editing YAML:

```bash
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_decision_image.yaml \
  --checkpoint /ssd/wyzhang/all_models/Intern-Decision-4B \
  --device cuda:0
```

The same worker device is now used for the local Qwen answer/EAC model and,
for planner conditions, the VLM used to materialize GOE/FBE candidates. An
explicit `--device` therefore keeps a worker on one GPU instead of allowing
Transformers `device_map="auto"` to shard Qwen across every visible GPU.

`decision_objects` enables Habitat's annotated semantic scene configuration and
requires the HM3D semantic assets. It serializes only object ids present in the
current semantic sensor frame, category, and optional agent-relative geometry;
it does not send a full scene graph or target metadata.

## Direct autonomous action ablation

This mode does not use VLM/TSDF candidate generation for the decision. Intern-
Decision receives the current RGB frame and chooses one validated Habitat action
from `move_forward`, `turn_left`, `turn_right`, or `stop`:

```bash
# Full dataset
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_direct_actions.yaml

# One-question check: add --indices-file /tmp/express_one.json
```

One model choice produces at most one simulator transition. Invalid actions use
the configured safe rotation; a collision after `move_forward` is recorded and
the next decision cycle uses a deterministic turn. Direct mode never silently
switches to the planner. `max_steps` limits the episode; fallback transitions
count toward that limit.

## Final-answer backend

All supplied configs use `answer_backend: qwen_vl`, preserving the existing
Qwen answer generation and EAC scoring. Intern-Decision is a structured choice
model rather than a free-form generator. To use it for final answers, add a
finite, static, non-gold vocabulary under `answer_options` (or configure a
per-question-type schema), then run with:

```yaml
intern_decision:
  answer_backend: intern_decision
  answer_options:
    existence: ["yes", "no"]
    state:
      type: noul
    attribute:
      type: choice
      criteria:
        white: "white"
        black: "black"
```

For open-ended question types without configured options, the answer adapter
explicitly falls back to Qwen. If both local Intern and Qwen answer calls fail,
the episode still writes `result.pkl` with `gen_answer: ""`,
`answer_status: error`, and the exception class; the decision trace is retained.

## Outputs and resume

Each run writes:

- `<output>/intern_decisions.jsonl`: sanitized per-step decision traces;
- `<output>/<question_ind>/result.pkl`: the existing EXPRESS-Bench episode result;
- `<output>/progress.json`: progress state used by the existing parallel launcher;
- `<output>/intern_decision.log`: launcher/Habitat log.

The existing `Jev_scripts/run_parallel.py` can launch independent workers using
an Intern config because the config and runner preserve `--indices-file`,
`--output-dir`, and progress-file semantics. For example:

```bash
python Jev_scripts/run_parallel.py \
  --cfg-file InternDecision_scripts/config_decision_image.yaml \
  --num-gpus 4 --gpus 0,1,2,3 \
  --output-dir experiment_result_intern_decision_image_4gpu
```

The launcher now merges both `jev_decisions.jsonl` and
`intern_decisions.jsonl` when present. Keep fair/image/object/direct outputs in
separate directories. Use `--dry-run` on the Intern launcher to print the
resolved config.

## Important switches

- `evidence_mode`: `vlm_state`, `decision_image`, or `decision_objects`.
- `action_mode`: `planner_candidates` or `direct_actions`.
- `answer_backend`: `qwen_vl` or `intern_decision`.
- `stop_threshold`: probability of the structured `stop` yes decision.
- `min_choice_confidence`: confidence floor for mode/target/action choices.
- `max_steps`: optional episode step cap; `0` uses the benchmark-derived budget.
- `checkpoint`, `device`, `dtype`, `temperature`, `max_length`, and
  `attn_implementation`: local model runtime settings.

## Evidence-aware active-perception path

The new fair RGB-history runner does not execute Fine-EQA perception. It uses
RGB-D only for geometry, stores a bounded current/key-frame history, and sends
at most one Intern-Decision request per simulator decision. Forced budget or no-progress stops make zero requests. The request contains one
`verifier` field plus `value_<candidate>` and `cost_<candidate>` fields for all
current candidates. Candidate selection is local, so the model never returns a
cross-mode target ID.

```bash
# Offline contract smoke test
python -m InternDecision_scripts.run_evidence_aware --smoke

# Print the resolved fair configuration
python -m InternDecision_scripts.run_evidence_aware \
  --cfg-file InternDecision_scripts/config_evidence_aware.yaml --dry-run

# One-question Habitat run
printf '[0]\n' > /tmp/express_one.json
LD_LIBRARY_PATH=/ssd/wyzhang/miniconda3/envs/express-intern/lib \
python -m InternDecision_scripts.run_evidence_aware \
  --cfg-file InternDecision_scripts/config_evidence_aware.yaml \
  --indices-file /tmp/express_one.json
```

The active path forces `semantic_sensor: false`, disables Jev and the legacy
Intern router, and never calls Fine-EQA `VLM.get_loss`, semantic-region
integration, `find_next_pose`, or `find_next_point_region`. Geometry frontiers
are checked with Habitat PathFinder and executed through a bounded waypoint
using the same `agent.set_state` pose-placement convention as the original
planner. Route distance and applied pose displacement are logged separately;
no physical collision test is claimed.
The default `goe_provider: observed_viewpoints` proposes seen-space
re-observation locations; the shared model rates them against the question.
Set `goe_provider: none` for FBE-only ablation. When no frontier exists, a
same-pose scan candidate is used and the evidence
no-progress patience rule prevents indefinite repetition.

The current Intern-Decision checkpoint supports a maximum of 16 structured
fields. With one verifier and two fields per candidate, the default maximum is
seven candidates per forward. Traces include `decision_fields`, `model_calls`,
latency, token usage, candidate scores, pose-placement status, and evidence
novelty. They do not contain gold answers or goal metadata.

A fixed-index comparison can use the same runner with a JSON array of 100
question indices in `--indices-file` and a separate `--output-dir`. After the
run, evaluate `results.pkl` with `evaluation.score` in the express-intern
environment. A one-question smoke run verifies wiring only; it cannot establish
an exploration or accuracy gain. EAC model failures are stored as `EAC: "0,0"`
with `eac_status` to keep result aggregation readable.
