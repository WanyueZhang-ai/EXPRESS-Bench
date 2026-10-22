# Evidence-Aware Active Perception for EXPRESS-Bench

## Goal

Add a new EXPRESS-Bench execution path in which exploration decisions use the
question, current RGB observations, and a compact exploration history. The new
path must not use Fine-EQA's VLM region recognition, LSV/GSV scoring, semantic
region integration, or Fine-EQA stop decision. It retains RGB-D geometry,
Habitat reachability/navigation execution, the existing result contract, and a
separate answering VLM.

The decision model uses at most one forward pass per simulator decision.
Forced stops for budget, absent candidates, or no progress skip inference and
record zero model calls. An inference request contains a Verifier field plus Value and Cost fields for every executable
candidate. The local policy chooses STOP or the best valid candidate from those
scores; the model never emits a free-form mode/target pair.

## Scope and non-goals

### In scope

- A safe state encoder for current RGB, bounded key-frame history, action/path
  history, novelty, collision, and geometry statistics.
- Geometry-only FBE candidates and a configured, fair GOE proposal source.
  The default `observed_viewpoints` source proposes revisits to observed free
  space; `none` disables GOE. The shared backbone rates each proposal against
  the question and RGB history. The runtime never fabricates semantic labels.
- A structured Intern-Decision backend that asks all decision fields in one
  request and maps discrete score bins to Verifier, Value, and Cost values.
- A deterministic policy combining the three scores, novelty, progress, and
  exploration cost.
- A new online runner/configuration path, while preserving old Fine-EQA, Jev,
  and Intern-Decision experiment entry points for reproducibility.
- Multi-frame evidence for the answering VLM, while preserving the existing
  EAC/result serialization contract.

### Out of scope for the first runtime change

- Training a new neural checkpoint with literal Verifier/Value/Cost heads.
- Using Habitat's privileged semantic sensor or the full scene object graph in
  fair RGB experiments.
- Replacing the Habitat simulator, TSDF geometry fusion, or navigation control.

The backbone interface is designed so a later learned-head backend can replace
the structured backend without changing state encoding or policy code. The
current Intern-Decision checkpoint exposes structured logits, not hidden states
or trained value heads, so the first implementation must not claim otherwise.

## Architecture

```text
question + current RGB + bounded history
                  |
        EvidenceStateEncoder
                  |
     geometry candidates + safe state
                  |
       InternDecisionBackbone
       verifier/value_i/cost_i
                  |
        EvidenceAwarePolicy
             STOP / EXPLORE
                  |
     existing Habitat navigation controller
                  |
        key-frame memory + answer VLM
```

### State encoder

`EvidenceStateEncoder` owns an episode-local `ExplorationMemory`. Each step it
accepts the absolute current RGB path, simulator step, pose/action metadata,
actual displacement, collision status, TSDF novelty statistics, and candidate
summaries. It retains at most `history_window` records and
`max_history_images` key frames. It uses deterministic image digests and pose
bins to identify repeated views; it never serializes answer labels, goal poses,
reference paths, geodesic distances, or full-scene metadata.

The emitted state contains the question, question type, current-step ID,
compact history, current image(s), and safe candidate records. Image/history
paths are validated absolute local files. Candidate IDs and executor indices
are kept in separate structures so model output cannot address stale planner
indices.

### Candidate generation

`GeometryCandidateGenerator` extracts reachable frontier proposals from TSDF
occupancy/exploration volumes and Habitat path checks. FBE features include
stable ID, relative direction, distance bin, frontier/new-voxel estimates, and
feasibility. It does not use semantic values produced by Fine-EQA. The
checkpoint supports at most 16 decision fields, so a single-forward request
contains at most seven candidates (one Verifier and two fields per candidate).

GOE uses the explicitly configured `goe_provider`. The default
`observed_viewpoints` provider proposes navigable re-observation locations
from seen free space without claiming semantic goal knowledge. The backbone
uses the question and RGB evidence to rate their expected value. Setting the
provider to `none` yields an FBE-only ablation.

### One-forward decision backbone

`EvidenceAwareDecisionBackbone` compiles one request containing:

- `verifier`: `noul` yes/no, where yes means current evidence is sufficient;
- `value_<candidate_id>`: a finite score question mapped to expected evidence
  value in `[0, 1]`;
- `cost_<candidate_id>`: a finite score question mapped to normalized
  exploration cost in `[0, 1]`.

All fields share the same state and image list, and are submitted to one
`DecisionEngine.predict` call. The structured backend uses score bins supported
by the checkpoint and records the bin-to-float calibration. A future learned
head backend can return the same `BackboneScores` object.

The model does not output `mode` or `target`. The policy filters valid current
candidate IDs and selects locally, eliminating the current cross-mode
`invalid_target` failure class.

### Policy

For candidate `i`, the initial utility is:

```text
U_i = value_i - cost_weight * cost_i + novelty_weight * novelty_i
```

The policy stops when Verifier probability crosses the configured threshold,
the best utility is below the exploration threshold, no candidate is feasible,
the no-progress patience limit is reached, or the episode budget is exhausted.
Otherwise it executes the highest-utility candidate with deterministic ascending lexical
tie breaking. Every forced stop, fallback, and no-progress decision has an
explicit reason.

### Navigation and answering

The active-perception branch calls `TSDFPlanner.integrate` only for RGB-D
geometry and executes the selected candidate point with a geometry-only PathFinder route
and bounded waypoint. Following the existing EXPRESS-Bench planner convention,
the active runner places the agent at that navigable waypoint using
`agent.set_state`; this is a simulator pose placement, not a collision-checked
physical action sequence. It records the applied pose displacement, route
length, and explicit `unreachable` failures. It never
calls `find_next_pose` or `find_next_point_region`, because both contain
Fine-EQA semantic target selection. It does not instantiate or call the
Fine-EQA Prismatic VLM. On
STOP or forced termination, the answer backend receives the selected current
frame plus bounded key frames. Qwen remains the default answering and EAC
backend; result pickle fields remain compatible with `evaluation.py`.

## Files and interfaces

### New files

- `InternDecision_scripts/evidence_state_encoder.py`: memory and safe state
  construction.
- `InternDecision_scripts/geometry_candidates.py`: frontier/GOE candidate
  interfaces and executor mapping.
- `InternDecision_scripts/evidence_backbone.py`: one-forward structured
  Verifier/Value/Cost adapter.
- `InternDecision_scripts/evidence_policy.py`: stop/explore utility policy.
- `InternDecision_scripts/geometry_navigation.py`: geometry-only bounded
  waypoint selection and target failure accounting.
- `InternDecision_scripts/active_perception_hook.py`: online composition of the
  encoder, backbone, and policy.
- `InternDecision_scripts/active_episode.py`: independent Habitat episode loop,
  geometry proposals, PathFinder waypoint placement, answer and EAC output.
- `InternDecision_scripts/run_evidence_aware.py`: CLI runner.
- `InternDecision_scripts/config_evidence_aware.yaml`: fair RGB-history config.
- Focused unit/integration tests for each new component.

### Modified files

- `gpt.py`: accept a bounded list of image paths for answering while retaining
  single-image compatibility.
- `InternDecision_scripts/README.md`: document the new command, provider
  switch, and navigation measurement semantics.

The new runner is independent of `main.py`, so the legacy experiment entry
points and their results remain unchanged. It reads TSDF geometry through the
existing planner instance and does not call semantic region methods.

## Fairness and error handling

- Active RGB-history mode disables semantic sensor input and scene semantic
  dataset loading.
- State serialization rejects forbidden keys and non-finite values.
- Missing images, malformed model responses, stale candidate IDs, low
  confidence, or inference errors produce explicit fallbacks; they never
  silently switch to Fine-EQA.
- Each scored decision makes one model call. Forced stops make zero calls.
  History capture adds no simulator transition.
- Candidate count, decision field count, input tokens, inference latency,
  fallback reason, stop reason, and actual transition results are logged in a
  secret-free JSONL trace.

## Verification

1. Run pure unit tests with fake engines and no Habitat/model weights.
2. Run the full available unittest suite in the `express-intern` environment.
3. Run one fixed-index Habitat episode with the active config.
4. Inspect the trace for at most one call per decision, no forbidden state fields,
   no `invalid_target` fallback field, and accurate candidate-to-executor mapping.
5. In a later fixed 100-question comparison, report C, C*, E_path, d_T, actual
   path, stop rate, no-progress tail, collision rate, fallback rate, and
   evidence gain. Keep the old Fine-EQA results as a comparison only; they are
   not used to execute the new branch.

## Risks

- The current checkpoint's structured score calibration may be weak. The trace
  records raw bins/probabilities so thresholds can be calibrated before any
  learned-head training.
- GOE quality depends on question relevance inferred by the shared backbone
  for geometry-only revisit proposals. The `none` switch exposes FBE-only
  ablation without Fine-EQA semantics.
- Multi-frame answering increases image-token cost. `max_history_images` and
  deterministic key-frame selection keep this bounded and measurable.
