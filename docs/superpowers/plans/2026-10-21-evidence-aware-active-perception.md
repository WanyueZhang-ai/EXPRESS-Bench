# Evidence-Aware Active Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a fair RGB-history exploration path in EXPRESS-Bench whose one-forward Intern-Decision request scores evidence sufficiency and each executable candidate's value and cost, then uses existing geometry navigation and Qwen answering.

**Architecture:** New focused modules encode bounded history, enumerate reachable geometric candidates, compile one structured request, and choose STOP/EXPLORE locally. A separate active runner owns the fair episode loop; the legacy Fine-EQA/Jev/Intern branches remain reproducible. The new branch does not instantiate the Prismatic VLM or call Fine-EQA semantic exploration methods.

**Tech Stack:** Python 3.12, unittest, NumPy, Habitat-Sim, OmegaConf, local Intern-Decision-4B, local Qwen3-VL.

**Spec:** `docs/superpowers/specs/2026-10-21-evidence-aware-active-perception-design.md`

---

### Task 1: Establish contracts and safe episode memory

**Files:**
- Create: `InternDecision_scripts/evidence_state_encoder.py`
- Create: `tests/test_evidence_state_encoder.py`

- [x] Define `ObservationRecord`, `CandidateRecord`, and `EvidenceState` with explicit fields and no arbitrary metadata passthrough.
- [x] Add tests for bounded deterministic history, novelty, and no-progress counts.
- [x] Run `/ssd/wyzhang/miniconda3/envs/express-intern/bin/python -m unittest tests.test_evidence_state_encoder -v`; expect a missing-module/behavior failure.
- [x] Implement bounded memory, image digest/pose-bin repeat detection, action/displacement/collision updates, and deterministic key-frame selection.
- [x] Add tests for forbidden fields, non-finite values, missing image paths, and history/window limits; implement validation.
- [x] Commit the focused change.

### Task 2: Enumerate executable geometry candidates

**Files:**
- Create: `InternDecision_scripts/geometry_candidates.py`
- Create: `tests/test_geometry_candidates.py`

- [x] Add tests for frontier candidate IDs, feasibility, stable order, geometry-only features, and ID-to-target mapping.
- [x] Use pure geometry extraction over the TSDF occupancy/exploration masks; do not read `_val_vol_cpu`, `_val_region_cpu`, `exp_list`, or target metadata.
- [x] Generate a bounded list of reachable FBE frontier points from explored/free-space geometry, with Habitat path checks delegated through an injected callback.
- [x] Add the configured fair GOE provider switch. `observed_viewpoints` proposes geometry-only re-observation points rated by the shared backbone; `none` provides an FBE-only ablation.
- [x] Add tests that candidate IDs never collide, stale IDs are rejected, and empty geometry uses an explicit same-pose scan fallback.
- [x] Run focused tests and commit.

### Task 3: Geometry-only target navigation

**Files:**
- Create: `InternDecision_scripts/geometry_navigation.py`
- Create: `tests/test_geometry_navigation.py`

- [x] Add tests for bounded waypoint movement, accurate traveled distance, and unreachable targets.
- [x] Implement a pure waypoint selector over a PathFinder route and online pose-placement adapter.
- [x] Ensure selected points are validated through the current candidate ID map; do not call `find_next_pose` or `find_next_point_region`.
- [x] Run focused tests and commit.

### Task 4: One-forward Intern-Decision scoring

**Files:**
- Create: `InternDecision_scripts/evidence_backbone.py`
- Create: `tests/test_evidence_backbone.py`
- Reuse: `InternDecision_scripts/intern_decision_engine.py`

- [x] Add a fake-engine test asserting one `predict` call with one shared `state`/`images` payload and fields `verifier`, `value_<id>`, `cost_<id>` for all candidates.
- [x] Encode score fields with checkpoint-supported finite bins and map answers to normalized floats.
- [x] Validate missing/non-finite/out-of-range fields, candidate ID mismatch, usage, and latency. Preserve raw safe score distributions in the trace.
- [x] Run focused tests and commit.

### Task 5: Deterministic STOP/EXPLORE policy

**Files:**
- Create: `InternDecision_scripts/evidence_policy.py`
- Create: `tests/test_evidence_policy.py`

- [x] Add tests for verifier STOP, maximum budget, no candidates, no-progress patience, utility threshold, and deterministic target tie breaking.
- [x] Implement `U_i = value_i - cost_weight * cost_i + novelty_weight * novelty_i`, with hard feasibility filtering and explicit stop/fallback reasons.
- [x] Ensure malformed inference never calls Fine-EQA fallback; use a deterministic safe candidate or terminate with an error/status reason.
- [x] Run focused tests and commit.

### Task 6: Multi-frame Qwen answer input

**Files:**
- Modify: `gpt.py`
- Create: `tests/test_answer_evidence_memory.py`

- [x] Write a processor-message test that multiple absolute image paths are normalized in deterministic order and bounded.
- [x] Extend `qwen3_vl` to accept a bounded list of images while preserving the old single-path signature and single-image behavior.
- [x] Pass selected key frames through the answer backend; record frame IDs/count but no raw answer or gold field in decision traces.
- [x] Run focused and existing Qwen tests, then commit.

### Task 7: Online integration and runner

**Files:**
- Create: `InternDecision_scripts/active_perception_hook.py`
- Create: `InternDecision_scripts/run_evidence_aware.py`
- Create: `InternDecision_scripts/config_evidence_aware.yaml`
- Modify: `InternDecision_scripts/README.md`
- Create: `tests/test_active_perception_hook.py`
- Create: `tests/test_active_perception_runner.py`

- [x] Add offline hook tests that one RGB step yields one model request, one executable candidate selection, and one trace record.
- [x] Add explicit `active_perception.enabled` validation and a runner with CLI indices/output/progress conventions.
- [x] In the independent runner, skip Prismatic VLM construction, region prompting, LSV/GSV, `integrate_sem`, `integrate_reg`, `get_exp_state`, `find_next_point_region`, and Fine-EQA stop/fallback; keep TSDF geometric fusion and PathFinder navigation.
- [x] Execute the chosen candidate through its validated point and bounded PathFinder route. One decision causes at most one pose-placement transition; record route distance, applied displacement, and that collision was not checked.
- [x] Answer on policy STOP or forced termination with bounded current/key-frame evidence and preserve result field types; normalize failed EAC to `0,0` with status.
- [x] Log one schema-versioned JSONL row per decision with shared-field count, token count, latency, candidate scores, stop/fallback reason, and pose-placement outcome.
- [x] Run hook/runner tests and existing result compatibility tests, then commit.

### Task 8: Smoke, regression, and online validation

**Files:**
- Create: `InternDecision_scripts/run_evidence_aware_smoke.py`
- Modify: `InternDecision_scripts/README.md`

- [x] Add no-GPU/no-Habitat fake-engine smoke covering current RGB, history, GOE/FBE candidate fields, malformed model output, and forced stop.
- [x] Run `/ssd/wyzhang/miniconda3/envs/express-intern/bin/python -m InternDecision_scripts.run_evidence_aware_smoke`; exit 0 with one predict call for the scored decision.
- [x] Run `/ssd/wyzhang/miniconda3/envs/express-intern/bin/python -m unittest discover -s tests -p 'test_*.py' -q`; expect zero failures.
- [x] Run a fixed-index Habitat/model episode with accessible assets. Inspect trace for no `invalid_target` field, candidate mapping, no privileged inputs, and at most one forward per step; forced stops have zero calls.
- [x] Record a reproducible fixed-100-question command and metrics script invocation; do not claim benchmark gains before this run completes.
- [x] Review `git diff`, verify no changes to the original dirty workspace, and commit documentation.

### Task 9: Final review

- [x] Review the implementation against every spec section and the user's one-forward speed requirement.
- [x] Re-run focused and full verification commands fresh after the last edit.
- [x] Inspect branch status and provide exact worktree/branch paths, commands, results, limitations, and any next experiment needed.
