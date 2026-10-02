# Intern-Decision-4B Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local Intern-Decision-4B decision backend with VLM-state, image, object, and direct-action experiments while preserving EXPRESS-Bench result and Jev compatibility.

**Architecture:** A focused `InternDecision_scripts` package wraps the checkpoint's local `DecisionEngine`, builds safe evidence requests, validates structured decisions, and exposes an online runner. `main.py` gets a minimal hook at the existing Jev decision boundary; the old Jev path remains unchanged. A fake-engine smoke suite verifies all contracts without Habitat or GPU.

**Tech Stack:** Python 3.9+, PyTorch/Transformers from the local model environment, Habitat-Sim, OmegaConf, PIL, NumPy, unittest/pytest.

---

### Task 1: Add package skeleton and model wrapper

**Files:**
- Create: `InternDecision_scripts/__init__.py`
- Create: `InternDecision_scripts/intern_decision_engine.py`
- Create: `tests/test_intern_decision_engine.py`

- [ ] Add package exports and constants for the four evidence/action modes.
- [ ] Implement a lazy `InternDecisionEngine` that resolves `inference.py` relative to the configured checkpoint with `importlib.util.spec_from_file_location`, imports `DecisionEngine`, validates local files, and loads one model instance per process.
- [ ] Pass checkpoint, device, dtype, temperature, media root, max length, and attention implementation without network access; expose `predict(request)` and normalized error types.
- [ ] Add fake-module tests for lazy import, missing checkpoint, request immutability, and local-only settings.
- [ ] Run `pytest tests/test_intern_decision_engine.py -q` and confirm the wrapper tests pass without loading GPU weights.

### Task 2: Implement safe evidence adapters

**Files:**
- Create: `InternDecision_scripts/evidence_adapters.py`
- Create: `tests/test_intern_decision_evidence.py`

- [ ] Define dataclasses for decision config, planner candidates, and same-timestep evidence snapshots.
- [ ] Normalize GOE/FBE candidates in planner order, truncate before prompt construction, assign stable IDs, expose only safe attributes, and keep a separate ID-to-executor index map.
- [ ] Implement recursive JSON allowlisting/redaction for history, planner features, current semantic objects, and forbidden privileged fields.
- [ ] Extract only object IDs present in the current semantic sensor frame; serialize category and agent-relative center/extent in meters, excluding hidden/offscreen/full-scene data.
- [ ] Implement image snapshot validation that persists/returns an absolute readable RGB path even when `save_obs` is false.
- [ ] Add tests for all evidence modes, candidate redaction, hidden-object exclusion, image lifecycle, and same-step metadata.
- [ ] Run `pytest tests/test_intern_decision_evidence.py -q` and confirm no gold/goal/reference/geodesic fields survive.

### Task 3: Implement structured planner/direct router

**Files:**
- Create: `InternDecision_scripts/decision_router.py`
- Create: `tests/test_intern_decision_router.py`

- [ ] Build Intern-Decision request questions: planner `stop/mode/target`; direct `stop/action`, omitting empty candidate questions.
- [ ] Validate `noul`, `choice`, probability, confidence, stable IDs, mode prefixes, and direct whitelist membership with deterministic lexical tie behavior.
- [ ] Implement planner fallback to the exact baseline action/target and direct safe-rotation fallback with no planner switch; log normalized fallback reasons and usage/latency.
- [ ] Implement direct action state machine: one transition per choice, collision signal handling, deterministic next-cycle turn, max action/step forced stop, and answer callback invocation.
- [ ] Add answer adapter using `DecisionEngine.predict` with static non-gold `answer_options`/per-type schema, same-timestep state and optional image, Qwen fallback, and dual-failure error status.
- [ ] Run `pytest tests/test_intern_decision_router.py -q` and verify malformed responses, collisions, no-candidate cases, and both answer backends.

### Task 4: Add online runner and configuration files

**Files:**
- Create: `InternDecision_scripts/run_intern_decision.py`
- Create: `InternDecision_scripts/run_intern_decision_smoke.py`
- Create: `InternDecision_scripts/config_vlm_state.yaml`
- Create: `InternDecision_scripts/config_decision_image.yaml`
- Create: `InternDecision_scripts/config_decision_objects.yaml`
- Create: `InternDecision_scripts/config_direct_actions.yaml`
- Modify: `main.py` at the existing Jev hook boundary only if needed for shared executor helpers.
- Create: `InternDecision_scripts/README.md`

- [ ] Add CLI flags for config, evidence mode, action mode, answer backend, indices file, output directory, progress file, resume/trace append, and dry-run/smoke.
- [ ] Reuse existing Habitat setup, observation capture, TSDF candidate generation, GOE/FBE target index mapping, result pickle writes, progress files, and evaluation fields.
- [ ] Enable semantic sensor and annotated scene dataset only for object evidence; keep current RGB/depth timing identical.
- [ ] Persist decision RGB frames to absolute paths, call router immediately after same-step candidates/objects exist, and never add a simulator transition for capture.
- [ ] Map direct action names to validated Habitat action-space entries and configured turn/forward settings; write one sanitized JSONL trace per decision.
- [ ] Add finite static answer-option examples and explicit fallback behavior to configs; keep Qwen answer generation as default.
- [ ] Implement fake-engine smoke CLI that runs all modes without Habitat/GPU/network.
- [ ] Run `python -m InternDecision_scripts.run_intern_decision_smoke` and inspect generated trace/result fixtures.

### Task 5: Documentation, regression tests, and integration verification

**Files:**
- Modify: `InternDecision_scripts/README.md`
- Create: `tests/test_intern_decision_result_compat.py`
- Optionally modify: `README.md` with a short link to the new experiment guide.

- [ ] Document one command for each planner evidence mode, one direct-action command, answer backend options, model path/device overrides, single-index smoke, and multi-GPU usage through the existing launcher if supported.
- [ ] Add fixture assertions for planner/direct `result.pkl` keys/types and `evaluation.py` compatibility, including Qwen, Intern answer, and dual-failure statuses.
- [ ] Run focused Intern-Decision tests, existing Jev router tests, and the full available test suite.
- [ ] Run a one-question Habitat integration command when model dependencies and scene assets are available; otherwise report the exact blocked dependency without claiming integration success.
- [ ] Record verified commands and outputs in the final response.

### Task 6: Review and finish

- [ ] Inspect `git diff` and `git status` for unintended changes; preserve existing user modifications.
- [ ] Run the verification commands fresh after the final edit.
- [ ] Summarize files, experiment commands, model-loading requirements, fallback behavior, and any integration limitations.
