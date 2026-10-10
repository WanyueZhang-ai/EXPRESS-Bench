# Intern-Decision Runtime Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair the four observed Intern-Decision experiment defects without changing the result schema: direct-action path accounting, planner candidate availability, per-worker Qwen device placement, and durable worker logging/evidence observability.

**Architecture:** Keep the existing Habitat loop and router APIs. Add small pure helpers for direct path accounting, preserve the planner's existing candidate snapshot for all planner evidence modes, pass the worker device explicitly to Qwen, and configure a named file logger with explicit handler ownership. Extend offline tests before changing production code.

**Tech Stack:** Python 3.12, unittest, OmegaConf, Habitat-Sim, PyTorch/Transformers.

---

### Task 1: Direct action path accounting

**Files:** `main.py`, `tests/test_intern_decision_actions.py`

- [x] Write a failing test that a stepped forward action contributes the observed displacement to episode path length, while turns and collisions contribute zero displacement.
- [x] Run the focused test and confirm it fails against the current code.
- [x] Add the minimal accounting helper/integration and preserve one-transition-per-decision behavior.
- [x] Run focused and full Intern tests.

### Task 2: Planner evidence candidate parity

**Files:** `main.py`, `tests/test_intern_decision_integration.py`

- [x] Write a failing source-level/integration test showing image/object planner conditions must materialize the same GOE/FBE candidate snapshot rather than force an empty GOE list.
- [x] Run it and confirm the current branch fails.
- [x] Change candidate-state preparation so evidence mode does not silently disable planner state; keep object/image evidence separate from privileged answer data.
- [x] Verify candidate IDs and executor mapping remain stable.

### Task 3: Explicit Qwen worker device

**Files:** `gpt.py`, `main.py`, configs/tests as needed.

- [x] Write a failing test for explicit device selection and rejection of ambiguous `device_map=auto` in worker mode.
- [x] Run it and confirm failure.
- [x] Add a `QWEN3_VL_DEVICE`/runtime device path, use a single-device map, and pass the worker device from config/CLI.
- [x] Verify the loader remains offline and CPU-testable without model weights.

### Task 4: Durable logging and evidence observability

**Files:** `InternDecision_scripts/run_intern_decision.py`, `main.py`, `InternDecision_scripts/evidence_adapters.py`, tests.

- [x] Write failing tests for a non-empty worker log handler and serialized object-count metadata.
- [x] Run them red.
- [x] Implement explicit logger setup (`force=True` or named logger with owned handlers), flush trace writes, and include object-count/evidence summary metadata without exposing gold labels.
- [x] Run smoke, focused unittest suite, compile checks, and source-level validation.

### Task 5: Integration handoff

- [x] Copy only repaired source/tests/docs back to the active workspace after review.
- [x] Do not restart or kill the four currently running processes automatically.
- [x] Report exact restart commands and verification evidence.
