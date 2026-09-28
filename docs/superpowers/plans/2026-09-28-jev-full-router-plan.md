# EXPRESS-Bench Jev Full Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Jev-full-router decision layer and an oracle-semantic upper-bound experiment for EXPRESS-Bench.

**Architecture:** A small HTTP client calls Jev with JSON state and typed questions. A pure router converts VLM/TSDF summaries and candidates into stop/mode/target decisions with deterministic baseline fallback. An opt-in integration hook and replay CLI produce auditable logs without changing disabled baseline behavior.

**Tech Stack:** Python 3.9-compatible standard library HTTP/JSON, NumPy, existing Habitat/TSDF/VLM stack, pytest-compatible unit tests.

---

### Task 1: Add failing tests for Jev response parsing and router behavior

**Files:**
- Create: `tests/test_jev_router.py`

- [ ] **Step 1: Write tests for typed response parsing, auth/request fixtures, state redaction, fair/oracle parity, dynamic Choice ids, stop threshold mapping (.60/.75/.90), malformed/unknown values, network/HTTP/schema failures, deterministic fallback, and trace status.**
- [ ] **Step 2: Run `python -m pytest tests/test_jev_router.py -q` and verify it fails because `JevClient` and `DecisionRouter` do not exist.**

### Task 2: Implement the Jev client

**Files:**
- Create: `Jev_scripts/__init__.py`
- Create: `Jev_scripts/jev_client.py`

- [ ] **Step 1: Implement a Python 3.9-compatible client using `urllib.request` with environment-key validation, timeout, bounded transient retries, and no secret logging.**
- [ ] **Step 2: Implement typed answer normalization for Noul, Choice, and Score plus latency/token metadata, with pinned endpoint/auth/timeout/retry behavior and explicit rate-limit handling.**
- [ ] **Step 3: Run the Task 1 client tests and confirm they pass.**

### Task 3: Implement pure state building and routing

**Files:**
- Create: `Jev_scripts/decision_router.py`
- Create: `Jev_scripts/tsdf_candidates.py`

- [ ] **Step 1: Add state normalization/redaction and schema versioning; reject answer/goal/reference/geodesic/full-scene fields.**
- [ ] **Step 2: Add fair and current-FOV-only oracle-semantic evidence modes, bounded GOE/FBE candidate summaries with identical IDs, and typed questions.**
- [ ] **Step 3: Add thresholded stop, feasible mode/target selection, confidence checks, and baseline fallback for stop inconsistency or stale candidate IDs.**
- [ ] **Step 4: Run router tests and the existing test suite.**

### Task 4: Add online runner/replay smoke CLI and documentation

**Files:**
- Create: `Jev_scripts/run_jev_replay.py`
- Create: `Jev_scripts/README.md`
- Create: `Jev_scripts/fixtures/sample_state.json`

- [ ] **Step 1: Add an online-compatible runner interface in which the chosen action is returned to the Habitat loop, plus a deterministic offline replay path that does not call Jev unless `--live` is supplied.**
- [ ] **Step 2: Add JSONL decision logs with episode/step/seed/model/schema/status and aggregate CSV/JSON metrics including fallback, latency, token, and call-budget fields.**
- [ ] **Step 3: Document fair versus current-FOV-only oracle semantic protocol, installation, API-key handling, and commands.**
- [ ] **Step 4: Run offline CLI smoke tests.**

### Task 5: Add opt-in main-loop integration

**Files:**
- Modify: `main.py`
- Modify: `fine_eqa.yaml`

- [ ] **Step 1: Add an opt-in config block with router mode, thresholds, timeout, and log path.**
- [ ] **Step 2: Make the integration CLI/config path explicit, instantiate the router only when enabled, and call it after local perception/candidate summaries are available; pass the selected action into the same GOE/FBE executor so later observations follow Jev's trajectory.**
- [ ] **Step 3: Build oracle semantic evidence only from the current Habitat semantic observation/local FOV map and use the same candidate IDs, feasibility checks, and deterministic fallback as fair mode.**
- [ ] **Step 4: Persist the decision trace and use baseline behavior when disabled or when Jev fails.**
- [ ] **Step 5: Run import/compile checks and the existing tests without requiring Habitat or Jev.**

### Task 6: Verify and report

**Files:**
- Modify: `Jev_scripts/README.md` if command output reveals corrections.

- [ ] **Step 1: Run focused tests, the complete existing test suite, compile checks, and offline replay; verify aggregate output includes C_avg, C_star_avg, E_path, d_T_avg, paired episode ids/strata, fallback, latency, and tokens.**
- [ ] **Step 2: Run a minimal live Jev smoke test only if the environment key and network are available; redact response state in logs.**
- [ ] **Step 3: Inspect `git diff` and report exact verification output, limitations, and live-test status.**

