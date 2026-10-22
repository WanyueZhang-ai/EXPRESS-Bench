# Evidence-aware active perception: implementation record

**Goal:** Execute EXPRESS-Bench exploration from current RGB, bounded visual history, and reachable geometry candidates, without Fine-EQA perception or stop logic. Score evidence sufficiency and candidate Value/Cost in a single shared Intern-Decision request.

**Design:** [Evidence-aware active perception spec](../specs/2026-10-21-evidence-aware-active-perception-design.md)

## Implementation

1. `evidence_state_encoder.py` builds a safe, bounded episode state with current image, recent images, novelty, map gain, path displacement, and candidate summaries. It rejects forbidden gold/goal fields, missing images, and non-finite values.
2. `geometry_candidates.py` proposes FBE frontiers from observed free geometry and GOE re-observation viewpoints from seen free space. `goe_provider: none` disables GOE. PathFinder validates points before they receive stable IDs; exact points remain private to the executor. An empty snapshot uses a same-pose scan proposal.
3. `evidence_backbone.py` creates one `DecisionEngine.predict` request with a Verifier field and distinct Value and Cost fields for each candidate. The checkpoint's 16-field cap allows at most seven candidates. These are structured score fields, not newly trained neural heads.
4. `evidence_policy.py` selects STOP or the highest-utility current candidate using `value - cost_weight * cost + novelty_weight * novelty`. Budget and no-progress stops skip inference; model failures use a valid geometry candidate or stop.
5. `active_episode.py` runs a separate fair Habitat loop. It uses TSDF geometry and PathFinder routes, then applies one bounded waypoint with the original planner's `agent.set_state` convention. It records route length and applied displacement separately; this placement does not test physical collisions. It never imports or calls Fine-EQA VLM, semantic integration, `find_next_pose`, or `find_next_point_region`.
6. `gpt.py` accepts bounded current/history image lists for Qwen answers while keeping single-image calls compatible. The active runner saves the existing pickle result fields, status for answer/EAC failures, and a schema-versioned JSONL decision trace.
7. `run_evidence_aware.py` and `config_evidence_aware.yaml` provide opt-in CLI/configuration. The old entry points are unchanged.

## Verified

- Full `unittest` discovery in `express-intern`: 95 tests passed after the final score-field edit.
- No-GPU smoke: one model call and three fields for one candidate.
- Config dry-run, Python compile, and `git diff --check` passed.
- Fixed question index 0 on real Habitat with Intern-Decision and Qwen: first decision made one forward with five fields for two candidates; budget stop made zero calls; PathFinder waypoint placement and result pickle succeeded. The run is wiring validation, not a benchmark comparison.

## Follow-up experiment

Run a fixed 100-question set with a dedicated output directory, then compare `evaluation.score` results and trace-level stop rate, path length, no-progress tail, fallback rate, latency, input tokens, and evidence gain against the saved Fine-EQA condition. This is the next evidence needed to tune thresholds and claim any quality or speed gain. Physical collision metrics require a simulator action controller rather than pose placement.
