import json
import unittest
from pathlib import Path


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        return self.response


class TestJevClient(unittest.TestCase):
    def test_evaluate_sends_typesafe_request_and_parses_typed_answers(self):
        from Jev_scripts.jev_client import JevClient

        calls = []

        def transport(url, headers, payload, timeout):
            calls.append((url, headers, payload, timeout))
            return {
                "model": "jev-1.13.0",
                "answers": {
                    "stop": {"type": "noul", "noul": 0.2},
                    "mode": {
                        "type": "choice",
                        "choice": "goe",
                        "confidence": 0.91,
                        "probabilities": {"goe": 0.91, "fbe": 0.09},
                    },
                },
                "usage": {"input_tokens": 12, "output_tokens": 8},
            }

        client = JevClient(api_key="unit-secret", transport=transport, timeout=3)
        response = client.evaluate(
            {"question": "where?"},
            {"stop": {"type": "noul", "instructions": "stop?"}},
        )

        self.assertEqual(calls[0][0], "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(calls[0][1]["Authorization"], "Bearer unit-secret")
        self.assertNotIn("unit-secret", json.dumps(calls[0][2]))
        self.assertEqual(calls[0][3], 3)
        self.assertEqual(response.answers["mode"]["choice"], "goe")
        self.assertEqual(response.usage["input_tokens"], 12)

    def test_missing_key_is_rejected_before_network_call(self):
        from Jev_scripts.jev_client import JevClient, JevConfigurationError

        with self.assertRaises(JevConfigurationError):
            JevClient(api_key="")


class TestDecisionRouter(unittest.TestCase):
    def _state_parts(self):
        return dict(
            question="Where is the faucet?",
            question_type="state",
            history={"step": 2, "max_steps": 30, "path_length": 1.5},
            perception={"stop_vlm_label": "no", "stop_vlm_score": 0.2},
            goe_candidates=[{"id": "goe_0", "region": "kitchen", "count": 2}],
            fbe_candidates=[{"id": "fbe_0", "distance": 2.0, "frontier_value": 0.5}],
        )

    def test_stop_threshold_returns_stop_and_no_target(self):
        from Jev_scripts.decision_router import DecisionRouter, RouterConfig
        from Jev_scripts.jev_client import JevResponse

        fake = _FakeClient(
            JevResponse(
                model="jev",
                answers={
                    "stop": {"type": "noul", "noul": 0.8},
                    "mode": {"type": "choice", "choice": "goe", "confidence": 0.9},
                    "target": {"type": "choice", "choice": "goe_0", "confidence": 0.9},
                },
                usage={},
                latency_ms=4.0,
                request_id=None,
            )
        )
        decision = DecisionRouter(fake, RouterConfig(stop_threshold=0.75)).decide(
            **self._state_parts(), baseline_action="fbe", baseline_target_id="fbe_0"
        )
        self.assertEqual(decision.action, "stop")
        self.assertIsNone(decision.target_id)
        self.assertFalse(decision.fallback)

    def test_mode_and_target_are_selected_from_current_feasible_ids(self):
        from Jev_scripts.decision_router import DecisionRouter
        from Jev_scripts.jev_client import JevResponse

        fake = _FakeClient(
            JevResponse(
                model="jev",
                answers={
                    "stop": {"type": "noul", "noul": 0.1},
                    "mode": {"type": "choice", "choice": "goe", "confidence": 0.8},
                    "target": {"type": "choice", "choice": "goe_0", "confidence": 0.8},
                },
                usage={},
                latency_ms=4.0,
                request_id=None,
            )
        )
        decision = DecisionRouter(fake).decide(
            **self._state_parts(), baseline_action="fbe", baseline_target_id="fbe_0"
        )
        self.assertEqual((decision.action, decision.target_id), ("goe", "goe_0"))
        state, questions = fake.calls[0]
        self.assertEqual(
            set(questions["target"]["criteria"]), {"goe_0", "fbe_0"}
        )
        self.assertNotIn("answer", json.dumps(state))
        self.assertNotIn("goal_position", json.dumps(state))
        self.assertNotIn("geodesic_distance", json.dumps(state))

    def test_invalid_target_uses_exact_baseline_and_records_fallback(self):
        from Jev_scripts.decision_router import DecisionRouter
        from Jev_scripts.jev_client import JevResponse

        fake = _FakeClient(
            JevResponse(
                model="jev",
                answers={
                    "stop": {"type": "noul", "noul": 0.0},
                    "mode": {"type": "choice", "choice": "goe", "confidence": 0.8},
                    "target": {"type": "choice", "choice": "stale", "confidence": 0.8},
                },
                usage={},
                latency_ms=4.0,
                request_id=None,
            )
        )
        decision = DecisionRouter(fake).decide(
            **self._state_parts(), baseline_action="fbe", baseline_target_id="fbe_0"
        )
        self.assertEqual((decision.action, decision.target_id), ("fbe", "fbe_0"))
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.fallback_reason, "invalid_target")

    def test_api_failure_uses_baseline_without_leaking_oracle_scene_labels(self):
        from Jev_scripts.decision_router import DecisionRouter, build_state
        from Jev_scripts.jev_client import JevApiError

        state = build_state(
            **self._state_parts(),
            evidence_mode="oracle_semantic",
            oracle_semantic={"current_fov_labels": ["kitchen"], "semantic_source": "habitat_current_fov"},
        )
        self.assertIn("current_fov_labels", state["perception"]["oracle_semantic"])
        with self.assertRaises(ValueError):
            build_state(
                **self._state_parts(),
                evidence_mode="oracle_semantic",
                oracle_semantic={"full_scene_labels": ["kitchen"]},
            )

        fake = _FakeClient(error=JevApiError("offline"))
        decision = DecisionRouter(fake).decide(
            **self._state_parts(), baseline_action="fbe", baseline_target_id="fbe_0"
        )
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.fallback_reason, "jev_error")
        self.assertEqual((decision.action, decision.target_id), ("fbe", "fbe_0"))


if __name__ == "__main__":
    unittest.main()

class TestCandidateSummaries(unittest.TestCase):
    def test_summarize_planner_candidates_is_bounded_and_stable(self):
        from Jev_scripts.tsdf_candidates import summarize_goe

        result = summarize_goe(
            [
                {"region": 4, "count": 1, "dir": 0, "point": [[3, 4]]},
                {"region": 7, "count": 0, "dir": 2, "point": []},
            ],
            region_names={4: "kitchen", 7: "bedroom"},
        )
        self.assertEqual([item["id"] for item in result], ["goe_0", "goe_1"])
        self.assertEqual(result[0]["region"], "kitchen")
        self.assertEqual(result[0]["point_count"], 1)

class TestJevFailureContracts(unittest.TestCase):
    def test_retryable_http_failure_is_bounded_and_then_succeeds(self):
        from Jev_scripts.jev_client import JevClient, JevHTTPError

        attempts = []
        sleeps = []

        def transport(url, headers, payload, timeout):
            attempts.append(1)
            if len(attempts) == 1:
                raise JevHTTPError(503, "busy")
            return {"model": "jev", "answers": {"stop": {"type": "noul", "noul": 0.1}}}

        response = JevClient(
            api_key="secret", transport=transport, max_retries=1, sleep=sleeps.append
        ).evaluate("state", {"stop": {"type": "noul", "instructions": "stop"}})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(response.answers["stop"]["noul"], 0.1)

    def test_unknown_answer_type_is_a_schema_error(self):
        from Jev_scripts.jev_client import JevClient, JevSchemaError

        with self.assertRaises(JevSchemaError):
            JevClient.parse_response(
                {"model": "jev", "answers": {"mode": {"type": "unknown"}}}
            )


class TestOracleEvidence(unittest.TestCase):
    def test_oracle_summary_contains_only_current_observation_labels(self):
        from Jev_scripts.tsdf_candidates import current_fov_oracle_semantic

        result = current_fov_oracle_semantic([[0, 2, 2], [3, -1, 0]], category_names={2: "kitchen"})
        self.assertEqual(result["semantic_source"], "habitat_current_fov")
        self.assertEqual(result["current_fov_labels"], ["kitchen", 3])

class TestReplayLogging(unittest.TestCase):
    def test_aggregate_includes_express_metrics_and_router_metadata(self):
        from Jev_scripts.run_jev_replay import aggregate_records

        summary = aggregate_records(
            [
                {
                    "episode_id": 7,
                    "question_type": "state",
                    "action": "goe",
                    "fallback": False,
                    "latency_ms": 10.0,
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                    "express_result": {
                        "path_len": 2.0,
                        "geodesic_distance": 3.0,
                        "EAC": "1,1",
                        "goal_dis": 4.0,
                    },
                }
            ]
        )
        self.assertEqual(summary["episode_count"], 1)
        self.assertEqual(summary["C_avg"], 20.0)
        self.assertEqual(summary["C_star_avg"], 20.0)
        self.assertEqual(summary["E_path"], 20.0)
        self.assertEqual(summary["d_T_avg"], 4.0)
        self.assertEqual(summary["total_input_tokens"], 3)
        self.assertEqual(summary["fallback_rate"], 0.0)

class TestDecisionTrace(unittest.TestCase):
    def test_trace_contains_execution_metadata_without_full_state(self):
        from Jev_scripts.decision_router import RouterDecision, decision_to_trace

        trace = decision_to_trace(
            RouterDecision("fbe", "fbe_0", False, latency_ms=3.0, usage={"input_tokens": 4}),
            episode_id=9,
            step=2,
            seed=42,
            router_mode="fair",
        )
        self.assertEqual(trace["episode_id"], 9)
        self.assertEqual(trace["semantic_source"], "vlm_tsdf")
        self.assertNotIn("state", trace)
        self.assertEqual(trace["candidate_ids"], {"goe": [], "fbe": []})

class TestOnlineHook(unittest.TestCase):
    def test_disabled_config_does_not_construct_a_client(self):
        from Jev_scripts.online_hook import router_from_config

        self.assertIsNone(router_from_config({"enabled": False}))

    def test_baseline_target_mapping_is_deterministic(self):
        from Jev_scripts.online_hook import baseline_route, target_index

        self.assertEqual(baseline_route(False, ["goe_0"]), ("goe", "goe_0"))
        self.assertEqual(baseline_route(True, ["fbe_0"]), ("fbe", "fbe_0"))
        self.assertEqual(target_index("goe_3", "goe"), 3)
        self.assertIsNone(target_index("fbe_0", "goe"))

class TestPlannerIntegrationSurface(unittest.TestCase):
    def test_region_selector_accepts_jev_region_index(self):
        import ast

        tree = ast.parse(Path("src/tsdf.py").read_text())
        methods = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "find_next_point_region"
        ]
        self.assertEqual(len(methods), 1)
        names = [arg.arg for arg in methods[0].args.args]
        self.assertIn("region_index", names)

class TestSemanticSensorConfig(unittest.TestCase):
    def test_habitat_config_has_explicit_semantic_sensor_branch(self):
        source = Path("src/habitat.py").read_text()
        self.assertIn('settings.get("semantic_sensor", False)', source)
        self.assertIn('SensorType.SEMANTIC', source)
        self.assertIn('scene_dataset_config_file', source)

class TestFallbackReasons(unittest.TestCase):
    def test_rate_limit_error_is_distinguished_in_trace(self):
        from Jev_scripts.decision_router import DecisionRouter
        from Jev_scripts.jev_client import JevHTTPError

        fake = _FakeClient(error=JevHTTPError(429, "rate limited"))
        decision = DecisionRouter(fake).decide(
            question="q",
            question_type="state",
            history={},
            perception={},
            goe_candidates=[],
            fbe_candidates=[{"id": "fbe_0"}],
            baseline_action="fbe",
            baseline_target_id="fbe_0",
        )
        self.assertEqual(decision.fallback_reason, "jev_http_429")

class TestSemanticCategoryMapping(unittest.TestCase):
    def test_mapping_is_internal_and_can_label_current_ids(self):
        from Jev_scripts.online_hook import semantic_category_names
        from Jev_scripts.tsdf_candidates import current_fov_oracle_semantic

        class Category:
            def name(self):
                return "kitchen"

        class Object:
            id = "object_2"
            category = Category()

        class Scene:
            objects = [Object()]

        mapping = semantic_category_names(type("Simulator", (), {"semantic_scene": Scene()})())
        result = current_fov_oracle_semantic([2], category_names=mapping)
        self.assertEqual(result["current_fov_labels"], ["kitchen"])

class TestPairedBootstrap(unittest.TestCase):
    def test_paired_bootstrap_matches_episode_ids_and_returns_interval(self):
        from Jev_scripts.run_jev_replay import paired_bootstrap_ci

        left = [
            {"episode_id": "a", "express_result": {"path_len": 1, "geodesic_distance": 1, "EAC": "1,1", "goal_dis": 1}},
            {"episode_id": "b", "express_result": {"path_len": 1, "geodesic_distance": 1, "EAC": "2,1", "goal_dis": 1}},
        ]
        right = [
            {"episode_id": "a", "express_result": {"path_len": 1, "geodesic_distance": 1, "EAC": "2,1", "goal_dis": 1}},
            {"episode_id": "b", "express_result": {"path_len": 1, "geodesic_distance": 1, "EAC": "3,1", "goal_dis": 1}},
        ]
        result = paired_bootstrap_ci(left, right, "C_avg", n_bootstrap=100, seed=7)
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(result["mean_difference"], 20.0)
        self.assertLessEqual(result["low"], result["mean_difference"])
        self.assertGreaterEqual(result["high"], result["mean_difference"])

class TestMalformedProbability(unittest.TestCase):
    def test_non_numeric_probability_is_schema_error(self):
        from Jev_scripts.jev_client import JevClient, JevSchemaError

        with self.assertRaises(JevSchemaError):
            JevClient.parse_response(
                {
                    "model": "jev",
                    "answers": {
                        "mode": {
                            "type": "choice",
                            "choice": "goe",
                            "probabilities": {"goe": "not-a-number"},
                        }
                    },
                }
            )

class TestParallelRunner(unittest.TestCase):
    def test_split_indices_is_balanced_and_preserves_order(self):
        from Jev_scripts.parallel_runner import split_indices

        shards = split_indices(2044, 8)
        self.assertEqual([len(shard) for shard in shards], [256, 256, 256, 256, 255, 255, 255, 255])
        self.assertEqual([index for shard in shards for index in shard], list(range(2044)))

    def test_split_indices_supports_one_and_four_workers(self):
        from Jev_scripts.parallel_runner import split_indices

        self.assertEqual(split_indices(5, 1), [list(range(5))])
        self.assertEqual([len(shard) for shard in split_indices(10, 4)], [3, 3, 2, 2])

    def test_gpu_selection_accepts_explicit_physical_ids(self):
        from Jev_scripts.parallel_runner import resolve_gpu_ids

        self.assertEqual(resolve_gpu_ids(4, "0,2,5,7"), [0, 2, 5, 7])
        self.assertEqual(resolve_gpu_ids(2, None), [0, 1])
        with self.assertRaises(ValueError):
            resolve_gpu_ids(2, "0,0")
        with self.assertRaises(ValueError):
            resolve_gpu_ids(4, "0,1")

    def test_merge_results_detects_missing_and_duplicate_question_indices(self):
        import pickle
        import tempfile

        from Jev_scripts.parallel_runner import merge_results

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shard_a = root / "worker_00"
            shard_b = root / "worker_01"
            shard_a.mkdir()
            shard_b.mkdir()
            with (shard_a / "results.pkl").open("wb") as handle:
                pickle.dump([{"question_ind": 2}, {"question_ind": 0}], handle)
            with (shard_b / "results.pkl").open("wb") as handle:
                pickle.dump([{"question_ind": 1}, {"question_ind": 3}], handle)

            merged = merge_results([shard_a, shard_b], root, expected_indices=range(4))
            self.assertEqual([row["question_ind"] for row in merged], [0, 1, 2, 3])
            self.assertTrue((root / "results.pkl").exists())

            with (shard_b / "results.pkl").open("wb") as handle:
                pickle.dump([{"question_ind": 1}, {"question_ind": 2}], handle)
            with self.assertRaises(ValueError):
                merge_results([shard_a, shard_b], root, expected_indices=range(4))

    def test_worker_command_passes_index_and_output_overrides(self):
        from Jev_scripts.parallel_runner import build_worker_command

        command = build_worker_command(
            python_executable="python",
            main_script=Path("main.py"),
            cfg_file=Path("fine_eqa.yaml"),
            indices_file=Path("indices.json"),
            output_dir=Path("shards/worker_00"),
            progress_file=Path("shards/worker_00/progress.json"),
        )
        self.assertEqual(command[:2], ["python", "main.py"])
        self.assertIn("--indices-file", command)
        self.assertIn("--output-dir", command)
        self.assertIn("--progress-file", command)

class TestParallelProgress(unittest.TestCase):
    def test_progress_file_is_readable_and_contains_timestamp(self):
        import tempfile

        from Jev_scripts.parallel_runner import read_progress, write_progress

        with tempfile.TemporaryDirectory() as temp_dir:
            progress_path = Path(temp_dir) / "worker" / "progress.json"
            write_progress(
                progress_path,
                {"status": "running", "completed_questions": 3},
            )
            state = read_progress(progress_path)
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["completed_questions"], 3)
            self.assertIsInstance(state["updated_at"], float)

class TestParallelResume(unittest.TestCase):
    def test_resume_only_keeps_question_indices_without_episode_results(self):
        import pickle
        import tempfile

        from Jev_scripts.parallel_runner import prepare_workers

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            initial = prepare_workers(
                total_questions=6,
                gpu_ids=[0, 1],
                output_dir=root,
            )
            completed_dir = initial[0].output_dir / "0"
            completed_dir.mkdir()
            with (completed_dir / "result.pkl").open("wb") as handle:
                pickle.dump({"question_ind": 0}, handle)

            resumed = prepare_workers(
                total_questions=6,
                gpu_ids=[0, 1],
                output_dir=root,
                resume=True,
            )
            self.assertEqual(resumed[0].indices, [0, 1, 2])
            self.assertEqual(resumed[0].pending_indices, [1, 2])
            self.assertEqual(resumed[1].pending_indices, [3, 4, 5])

    def test_merge_results_includes_existing_episode_files_during_resume(self):
        import pickle
        import tempfile

        from Jev_scripts.parallel_runner import merge_results

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            worker = root / "worker_00"
            worker.mkdir()
            old_dir = worker / "0"
            old_dir.mkdir()
            with (old_dir / "result.pkl").open("wb") as handle:
                pickle.dump({"question_ind": 0, "source": "old"}, handle)
            with (worker / "results.pkl").open("wb") as handle:
                pickle.dump([{"question_ind": 1, "source": "new"}], handle)

            merged = merge_results([worker], root, expected_indices=range(2))
            self.assertEqual([row["question_ind"] for row in merged], [0, 1])
            self.assertEqual(merged[0]["source"], "old")
