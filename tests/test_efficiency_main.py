"""Execute the real main() body with synthetic scene/model boundaries.

The AST loader avoids importing unavailable Habitat/Prismatic dependencies;
the benchmark's loops, break/continue handling and profiler are unchanged.
"""
import ast
import json
import logging
import math
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import pytest

from src.efficiency import current_profiler, profiled_run, apply_efficiency_cli
from src.efficiency_models import model_call
from src.efficiency_report import read_jsonl
from InternDecision_scripts.action_executor import DirectActionExecutor, path_increment
from InternDecision_scripts.runtime_policy import use_vlm_planner_perception
from InternDecision_scripts.evidence_adapters import build_evidence_state


def _run_main(tmp_path, monkeypatch, *, action="stop", black=False, fail=False, planner=False, baseline=False, goe=False, enabled=True):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "InternDecision_scripts/config_direct_actions.yaml")
    cfg.output_dir = str(tmp_path / "run")
    cfg.dataset_path = str(tmp_path / "dataset.json")
    cfg.scene_data_path = str(tmp_path / "data")
    cfg.img_width = cfg.img_height = 8
    cfg.max_step_room_size_ratio = 1
    cfg.intern_decision.device = "cpu"
    cfg.vlm.device = "cpu"
    cfg.intern_decision.enabled = not baseline
    cfg.intern_decision.action_mode = "planner_candidates" if planner else "direct_actions"
    cfg.efficiency.enabled = enabled
    cfg.efficiency.mode = "detailed"
    Path(cfg.dataset_path).write_text(json.dumps([{"question": "Is there a chair?", "answer": "yes", "type": "existence", "scene_id": "00000-synthetic",
                                                  "start_position": [0., 0., 0.], "start_rotation": [1., 0., 0., 0.],
                                                  "goal_position": [1., 0., 0.], "geodesic_distance": 1.}]))
    class AgentState:
        def __init__(self):
            self.position = np.zeros(3)
            self.rotation = [1, 0, 0, 0]
            self.sensor_states = {"depth_sensor": NS(rotation=None, position=np.zeros(3))}
    class Agent:
        agent_config = NS(action_space={"move_forward", "turn_left", "turn_right"})
        def __init__(self):
            self.state = AgentState()
        def get_state(self):
            return self.state
        def set_state(self, state):
            self.state = state
            state.position = np.asarray(state.position, dtype=float)
    class Finder:
        def seed(self, *args):
            pass
        def load_nav_mesh(self, *args):
            pass
        def snap_point(self, point):
            return np.asarray(point)
        def find_path(self, path):
            path.geodesic_distance = 1.
            return True
    class Simulator:
        def __init__(self, settings):
            self.agent, self.pathfinder = Agent(), Finder()
            self.transitions = 0
        def initialize_agent(self, index):
            return self.agent
        def get_sensor_observations(self):
            return {"color_sensor": np.full((8, 8, 4), 0 if black else 255, dtype=np.uint8), "depth_sensor": np.ones((8, 8))}
        def step(self, action):
            self.transitions += 1
            if action == "move_forward":
                self.agent.state.position = self.agent.state.position + np.array([.25, 0., 0.])
            return self.get_sensor_observations()
        def close(self):
            pass
    class Planner:
        def __init__(self, **kwargs):
            pass
        def integrate(self, **kwargs):
            pass
        def get_exp_state(self):
            return False, [{"count": 1, "dir": 1, "region": 4}]
        def find_prompt_points_within_view(self, *args, **kwargs):
            return [], None
        def find_next_point_region(self, *args, **kwargs):
            return np.array([.2, 0.]), np.array([1, 1]), None
        def find_next_pose(self, **kwargs):
            axes = [NS(plot=lambda *args, **kwargs: None, scatter=lambda *args, **kwargs: None) for _ in range(5)]
            return np.array([.2, 0.]), 0., np.array([1, 1]), NS(axes=axes)
    class Runtime:
        def evidence(self, **kwargs):
            return build_evidence_state(mode="decision_image", **kwargs)
        def decide(self, state, **kwargs):
            with model_call("synthetic_decision", "decision", input_modality="text_image", output_kind="decision"):
                if fail:
                    raise RuntimeError("synthetic failure")
            return NS(action=action, fallback=False, fallback_reason=None, target_id="goe_0" if action == "goe" else None)
        def trace(self, decision, **kwargs):
            return {"action": decision.action, **kwargs}
        def answer(self, *args):
            return NS(text="yes", status="ok", error_type=None)
    def qwen(prompt, *args, **kwargs):
        if "region" in prompt:
            return "kitchen" if goe else "unknown"
        if "explore" in prompt:
            return "yes"
        if "evaluation" in prompt:
            return "5, 1"
        return "yes"
    def save_image(path, pixels):
        Image.fromarray(pixels).save(path)
    def progress(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))
    namespace = dict(json=json, sys=sys, time=time, os=os, pickle=pickle, logging=logging, math=math, np=np,
                     Image=Image, ImageFont=NS(truetype=lambda *args: None),
                     plt=NS(imsave=save_image, close=lambda fig: None), quaternion=NS(as_rotation_matrix=lambda x: np.eye(3)),
                     habitat_sim=NS(Simulator=Simulator, AgentState=AgentState, ShortestPath=NS),
                     profiled_run=profiled_run, current_profiler=current_profiler, apply_efficiency_cli=apply_efficiency_cli,
                     router_from_config=lambda cfg: None, runtime_from_config=lambda *args, **kwargs: Runtime(),
                     use_vlm_planner_perception=use_vlm_planner_perception, get_cam_intr=lambda *args: np.eye(3),
                     make_simple_cfg=lambda settings: settings, get_scene_bnds=lambda *args: (np.zeros((3, 2)), 1),
                     pos_habitat_to_normal=np.asarray, pos_normal_to_habitat=np.asarray,
                     pose_habitat_to_normal=lambda value: value, pose_normal_to_tsdf=lambda value: value,
                     VLM=lambda cfg: None, TSDFPlanner=Planner, qwen3_vl=qwen,
                     tqdm=lambda value, **kwargs: value, write_progress=progress,
                     summarize_goe=lambda *args, **kwargs: [], summarize_fbe_from_planner=lambda *args, **kwargs: [],
                     DirectActionExecutor=DirectActionExecutor, path_increment=path_increment,
                     save_and_close_figure=lambda *args: None, score=lambda results: (1, 1, 1, 1),
                     target_index=lambda *args: 0, quat_to_coeffs=lambda x: NS(tolist=lambda: [1, 0, 0, 0]),
                     quat_from_angle_axis=lambda *args: 1)
    monkeypatch.setattr(np, "quaternion", lambda *args: args, raising=False)
    module = ast.parse((root / "main.py").read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(root / "main.py"), "exec"), namespace)
    if fail:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            namespace["main"](cfg)
    else:
        namespace["main"](cfg)
    return Path(cfg.output_dir)


@pytest.mark.parametrize("action,transitions", [("stop", 0), ("move_forward", 1)])
def test_actual_main_finalizes_stop_and_direct_continue(tmp_path, monkeypatch, action, transitions):
    output = _run_main(tmp_path, monkeypatch, action=action)
    steps = read_jsonl(output / "efficiency_steps.jsonl")
    assert len(steps) == 1
    assert steps[0]["simulator_transition_count"] == transitions
    assert steps[0]["status"] == "ok"
    assert steps[0]["observation_count"] == 1
    episode = read_jsonl(output / "efficiency_episodes.jsonl")[0]
    assert episode["task_completion_ms"] <= episode["wall_ms"]
    assert episode["evaluation_ms"] >= 0
    assert (output / "0/result.pkl").exists()


def test_actual_main_preserves_black_frame_step(tmp_path, monkeypatch):
    output = _run_main(tmp_path, monkeypatch, black=True)
    step = read_jsonl(output / "efficiency_steps.jsonl")[0]
    assert step["perception_skipped"] == "black_image"
    assert step["model_call_count"] == 1


def test_actual_main_records_error_before_episode_result(tmp_path, monkeypatch):
    output = _run_main(tmp_path, monkeypatch, fail=True)
    assert read_jsonl(output / "efficiency_steps.jsonl")[0]["status"] == "error"
    assert read_jsonl(output / "efficiency_episodes.jsonl")[0]["status"] == "error"
    assert not (output / "0/result.pkl").exists()


def test_actual_main_counts_three_rotations_in_one_goe_step(tmp_path, monkeypatch):
    output = _run_main(tmp_path, monkeypatch, planner=True, action="goe", goe=True)
    steps = read_jsonl(output / "efficiency_steps.jsonl")
    assert len(steps) == 1
    assert steps[0]["simulator_transition_count"] == 3


def test_actual_main_baseline_stop_path_remains_supported(tmp_path, monkeypatch):
    output = _run_main(tmp_path, monkeypatch, baseline=True)
    assert read_jsonl(output / "efficiency_steps.jsonl")[0]["outcome"] == "answered"


def test_actual_main_without_profiling_creates_no_efficiency_files(tmp_path, monkeypatch):
    output = _run_main(tmp_path, monkeypatch, enabled=False)
    assert (output / "0/result.pkl").exists()
    assert not list(output.glob("efficiency*"))
