"""Habitat episode loop for the fair evidence-aware active-perception path."""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from evaluation import score
from Jev_scripts.parallel_runner import write_progress

from .active_perception_hook import ActivePerceptionController
from .geometry_candidates import (
    GeometryCandidateGenerator,
    geometry_frontiers,
    observed_viewpoints,
)
from .geometry_navigation import bounded_waypoint, NavigationStep, yaw_toward
from .intern_decision_engine import InternDecisionEngine


def prepare_candidate_snapshot(
    *,
    free: np.ndarray,
    observed: np.ndarray,
    current_xy: Sequence[float],
    origin_xy: Sequence[float],
    voxel_size: float,
    floor_height: float,
    path_checker: Callable[[tuple[float, float, float]], bool],
    max_candidates: int = 7,
    max_fbe: int = 4,
    max_goe: int = 3,
    goe_provider: str = "observed_viewpoints",
):
    if goe_provider not in {"observed_viewpoints", "none"}:
        raise ValueError("goe_provider must be observed_viewpoints or none")
    fbe = []
    if max_fbe > 0:
        fbe = geometry_frontiers(
            free, observed, current_xy=current_xy, origin_xy=origin_xy,
            voxel_size=voxel_size, floor_height=floor_height,
            max_candidates=max_fbe,
        )
    goe = []
    if goe_provider == "observed_viewpoints" and max_goe > 0:
        goe = observed_viewpoints(
            free, observed, current_xy=current_xy, origin_xy=origin_xy,
            voxel_size=voxel_size, floor_height=floor_height,
            max_candidates=max_goe,
        )
    if not fbe and not goe:
        fbe = [{"point": [float(current_xy[0]), float(current_xy[1]), float(floor_height)],
                "distance_m": 0.0, "view_bearing_rad": math.pi / 2,
                "expected_new_fraction": 0.0, "novelty": 0.2}]
    generator = GeometryCandidateGenerator(max_candidates=max_candidates)
    return generator.build(fbe=fbe, goe=goe, path_checker=path_checker), generator



def exploration_gain(before: np.ndarray, after: np.ndarray) -> float:
    """Share of currently observed cells that became observed this step."""
    before = np.asarray(before, dtype=bool)
    after = np.asarray(after, dtype=bool)
    if before.shape != after.shape:
        raise ValueError("exploration masks must have matching shape")
    new = np.count_nonzero(after & ~before)
    return float(new / max(1, np.count_nonzero(after)))


def _active_cfg(cfg: Any) -> Any:
    return cfg.active_perception if hasattr(cfg, "active_perception") else cfg["active_perception"]


def _get(section: Any, key: str, default: Any) -> Any:
    try:
        return section.get(key, default)
    except AttributeError:
        return section[key] if key in section else default


def route_points(path: Any) -> list[tuple[float, float, float]] | None:
    points = [tuple(float(item) for item in point) for point in getattr(path, "points", [])]
    if len(points) < 2 or any(len(point) != 3 or not all(math.isfinite(item) for item in point) for point in points):
        return None
    return points


def _route(simulator: Any, start: np.ndarray, target: np.ndarray):
    import habitat_sim
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
        return None
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=float)
    path.requested_end = np.asarray(target, dtype=float)
    if not simulator.pathfinder.find_path(path):
        return None
    points = route_points(path)
    return (path, points) if points is not None else None


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array[..., :3], mode="RGB").save(path)


def _qwen_call(prompt_path: str, prompt: str, images: Any, device: str) -> str:
    from gpt import qwen3_vl
    result = qwen3_vl(prompt_path, prompt, images, device=device)
    return result.replace("A:", "").strip()



def safe_model_text(callable_fn: Callable[..., str], prompt_path: str, prompt: str,
                    images: Any, device: str) -> tuple[str, str]:
    try:
        return callable_fn(prompt_path, prompt, images, device), "ok"
    except Exception as exc:
        logging.error("answer model failure: %s", exc.__class__.__name__)
        return "", exc.__class__.__name__


def normalize_eac(value: str, status: str) -> tuple[str, str]:
    """Keep result.pkl compatible when Qwen scoring fails or returns bad text."""
    if status != "ok":
        return "0,0", status
    try:
        cleaned = value.replace("Your mark:", "").strip()
        parts = [part.strip() for part in cleaned.split(",")]
        if len(parts) != 2:
            raise ValueError("EAC needs two fields")
        grounded, accuracy = float(parts[0]), int(parts[1])
        if not math.isfinite(grounded) or grounded not in (0.0, 0.5, 1.0) or not 1 <= accuracy <= 5:
            raise ValueError("EAC score is out of range")
        return f"{grounded:g},{accuracy}", "ok"
    except (TypeError, ValueError):
        return "0,0", "invalid_format"


def run_active(cfg: Any, *, question_indices: list[int] | None = None,
               progress_path: str | None = None, append_trace: bool = False) -> list[dict[str, Any]]:
    import habitat_sim
    from src.habitat import pos_habitat_to_normal, pos_normal_to_habitat
    from src.geom import get_cam_intr, get_scene_bnds
    from src.tsdf import TSDFPlanner

    active = _active_cfg(cfg)
    questions = json.loads(Path(str(cfg.dataset_path)).read_text(encoding="utf-8"))
    indices = list(range(len(questions))) if question_indices is None else [int(item) for item in question_indices]
    output_dir = Path(str(cfg.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "evidence_aware_decisions.jsonl"
    if not append_trace:
        trace_path.write_text("", encoding="utf-8")
    progress_path = str(progress_path or output_dir / "progress.json")
    progress = {"status": "loading_model", "phase": "loading_model", "completed_questions": 0,
                "total_questions": len(indices), "current_question": None, "current_step": None}
    write_progress(progress_path, progress)

    engine = InternDecisionEngine(
        checkpoint=str(_get(active, "checkpoint", "/ssd/wyzhang/all_models/Intern-Decision-4B")),
        device=str(_get(active, "device", "cuda:0")), dtype=str(_get(active, "dtype", "bfloat16")),
        temperature=float(_get(active, "temperature", 1.99241824)),
        media_root=str(_get(active, "media_root", ".")),
        max_length=int(_get(active, "max_length", 8192)),
        attn_implementation=str(_get(active, "attn_implementation", "sdpa")),
    )
    qwen_device = str(_get(active, "answer_device", _get(active, "device", "cuda:0")))
    answer_prompt = "./prompt/answer.txt"
    score_prompt = "./prompt/evaluation.txt"
    simulator = None
    results: list[dict[str, Any]] = []
    for position, question_ind in enumerate(indices):
        data = questions[question_ind]
        question = data["question"]
        question_type = data["type"]
        scene_id = data["scene_id"]
        scene = os.path.join(str(cfg.scene_data_path), scene_id, scene_id.split("-")[1] + ".basis.glb")
        navmesh = scene.replace("glb", "navmesh")
        episode_dir = output_dir / str(question_ind)
        episode_dir.mkdir(parents=True, exist_ok=True)
        if simulator is not None:
            simulator.close()
        settings = {
            "scene": scene, "scene_dataset": None, "default_agent": 0,
            "sensor_height": float(cfg.camera_height), "width": int(cfg.img_width),
            "height": int(cfg.img_height), "hfov": float(cfg.hfov),
            "color_sensor": True, "depth_sensor": True, "semantic_sensor": False,
        }
        from src.habitat import make_simple_cfg
        simulator = habitat_sim.Simulator(make_simple_cfg(settings))
        simulator.pathfinder.seed(int(cfg.seed))
        simulator.pathfinder.load_nav_mesh(navmesh)
        agent = simulator.initialize_agent(0)
        state = habitat_sim.AgentState()
        pts = np.asarray(data["start_position"], dtype=float)
        rotation = np.quaternion(*data["start_rotation"])
        yaw = 0.0
        state.position, state.rotation = pts, rotation
        agent.set_state(state)
        pts_normal = pos_habitat_to_normal(pts)
        floor_height = float(pts_normal[-1])
        bounds, scene_size = get_scene_bnds(simulator.pathfinder, floor_height)
        max_steps = int(_get(active, "max_steps", 0)) or max(1, int(math.sqrt(scene_size) * float(cfg.max_step_room_size_ratio)))
        cam_intr = get_cam_intr(float(cfg.hfov), int(cfg.img_height), int(cfg.img_width))
        planner = TSDFPlanner(bounds, float(cfg.tsdf_grid_size), regs=[], max_exp=0,
                              floor_height_offset=0, pts_init=pts_normal,
                              init_clearance=float(cfg.init_clearance) * 2)
        controller = ActivePerceptionController(
            question, question_type, engine=engine,
            max_candidates=int(_get(active, "max_candidates", 7)), max_steps=max_steps,
            history_window=int(_get(active, "history_window", 8)),
            max_history_images=int(_get(active, "max_history_images", 2)),
            progress_threshold=float(_get(active, "progress_threshold", 0.01)),
            stop_threshold=float(_get(active, "stop_threshold", 0.75)),
            utility_threshold=float(_get(active, "utility_threshold", 0.05)),
            no_progress_patience=int(_get(active, "no_progress_patience", 3)),
            cost_weight=float(_get(active, "cost_weight", 0.5)),
            novelty_weight=float(_get(active, "novelty_weight", 0.1)),
        )
        result: dict[str, Any] = {"question_ind": question_ind, "question": question,
                                  "answer": data["answer"], "type": question_type,
                                  "path": {}, "intern_decisions": [], "jev_decisions": [],
                                  "active_decisions": []}
        path_len = 0.0
        previous = {"action": None, "displacement_m": 0.0, "collided": False}
        gen_answer = ""
        last_image = None
        last_evidence_images: list[str] = []
        find_answer = False
        for step in range(max_steps):
            progress.update(status="running", phase="question", current_question=question_ind,
                            current_step=step)
            write_progress(progress_path, progress)
            state.position, state.rotation = pts, rotation
            agent.set_state(state)
            result["path"][f"step_{step}"] = {"pts": np.asarray(pts).copy(), "rotation": rotation}
            obs = simulator.get_sensor_observations()
            rgb, depth = obs["color_sensor"], obs["depth_sensor"]
            image_path = episode_dir / f"{step}.png"
            _save_rgb(image_path, rgb)
            last_image = image_path
            sensor = agent.get_state().sensor_states["depth_sensor"]
            cam_pose = np.eye(4)
            import quaternion as np_quaternion
            cam_pose[:3, :3] = np_quaternion.as_rotation_matrix(sensor.rotation)
            cam_pose[:3, 3] = sensor.position
            from src.habitat import pose_habitat_to_normal, pose_normal_to_tsdf
            observed_before = np.sum(planner._explore_vol_cpu, axis=2) > 0
            if np.sum(np.sum(rgb, axis=-1) == 0) < float(cfg.black_pixel_ratio) * int(cfg.img_width) * int(cfg.img_height):
                planner.integrate(color_im=rgb, depth_im=depth, cam_intr=cam_intr,
                                  cam_pose=pose_normal_to_tsdf(pose_habitat_to_normal(cam_pose)),
                                  obs_weight=1.0, margin_h=int(float(cfg.margin_h_ratio) * int(cfg.img_height)),
                                  margin_w=int(float(cfg.margin_w_ratio) * int(cfg.img_width)))
            current_normal = pos_habitat_to_normal(pts)
            observed_after = np.sum(planner._explore_vol_cpu, axis=2) > 0
            current_gain = exploration_gain(observed_before, observed_after)
            candidate_status = "ok"
            try:
                free, _ = planner.get_island_around_pts(current_normal)
                observed = observed_after
                def path_checker(point):
                    target = simulator.pathfinder.snap_point(pos_normal_to_habitat(np.asarray(point, dtype=float)))
                    if not np.all(np.isfinite(target)):
                        return False
                    if np.linalg.norm(np.asarray(target) - pts) < 0.05:
                        return True
                    return _route(simulator, pts, target) is not None
                candidates, generator = prepare_candidate_snapshot(
                    free=free, observed=observed, current_xy=current_normal[:2],
                    origin_xy=planner._vol_origin[:2], voxel_size=planner._voxel_size,
                    floor_height=floor_height, path_checker=path_checker,
                    max_candidates=int(_get(active, "max_candidates", 7)),
                    max_fbe=int(_get(active, "max_fbe_candidates", 4)),
                    max_goe=int(_get(active, "max_goe_candidates", 3)),
                    goe_provider=str(_get(active, "goe_provider", "observed_viewpoints")),
                )
            except Exception as exc:
                candidate_status = f"fallback_{exc.__class__.__name__}"
                logging.warning("geometry candidate generation failed: %s", exc.__class__.__name__)
                candidates, generator = prepare_candidate_snapshot(
                    free=np.zeros((1, 1), dtype=bool), observed=np.zeros((1, 1), dtype=bool),
                    current_xy=current_normal[:2], origin_xy=current_normal[:2],
                    voxel_size=float(cfg.tsdf_grid_size), floor_height=floor_height,
                    path_checker=lambda point: True,
                    max_candidates=int(_get(active, "max_candidates", 7)),
                    max_fbe=1, max_goe=0,
                )
            step_result = controller.step(
                step=step, image_path=image_path, pose=pts,
                candidates=candidates, newly_explored_fraction=current_gain, **previous,
            )
            trace = step_result.trace
            trace["candidate_status"] = candidate_status
            last_evidence_images = step_result.state["images"]
            if step_result.decision.action == "stop":
                trace["transition"] = {"status": "not_executed", "distance_m": 0.0}
                result["active_decisions"].append(trace)
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"question_index": question_ind, **trace}, ensure_ascii=False) + "\n")
                images = step_result.state.get("images", [str(image_path)])
                gen_answer, answer_status = safe_model_text(
                    _qwen_call, answer_prompt, f"Q: {question}\nA: ", images, qwen_device
                )
                result["answer_status"] = answer_status
                find_answer = True
                break
            target_normal = generator.target_point(step_result.decision.target_id)
            target_habitat = simulator.pathfinder.snap_point(pos_normal_to_habitat(np.asarray(target_normal)))
            if not np.all(np.isfinite(target_habitat)):
                trace["transition"] = {"status": "invalid_snap", "distance_m": 0.0}
                result["active_decisions"].append(trace)
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"question_index": question_ind, **trace}, ensure_ascii=False) + "\n")
                previous = {"action": step_result.decision.target_id, "displacement_m": 0.0, "collided": True}
                continue
            stationary_scan = np.linalg.norm(np.asarray(target_habitat) - pts) < 0.05
            route = None if stationary_scan else _route(simulator, pts, target_habitat)
            if route is None and not stationary_scan:
                trace["transition"] = {"status": "unreachable", "distance_m": 0.0}
                result["active_decisions"].append(trace)
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"question_index": question_ind, **trace}, ensure_ascii=False) + "\n")
                previous = {"action": step_result.decision.target_id, "displacement_m": 0.0,
                            "collided": True}
                continue
            if stationary_scan:
                waypoint = NavigationStep(tuple(float(item) for item in pts), 0.0, True)
            else:
                _, route_points = route
                waypoint = bounded_waypoint(route_points, max_step_m=float(_get(active, "max_move_m", 3.0)))
            before = np.asarray(pts, dtype=float).copy()
            next_position = np.asarray(waypoint.position, dtype=float)
            path_len += waypoint.distance_m
            before_normal = pos_habitat_to_normal(before)
            after_normal = pos_habitat_to_normal(next_position)
            yaw = yaw_toward(before_normal[:2], after_normal[:2], current_yaw=yaw)
            from habitat_sim.utils.common import quat_from_angle_axis
            rotation = quat_from_angle_axis(yaw, np.array([0, 1, 0])) * quat_from_angle_axis(
                float(cfg.camera_tilt_deg) * math.pi / 180, np.array([1, 0, 0])
            )
            state.position, state.rotation = next_position, rotation
            agent.set_state(state)
            realized_state = agent.get_state()
            pts = np.asarray(realized_state.position, dtype=float)
            rotation = realized_state.rotation
            realized_displacement = float(np.linalg.norm(pts - before))
            trace["transition"] = {
                "status": "waypoint_applied", "execution_mode": "pathfinder_pose_placement",
                "route_distance_m": waypoint.distance_m, "displacement_m": realized_displacement,
                "collision_checked": False,
                "reached_target": waypoint.reached_target, "yaw_rad": yaw,
            }
            result["active_decisions"].append(trace)
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"question_index": question_ind, **trace}, ensure_ascii=False) + "\n")
            previous = {"action": step_result.decision.target_id,
                        "displacement_m": realized_displacement,
                        "collided": False}
        if not find_answer:
            if last_image is not None:
                gen_answer, answer_status = safe_model_text(
                    _qwen_call, answer_prompt, f"Q: {question}\nA: ", last_evidence_images or [str(last_image)], qwen_device
                )
                result["answer_status"] = answer_status
            else:
                gen_answer = ""
                result["answer_status"] = "error"
        eac_prompt = f"Question: {question}\nAnswer: {data['answer']}\nResponse: {gen_answer}\nYour mark: "
        eac, eac_status = safe_model_text(
            _qwen_call, score_prompt, eac_prompt,
            str(last_image) if last_image else None, qwen_device
        )
        eac_raw = eac
        eac, eac_status = normalize_eac(eac, eac_status)
        result["eac_status"] = eac_status
        if eac_status != "ok":
            result["eac_raw"] = eac_raw
        goal_path = _route(simulator, pts, np.asarray(data["goal_position"], dtype=float))
        goal_dis = goal_path[0].geodesic_distance if goal_path is not None else float("inf")
        result.update({"scene": scene, "geodesic_distance": data["geodesic_distance"],
                       "cnt_step": step, "gen_answer": gen_answer, "EAC": eac,
                       "end_pts": pts, "goal_dis": goal_dis, "path_len": path_len})
        with (episode_dir / "result.pkl").open("wb") as handle:
            pickle.dump(result, handle)
        results.append(result)
        progress.update(completed_questions=position + 1, current_question=None, current_step=None)
        write_progress(progress_path, progress)
    if simulator is not None:
        simulator.close()
    with (output_dir / "results.pkl").open("wb") as handle:
        pickle.dump(results, handle)
    if results:
        logging.info("active metrics: %s", score(results))
    progress.update(status="completed", phase="completed", completed_questions=len(results), current_question=None, current_step=None)
    write_progress(progress_path, progress)
    return results
