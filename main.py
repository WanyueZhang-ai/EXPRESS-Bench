import json
import sys
import time
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"
)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MAGNUM_LOG"] = "quiet"

import numpy as np
np.set_printoptions(precision=3)
import csv
import pickle
import logging
import math
import quaternion
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from habitat_sim.utils import viz_utils as vut
import magnum as mn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    store_obs
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner
from gpt import qwen3_vl
from evaluation import score
from src.plotting import save_and_close_figure
from src.efficiency import current_profiler, profiled_run, apply_efficiency_cli
from Jev_scripts.online_hook import (
    oracle_evidence_from_observation,
    router_from_config,
    semantic_category_names,
    target_index,
    trace_for_main,
)
from Jev_scripts.tsdf_candidates import summarize_fbe_from_planner, summarize_goe
from Jev_scripts.parallel_runner import write_progress
from InternDecision_scripts.action_executor import DirectActionExecutor, path_increment
from InternDecision_scripts.online_hook import runtime_from_config
from InternDecision_scripts.evidence_adapters import visible_object_summaries
from InternDecision_scripts.runtime_policy import use_vlm_planner_perception
from InternDecision_scripts.runtime_utils import scene_dataset_config_path


@profiled_run
def main(cfg, question_indices=None, progress_path=None, append_trace=False):
    efficiency = current_profiler()
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    jev_cfg = cfg.get("jev", {})
    jev_router = router_from_config(jev_cfg)
    jev_enabled = jev_router is not None
    jev_mode = str(jev_cfg.get("mode", "fair"))
    intern_cfg = cfg.get("intern_decision", {})
    intern_enabled = bool(intern_cfg.get("enabled", False))
    if jev_enabled and intern_enabled:
        raise ValueError("Enable either jev or intern_decision, not both")
    intern_evidence_mode = str(intern_cfg.get("evidence_mode", "vlm_state"))
    intern_action_mode = str(intern_cfg.get("action_mode", "planner_candidates"))
    intern_trace_path = os.path.join(str(cfg.output_dir), "intern_decisions.jsonl")
    jev_trace_path = os.path.join(str(cfg.output_dir), "jev_decisions.jsonl")
    if jev_enabled:
        os.makedirs(str(cfg.output_dir), exist_ok=True)
        if not append_trace:
            open(jev_trace_path, "w", encoding="utf-8").close()
    if intern_enabled:
        os.makedirs(str(cfg.output_dir), exist_ok=True)
        if not append_trace:
            open(intern_trace_path, "w", encoding="utf-8").close()
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)
    scene_data_path = cfg.scene_data_path
    output_dir = cfg.output_dir
    seed = cfg.seed
    qwen_device = str(intern_cfg.get("device") or cfg.vlm.get("device", "cuda:0") or "cuda:0")

    def _qwen_call(prompt_path, ex_prompt, img_path=None):
        return qwen3_vl(prompt_path, ex_prompt, img_path, device=qwen_device)

    # Prompt
    region_prompt = "./prompt/region.txt"
    explore_prompt = "./prompt/explore.txt"
    answer_prompt = "./prompt/answer.txt"
    random_answer_prompt = "./prompt/random_answer.txt"
    score_prompt = "./prompt/evaluation.txt"

    # Load dataset
    with open(cfg.dataset_path, 'r', encoding='utf-8') as file:
        questions_data = json.load(file)
    if question_indices is None:
        question_indices = list(range(len(questions_data)))
    else:
        question_indices = [int(index) for index in question_indices]
        invalid_indices = [
            index for index in question_indices
            if index < 0 or index >= len(questions_data)
        ]
        if invalid_indices:
            raise ValueError(
                "question_indices contains out-of-range values: "
                + repr(invalid_indices[:10])
            )
        if len(set(question_indices)) != len(question_indices):
            raise ValueError("question_indices must not contain duplicates")
    logging.info(
        f"Loaded {len(questions_data)} questions; "
        f"processing {len(question_indices)} selected indices."
    )

    progress_path = str(progress_path or os.path.join(output_dir, "progress.json"))
    progress_state = {
        "status": "loading_model",
        "phase": "loading_model",
        "completed_questions": 0,
        "total_questions": len(question_indices),
        "current_question": None,
        "current_question_position": None,
        "current_step": None,
        "last_completed_question": None,
    }
    last_progress_write = 0.0

    def update_progress(force=False, **updates):
        nonlocal last_progress_write
        progress_state.update(updates)
        now = time.monotonic()
        if force or now - last_progress_write >= 2.0:
            efficiency.measure("io.progress", write_progress, progress_path, progress_state, category="io")
            last_progress_write = now

    update_progress(force=True)

    # Load the decision model once. Non-vlm_state modes do not load the
    # Prismatic VLM used for perception; Qwen remains available for answers/EAC.
    intern_runtime = None
    if intern_enabled:
        def _qwen_answer(question_text, question_type_text, answer_state):
            image_paths = answer_state.get("images", []) if isinstance(answer_state, dict) else []
            image_for_answer = image_paths[0] if image_paths else None
            text = _qwen_call(answer_prompt, f"Q: {question_text}\nA: ", image_for_answer)
            return text.replace("A:", "").strip()
        intern_runtime = efficiency.measure("initialization.intern_decision", runtime_from_config, intern_cfg,
                                            qwen_fn=_qwen_answer, category="initialization")
    use_planner_perception = use_vlm_planner_perception(
        intern_enabled, intern_action_mode, evidence_mode=intern_evidence_mode
    )
    if intern_enabled and use_planner_perception:
        # Keep the planner VLM on the same worker GPU as Intern/Qwen.
        cfg.vlm.device = qwen_device
    vlm = efficiency.measure("initialization.planner_vlm", VLM, cfg.vlm, category="initialization") if use_planner_perception else None
    update_progress(force=True, status="running", phase="exploration")

    # Run all questions
    results_all = []
    simulator = None
    for question_position, question_ind in enumerate(
        tqdm(question_indices, total=len(question_indices))
    ):
        episode_scope = efficiency.begin_episode(question_ind, question_type=questions_data[question_ind]["type"])
        update_progress(
            force=True,
            status="running",
            phase="question",
            current_question=question_ind,
            current_question_position=question_position + 1,
            current_step=None,
        )

        result = {"question_ind": question_ind, "jev_decisions": [], "intern_decisions": []}
        question_data = questions_data[question_ind]
        question = question_data["question"]
        answer = question_data["answer"]
        question_type = question_data["type"]
        scene = os.path.join(scene_data_path, question_data["scene_id"], question_data["scene_id"].split("-")[1]+".basis.glb")
        navmesh_file = scene.replace("glb", "navmesh")
        init_pts = question_data["start_position"]
        init_rotation = question_data["start_rotation"]
        goal_pts = question_data["goal_position"]
        init_angle = 0

        logging.info(f"\n========\nIndex: {question_ind}\nScene: {scene}\nQuestion: {question}\nAnswer: {answer}")
        episode_data_dir = os.path.join(output_dir, str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)

        # Set up scene in Habitat
        if simulator is not None:
            efficiency.measure("simulator.close", simulator.close, category="simulator")
            simulator = None
        semantic_scene_dataset = None
        if (jev_enabled and jev_mode == "oracle_semantic") or (
            intern_enabled and intern_evidence_mode == "decision_objects"
        ):
            semantic_scene_dataset = scene_dataset_config_path(scene)
            if not semantic_scene_dataset.is_file():
                raise FileNotFoundError(
                    "HM3D semantic scene dataset config not found: "
                    f"{semantic_scene_dataset} (scene: {scene})"
                )
        sim_settings = {
            "scene": scene,
            "scene_dataset": str(semantic_scene_dataset) if semantic_scene_dataset else None,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
            "color_sensor": cfg.rgb_sensor,
            "depth_sensor": cfg.depth_sensor,
            # Keep the baseline/fair runs RGB-D only. The oracle condition
            # explicitly enables a current-FOV semantic sensor.
            "semantic_sensor": bool(
                cfg.semantic_sensor and (
                    (jev_enabled and jev_mode == "oracle_semantic")
                    or (intern_enabled and intern_evidence_mode == "decision_objects")
                )
            )
        }
        sim_cfg = make_simple_cfg(sim_settings)
        simulator = efficiency.measure("initialization.simulator", habitat_sim.Simulator, sim_cfg, category="initialization")
        oracle_category_names = (
            semantic_category_names(simulator)
            if jev_enabled and jev_mode == "oracle_semantic"
            else {}
        )
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        efficiency.measure("initialization.navmesh", pathfinder.load_nav_mesh, navmesh_file, category="initialization")
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        pts = init_pts
        rotation = np.quaternion(init_rotation[0], init_rotation[1], init_rotation[2], init_rotation[3])
        angle = init_angle

        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
        if intern_enabled:
            configured_max_steps = int(intern_cfg.get("max_steps", 0))
            if configured_max_steps > 0:
                num_step = min(num_step, configured_max_steps)
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )

        # Identify task-related regions
        regions_dict = {'A':"bathroom", 'B': "bedroom", 'C': "dining room", 'D':"garage", 'E': "kitchen", 'F': "laundry room", "G": "living room", "H": "office", "I": "rec room", "J": "study", "K": "hallway", "L": "entryway", "M": "laboratory", "N": "workout room", "O": "warehouse", "P": "lounge", "Q": "balcony", "R": "staircase", "S": "cloakroom", "T": "unknown"}
        letters = list(regions_dict.keys())
        regions = list(regions_dict.values())
        if not use_planner_perception:
            regs = []
            regs_list = []
            logging.info("Intern-Decision evidence mode %s: skipping VLM region recognition", intern_evidence_mode)
        else:
            ex_prompt = f"QUESTION: {question}\nREGION: "
            regs = _qwen_call(region_prompt, ex_prompt)
            regs = regs.split(",")
            regs_list = []
            for r in regs:
                if (r != 'unknown') and (r in regions):
                    regs_list.append(regions.index(r))
            logging.info(f"regs: {regs}")
        result["regs_list"] = regs
        result["path"] = {}

        # Initialize TSDF
        tsdf_planner = efficiency.measure("initialization.tsdf", TSDFPlanner,
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            regs=regs_list,
            max_exp=3,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
            category="initialization",
        )

        pts_pixs = np.empty((0, 2))
        last_pts = pts
        path_len = 0
        find_answer = False
        exp_list = []
        intern_state = None
        direct_forced_action = None

        for cnt_step in range(num_step):
            step_scope = efficiency.begin_step(cnt_step, memory_device=qwen_device)
            update_progress(
                phase="question",
                current_question=question_ind,
                current_question_position=question_position + 1,
                current_step=cnt_step,
            )
            logging.info(f"\n== step: {cnt_step}")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")
            agent_state.position = pts
            agent_state.rotation = rotation
            efficiency.measure("simulator.set_pose", agent.set_state, agent_state, category="simulator")
            pts_normal = pos_habitat_to_normal(pts)
            result["path"][step_name] = {"pts": agent_state.position, "rotation": agent_state.rotation}

            # Update camera info
            sensor = agent.get_state().sensor_states["depth_sensor"]
            quaternion_0 = sensor.rotation
            translation_0 = sensor.position
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
            cam_pose[:3, 3] = translation_0
            cam_pose_normal = pose_habitat_to_normal(cam_pose)
            cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

            # Get observation at current pose - skip black image, meaning robot is outside the floor
            obs = efficiency.measure("simulator.observe", simulator.get_sensor_observations, category="simulator")
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]

            if cfg.save_obs or intern_enabled:
                efficiency.measure("io.save_observation", plt.imsave,
                    os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgb, category="io"
                )
            num_black_pixels = np.sum(
                np.sum(rgb, axis=-1) == 0
            )

            perception_summary = {}
            img_path = os.path.abspath(os.path.join(episode_data_dir, f"{cnt_step}.png"))
            if num_black_pixels < cfg.black_pixel_ratio * img_width * img_height:
                # TSDF fusion
                efficiency.measure("mapping.tsdf", tsdf_planner.integrate,
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=cam_pose_tsdf,
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width), category="mapping"
                )

                # Terminate exploration with the VLM only for the vlm_state
                # condition. Image/object Intern conditions pass evidence directly
                # to the decision model.
                ex_prompt = f"QUESTION: {question}"
                stop_explore = (
                    "no"
                    if intern_enabled and intern_evidence_mode != "vlm_state"
                    else _qwen_call(explore_prompt, ex_prompt, img_path)
                )
                stop_vlm_yes = "yes" in stop_explore.lower()
                perception_summary.update({
                    "stop_vlm_label": "yes" if stop_vlm_yes else "no",
                    "stop_vlm_raw": stop_explore,
                })
                if stop_vlm_yes and not jev_enabled and not intern_enabled:
                    ex_prompt = f"Q: {question}\nA: "
                    gen_answer = _qwen_call(answer_prompt, ex_prompt, img_path)
                    gen_answer = gen_answer.replace("A:", "").strip()
                    logging.info(f"gen_answer: {gen_answer}")
                    find_answer = True
                    step_scope.update(outcome="answered", selected_action="stop")
                    efficiency.end_step()
                    break

                if use_planner_perception:
                    # Semantic map. In Jev mode we continue this pass even when
                    # the VLM thinks the answer is ready so Jev sees the same
                    # current-step map/candidate evidence.
                    prompt_points_pix = []
                    prompt_points_pix, fig1 = (
                        efficiency.measure("planning.prompt_candidates", tsdf_planner.find_prompt_points_within_view,
                            pts_normal,
                            img_width,
                            img_height,
                            cam_intr,
                            cam_pose_tsdf,
                            **cfg.visual_prompt, category="planning"
                        )
                    )
                    save_and_close_figure(
                        fig1,
                        os.path.join(episode_data_dir, "{}_prompt_points.png".format(cnt_step)),
                    )

                    # Get VLM prediction
                    rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
                    draw_letters = ["A", "B", "C", "D"]
                    fnt = efficiency.measure("io.load_font", ImageFont.truetype,
                        "data/Open_Sans/static/OpenSans-Regular.ttf",
                        30, category="io"
                    )
                    actual_num_prompt_points = len(prompt_points_pix)
                    if actual_num_prompt_points >= cfg.visual_prompt.min_num_prompt_points:
                        rgb_im_draw = rgb_im.copy()
                        draw = ImageDraw.Draw(rgb_im_draw)
                        for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
                            draw.ellipse(
                                (
                                    point_pix[0] - cfg.visual_prompt.circle_radius,
                                    point_pix[1] - cfg.visual_prompt.circle_radius,
                                    point_pix[0] + cfg.visual_prompt.circle_radius,
                                    point_pix[1] + cfg.visual_prompt.circle_radius,
                                ),
                                fill=(200, 200, 200, 255),
                                outline=(0, 0, 0, 255),
                                width=3,
                            )
                            draw.text(
                                tuple(point_pix.astype(int).tolist()),
                                draw_letters[prompt_point_ind],
                                font=fnt,
                                fill=(0, 0, 0, 255),
                                anchor="mm",
                                font_size=12,
                            )
                        efficiency.measure("io.save_prompt_image", rgb_im_draw.save,
                            os.path.join(episode_data_dir, f"{cnt_step}_draw.png"), category="io"
                        )

                        prompt_lsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Answer with a single letter."
                        lsv = vlm.get_loss(
                            rgb_im_draw,
                            prompt_lsv,
                            draw_letters[:actual_num_prompt_points],
                            purpose="local_direction_scoring",
                        )
                        lsv *= actual_num_prompt_points / 3
                        prompt_gsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
                        gsv = vlm.get_loss(rgb_im, prompt_gsv, ["Yes", "No"], purpose="global_exploration_scoring")[0]
                        gsv = (
                            np.exp(gsv / cfg.gsv_T) / cfg.gsv_F
                        )
                        sv = lsv * gsv
                        logging.info(f"Exp - LSV: {lsv} GSV: {gsv} SV: {sv}")
                        perception_summary.update({
                            "lsv": np.asarray(lsv).astype(float).tolist(),
                            "gsv": float(gsv),
                            "sv": np.asarray(sv).astype(float).tolist(),
                        })

                        # Integrate semantics only if there is any prompted point
                        efficiency.measure("mapping.semantic", tsdf_planner.integrate_sem,
                            sem_pix=sv,
                            radius=1.0,
                            obs_weight=1.0, category="mapping"
                        )

                        # Get region prediction
                        prompt_reg = f"You need to determine what region the picture shows. All possible regions are shown in the dictionary {regions_dict}. Answer with the key corresponding to the region in the dictionary."
                        pred = vlm.get_loss(
                            rgb_im,
                            prompt_reg,
                            letters,
                            purpose="region_scoring",
                        )
                        array_reg = np.array(pred)
                        region = regions[np.argmax(array_reg)]
                        region_confidence = float(np.max(array_reg))
                        perception_summary.update({
                            "visible_region": region,
                            "visible_region_confidence": region_confidence,
                        })
                        logging.info(f"Predicted region: {region}, Confidence: {region_confidence}")

                        # Update TSDF with the predicted region
                        if region != "unknown":
                            prompt_pt = f"Based on the content of the picture, please judge which letter indicates the location that belongs to the {region} area. Answer with a single letter."
                            pred = vlm.get_loss(
                                rgb_im_draw,
                                prompt_pt,
                                draw_letters[:actual_num_prompt_points],
                                purpose="region_point_scoring",
                            )
                            array_pt = np.array(pred)
                            point = draw_letters[np.argmax(array_pt)]
                            point_confidence = float(np.max(array_pt))
                            perception_summary.update({
                                "visible_point": point,
                                "visible_point_confidence": point_confidence,
                            })
                            logging.info(f"Predicted point: {point}, Confidence: {point_confidence}")

                            # Update only the region with the highest probability and its coordinates
                            efficiency.measure("mapping.region", tsdf_planner.integrate_reg,
                                [np.argmax(array_pt)],
                                [np.argmax(array_reg)],
                                radius=1.0, category="mapping"
                            )

                        # Draw semantic map and region map
                        fig2 = efficiency.measure("visualization.semantic_map", tsdf_planner.draw_map, regions_dict, category="visualization")
                        save_and_close_figure(
                            fig2,
                            os.path.join(episode_data_dir, "{}_map.png".format(cnt_step + 1)),
                        )

            else:
                logging.info("Skipping black image!")
                step_scope.update(perception_skipped="black_image")

            # Determine whether to use GOE or FBE. The Intern branch shares
            # this exact candidate snapshot with Jev and runs before the
            # existing executor. The two backends are mutually exclusive.
            if regs_list != []:
                state, exp_list = efficiency.measure("planning.exploration_state", tsdf_planner.get_exp_state, category="planning")
            else:
                state = True
                exp_list = []
            found_path = True
            execution_state = state
            jev_decision = None
            jev_region_index = None
            intern_decision = None
            intern_state = None
            if intern_enabled:
                max_candidates = int(intern_cfg.get("max_candidates", 8))
                region_names = {index: name for index, name in enumerate(regions)}
                goe_candidates = efficiency.measure("evidence.goe_candidates", summarize_goe,
                    exp_list, region_names=region_names, max_candidates=max_candidates, category="evidence"
                )
                fbe_candidates = efficiency.measure("evidence.fbe_candidates", summarize_fbe_from_planner,
                    tsdf_planner, max_candidates=max_candidates, category="evidence"
                )
                baseline_mode = "fbe" if state else "goe"
                if baseline_mode == "goe" and not exp_list:
                    baseline_mode = "fbe"
                if baseline_mode == "goe" and exp_list:
                    baseline_idx = int(np.argmax([item.get("count", 0) for item in exp_list]))
                    baseline_target = "goe_{}".format(baseline_idx)
                else:
                    baseline_target = "fbe_0" if fbe_candidates else None
                object_rows = None
                if intern_evidence_mode == "decision_objects":
                    semantic_observation = obs.get("semantic_sensor")
                    if semantic_observation is None:
                        raise RuntimeError(
                            "intern decision_objects requires the current Habitat semantic_sensor observation"
                        )
                    scene_objects = getattr(getattr(simulator, "semantic_scene", None), "objects", [])
                    object_rows = efficiency.measure("evidence.objects", visible_object_summaries,
                        semantic_observation,
                        scene_objects,
                        agent_position=agent.get_state().position, category="evidence"
                    )
                intern_state = efficiency.measure("evidence.prepare", intern_runtime.evidence,
                    question=question,
                    question_type=question_type,
                    history={
                        "step": cnt_step,
                        "max_steps": num_step,
                        "path_length": float(path_len),
                    },
                    perception=perception_summary if intern_evidence_mode == "vlm_state" else {},
                    goe_candidates=([] if intern_action_mode == "direct_actions" else goe_candidates),
                    fbe_candidates=([] if intern_action_mode == "direct_actions" else fbe_candidates),
                    max_candidates=max_candidates,
                    image_path=img_path if intern_evidence_mode == "decision_image" else None,
                    objects=object_rows,
                    step=cnt_step,
                    decision_id=f"{question_ind}:{cnt_step}", category="evidence"
                )
                intern_decision = intern_runtime.decide(
                    intern_state,
                    baseline_action=(
                        intern_cfg.get("safe_direct_action", "turn_right")
                        if intern_action_mode == "direct_actions" else baseline_mode
                    ),
                    baseline_target_id=None if intern_action_mode == "direct_actions" else baseline_target,
                )
                if intern_action_mode == "direct_actions" and direct_forced_action is not None:
                    intern_decision.action = direct_forced_action
                    intern_decision.target_id = None
                    intern_decision.fallback = True
                    intern_decision.fallback_reason = "collision_safe_turn"
                    direct_forced_action = None
                trace = intern_runtime.trace(
                    intern_decision,
                    question_index=question_ind,
                    step=cnt_step,
                    seed=seed,
                )
                result["intern_decisions"].append(trace)
                step_scope.update(selected_action=intern_decision.action, decision_fallback=intern_decision.fallback,
                                  fallback_reason=intern_decision.fallback_reason)
                if intern_decision.action == "stop":
                    answer_state = dict(intern_state)
                    if os.path.isfile(img_path):
                        answer_state["images"] = [img_path]
                    answer_result = intern_runtime.answer(question, question_type, answer_state)
                    gen_answer = answer_result.text
                    result["answer_status"] = answer_result.status
                    if answer_result.error_type:
                        result["answer_error_type"] = answer_result.error_type
                    logging.info("Intern answer status=%s: %s", answer_result.status, gen_answer)
                    with efficiency.span("io.decision_trace", category="io"), open(intern_trace_path, "a", encoding="utf-8") as trace_file:
                        trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    find_answer = True
                    step_scope.update(outcome="answered")
                    efficiency.end_step()
                    break
                if intern_action_mode == "direct_actions":
                    configured_space = getattr(getattr(agent, "agent_config", None), "action_space", None)
                    if configured_space is None:
                        configured_space = {"move_forward", "turn_left", "turn_right"}
                    direct_executor = DirectActionExecutor(
                        action_space=configured_space,
                        action_whitelist=tuple(intern_cfg.get("direct_action_whitelist", ("move_forward", "turn_left", "turn_right", "stop"))),
                        safe_action=str(intern_cfg.get("safe_direct_action", "turn_right")),
                    )
                    before = np.asarray(agent.get_state().position, dtype=float).copy()
                    def _step_direct(action_name):
                        raw_observation = efficiency.measure("simulator.action", simulator.step, action_name, category="simulator")
                        after = np.asarray(agent.get_state().position, dtype=float)
                        collided = action_name == "move_forward" and bool(np.linalg.norm(after - before) < 1e-6)
                        return {"observation": raw_observation, "collided": collided}
                    action_result = direct_executor.execute(intern_decision.action, _step_direct)
                    result["intern_decisions"][-1]["executed_action"] = action_result.action
                    result["intern_decisions"][-1]["transition"] = {
                        "stepped": action_result.stepped,
                        "collided": action_result.collided,
                        "fallback": action_result.fallback,
                        "fallback_reason": action_result.fallback_reason,
                    }
                    if action_result.stepped:
                        pts = np.asarray(agent.get_state().position).copy()
                        path_len += path_increment(
                            action_result.action, before, pts, stepped=True
                        )
                        rotation = agent.get_state().rotation
                        last_pts = pts
                    if action_result.collided:
                        direct_forced_action = direct_executor.safe_action_for_next_cycle(action_result)
                    with efficiency.span("io.decision_trace", category="io"), open(intern_trace_path, "a", encoding="utf-8") as trace_file:
                        trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    step_scope.update(outcome="action_executed", collision=action_result.collided,
                                      executed_action=action_result.action)
                    efficiency.end_step()
                    continue
                with efficiency.span("io.decision_trace", category="io"), open(intern_trace_path, "a", encoding="utf-8") as trace_file:
                    trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                execution_state = intern_decision.action == "fbe"
                if intern_decision.action == "goe" and not intern_decision.fallback:
                    jev_region_index = target_index(intern_decision.target_id, "goe")
            elif jev_enabled:
                region_names = {index: name for index, name in enumerate(regions)}
                goe_candidates = efficiency.measure("evidence.goe_candidates", summarize_goe,
                    exp_list, region_names=region_names, max_candidates=int(jev_cfg.get("max_candidates", 8)), category="evidence"
                )
                fbe_candidates = efficiency.measure("evidence.fbe_candidates", summarize_fbe_from_planner,
                    tsdf_planner, max_candidates=int(jev_cfg.get("max_candidates", 8)), category="evidence"
                )
                baseline_mode = "fbe" if state else "goe"
                if baseline_mode == "goe" and not exp_list:
                    baseline_mode = "fbe"
                if baseline_mode == "goe" and exp_list:
                    baseline_idx = int(np.argmax([item.get("count", 0) for item in exp_list]))
                    baseline_target = "goe_{}".format(baseline_idx)
                else:
                    baseline_target = "fbe_0" if fbe_candidates else None
                oracle_semantic = None
                if jev_mode == "oracle_semantic":
                    semantic_observation = obs.get("semantic_sensor")
                    if semantic_observation is None:
                        raise RuntimeError(
                            "oracle_semantic requires the current Habitat semantic_sensor observation"
                        )
                    oracle_semantic = oracle_evidence_from_observation(
                        semantic_observation, category_names=oracle_category_names
                    )
                jev_decision = jev_router.decide(
                    question=question,
                    question_type=question_type,
                    history={
                        "step": cnt_step,
                        "max_steps": num_step,
                        "path_length": float(path_len),
                    },
                    perception=perception_summary,
                    goe_candidates=goe_candidates,
                    fbe_candidates=fbe_candidates,
                    baseline_action=baseline_mode,
                    baseline_target_id=baseline_target,
                    evidence_mode=jev_mode,
                    oracle_semantic=oracle_semantic,
                )
                trace = trace_for_main(
                    jev_decision,
                    episode_id=question_ind,
                    step=cnt_step,
                    seed=seed,
                    mode=jev_mode,
                )
                result["jev_decisions"].append(trace)
                step_scope.update(selected_action=jev_decision.action, decision_fallback=jev_decision.fallback,
                                  fallback_reason=jev_decision.fallback_reason)
                with efficiency.span("io.decision_trace", category="io"), open(jev_trace_path, "a", encoding="utf-8") as trace_file:
                    trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                if jev_decision.action == "stop":
                    ex_prompt = f"Q: {question}\nA: "
                    gen_answer = _qwen_call(answer_prompt, ex_prompt, img_path)
                    gen_answer = gen_answer.replace("A:", "").strip()
                    logging.info(f"gen_answer: {gen_answer}")
                    find_answer = True
                    step_scope.update(outcome="answered")
                    efficiency.end_step()
                    break
                execution_state = jev_decision.action == "fbe"
                if jev_decision.action == "goe" and not jev_decision.fallback:
                    jev_region_index = target_index(jev_decision.target_id, "goe")

            # GOE
            if execution_state == False:
                step_scope.update(execution_mode="goe")
                agent_state = agent.get_state()
                cur_pt = agent_state.position
                pts_normal, pts_pix, fig3 = efficiency.measure("planning.goe", tsdf_planner.find_next_point_region,
                    cur_pt, regions_dict, region_index=jev_region_index, category="planning"
                )
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)
                pts = efficiency.measure("planning.snap_point", pathfinder.snap_point, pts, category="planning")
                reg = [i["count"] for i in exp_list]
                idx = jev_region_index if jev_region_index is not None else int(np.argmax(reg))
                direction = exp_list[idx]["dir"]
                if direction == 0:
                    pts = efficiency.measure("planning.snap_point", pathfinder.snap_point, pts, category="planning")
                    path = habitat_sim.ShortestPath()
                    path.requested_start = np.array(last_pts)
                    path.requested_end = pts
                    found_path = efficiency.measure("planning.find_path", simulator.pathfinder.find_path, path, category="planning")
                    start_time = time.time()
                    while not found_path:
                        end_time = time.time()
                        if end_time - start_time > 60:
                            break
                        plt.close(fig3)
                        pts_normal, pts_pix, fig3 = efficiency.measure("planning.goe", tsdf_planner.find_next_point_region,
                            cur_pt, regions_dict, region_index=jev_region_index, category="planning"
                        )
                        pts_normal = np.append(pts_normal, floor_height)
                        pts = pos_normal_to_habitat(pts_normal)
                        pts = efficiency.measure("planning.snap_point", pathfinder.snap_point, pts, category="planning")
                        simulator.pathfinder.seed(seed)
                        path = habitat_sim.ShortestPath()
                        path.requested_start = np.array(last_pts)
                        path.requested_end = pts
                        found_path = efficiency.measure("planning.find_path", simulator.pathfinder.find_path, path, category="planning")
                    if found_path:
                        path_len += path.geodesic_distance
                        save_and_close_figure(
                            fig3,
                            os.path.join(episode_data_dir, "{}_region.png".format(cnt_step + 1)),
                        )
                    else:
                        plt.close(fig3)
                else:
                    plt.close(fig3)
                    for _ in range(3):
                        obs = efficiency.measure("simulator.action", simulator.step, "turn_left", category="simulator")
                    agent_state = agent.get_state()
                    pts = agent_state.position
                    rotation = agent_state.rotation

                region = regions[exp_list[idx]["region"]]
                logging.info(f"Region: {region}")

            # FBE
            if (execution_state == True) or (found_path == False):
                step_scope.update(execution_mode="fbe")
                pts_normal, angle, pts_pix, fig4 = efficiency.measure("planning.fbe", tsdf_planner.find_next_pose,
                    pts=pts_normal,
                    angle=angle,
                    flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                    **cfg.planner, category="planning"
                )
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)

                # Add path to ax5, with colormap to indicate order
                with efficiency.span("visualization.path_overlay", category="visualization"):
                    ax5 = fig4.axes[4]
                    ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                    ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                save_and_close_figure(
                    fig4,
                    os.path.join(episode_data_dir, "{}_semantic.png".format(cnt_step + 1)),
                )

                rotation = quat_to_coeffs(
                    quat_from_angle_axis(angle, np.array([0, 1, 0]))
                    * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                ).tolist()

                pts = efficiency.measure("planning.snap_point", pathfinder.snap_point, pts, category="planning")
                if np.isnan(pts).any():
                    pts = efficiency.measure("planning.random_point", pathfinder.get_random_navigable_point_near, last_pts, 3, category="planning")
                path = habitat_sim.ShortestPath()
                path.requested_start = np.array(last_pts)
                path.requested_end = pts
                found_path = efficiency.measure("planning.find_path", simulator.pathfinder.find_path, path, category="planning")
                if found_path:
                    path_len += path.geodesic_distance

            last_pts = pts
            step_scope.update(outcome="planned", path_found=bool(found_path))
            efficiency.end_step()


        if not find_answer:
            if intern_enabled and intern_state is not None:
                answer_state = dict(intern_state)
                if os.path.isfile(img_path):
                    answer_state["images"] = [img_path]
                answer_result = intern_runtime.answer(question, question_type, answer_state)
                gen_answer = answer_result.text
                result["answer_status"] = answer_result.status
                if answer_result.error_type:
                    result["answer_error_type"] = answer_result.error_type
            else:
                gen_answer = _qwen_call(random_answer_prompt, question)
                gen_answer = gen_answer.replace("A:", "").strip()
            logging.info(f"gen_answer: {gen_answer}")

        episode_scope.update(task_completion_ms=episode_scope.elapsed_ms())
        ex_prompt = f"Question: {question}\nAnswer: {answer}\nResponse: {gen_answer}\nYour mark: "
        img_path = os.path.join(episode_data_dir, f"{cnt_step}.png")
        with efficiency.span("evaluation.eac", category="evaluation") as evaluation_scope:
            EAC = _qwen_call(score_prompt, ex_prompt, img_path)
            episode_scope.update(evaluation_ms=evaluation_scope.elapsed_ms())

        # Episode summary
        logging.info(f"\n== Episode Summary")
        logging.info(f"Index: {question_ind}")
        logging.info(f"Scene: {scene}")
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Find_answer: {find_answer}")
        logging.info(f"Gen_answer: {gen_answer}")
        logging.info(f"EAC: {EAC}")
        logging.info(f"Path_len: {path_len}")

        # Distance to target point
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(pts)
        path.requested_end = np.array(goal_pts)
        found_path = efficiency.measure("planning.find_path", simulator.pathfinder.find_path, path, category="planning")
        goal_dis = path.geodesic_distance

        result.update({
            "scene": scene,
            "type": question_type,
            "question": question,
            "answer": answer,
            "geodesic_distance": question_data["geodesic_distance"],
            "cnt_step": cnt_step,
            "gen_answer": gen_answer,
            "EAC": EAC,
            "end_pts": pts,
            "goal_dis": goal_dis,
            "path_len": path_len,
        })
        episode_scope.update(path_length_m=float(path_len), goal_distance_m=float(goal_dis),
                             answer_status=result.get("answer_status", "ok"), eac=EAC,
                             geodesic_distance=float(question_data["geodesic_distance"]))

        # Save data
        results_all.append(result)
        with open(os.path.join(episode_data_dir, f"result.pkl"), "wb") as f:
            efficiency.measure("io.save_results", pickle.dump, result, f, category="io")
        if (question_ind+1) % cfg.save_freq == 0:
            with open(os.path.join(output_dir, f"results_{question_ind+1}.pkl"), "wb") as f:
                efficiency.measure("io.save_results", pickle.dump, results_all, f, category="io")
        update_progress(
            force=True,
            phase="question",
            completed_questions=question_position + 1,
            current_question=None,
            current_question_position=question_position + 1,
            current_step=None,
            last_completed_question=question_ind,
        )
        efficiency.end_episode()

    if simulator is not None:
        efficiency.measure("simulator.close", simulator.close, category="simulator")
        simulator = None
    # Save all data again
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        efficiency.measure("io.save_results", pickle.dump, results_all, f, category="io")
    C_avg, C_star_avg, E_path, d_T_avg = score(results_all)
    logging.info(f"\nC_avg: {C_avg}\nC_star_avg: {C_star_avg}\nE_path: {E_path}\nd_T_avg: {d_T_avg}")
    update_progress(
        force=True,
        status="completed",
        phase="completed",
        completed_questions=len(question_indices),
        current_question=None,
        current_question_position=len(question_indices),
        current_step=None,
    )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", "--cfg-file",
        help="cfg file path", default="", type=str
    )
    parser.add_argument(
        "--indices-file",
        help="JSON list of original question indices to process",
        default=None, type=str,
    )
    parser.add_argument(
        "--output-dir",
        help="override cfg.output_dir (used by parallel workers)",
        default=None, type=str,
    )
    parser.add_argument(
        "--progress-file",
        help="JSON progress file written by this worker",
        default=None, type=str,
    )
    parser.add_argument(
        "--append-trace",
        help="append Jev decisions when resuming an existing shard",
        action="store_true",
    )
    parser.add_argument("--efficiency", action="store_true", help="enable efficiency traces and reports")
    parser.add_argument("--efficiency-mode", choices=["light", "detailed"], default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    apply_efficiency_cli(cfg, enabled=args.efficiency, mode=args.efficiency_mode)
    OmegaConf.resolve(cfg)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    question_indices = None
    if args.indices_file is not None:
        with open(args.indices_file, "r", encoding="utf-8") as handle:
            question_indices = json.load(handle)
        if not isinstance(question_indices, list):
            raise ValueError("--indices-file must contain a JSON list")

    # Set up logging
    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logging_path = os.path.join(cfg.output_dir, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    main(
        cfg,
        question_indices=question_indices,
        progress_path=args.progress_file,
        append_trace=args.append_trace,
    )
