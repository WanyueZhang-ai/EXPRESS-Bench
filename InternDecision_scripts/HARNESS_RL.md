# Intern-Decision harness 与在线 RL

## 当前实现范围

这个分支提供能够接入现有评估入口的第一版实现。

- `harness.py` 管理 episode 隔离、版本化状态、有限容量的历史帧与对象记录、证据来源、冲突和检索。
- `joint_policy.py` 在同一请求中输出行动、证据充分性、每个候选的收益和成本。利用 Intern-Decision 原生 decision-token 评分，底层 HF 后端共享一次前向。
- `online_hook.py` 将共享状态接入现有主循环。最终回答使用冻结 Qwen-VL，多帧通过 `gpt.py` 输入。
- `rl/policy.py` 在 checkpoint 原生 `backend.encode()` 之上执行可微分 forward，直接更新 Intern-Decision 参数。
- `rl/trainer.py` 实现监督预训练、on-policy 采样、PPO、GAE、辅助监督和独立特权 critic。
- `rl/environment.py` 提供静态 Habitat 场景训练适配，支持独立分支的一步候选可见性监督。
- `rl/run.py` 提供 collect、sft、ppo 三种入口，以及训练和 held-out 场景交集检查。

默认运行配置为 RGB 图像模式，actor 不读取语义真值。已有 `decision_objects` 模式使用仿真器可见对象标注，属于 oracle-perception 条件，需要单列结果。

## 明确的范围限制

这是原方案的可训练基线。当前包含视觉 episodic memory 和结构化对象记录，尚未实现学习式实例关联、完整 3D scene graph、任务证据集合自动发现、长期语义记忆或可执行的对象重访宏动作。图像模式没有视觉描述抽取器，因此其帧检索主要按时间选择；对象/感知模式额外采用词汇相关性。

现有 Fine-EQA frontier 接口只公开 `fbe_0`，内部规划器执行 frontier 选择。本分支保留其真实语义。默认 RL 使用 `move_forward`、`turn_left`、`turn_right`、`stop`。规划候选模式可使用共享 harness，但尚未接入该 RL 环境。

收益和成本为离散 score 字段，充分性为 noul 字段，行动为统一 choice 字段。它们通过多任务监督共享表示。原生结构化输出接口每次支持有限字段，因此最多 7 个移动候选。多头预测不保证相互一致，必须在验证集上检查。

当前 critic 使用预算、路程、证据覆盖及目标距离的六维特征。目标距离只进入 critic。未来可以加入更丰富的可见状态编码。

默认终局 QA reward 为固定 VLM 答案与 gold 的规范化精确匹配，适合先跑通短答案。开放式回答应替换 `answer_reward(task, state)` 回调为经过验证的语义评审器。未提供证据标注时训练使用答案奖励及动作成本，充分性/收益辅助标签缺失时跳过。此模式不能声称具有 grounding reward。

本地验证结果为 10 项新增测试和 29 项 Intern-Decision 回归测试通过，四种原有路由 smoke case 通过，配置 dry-run 通过。完整旧测试中两个模块因缺少 Prismatic 无法导入。尚未运行真实模型训练或 Habitat 数据集评估。

## 环境要求

使用仓库的 Linux / Habitat 环境和本地模型资产。Intern-Decision checkpoint 需要包含 `inference.py`，且其 HF backend 暴露 `encode(row)`、`model`、`tokenizer`、`processor`。不支持该接口时训练会明确报错。训练环境中安装与 checkpoint 匹配的 PyTorch / Transformers，不能使用仓库旧版依赖直接推断模型兼容性。

PPO 默认全参数训练，需要为 4B 模型权重、梯度、优化器和多帧激活预留显存。建议把回答 VLM 放在另一张 GPU。当前没有 LoRA、分布式训练和断点恢复命令，保存的 training_state 可供后续实现恢复。

## 评估运行

从仓库根目录执行。

```bash
python -m InternDecision_scripts.run_intern_decision \
  -cf InternDecision_scripts/config_harness.yaml \
  --checkpoint /path/to/Intern-Decision-4B \
  --output-dir ./experiment_result_harness
```

配置默认 temperature 为 1.0，与训练一致。旧配置继续使用旧 router。运行日志中新增 shared_state、state_version 和 joint_response，可检查实际输入与所有字段输出。

最终 VLM 使用最多 8 帧。必须对所有比较方法使用相同图像预算、模型与提示词，并计入计算成本。当前主循环中碰撞恢复属于环境执行层，在线训练中碰撞通过 reward 反馈，报告结果时要说明这一差异。

## 训练数据

训练文件是 JSON 列表，每项继承 benchmark 的 question、type、answer、scene_id、start_position、start_rotation 等字段，另加 `split: "train"`。held-out 文件同样为 JSON 列表。程序按 scene_id 拒绝交集，仍需人工确认场景资产别名与数据来源。

可选 `evidence_sets` 表示已经审核的问题有效可见性证据，例如 `[[12, 27], [35]]` 表示观察到 12 和 27，或者观察到 35。值必须是该场景 semantic sensor 的实际整数 ID。

这类标注只适用于已经核实“对象可见性足以支持答案”的问题。颜色可读性、精确计数、否定存在、复杂关系需要更丰富的证据验证器，不能仅用对象出现判定充分性。禁止把完整 benchmark gold 或场景语义图放入 actor state。

种子采集使用一步可见性增量的启发式教师，收益相同时随机探索。它不提供全局最短路径专家保证。

## 三阶段命令

每个命令使用新的 output 目录。seed 文件引用原始观察图像绝对路径，训练时保留对应目录。

```bash
python -m InternDecision_scripts.rl.run --phase collect \
  --checkpoint /path/to/Intern-Decision-4B \
  --tasks train_questions.json --holdout-tasks heldout_questions.json \
  --scene-root ./data --output ./runs/seed --updates 100

python -m InternDecision_scripts.rl.run --phase sft \
  --checkpoint /path/to/Intern-Decision-4B \
  --tasks train_questions.json --holdout-tasks heldout_questions.json \
  --scene-root ./data --seed-data ./runs/seed/seed.jsonl \
  --output ./runs/sft --updates 100

python -m InternDecision_scripts.rl.run --phase ppo \
  --checkpoint ./runs/sft/checkpoint \
  --tasks train_questions.json --holdout-tasks heldout_questions.json \
  --scene-root ./data --output ./runs/ppo --updates 100 \
  --device cuda:0 --answer-device cuda:1
```

collect 和 PPO 会调用本地 Qwen-VL。通过 `QWEN3_VL_MODEL_PATH` 指定答案模型位置。PPO 每次更新采集完整 episode，旧 rollout 不跨更新重复使用。有限预算耗尽作为任务终态，GAE 实现另支持外部截断时的 bootstrap。反事实分支只生成辅助监督，不进入 on-policy action 样本。

静态环境分支仅恢复 agent pose，适用于没有物体交互和随机动力学的该环境。加入动态物体或随机控制器后，需要保存并恢复完整 simulator/RNG/controller 状态。

模型输出保存到 `checkpoint/`，critic 与优化器保存到 `training_state.pt`，训练指标保存到 `metrics.jsonl`。

## 验证

```bash
python -m unittest discover -s tests -p 'test_harness_rl.py' -v
python -m unittest discover -s tests -p 'test_intern*.py' -v
python -m InternDecision_scripts.run_intern_decision --smoke
```

测试覆盖记忆隔离、幂等性、容量、原始帧检索、特权字段拒绝、单次联合请求、最终 VLM 多帧输入、原生 token adapter 梯度、PPO 参数更新、GAE 和反事实恢复。小型模型测试用于验证计算逻辑，真实 Intern-Decision / Habitat / HM3D 的集成运行需要在具备模型与数据的机器上完成。
