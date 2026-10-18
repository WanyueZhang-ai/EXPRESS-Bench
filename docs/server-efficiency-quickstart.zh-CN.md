# 服务器全流程：先随机验证 10 道题

分支：`codex/efficiency-analysis`。下文只运行 **10 道题**，使用
`--sample-size 10 --sample-seed 42`，保留原始题号。省略 `--sample-size` 和
`--indices-file` 的原有命令仍会运行全集。

## 1. 下载分支，或应用补丁

推荐下载到新目录，避免影响已有代码：

```bash
git clone --branch codex/efficiency-analysis --single-branch \
  https://github.com/WanyueZhang-ai/EXPRESS-Bench.git EXPRESS-Bench-efficiency
cd EXPRESS-Bench-efficiency
git log -2 --oneline
```

若使用已有克隆，先保存自己的未提交修改，再执行：

```bash
cd /你的路径/EXPRESS-Bench
git fetch origin
git switch codex/efficiency-analysis
git pull --ff-only origin codex/efficiency-analysis
```

**下载分支已经包含全部修改，不需要再打补丁。** 仅当改用本次交付的
`efficiency-analysis.patch` 时，把补丁传到服务器，并从原始提交新建分支应用：

```bash
cd /你的路径/EXPRESS-Bench
git switch -c efficiency-local d37267591abb2d4f63d8c184a4e5c305cefca4cc
git am /补丁所在路径/efficiency-analysis.patch
```

补丁包含效率分析和抽题功能。若冲突，解决后 `git am --continue`，或用
`git am --abort` 退出本次应用。不要在已包含修改的目录重复打补丁。

## 2. 检查环境、数据、模型

效率分析和抽题没有新增第三方依赖。优先激活服务器原本能运行该实验的环境，
始终从仓库根目录执行命令：

```bash
conda activate 你的现有环境名
nvidia-smi
python -c "import sys, torch, habitat_sim, transformers; print(sys.version); print('CUDA:', torch.cuda.is_available()); print('Transformers:', transformers.__version__)"
python -m InternDecision_scripts.run_intern_decision_smoke
```

最后一个命令不加载模型或场景，只检查路由逻辑。当前 Intern checkpoint 要求
Python 3.12+，应采用 checkpoint 自带的 `requirements.txt`，同时安装兼容的
Habitat。原始 Fine-EQA 的 Python 3.9 环境说明不能直接替代这个要求。

检查 `InternDecision_scripts/config_decision_image.yaml`：

| 配置 | 默认位置/要求 |
| --- | --- |
| `dataset_path` | `./data/express-bench.json` |
| `scene_data_path` | `./data`，包含 `episode/`、HM3D 场景和导航文件 |
| `intern_decision.checkpoint` | `/ssd/wyzhang/all_models/Intern-Decision-4B`，含 `inference.py` 和权重 |
| Qwen 回答/EAC 模型 | `/ssd/wyzhang/all_models/Qwen3-VL-8B-Instruct` |
| `vlm.model_id` / `local_model_root` | `vlm_state` 额外使用的 Prismatic 及其模型文件 |

数据和权重不会随 Git 自动下载。新克隆可复用现有数据：若新目录没有 `data/`，
可创建指向现有数据目录的软链接；若已有 `data/`，检查并补齐 `episode/`、`hm3d/`，
不要覆盖已有内容。语义对象条件需要 HM3D 语义资产。

路径不同时修改 YAML，或复制为自己的服务器配置并替换下文 `--cfg-file`。
Qwen 路径通过环境变量覆盖：

```bash
export QWEN3_VL_MODEL_PATH=/你的模型目录/Qwen3-VL-8B-Instruct
```

## 3. 先检查 10 题选择，不加载模型

```bash
python -m InternDecision_scripts.run_intern_decision \
  --cfg-file InternDecision_scripts/config_decision_image.yaml \
  --sample-size 10 --sample-seed 42 \
  --efficiency-mode detailed --dry-run
```

输出末尾 `Question selection` 应包含 `selected_count: 10` 和 10 个原始题号。
此检查仅读取配置及题目 JSON，不验证模型权重、GPU 或具体场景是否可用。
抽样使用独立随机数生成器，不改变模型/仿真器的随机状态。

## 4. 单卡运行 10 题

推荐即使单卡也用并行启动器，以获得日志、进度、自动合并和恢复功能。
`--gpus 0` 是物理 GPU 编号；worker 内映射为 `cuda:0`。

```bash
export PYTHONUNBUFFERED=1
python -u Jev_scripts/run_parallel.py \
  --cfg-file InternDecision_scripts/config_decision_image.yaml \
  --num-gpus 1 --gpus 0 \
  --sample-size 10 --sample-seed 42 \
  --efficiency-mode detailed \
  --output-dir experiment_result/eff_image10_detailed
```

若需断开 SSH 后继续，**用下面的后台命令替代上述命令**，不要重复启动：

```bash
mkdir -p logs
export PYTHONUNBUFFERED=1
nohup python -u Jev_scripts/run_parallel.py \
  --cfg-file InternDecision_scripts/config_decision_image.yaml \
  --num-gpus 1 --gpus 0 \
  --sample-size 10 --sample-seed 42 \
  --efficiency-mode detailed \
  --output-dir experiment_result/eff_image10_detailed \
  > logs/eff_image10_detailed.log 2>&1 &
echo $! > logs/eff_image10_detailed.pid
```

首次运行使用新输出目录。已有任务目录会被拒绝，以免覆盖结果。
10 题减少总评估量，模型权重本身的常驻显存需求不变。

## 5. 实时查看流程

后台运行时查看总进度：

```bash
tail -f logs/eff_image10_detailed.log
```

应看到 `selected 10/数据集总数 questions`，随后是 `questions 0/10`。
加载模型时停留在 `0/10` 正常。详细模型/场景加载、决策和错误日志：

```bash
tail -f experiment_result/eff_image10_detailed/shards/worker_00/worker_stdout.log
```

另开终端查看结构化进度和 GPU：

```bash
watch -n 2 cat experiment_result/eff_image10_detailed/shards/worker_00/progress.json
watch -n 2 nvidia-smi
```

`phase` 显示当前阶段；`current_question` 是原始题号，`current_step` 是当前探索
循环编号。`completed_questions` 是 worker 本次完成数量。恢复运行时，启动器
总进度会加回旧的完成数量，并保持以原来选定的 10 题为分母；worker 原始进度文件
只描述本次剩余部分。`Ctrl+C` 可退出 `tail/watch` 查看。

查看抽题清单及种子：

```bash
cat experiment_result/eff_image10_detailed/selected_indices.json
cat experiment_result/eff_image10_detailed/question_selection.json
```

## 6. 中断后恢复

确认旧任务已停止，重复原命令并加 `--resume`，保持相同配置、抽样参数、GPU 列表
和输出目录：

```bash
python -u Jev_scripts/run_parallel.py \
  --cfg-file InternDecision_scripts/config_decision_image.yaml \
  --num-gpus 1 --gpus 0 \
  --sample-size 10 --sample-seed 42 \
  --efficiency-mode detailed \
  --output-dir experiment_result/eff_image10_detailed --resume
```

已持久化 `result.pkl` 的题目会跳过，未完成题目重跑。遗漏抽样参数、换种子导致
题目集合变化或换 GPU 列表时，恢复检查会拒绝执行。历史失败尝试仍保留，正式统计
优先选择每题最新成功尝试。worker 失败时也会生成部分效率报告。

## 7. 查看效果和效率结果

| 总输出目录下的文件 | 内容 |
| --- | --- |
| `selected_indices.json` / `question_selection.json` | 抽题清单与来源 |
| `manifest.json` / `parallel_summary.json` | 分片安排、合并题数、任务质量指标 |
| `results.pkl` | 合并后的答案、EAC、路径等结果 |
| `intern_decisions.jsonl` | 逐步动作、决策及回退 |
| `efficiency_report.html` | 可离线查看的效率图表和模型调用表 |
| `efficiency_summary.json` / `.csv` | 分组统计、P50/P95、吞吐与 token 覆盖 |
| `efficiency_calls/steps/episodes/runs.jsonl` | 四类原始测量明细 |
| `shards/worker_00/原始题号/` | 单题结果和图像/轨迹等已有输出 |

检查完成数、质量指标和编码可观测性：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('experiment_result/eff_image10_detailed')
selection = json.loads((p / 'selected_indices.json').read_text())
parallel = json.loads((p / 'parallel_summary.json').read_text())
summary = json.loads((p / 'efficiency_summary.json').read_text())
print('选中题数:', len(selection), '合并结果数:', parallel['result_count'])
print('任务质量指标:', parallel['metrics'])
print('Episode 状态:', summary['episodes']['status_counts'])
print('Step 数:', summary['steps']['count'], '模型调用数:', summary['calls']['count'])
print('图像编码状态:', summary['diagnostics']['encoding_status_counts'])
print('Token 覆盖:', summary['calls']['tokens'])
PY
```

正常完成后，选中题数和合并结果数都应为 **10**。还应查看单题 `answer_status`、
模型错误/回退及调用明细中的 `observed_module_paths`。编码字段为 `null` 表示
该 checkpoint/路径无法观察该边界，不代表耗时为零。非生成式决策的
`generated_tokens: 0` 正常，决策字段数单独记录。

重新合并/生成报告：

```bash
python tools/build_efficiency_report.py \
  experiment_result/eff_image10_detailed/shards/worker_* \
  --output-dir experiment_result/eff_image10_detailed --merge-traces
```

在自己的电脑终端下载，再双击 HTML 用浏览器打开，无需在服务器开放网页端口：

```bash
scp 用户名@服务器地址:/仓库绝对路径/experiment_result/eff_image10_detailed/efficiency_report.html .
scp 用户名@服务器地址:/仓库绝对路径/experiment_result/eff_image10_detailed/efficiency_summary.csv .
```

## 8. 比较不同条件，复用同一批 10 题

例如文本状态决策，使用第一轮的清单，改变配置和输出目录：

```bash
python -u Jev_scripts/run_parallel.py \
  --cfg-file InternDecision_scripts/config_vlm_state.yaml \
  --num-gpus 1 --gpus 0 \
  --indices-file experiment_result/eff_image10_detailed/selected_indices.json \
  --efficiency-mode detailed \
  --output-dir experiment_result/eff_text10_detailed
```

同理可换为 `config_decision_objects.yaml`、`config_direct_actions.yaml` 或
`fine_eqa.yaml`。`--indices-file` 与 `--sample-size` 互斥，不同条件使用不同输出目录。
`vlm_state` 还需要 Prismatic 及其依赖。多卡只需将 GPU 参数改为例如
`--num-gpus 2 --gpus 0,1`，10 题会按原始题号分给两个独立 worker；每卡各加载模型。

也可以预先独立抽题，不安装模型库：

```bash
python tools/select_questions.py --dataset data/express-bench.json \
  --sample-size 10 --sample-seed 42 --output-file sample10_seed42.json
```

10 题用于检查流程、模型可用性和计时边界，不能替代完整评测结论。`detailed` 有同步
与探针开销；验证后可在同一清单上另跑 `light`，使用新目录如 `eff_image10_light`。
比较效率时保持测量模式一致。
