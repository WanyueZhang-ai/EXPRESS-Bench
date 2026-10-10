# EXPRESS 实验结果分析报告设计

## 目标

生成一个无需本地服务器、双击即可打开的自包含 HTML 文件，系统分析 baseline、JEV fair/8GPU、JEV oracle 的当前实验结果，并保留未完成分片的完成度与不确定性。

## 结构

1. **总览仪表盘**：数据覆盖、成功率、C、C*、E_path、d_T、JEV 路由行为和 GPU 完成度。
2. **论文式分析**：复现论文 Table 2，解释指标计算口径，给出当前结果与论文基线的可比性、完成度和限制。
3. **轨迹探索器**：按 question type、GPU、状态、动作、fallback、轨迹长度筛选，并展示样本级摘要和轨迹缩略图/文件链接。

## 数据口径

- baseline 使用 `/ssd/wyzhang/EXPRESS-Bench/experiment_result` 的每题结果。
- JEV fair 使用 `experiment_result_jev`，8GPU 分片使用 `experiment_result_jev_8gpu`；Oracle 使用 `experiment_result_jev_oracle`。
- 未完成样本不计入性能均值；报告同时显示已完成样本观测值、目标总量和分片完成率。
- C/C*/E_path/d_T 使用仓库 `evaluation.py` 的定义，并在 HTML 中明确标注路径样本与目标距离样本的分母不同。
- JEV 决策日志使用 JSONL；pickle 仅通过不执行代码的读取器提取字段。

## 可视化

- 指标卡片和模型对比条形图。
- question type 组成与分类型性能表。
- 8GPU 分片堆叠进度条、worker 热力表、吞吐/延迟/错误原因统计。
- 轨迹长度、goal distance、EAC 分布和动作流。
- 论文原始表格图片以内嵌 base64 方式插入，保证 HTML 可独立打开。

## 风险和限制

- JEV 目录可能没有最终合并 `results.pkl`，因此按每题 `result.pkl`、JSONL 和分片元数据汇总。
- 如果某个 JEV 题目只有决策日志而没有完整环境结果，则只纳入路由/轨迹诊断，不伪造 C 类指标。
- Oracle 是 privileged upper bound，应与 fair 分开解释，不直接当作公平提升。
