# EXPRESS 实验结果分析 HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 EXPRESS-Bench 的 baseline、JEV fair/8GPU 和 Oracle 文件生成一份自包含、可筛选的中文 HTML 分析报告。

**Architecture:** 用一个 Python 汇总脚本读取 JSON/JSONL、分片 progress 和 pickle 中的标量字段，生成 `express_results_report.html`。HTML 内嵌 JSON 数据、CSS、原始论文表格图片和原生 JavaScript；浏览器端完成筛选、表格排序、SVG/CSS 图表与轨迹详情展示。

**Tech Stack:** Python 3 标准库；HTML/CSS/Vanilla JavaScript；pickletools 安全解析；base64 图片嵌入。

---

### Task 1: 安全数据读取与指标汇总

**Files:**
- Create: `tools/build_express_report.py`
- Read: `evaluation.py`, `data/express-bench.json`, `experiment_result*/`

- [ ] 实现不执行 pickle 全局对象的标量字段解析器。
- [ ] 汇总 baseline 每题的 question、type、EAC、path_len、geodesic_distance、goal_dis、cnt_step 和轨迹坐标摘要。
- [ ] 汇总 JEV fair/oracle 的每题决策、fallback、action、latency、token、分片进度。
- [ ] 计算论文指标、分母、缺失值、完成度和 question-type 分层表。

### Task 2: 自包含 HTML 报告

**Files:**
- Create: `express_results_report.html`
- Modify: `tools/build_express_report.py`

- [ ] 生成总览、论文式分析、轨迹探索三个导航页。
- [ ] 内嵌所有 JSON 数据和用户提供的 Table 2 图片。
- [ ] 加入筛选器、排序、指标切换、GPU 分片进度、JEV 日志统计和样本详情。
- [ ] 清晰区分 observed、incomplete、oracle、not-comparable。

### Task 3: 验证与交付

**Files:**
- Create: `report_validation.json`

- [ ] 运行脚本并验证 HTML 为完整文档、JSON 可解析、指标计数与源文件数量一致。
- [ ] 检查无外部 CDN、无绝对路径依赖，双击打开仍能渲染。
- [ ] 输出报告文件路径、统计口径和当前实验完成度。
