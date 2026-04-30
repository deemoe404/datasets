# datasets

本仓库的作用是整理、缓存、测试和使用风电数据集。目标时空序列风功率任务：6H ahead + 24H look back。使用 `./.conda` 管理数据集处理部分的依赖。

主线实验是experiment/families/world_model_state_space_v1，其他都是辅助实验（例如测试、baseline等）。
如果对主线实验有新增模块（例如新增了一个版本的输出head），请如实记录在plan/wind_farm_design_revised_v3_input_reorg.tex里面
每次做完主线实验之后都把发现、收获给记录在experiment/families/world_model_state_space_v1/work_log.txt里面，每次新增的时候都带上日期。如果太长可以记录在别的文件里面，在experiment/families/world_model_state_space_v1/work_log.txt里添加一个文件地址指向过去，experiment/families/world_model_state_space_v1/work_log.txt里只保留摘要也可以。
开始新的任务之前也应该查看experiment/families/world_model_state_space_v1/work_log.txt，看看之前做到哪里了。

## 目录定义

- 确保先阅读 `./README.md` 了解数据集和仓库情况。
- 确保先阅读实验相关的内容 `./experiment/README.md` 了解实验情况。
- 数据集总根目录通过项目根目录下的 `wind_datasets.local.toml` 配置；源数据目录内都附有尽量多的官方支持文件（例如论文 PDF），并且数据集文件夹保持只读。
- 临时目录为本目录下的 `./cache`。这个目录应该可以随时被删除，支持重建。
- 数据集处理相关的代码放入本目录下的 `./src`。
- 数据集相关的测试代码放入本目录下的 `./test`。
- 实验放入本目录下的 `./experiment`。每个实验都应该拥有自己的子文件夹，并且子文件夹内独立管理属于该实验的 conda 环境，避免污染数据集处理环境，同时避免实验之间互相污染。

## 数据集处理原则

## 实验原则

- 不对数据集做任何改动：不允许丢弃/新增 key、插值等一切操作。如果缺少适合实验的数据集，立刻停止汇报。

## 实验产物同步原则

- Git 里应该保留足够让其他机器复现分析的轻量证据：实验代码、诊断/发布脚本、`experiment/artifacts/runs/<family_id>/<timestamp>/manifest.json`、`experiment/artifacts/published/<family_id>/` 下的 summary/seed/final-table/manifest/bootstrap/status CSV/JSON，以及 `figures/` 下日期标记的报告图和生成这些图的 `tools/` 脚本。
- `experiment/artifacts/scratch/**` 是可重建的运行工作区，默认不进 Git；checkpoint、shard `.npy/.npz`、中间日志、失败重试碎片、family-local `.work/` 也不要进普通 Git。需要跨机器恢复训练或 shard eval 时，使用 SMB/DVC/Git LFS 等外部 artifact 通道，而不是污染 Git history。
- published 目录里的 per-origin comparison 明细可以用于联合分析，但 raw CSV 往往很大。提交时应保留同名 `.csv.gz`，raw `*origin-comparison.csv` / `*origin-errors-comparison.csv` 只留本机并由 `.gitignore` 隔离。后续读取脚本应同时支持 `.csv` 与 `.csv.gz`，优先读 raw CSV，缺失时读同名 gzip。
- 多 seed 实验的主表/图表应同步 mean/std/seed_count；图里通常用 mean 作为柱高或点位，用 std、standard error 或 bootstrap CI 做误差线，并在 caption/table 里写明 `n` 和误差线定义。
- 提交 long-run/baseline 结果时优先拆成三类 commit：`code + tests + runners`、`published results + run manifests + work_log`、`reports + figures + plotting tools`。不要用 `git add .`；先检查 `git status --short --untracked-files=all`，确认 pycache、scratch、大 raw CSV、checkpoint 没被纳入。
- 提交前至少运行相关单测和轻量检查，例如 `./.conda/bin/python -m pytest -q test/test_world_model_official_baselines_v2.py`、报告脚本的 `py_compile`、以及 `git diff --check`。如果因为时间或环境无法运行，要在最终汇报里明确说明。
