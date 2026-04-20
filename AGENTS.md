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
