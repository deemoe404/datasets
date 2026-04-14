## 维护者速记

`README.md` 和 `experiment/README.md` 是当前仓库的正式说明；本文件只保留维护时常用的最短路径。

## 环境准备

项目根目录下创建或更新 `wind_datasets.local.toml`:

```toml
[paths]
source_data_root = "/path/to/Wind Power Forecasting"
```

建立项目根环境：

```shell
./scripts/create_env.sh
```

这个环境会安装仓库本体和测试依赖，可直接运行：

```shell
./.conda/bin/python -m pytest
```

## 缓存重建

常用命令：

```shell
./scripts/rebuild_cache.sh --clean
./scripts/rebuild_cache.sh --check
./scripts/rebuild_cache.sh hill_of_towie
```

可视化 notebook 重新生成：

```shell
./.conda/bin/python ./scripts/generate_visualization_notebooks.py
```

## 实验运行

当前 active tree 中只保留 `agcrn` family：

```shell
cd experiment/families/agcrn
./create_env.sh
./.conda/bin/python run_agcrn.py
```

正式运行默认会把结果写到 `experiment/artifacts/published/<family_id>/latest.csv`，并在
`experiment/artifacts/runs/<family_id>/<timestamp>/manifest.json` 记录本次调用的参数、
输出和 git 状态。正式 rerun 建议加上 `--run-label` 并在 clean commit
上执行。
