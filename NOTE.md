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

重建数据集缓存：

```shell
./scripts/rebuild_cache.sh --clean
./scripts/rebuild_cache.sh --check
```

## 运行实验

chronos-2 实验：

```shell
cd experiment/families/chronos-2
./create_env.sh
./.conda/bin/python run_power_only_full.py
```

chronos-2-exogenous 实验：

```shell
cd experiment/families/chronos-2-exogenous
./create_env.sh
./.conda/bin/python run_exogenous_full.py
```

ltsf-linear 实验：

```shell
cd experiment/families/ltsf-linear
./create_env.sh
./.conda/bin/python run_ltsf_linear_full.py
```

tft pilot 实验：

```shell
cd experiment/families/tft
./create_env.sh
./.conda/bin/python run_tft.py
```

AGCRN pilot 实验：

```shell
cd experiment/families/agcrn
./create_env.sh
./.conda/bin/python run_agcrn.py
```

正式运行默认会把结果写到 `experiment/artifacts/published/<family_id>/latest.csv`，并在
`experiment/artifacts/runs/<family_id>/<timestamp>/manifest.json` 记录本次调用的参数、
输出和 git 状态。正式 rerun 建议加上 `--run-label` 并在 clean commit
上执行。
