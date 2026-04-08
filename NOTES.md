# Notes

## 本机数据集根目录配置

项目根目录下创建或更新 `wind_datasets.local.toml`:

```toml
[paths]
source_data_root = "/home/sam/Documents/datasets/Wind Power Forecasting"
```

## 建立项目根环境

```shell
./create_env.sh
```

## 重建数据集缓存

```shell
./scripts/rebuild_cache.sh --clean --include-turbine
./scripts/rebuild_cache.sh --check
```

## 运行 chronos-2 实验

```shell
cd experiment/chronos-2
./create_env.sh
./.conda/bin/python run_power_only_full.py
```

## 运行 chronos-2-exogenous 实验

```shell
cd experiment/chronos-2-exogenous
./create_env.sh
./.conda/bin/python run_exogenous_full.py
```
