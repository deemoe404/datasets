# Notes

## 重建数据集缓存

```shell
cd "/Users/sam/Developer/Code/Wind Power Forecasting/datasets"
./scripts/rebuild_cache.sh --clean --include-turbine
./scripts/rebuild_cache.sh --check
```

## 运行 chronos-2 实验

```shell
cd "/Users/sam/Developer/Code/Wind Power Forecasting/datasets/experiment/chronos-2"
./.conda/bin/python run_power_only_full.py
```

## 运行 chronos-2-exogenous 实验

```shell
cd "/Users/sam/Developer/Code/Wind Power Forecasting/datasets/experiment/chronos-2-exogenous"
./.conda/bin/python run_exogenous_full.py
```
