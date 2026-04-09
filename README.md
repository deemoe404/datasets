# Notes

## 数据集概览

仅使用以下四个数据集。

| 数据集        | 机组数量 | 机组位置信息      | 时间分辨率 | 时间覆盖                   | 机组信息 |
| ------------- | -------: | ----------------- | ---------: | -------------------------: | -------: |
| Kelmarsh      |        6 | 海拔 + 绝对经纬度 |     10 min | 2016-01-03 - 2024-12-31    |  2050 kW |
| Penmanshiel   |       14 | 海拔 + 绝对经纬度 |     10 min | 2016-06-02 - 2024-12-31*   |  2050 kW |
| Hill of Towie |       21 |        绝对经纬度 |     10 min | 2016-01-01 - 2024-08-31**  |  2300 kW |
| sdwpf_kddcup  |      134 |          相对坐标 |     10 min | 2020-05-01 - 2020-12-31*** |  1500 kW |

*Penmanshiel 无 WT03。WT01/02/04/05/06/07/08/09/10 的最后观测日为 2023-12-31；WT11/12/13/14/15 的最后观测日为 2024-12-31。
**Hill of Towie 官方覆盖到 2024-08-31；因时间戳表示 10 分钟区间结束时刻，原始月表末行会出现 `2024-09-01 00:00:00`。
***`sdwpf_kddcup` 原始文件只给 `Day + Tmstamp`；仓库统一按 `Day 1 = 2020-05-01` 恢复工程用 `timestamp`，该日期锚点是仓库约定，不是原始 CSV 的显式字段。

| 数据集        | 机组级风功率预测目标                                         | 外生变量                                                   |
| ------------- | -----------------------------------------------------------: | ---------------------------------------------------------: |
| Kelmarsh      |              `Power (kW)` 的 min/max/stddev/mean over 10 min |               机组 SCADA 连续量 + 场级 PMU/电表 + 状态事件 |
| Penmanshiel   |              `Power (kW)` 的 min/max/stddev/mean over 10 min |               机组 SCADA 连续量 + 场级 PMU/电表 + 状态事件 |
| Hill of Towie | `wtc_ActPower_*` 的 min/max/stddev/endvalue/mean over 10 min |            机组/场级 SCADA 多表 + 告警事件 + AeroUp/TuneUp |
| sdwpf_kddcup  |                                      `Patv` mean over 10 min | `Wspd/Wdir/Etmp/Itmp/Ndir/Pab1/Pab2/Pab3/Prtv` + 质量 flag |

切分 train/val/test 时的注意事项（时间顺序按照 70/10/20 比例）：

- Hill of Towie 中，`wtc_ActualWindDirection_mean`/`days_since_tuneup_effective_start`/`days_since_tuneup_deployment_end` 在 train/val/test 上大量缺失。
- Hill of Towie 中，`tuneup_post_effective`/`tuneup_in_deployment_window` 在 train/val/test 上大量为 0。
- Hill of Towie 中，`farm_grid__frequency` 在 train 上波动很小，且会出现 0 值（可能是缺失哨兵）。
- Hill of Towie 中，`wtc_GridFreq_mean` 在时间轴上会出现 0 值（可能是缺失哨兵）。

- Kelmarsh 中，`farm_pmu__gms_power_setpoint_kw`/`farm_pmu__gms_grid_frequency_hz`/`farm_pmu__gms_voltage_v` 在 train 上几乎为常数，test 上会出现 0 值。
- Kelmarsh 中，`Grid frequency (Hz)` 在时间轴上会出现 0 值（可能是缺失哨兵）。

- Penmanshiel 中，`farm_pmu__gms_grid_frequency_hz`/`farm_pmu__gms_voltage_v` 在 train 上波动很小，test 上会出现 0 值。
- Penmanshiel 中，`farm_pmu__gms_power_setpoint_kw` 在 train 上接近常数。
- Penmanshiel 中，`Grid frequency (Hz)` 在时间轴上会出现 0 值（可能是缺失哨兵）。

## 滑窗式构建 `24H->6H` 窗口

`farm-synchronous` 基于共享 farm timestamp 轴统计；默认 farm task cache 也按同一时间轴聚窗，不受 `series.parquet` 物理行序影响。下表在此基础上再删除所有 `quality_flags != ""` 的目标行：

| 数据集         | 丢弃 flagged 行                | farm 时间点损失           | 可构造 `24H->6H` 窗口 | 窗口保留率 | 最长连续干净段          |
| -------------- | -----------------------------: | ------------------------: | --------------------: | ---------: | ----------------------: |
| Kelmarsh       |    54,948 / 2,839,104 =  1.94% | 15,187 / 473,184 =  3.21% |     422,717 / 473,005 |     89.37% | 14,489 步 ≈ 100.6 天   |
| Penmanshiel    |   685,431 / 6,318,676 = 10.85% | 90,032 / 451,334 = 19.95% |     326,816 / 451,155 |     72.44% | 15,651 步 ≈ 108.7 天   |
| Hill of Towie  |   120,118 / 9,574,005 =  1.25% | 50,574 / 455,905 = 11.09% |     331,852 / 455,726 |     72.82% |  8,629 步 ≈  59.9 天   |
| Hill of Towie* |   120,263 / 9,574,005 =  1.26% | 50,580 / 455,905 = 11.09% |     330,952 / 455,726 |     72.62% |  8,629 步 ≈  59.9 天   |
| sdwpf_kddcup   | 1,131,661 / 4,727,520 = 23.94% | 31,860 / 35,280  = 90.31% |           0 /  35,101 |      0.00% |     37 步 ≈   6.2 小时 |

`turbine`，删除所有 `quality_flags != ""` 的目标行：

| 数据集         | flagged 行损失                 | 可构造窗口            | 窗口保留率 | 0 窗口机组 |
| -------------- | -----------------------------: | --------------------: | ---------: | ---------: |
| Kelmarsh       |    54,948 / 2,839,104 =  1.94% | 2,641,667 / 2,838,030 |     93.08% |    0 /   6 |
| Penmanshiel    |   165,907 / 5,799,152 =  2.86% | 5,405,463 / 5,796,646 |     93.25% |    0 /  14 |
| Hill of Towie  |   120,118 / 9,574,005 =  1.25% | 9,115,020 / 9,570,246 |     95.24% |    0 /  21 |
| Hill of Towie* |   120,263 / 9,574,005 =  1.26% | 9,089,312 / 9,570,246 |     94.97% |    0 /  21 |
| sdwpf_kddcup   | 1,131,661 / 4,727,520 = 23.94% |   269,772 / 4,703,534 |      5.74% |    0 / 134 |

*Hill of Towie 数据集除了 `quality_flags` 外还有代表 sidecar 文件导致的质量问题的 `feature_quality_flags`。

---

## 环境准备

项目根目录下创建或更新 `wind_datasets.local.toml`:

```toml
[paths]
source_data_root = "/home/sam/Documents/datasets/Wind Power Forecasting"
```

建立项目根环境：

```shell
./scripts/create_env.sh
```

重建数据集缓存：

```shell
./scripts/rebuild_cache.sh --clean --include-turbine
./scripts/rebuild_cache.sh --check
```

## 运行实验

chronos-2 实验：

```shell
cd experiment/chronos-2
./create_env.sh
./.conda/bin/python run_power_only_full.py
```

chronos-2-exogenous 实验：

```shell
cd experiment/chronos-2-exogenous
./create_env.sh
./.conda/bin/python run_exogenous_full.py
```

ltsf-linear 实验：

```shell
cd experiment/ltsf-linear
./create_env.sh
./.conda/bin/python run_ltsf_linear_full.py
```
