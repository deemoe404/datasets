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

## 当前保留字段概览

下表按 `src/wind_datasets/data/source_column_policy/*.csv` 汇总当前会被保留到数据集处理链路中的原始字段，仅统计 `keep` 与 `keep+mask`。  
这张表回答的是“原始 source key 哪些还会进入后续处理”；如果你要看某个 `feature_protocol_id` 最终进入实验的数据列，还需要看 task bundle 的 `task_context.json`。

### Kelmarsh

| 源表/文件 | 当前保留字段 |
| -------- | ------------ |
| `Turbine_Data` | `Date and time`, `Power (kW)`, `Power, Minimum (kW)`, `Power, Maximum (kW)`, `Power, Standard deviation (kW)`, `Wind speed (m/s)`, `Wind direction (°)`, `Nacelle position (°)`, `Generator RPM (RPM)`, `Rotor speed (RPM)`, `Ambient temperature (converter) (°C)`, `Nacelle temperature (°C)`, `Power factor (cosphi)`, `Reactive power (kvar)`, `Blade angle (pitch position) A (°)`, `Blade angle (pitch position) B (°)`, `Blade angle (pitch position) C (°)` |
| `Device_Data_PMU` | `Date and time`, `GMS Current (A)`, `GMS Power (kW)`, `GMS Reactive power (kvar)` |
| `Device_Data_Grid_Meter` | `Date and time` |
| `Status` | `Timestamp start`, `Timestamp end`, `Status`, `Code`, `Service contract category`, `IEC category` |
| `WT_static` | `Title`, `Identity`, `Latitude`, `Longitude`, `Elevation (m)`, `Rated power (kW)`, `Hub Height (m)`, `Rotor Diameter (m)`, `Manufacturer`, `Model`, `Country`, `Commercial Operations Date` |

### Penmanshiel

| 源表/文件 | 当前保留字段 |
| -------- | ------------ |
| `Turbine_Data` | `Date and time`, `Power (kW)`, `Power, Minimum (kW)`, `Power, Maximum (kW)`, `Power, Standard deviation (kW)`, `Wind speed (m/s)`, `Wind direction (°)`, `Nacelle position (°)`, `Generator RPM (RPM)`, `Rotor speed (RPM)`, `Ambient temperature (converter) (°C)`, `Nacelle temperature (°C)`, `Power factor (cosphi)`, `Reactive power (kvar)`, `Blade angle (pitch position) A (°)`, `Blade angle (pitch position) B (°)`, `Blade angle (pitch position) C (°)` |
| `Device_Data_PMU` | `Date and time`, `GMS Current (A)`, `GMS Power (kW)`, `GMS Reactive power (kvar)` |
| `Device_Data_Grid_Meter` | `Date and time` |
| `Status` | `Timestamp start`, `Timestamp end`, `Status`, `Code`, `Service contract category`, `IEC category` |
| `WT_static` | `Title`, `Identity`, `Latitude`, `Longitude`, `Elevation (m)`, `Rated power (kW)`, `Hub Height (m)`, `Rotor Diameter (m)`, `Manufacturer`, `Model`, `Country`, `Commercial Operations Date` |

### Hill of Towie

| 源表/文件 | 当前保留字段 |
| -------- | ------------ |
| `Hill_of_Towie_turbine_metadata` | `Turbine Name`, `Station ID`, `Latitude`, `Longitude`, `Manufacturer`, `Model`, `Rated power (kW)`, `Hub Height (m)`, `Rotor Diameter (m)`, `Country`, `Commercial Operations Date` |
| `tblSCTurbine` | `TimeStamp`, `StationId`, `wtc_GenRpm_mean`, `wtc_MainSRpm_mean`, `wtc_PitchRef_BladeA_mean`, `wtc_PitchRef_BladeB_mean`, `wtc_PitchRef_BladeC_mean`, `wtc_PriAnemo_mean`, `wtc_SecAnemo_mean`, `wtc_TwrHumid_mean`, `wtc_YawPos_mean` |
| `tblSCTurGrid` | `TimeStamp`, `StationId`, `wtc_ActPower_mean`, `wtc_ActPower_min`, `wtc_ActPower_max`, `wtc_ActPower_stddev`, `wtc_ActPower_endvalue` |
| `tblSCTurFlag` | `TimeStamp`, `StationId` |
| `tblGrid` | `TimeStamp`, `ActivePower`, `ReActivePower`, `PowerFactor` |
| `tblGridScientific` | `TimeStamp` |
| `tblSCTurCount` | `TimeStamp`, `StationId` |
| `tblSCTurDigiIn` | `TimeStamp`, `StationId` |
| `tblSCTurDigiOut` | `TimeStamp`, `StationId` |
| `tblSCTurIntern` | `TimeStamp`, `StationId` |
| `tblSCTurPress` | `TimeStamp`, `StationId`, `wtc_HydPress_mean` |
| `tblSCTurTemp` | `TimeStamp`, `StationId`, `wtc_AmbieTmp_mean`, `wtc_NacelTmp_mean`, `wtc_GeOilTmp_mean` |
| `tblAlarmLog` | `TimeOn`, `TimeOff`, `StationNr`, `Alarmcode` |
| `Hill_of_Towie_AeroUp_install_dates` | `Turbine`, `First date of AeroUp works`, `Last date of AeroUp works` |
| `ShutdownDuration` | `TimeStamp_StartFormat`, `TurbineName`, `ShutdownDuration` |

### sdwpf_kddcup

| 源表/文件 | 当前保留字段 |
| -------- | ------------ |
| `sdwpf_main` | `TurbID`, `Day`, `Tmstamp`, `Patv`, `Wspd`, `Wdir (keep+mask)`, `Etmp`, `Itmp`, `Ndir (keep+mask)`, `Pab1 (keep+mask)`, `Pab2 (keep+mask)`, `Pab3 (keep+mask)`, `Prtv` |
| `sdwpf_location` | `TurbID`, `x`, `y` |

如果要看最精确的规则，请直接查看：

- `src/wind_datasets/data/source_column_policy/kelmarsh.csv`
- `src/wind_datasets/data/source_column_policy/penmanshiel.csv`
- `src/wind_datasets/data/source_column_policy/hill_of_towie.csv`
- `src/wind_datasets/data/source_column_policy/sdwpf_kddcup.csv`

切分 train/val/test 时的注意事项（时间顺序按照 70/10/20 比例）：

- Hill of Towie 中，`wtc_ActualWindDirection_mean`/`days_since_tuneup_effective_start`/`days_since_tuneup_deployment_end` 在 train/val/test 上大量缺失。
- Hill of Towie 中，`tuneup_post_effective`/`tuneup_in_deployment_window` 在 train/val/test 上大量为 0。
- Hill of Towie 中，`farm_grid__frequency` 在 train 上波动很小，且会出现 0 值（可能是缺失哨兵）。
- Hill of Towie 中，`wtc_GridFreq_mean` 在时间轴上会出现 0 值（可能是缺失哨兵）。
- Hill of Towie 中，`days_since_aeroup_start`/`days_since_aeroup_end` 在 train 上几乎全部缺失；到 val/test 时缺失显著减少，但分布范围也随之明显变化。

- Kelmarsh 中，`farm_pmu__gms_power_setpoint_kw`/`farm_pmu__gms_grid_frequency_hz`/`farm_pmu__gms_voltage_v` 在 train 上几乎为常数，test 上会出现 0 值。
- Kelmarsh 中，`Grid frequency (Hz)` 在时间轴上会出现 0 值（可能是缺失哨兵）。

- Penmanshiel 中，`farm_pmu__gms_grid_frequency_hz`/`farm_pmu__gms_voltage_v` 在 train 上波动很小，test 上会出现 0 值。
- Penmanshiel 中，`farm_pmu__gms_power_setpoint_kw` 在 train 上接近常数。
- Penmanshiel 中，`Grid frequency (Hz)` 在时间轴上会出现 0 值（可能是缺失哨兵）。
- Penmanshiel 中，`farm_pmu__gms_power_kw`/`farm_pmu__gms_reactive_power_kvar`/`farm_pmu__gms_current_a` 在 train 上大量缺失，val/test 上缺失很少。
- Penmanshiel 中，`Blade angle (pitch position) A (°)`/`Blade angle (pitch position) B (°)`/`Blade angle (pitch position) C (°)` 在 train 上大量缺失，val/test 上缺失显著更少。
- Penmanshiel 中，`farm_evt_active_count`/`farm_evt_total_overlap_seconds` 在 train 与 val/test 上分布差异很大；train 中 0 值更多，val/test 更集中于较高取值。

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
