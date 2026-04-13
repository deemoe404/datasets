# README

本文件是数据集侧的正式总览；维护速记见 `./NOTE.md`，实验结构见 `./experiment/README.md`。

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
| `Turbine_Data` | `Date and time`, `Power (kW)`, `Power, Minimum (kW)`, `Power, Maximum (kW)`, `Power, Standard deviation (kW)`, `Wind speed (m/s)`, `Wind direction (°)`, `Nacelle position (°)`, `Generator RPM (RPM)`, `Rotor speed (RPM)`, `Ambient temperature (converter) (°C)`, `Nacelle ambient temperature (°C)`, `Nacelle temperature (°C)`, `Power factor (cosphi)`, `Reactive power (kvar)`, `Blade angle (pitch position) A (°)`, `Blade angle (pitch position) B (°)`, `Blade angle (pitch position) C (°)` |
| `Device_Data_PMU` | `Date and time`, `GMS Current (A)`, `GMS Power (kW)`, `GMS Reactive power (kvar)` |
| `Device_Data_Grid_Meter` | `Date and time` |
| `Status` | `Timestamp start`, `Timestamp end`, `Status`, `Code`, `Service contract category`, `IEC category` |
| `WT_static` | `Title`, `Identity`, `Latitude`, `Longitude`, `Elevation (m)`, `Rated power (kW)`, `Hub Height (m)`, `Rotor Diameter (m)`, `Manufacturer`, `Model`, `Country`, `Commercial Operations Date` |

### Penmanshiel

| 源表/文件 | 当前保留字段 |
| -------- | ------------ |
| `Turbine_Data` | `Date and time`, `Power (kW)`, `Power, Minimum (kW)`, `Power, Maximum (kW)`, `Power, Standard deviation (kW)`, `Wind speed (m/s)`, `Wind direction (°)`, `Nacelle position (°)`, `Generator RPM (RPM)`, `Rotor speed (RPM)`, `Ambient temperature (converter) (°C)`, `Nacelle ambient temperature (°C)`, `Nacelle temperature (°C)`, `Power factor (cosphi)`, `Reactive power (kvar)`, `Blade angle (pitch position) A (°)`, `Blade angle (pitch position) B (°)`, `Blade angle (pitch position) C (°)` |
| `Device_Data_PMU` | `Date and time`, `GMS Current (A)`, `GMS Power (kW)`, `GMS Reactive power (kvar)` |
| `Device_Data_Grid_Meter` | `Date and time` |
| `Status` | `Timestamp start`, `Timestamp end`, `Status`, `Code`, `Service contract category`, `IEC category` |
| `WT_static` | `Title`, `Identity`, `Latitude`, `Longitude`, `Elevation (m)`, `Rated power (kW)`, `Hub Height (m)`, `Rotor Diameter (m)`, `Manufacturer`, `Model`, `Country`, `Commercial Operations Date` |

### Hill of Towie

| 源表/文件 | 当前保留字段 |
| -------- | ------------ |
| `Hill_of_Towie_turbine_metadata` | `Turbine Name`, `Station ID`, `Latitude`, `Longitude`, `Manufacturer`, `Model`, `Rated power (kW)`, `Hub Height (m)`, `Rotor Diameter (m)`, `Country`, `Commercial Operations Date` |
| `tblSCTurbine` | `TimeStamp`, `StationId`, `wtc_AcWindSp_mean`, `wtc_ActualWindDirection_mean`, `wtc_GenRpm_mean`, `wtc_MainSRpm_mean`, `wtc_PitcPosA_mean`, `wtc_PitcPosB_mean`, `wtc_PitcPosC_mean`, `wtc_PitchRef_BladeA_mean`, `wtc_PitchRef_BladeB_mean`, `wtc_PitchRef_BladeC_mean`, `wtc_PriAnemo_mean`, `wtc_SecAnemo_mean`, `wtc_TwrHumid_mean`, `wtc_YawPos_mean` |
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

## 滑窗式构建 `24H->6H` 窗口

`farm-synchronous` 基于共享 farm timestamp 轴统计；默认 farm task cache 也按同一时间轴聚窗，不受 `series.parquet` 物理行序影响。下表在此基础上再删除所有 `quality_flags != ""` 的目标行：

| 数据集         | 丢弃 flagged 行                | farm 时间点损失           | 可构造 `24H->6H` 窗口 | 窗口保留率 | 最长连续干净段          |
| -------------- | -----------------------------: | ------------------------: | --------------------: | ---------: | ----------------------: |
| Kelmarsh       |    54,948 / 2,839,104 =  1.94% | 15,187 / 473,184 =  3.21% |     422,717 / 473,005 |     89.37% | 14,489 步 ≈ 100.6 天   |
| Penmanshiel    |   685,431 / 6,318,676 = 10.85% | 90,032 / 451,334 = 19.95% |     326,816 / 451,155 |     72.44% | 15,651 步 ≈ 108.7 天   |
| Hill of Towie  |   120,034 / 9,574,005 =  1.25% | 50,570 / 455,905 = 11.09% |     332,356 / 455,726 |     72.93% |  8,629 步 ≈  59.9 天   |
| sdwpf_kddcup   | 1,131,661 / 4,727,520 = 23.94% | 31,860 / 35,280  = 90.31% |           0 /  35,101 |      0.00% |     37 步 ≈   6.2 小时 |
