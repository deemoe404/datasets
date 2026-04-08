# datasets

本仓库的作用是整理、缓存、测试和使用风电数据集。目标时空序列风功率任务：6H ahead + 24H look back。使用 `./.conda` 管理数据集处理部分的依赖。

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

Kelmarsh/Penmanshiel 的 Greenbyte 连续 CSV 不是天然一条时间戳一行，同一时间戳会出现很多条记录，必须先聚合整理
Hill of Towie 的月表在月边界有重复行，需要先审计再去重
默认模型就绪层是 farm-synchronous 长表和 farm-granularity 窗口；单机 `turbine` 语义只作为显式兼容模式保留
`sdwpf_kddcup` 的源文件只提供 `Day + Tmstamp`；仓库统一按 `Day 1 = 2020-05-01` 恢复工程用 `timestamp`
`sdwpf_kddcup` 如果 manifest 时间语义审计不通过，则只允许保留 `manifest/silver`，禁止继续构建 `gold/task`

## 目录定义

- 数据集总根目录通过项目根目录下的 `wind_datasets.local.toml` 配置；源数据目录内都附有尽量多的官方支持文件（例如论文 PDF），并且数据集文件夹保持只读。
- 临时目录为本目录下的 `./cache`。这个目录应该可以随时被删除，支持重建。
- 数据集处理相关的代码放入本目录下的 `./src`。
- 数据集相关的测试代码放入本目录下的 `./test`。
- 实验放入本目录下的 `./experiment`。每个实验都应该拥有自己的子文件夹，并且子文件夹内独立管理属于该实验的 conda 环境，避免污染数据集处理环境，同时避免实验之间互相污染。
