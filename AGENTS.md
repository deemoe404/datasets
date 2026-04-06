# datasets

本仓库的作用是整理、缓存、测试和使用风电数据集。目标时空序列风功率任务：6H ahead + 24H look back。使用 `./.conda` 管理数据集处理部分的依赖。

仅使用以下四个数据集。

| 数据集        | 机组数量 | 机组位置信息      | 时间分辨率 | 机组级风功率数据                               | 机组信息 |
| ------------- | -------: | ----------------- | ---------: | ---------------------------------------------: | -------: |
| Kelmarsh      |        6 | 绝对经纬度 + 海拔 |     10 min | Greenbyte 10 分钟 SCADA 功率相关通道           |  2050 kW |
| Penmanshiel   |       14 | 绝对经纬度 + 海拔 |     10 min | Greenbyte 10 分钟 SCADA 功率相关通道           |  2050 kW |
| Hill of Towie |       21 | 绝对经纬度        |     10 min | 官方 10 分钟 turbine/grid 统计表中的功率列     |  2300 kW |
| SDWPF_full    |      134 |   相对坐标 + 海拔 |     10 min | 官方主表 `Patv`                                |  1500 kW |

Kelmarsh/Penmanshiel 的 Greenbyte 连续 CSV 不是天然一条时间戳一行，同一时间戳会出现很多条记录，必须先聚合整理
Hill of Towie 的月表在月边界有重复行，需要先审计再去重
默认模型就绪层是 farm-synchronous 长表和 farm-granularity 窗口；单机 `turbine` 语义只作为显式兼容模式保留
`SDWPF_full` 如果 manifest 时间语义审计不通过，则只允许保留 `manifest/silver`，禁止继续构建 `gold/task`

## 目录定义

- 数据集的位置在 `/Users/sam/Developer/Datasets/Wind Power Forecasting`，数据集文件夹内都附有尽量多的官方支持文件（例如论文 PDF）。数据集文件夹保持只读。
- 临时目录为本目录下的 `./cache`。这个目录应该可以随时被删除，支持重建。
- 数据集处理相关的代码放入本目录下的 `./src`。
- 数据集相关的测试代码放入本目录下的 `./test`。
- 实验放入本目录下的 `./experiment`。每个实验都应该拥有自己的子文件夹，并且子文件夹内独立管理属于该实验的 conda 环境，避免污染数据集处理环境，同时避免实验之间互相污染。
