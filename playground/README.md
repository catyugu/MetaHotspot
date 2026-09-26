# Playground

探索性实验区：一个目录对应一条调研线，结论写在目录自己的 `README.md` 或
`records/` 里。生成的 JSON / VTU / 日志 / 图片不进入版本控制。

| 目录                     | 内容                                              | 状态               |
| ------------------------ | ------------------------------------------------- | ------------------ |
| `adaptive_bci_sampling/` | 确定性 BCI 参数采样与连续参数盒证书               | 当前方法，见目录 README |
| `bci_rom_testcase1/`     | FloTHERM BCI ROM Case 1：几何、参考矩阵、仿射模型 | 长期保留           |
| `simple_erom_case1/`     | 简单 EROM 连接案例与 FloTHERM 参考                | 长期保留           |
| `connecting_mor_models/` | 多个 BCI ROM 的连接（common patch、contour 单元） | 进行中             |

## 已删除的过时实验脚本

判定标准有两条：其方法论已经被后续实验取代；并且结论已经落到记录里（必要时连可
重建的实验定义一起给出）。只满足其中一条的脚本不删。

### `adaptive_bci_sampling/` 的采样线

`adaptive_zolotarev.py`、`certified_greedy.py`、`compare_extractors.py`、
`compare_sampling.py`、`compare_transient.py`、`compare_zolotarev.py`、
`probe_conditional_zolotarev.py`、`probe_dual_space.py`、
`probe_tangent_corners.py`、`probe_tangent_greedy.py`、`rational_surrogate.py`、
`transient_dual_certificate.py`、`verify.py` 以及四个旧测试套件，在确定性设计 +
整盒证书落地后被删除。它们对应的负结果、实测数字与复现命令全部保留在
`adaptive_bci_sampling/records/` 和对应的提交信息里。

### `bci_rom_testcase1/experiments/extrapolation.py`

HTC 训练范围之外的外推练习：三个训练区间，测试网格是 `1e-2 .. 1e6` 的 7x7 对数
网格（训练盒两侧各四个数量级）。指标是该参数下四个结温用**最大**温升归一化的
最坏误差，瞬态窗口 100 s。它对应的周报问题一直挂在
`docs/reports/weekly_report_0818.md` 的待办里，本次把它跑完并记录后删除：

```text
训练区间       全阶求解   ROM 阶   最坏稳态结温误差   100 s 内最坏瞬态偏差
1e-2 .. 1e6      108       41        0.0742%            0.0039%
1 .. 1e4         116       42        0.0620%            0.0028%
10 .. 1e3        110       40        0.0700%            0.0043%
```

最坏点都落在测试网格的同一侧角点 `(1e-2, 1)` 附近。结论：把 HTC 用到训练盒外四个
数量级并没有破坏这个模型的精度（三个训练区间都在 `0.075%` 以内），训练盒是
**保证**的边界而不是**有效**的边界。最宽的 `1e-2 .. 1e6` 反而略差于标准区间
`1 .. 1e4`，与 `records/POST_SVD_EFFECT.md` 记录的“加宽采样区间不会单调降低压缩后
误差”一致。瞬态数字不能与瞬态证书相提并论：窗口只有 100 s，远短于该叠层的热时间
常数，因此它衡量的是早期轨迹的绝对偏差，而不是稳态精度。

脚本删除后，实验定义保留在这里，以便重建（输出写在被忽略的 `results/` 下）：

```text
模型   Case1Model(max_xy_cell_mm=2.5, max_z_cell_mm=2.5, duration_s=100, dt_s=5)
       功率 [0.1, 0.2, 0.3, 0.4] W，环境 308.15 K，BDF1 20 步
ROM    build_parametric_basis(tolerance=1e-3, max_order=256, probe_rounds=2,
       seed=20260805)，两个 HTC 组
训练区 1e-2..1e6 / 1..1e4 / 10..1e3
测试   笛卡尔积 [1e-2, 1e-1, 1, 1e2, 1e4, 1e5, 1e6]^2，共 49 点
稳态   max_j |T_rom,j - T_ref,j| / max_j (T_ref,j - T_amb)
瞬态   max_t max_j |T_rom,j(t) - T_ref,j(t)| / (T_ref,j(inf) - T_amb)
```

### `hotspot_example2_reproduction/`

项目早期用 HotSpot 自带的 example 2 做的一次复现练习，含第三方输入
`ev6.flp` / `gcc.ptrace` / `gcc.steady`。两个官方案例（`bci_rom_testcase1` /
`simple_erom_case1`）确立之后，仓库里没有任何脚本或文档引用它，也没有留下未记录
的科学结论，因此整目录删除。
