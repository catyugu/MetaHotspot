# 2026-10-08/09 研究对照归档

具体实验数据及结果报告只保存在本地，不随 Git 分发；`records/` 和归档中的同名目录受忽略规则保护。仓库保留代码、证明、复现方法、文献核对及统一失败档案。历史记录中的提交号在历史压缩后可能不再可用。

这些方案退出默认实验入口。当前目标、代码与结果见[当前 README](../../README.md)。
证明随 Git 保留，数值证据与成本记录只在本地保留；归档不否定局部成立的定理，不将局部通过
改称全域成功。历史记录的“当前”“最新”按原写作时状态理解。

| 路线 | 主要入口与证明 | 退出主线的原因 / 保留用途 |
|---|---|---|
| 标量残差重构 | `residual_reconstruction.py`、[证明](RESIDUAL_RECONSTRUCTION_PROOF.md) | 作为对照保留；输入无关余项过松，共享数值函数已独立到当前 numerics.py |
| 参数多项式 / 速度轨迹 | `polynomial_trajectory.py`、`velocity_defect.py`、[证明](POLYNOMIAL_TRAJECTORY_PROOF.md) | 局部连续盒有通过结果；不能以提高阶数解决当前整个域尾项 |
| 静态提升 | `continuous_lift.py`、[证明](CONTINUOUS_LIFT_PROOF.md) | 全系数提升过松；稳态输入误差不能代替瞬态的严格反例仍保留 |
| 共同误差空间 / 单衰减 Gramian | `common_error_reachability.py`、`common_residual_gramian.py`、[证明](COMMON_RESIDUAL_GRAMIAN_PROOF.md) | 指定标量化证书族失败；采样低秩仅诊断，不能作连续域证书 |
| 辅助误差系统 | `error_dynamics_gate.py`、`continuous_auxiliary.py`、[证明](AUXILIARY_ENERGY_PROOF.md) | 有局部连续域双验收；原始整个域与成本优势未闭合 |
| Robin 输入反馈结构 | `boundary_feedback_gate.py`、[证明](BOUNDARY_FEEDBACK_PROOF.md) | 输入驱动候选可用，但完整能量球尾项过松；候选公共函数已独立到 port_basis.py |
| 耦合包络 / 投影残差尾项 | `coupled_feedback_enclosure.py`、`coupled_galerkin_tail.py`、[证明](COUPLED_FEEDBACK_PROOF.md) | 小模型原域稳态通过；正式原域稳态和全时间组合仍未通过 |

阶段报告及整理后的数据仅在本地本目录 `records/` 保留。
[MANIFEST.json](MANIFEST.json)给出清理前后的逐文件路径、来源提交及临时输出清理记录。
失败判决只维护[统一档案](../../records/FAILURE_ARCHIVE.md)，不另建重复综述。

保留原来的平面 Python 模块依赖。归档脚本优先导入同目录历史实现，再
访问当前共享 Case1 装配；当前入口不会反向导入本目录。复现历史结果时，
将报告中的旧顶层脚本路径改为此归档前缀，例如从仓库根目录运行：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python playground/adaptive_bci_sampling/archive/research20261008_09/continuous_lift.py --help
python playground/adaptive_bci_sampling/archive/research20261008_09/coupled_feedback_enclosure.py --help
```

历史所需候选 NPZ / pickle 未随 Git 交付，应按对应阶段报告重建并计入成本。
归档数据不依赖本地临时结果。没有新建单元测试，也不恢复旧测试目录。
