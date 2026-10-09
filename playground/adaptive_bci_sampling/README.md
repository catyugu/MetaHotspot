# 连续 HTC 域的全场误差认证研究

具体实验数据及结果报告只保存在本地，不随 Git 分发；`records/` 和归档中的同名目录受忽略规则保护。仓库保留代码、证明、复现方法、文献核对及统一失败档案。历史记录中的提交号在历史压缩后可能不再可用。

最新阶段是共同输入误差空间、正终端尾项与正式规模判决（本地记录：`records/POSITIVE_REACHABILITY_20261009.md`），
[完整推导](POSITIVE_SOURCE_TAIL_PROOF.md)包含连续盒、全时间参数尾项的
Poisson/Jensen 归一化，以及实际 Galerkin 空间遗漏的模态可达锥充分界。
参数尾项在约 ±10% 的子域得到紧界；全盒慢模、空间留出方向及符号损失
仍使新构造失败。**没有新的全域双认证提取器，也没有成本优势或新颖性结论。**
各失败原型只作判决入口，不替换已有固定 285 阶候选或生产提取器。

上一阶段是输出加权动态尾项重构（本地记录：`records/DYNAMIC_OUTPUT_20261009.md`），
[完整推导](DYNAMIC_OUTPUT_TAIL_PROOF.md)与
数值证据及成本（本地记录：`records/DYNAMIC_OUTPUT_DATA_20261009.json`）配套保存。

**尚未得到原始整个 HTC 域、全部时间的实用双认证提取算法。**
47,085 / 122,400 自由度固定 285 阶基底在三个参数点、三个时刻的独立
阶跃参考核查中通过。连续子域的一阶输出包络保持了真实输入耦合，
但高阶诱导范数尾项仍严重松弛；这不是完整连续域阶跃验收。
证明使用精确算术，数值为普通浮点，未做 outward rounding。
Case1 使用矩阵重构，native_validated=false。

## 验收条件

固定 HTC、零初值、原四个功率源的任意有符号常数组合，认证 SVD 前固定 V：

- 稳态全场 K-energy 相对误差 ≤ epsilon=0.001。
- 每个 t>0 的阶跃全场 C-energy 相对误差 ≤ sqrt(epsilon)。

两项必须覆盖同一个连续参数域。固定实移位、有限时间采样、冲激积分或
Hankel 范数各有用途，不能替代这个验收；也不使用 SVD 后二倍容差。
生产提取器保持不变。候选快照仅构造 V，不能提供连续域尾项保证。

## 当前代码入口

| 文件 | 职责 |
|---|---|
| `prepare_output_dynamics.py` | 重建固定输入驱动八极点候选 V，记录全部构造成本 |
| `audit_output_step.py` | BE 时间误差解析界 + 实际 AMG-CG 缺陷，核查指定 h、t |
| `dynamic_observability_tail.py` | 连续子域共同储能对照、一阶输入耦合输出与 Neumann 积分余项 |
| `audit_dynamic_identity.py` | 小模型抵消恒等式与全时间积分的独立数值核查 |
| `case1_system.py` | 共享 Case1 矩阵装配，无实验驱动依赖 |
| `numerics.py` | 共享 AMG-CG、衰减下界、C 正交化与输入 Gram 运算 |
| `port_basis.py` | Robin 端口与输入驱动端口候选构造，无验收功能 |

当前代码不导入任何历史证书驱动。清理只提取原有共享函数，没有改变
求解器容差、模型算子、候选构造或数学界。本目录不维护单元测试。

新研究的四个独立判决入口是 `positive_source_tail.py`、
`modal_reachable_tail.py`、`input_error_reachability.py`、
`audit_positive_reachability.py`；其作用域、失败结论和复现命令统一放在
阶段报告（本地记录：`records/POSITIVE_REACHABILITY_20261009.md`），不作为通过的默认算法。

## 复现

从仓库根目录运行，依赖 NumPy、SciPy、PyAMG。以小模型为例：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python playground/adaptive_bci_sampling/prepare_output_dynamics.py --mesh-mm 10 --rank 16 --output playground/adaptive_bci_sampling/results/output_small.npz
python playground/adaptive_bci_sampling/audit_output_step.py --mesh-mm 10 --candidate playground/adaptive_bci_sampling/results/output_small.npz --output playground/adaptive_bci_sampling/results/output_step_small.json
python playground/adaptive_bci_sampling/dynamic_observability_tail.py --mesh-mm 10 --candidate playground/adaptive_bci_sampling/results/output_small.npz --output playground/adaptive_bci_sampling/results/output_tail_small.json
python playground/adaptive_bci_sampling/audit_dynamic_identity.py --candidate playground/adaptive_bci_sampling/results/output_small.npz --output playground/adaptive_bci_sampling/results/output_identity_small.json
```

正式模型分别使用 `--mesh-mm 1.5` / `--mesh-mm 1`，候选名分别为
`output_47k` / `output_122k`；动态尾项使用 `--widths .01 .1 -1`。
独立恒等式核查仅用于小模型。正式 FOM 不作稠密谱分解或直接求解。
`results/` 是忽略提交的可重建缓存；完整标量证据只在本地 records 中保留。
重跑候选成本必须计入，不得把缓存复用当作免费提取。

## 失败判决与历史对照

- [失败档案](records/FAILURE_ARCHIVE.md)：集中记录精确失败命题、作用域与未被排除的方向。
- [结构性文献核对](records/STRUCTURAL_LITERATURE_20261009.md)：反馈、端口尾项及辅助误差系统的先行工作。
- [2026-10-08/09 研究归档](archive/research20261008_09/README.md)：标量余项、多项式轨迹、共同误差空间、辅助系统、Robin 反馈与耦合稳态方案。
- [更早历史归档](archive/legacy20261007/README.md)：Poisson–Loewner、冲激积分、旧提取器与稠密桥；保留用户要求的独立存档。

归档表示退出默认入口，不表示所有定理无效。历史材料中的“当前”“最新”
只描述当时状态；局部通过的连续盒仍有其原有作用域。下一步研究应收紧
整个输出加权反馈尾项的输入可达包络，而非提高阶数或优化 setup 数。

正终端尾项及共同输入误差空间的小模型复现命令（从仓库根目录执行）：

```bash
python playground/adaptive_bci_sampling/modal_reachable_tail.py --mesh-mm 10 --build --candidate playground/adaptive_bci_sampling/results/source_small.npz --output playground/adaptive_bci_sampling/results/modal_small.json
python playground/adaptive_bci_sampling/positive_source_tail.py --mesh-mm 10 --orders 2 4 8 --shifts 0 --widths .01 .1 -1 --output playground/adaptive_bci_sampling/results/positive_small_relative.json
python playground/adaptive_bci_sampling/audit_positive_reachability.py --candidate playground/adaptive_bci_sampling/results/source_small.npz --output playground/adaptive_bci_sampling/results/positive_oracle.json
python playground/adaptive_bci_sampling/input_error_reachability.py --mesh-mm 10 --candidate playground/adaptive_bci_sampling/results/source_small.npz --output playground/adaptive_bci_sampling/results/input_error_small.json
python playground/adaptive_bci_sampling/audit_output_step.py --mesh-mm 10 --candidate playground/adaptive_bci_sampling/results/source_small.npz --output playground/adaptive_bci_sampling/results/source_step_small.json
```
