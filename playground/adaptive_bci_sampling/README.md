# 全场传递算子研究与审计

2026-10-07 归档入口：[方法与算法](METHODS_20261007.md)、
[最新目标完整证明](STEADY_STEP_PROOF.md)、[复现说明](REPRODUCE_20261007.md)、
[整理后的实验总览](records/EXPERIMENTS_20261007.md)。只提交源码、测试和结果文档；
`results/` 中的矩阵、基底、JSON 和其他原始输出均由脚本重建，不纳入本次提交。

**冲激积分版本独立存档：** [基础有理时间推导](RATIONAL_DYNAMIC_PROOF.md)、
[Poisson–Loewner 完整证明](SIGNED_POISSON_PROOF.md)、
[数学判决实验](records/SIGNED_POISSON_EXPERIMENT_20261007.md) 与
[进一步确认](records/SIGNED_POISSON_CONFIRMATION_20261007.md)。对应实现、测试和复现命令
完整保留，不以最新阶跃目标替换这套证明。

目标是 `X(h,s) = (K+sC+sum_i h_i H_i)^-1 G`：功率输入到**全部温度自由度**的传递算子。
固定实移验收使用所有输入组合下的相对 `A(h,s)`-能量误差，容差 `tau` 对应平方缺陷阈值
`tau^2`。最新验收为最终基稳态全场 K-energy 相对误差 `2 epsilon`、
阶跃全场 C-energy 相对误差 `2 sqrt(epsilon)`，包括全部输入组合与所有时刻。
历史全场 impulse 的 C 加权时间积分目标保留独立推导，不能替代阶跃验收，见
[THEORY.md](THEORY.md)。结温、共址输出 H2/Hankel 与有限时刻 step 测量不自动给出上述保证。

生产 `python/metahotspot/macromodel/utils.py` 保持 stock Extended BCI FANTASTIC，实验算法
留在 playground。新方法只能接受容差与 HTC 范围作为控制参数；网格、物理模型和审计分辨率
属于验证条件，不能成为改善候选成绩的调参手段。固定 seed 是复现基线所需的历史设置，不能
作为新方法输入。

当前目录保留固定实移基线审计、数学支撑和动态认证研究原型；不再运行已排除的选点配方。

| 文件 | 用途 |
| --- | --- |
| `certify_extraction.py` | stock 最终基的 DC 连续盒上界与直接全场对照，输出原始 JSON |
| `certified_box.py` | 固定实移的 Riesz–Bernstein 单元上界；升阶与分支定界是研究工具 |
| `exact_error.py` | 对角 Robin 项的 Woodbury 误差映射，作为小模型独立核验工具 |
| `sparse_solve.py` | 大型 SPD 系统的 AMG 预条件 CG |
| `test_field_audit.py` | 全输入场误差、弱观测节点误差及完整空间的回归检查 |
| `records/FAILURE_ARCHIVE.md` | 更早的负结果、测量更正、旧证书与预算证据 |
| `rational_dynamic.py` | 正交有理时间系数、精确尾项及 Bernstein 连续动态单元上界的浮点原型 |
| `run_rational_dynamic.py` | 固定 stock/raw 基的连续盒、局部收敛与成本判决实验 |
| `test_rational_dynamic.py` | Parseval、独立时间积分、全部输入组合、坐标变换等核验 |
| `RATIONAL_DYNAMIC_PROOF.md` | 原型的精确算术接受证明与有限覆盖存在性 |
| `records/RATIONAL_DYNAMIC_EXPERIMENT_20261007.md` | 判决记录：小模型连续覆盖成功但成本不可接受；Case1 重建与耦合负结果 |
| `records/RATIONAL_DYNAMIC_OPTIMIZATION_20261007.md` | 后续优化：联合创新能量界、参数包围细分与完整 RHS 成本对照 |
| `matrix_innovation.py` | 保留带符号时间/空间/输入 Gram 的 Poisson–Loewner 连续谱包围 |
| `SIGNED_POISSON_PROOF.md` | 新矩阵余量界、连续参数证明、无界结构增益例及 SVD 后 2 tau 预算 |
| `svd_dynamic_guard.py` | 固定 stock 快照/SVD 方向，以连续动态证书验收最终截断阶数的研究原型 |
| `verify_guarded_svd_cells.py` | 固定最终 SVD 基的旧/新数学界对照及独立正时间积分参考 |
| `test_matrix_innovation.py` | Poisson 恒等式、连续谱支配、方向抵消、SVD 嵌套及输入组合核验 |
| `records/SIGNED_POISSON_EXPERIMENT_20261007.md` | 最终 76 阶 SVD 基的 2 tau 局部通过与同设置旧拒绝/新接受判决 |
| `confirm_signed_poisson.py` | 100 个稠密 SPD 系统、非正交坐标变换、最坏输入与正时间积分复验 |
| `confirm_native_and_mesh.py` | C++ 原生装配核对与 1200 单元网格最终 SVD 基直接认证 |
| `records/SIGNED_POISSON_CONFIRMATION_20261007.md` | 原生、多种子、更宽盒与细网格复验；区分直接证书与分预算失败 |
| `steady_step_audit.py` | 最新稳态 2 epsilon / 阶跃 2 sqrt(epsilon) 目标复验及相同验收条件的 stock 成本对照 |
| `records/STEADY_STEP_COST_20261007.md` | 48/48/53 阶、134/137/140 次逆作用；阶跃仍仅覆盖审计 HTC 点的全时刻 |
| `STEADY_STEP_PROOF.md` | 最新目标定义、连续稳态证书、初始/区间/无穷尾的全时刻阶跃证明 |
| `METHODS_20261007.md` | 三条认证路线的输入、输出、接受/拒绝条件、伪代码及成本口径 |
| `REPRODUCE_20261007.md` | 无原始数据依赖的环境、测试、实验命令和结果文档对应表 |

最新用户验收是最终 SVD 基稳态相对误差 `2*epsilon`、阶跃相对误差
`2*sqrt(epsilon)`，使用全场能量范数、全部输入组合和所有时刻。
当前已完成局部连续 HTC 稳态证书，以及 7 个 HTC 点的全时刻阶跃证书；
阶跃的连续 HTC 全盒覆盖尚未建立。不要将下面历史冲激范数结果当作新目标的保证。

此前冲激范数研究验收：提取 tolerance 为 tau，最终 SVD 基的动态误差接受阈值按用户指定为
2 tau；嵌套三角路线给 raw 基分配 tau，直接认证最终基不要求这个充分条件。标准快照 SVD cutoff 本身不蕴含此动态保证；Case1 stock
34 阶基仍超限，实验中通过的是受动态证书验收的另一个 76 阶 SVD 基。

删除旧的确定性种子/网格贪心提取配方及其预算 sweep、仅端口 step 驱动；它们没有建立满足
当前全场动态目标且优于基线的方法。历史公式或负结果不等于所有上位数学方向被排除。
历史脚本在 Git 中保留，档案中的旧命令仅标识当时实验。

从根目录运行：

```powershell
$env:PYTHONPATH = "$PWD/python"
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& E:/env/miniconda3/envs/numerical/python.exe playground/adaptive_bci_sampling/certify_extraction.py 5 --tolerance 1e-3 --output "$env:TEMP/metahotspot-field-audit.json"
& E:/env/miniconda3/envs/numerical/python.exe -m pytest python/tests playground/adaptive_bci_sampling/test_field_audit.py -q -p no:cacheprovider
```

审计固定在 `s=0`、每轴 8 单元、三阶 trial、每轴 4 个 Gram 锚点块；每单元检查上下角及两个
独立内部点。JSON 同时记录这些条件、基线原始设置、直接场误差和所有 RHS 计数。该分辨率上的
上界可能很松，审计成功仅表示未发现违反上界，不表示达到给定容差。浮点 AMG-CG 与稠密代数
未作区间误差包围，`floating_point_certified=False`。

成本分别报告 `N_extract`、`N_certificate`、`N_reference` 与其总和。候选提取必须把为了选择、
停止或认证而执行的全阶逆作用纳入 `N_FOM`；独立外部参考成本单列。AMG setup、谱 matvec、CG
迭代、墙钟和内存另计，不能用快照数代替端到端成本。

Case 1 的 `playground/bci_rom_testcase1/reproduce_case1.py` 仍报告实际整场稳态与瞬态最终时刻
恢复误差，以及热源区观测；其名义功率组合不能代替所有输入方向或整个时域的保证。
