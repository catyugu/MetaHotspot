# 全场传递算子逼近实验

研究目标是**全场传递算子族**

```text
X(p, s) = A(p, s)^-1 G,     A(p, s) = K + s C + sum_i p_i H_i,   p in P, s >= 0,
```

即功率端口到**整场**温度分布的状态级传递算子；验收判据是整场温度对真解的 `A(p,s)`-能量
相对误差，代价指标是最小化 `N_FOM := N_RHS`（全阶逆作用次数）。

固定实频移下，共址恒等式 `Z - Z_V = E^T A E = R^T A^-1 R >= 0` 说明状态级相对误差的**平方**
恰好等于共址端口缺陷 `lambda_max(Y(p) - Y_V(p), Y(p))`（`Y = G^T A^-1 G`）。所以本目录证书
报告的端口量就是场级目标的**定频实现**，两者是同一恒等式的两种读数，不是两个目标
（`THEORY.md` 命题 1）。`certified_box.py` 是连续 HTC 盒的严格后验证书；它**不**是 Hankel /
冲激响应系统范数的保证（该缺口见 `THEORY.md` P0）。

生产代码 `python/metahotspot/macromodel/utils.py` 保持原有逐端口裸 `K` 频率计划、随机 HTC
残差探针与 closing SVD；本目录的算法不进入生产路径。裸 `K` 的谱端点不保证覆盖 Robin 族，故
不能据此声称整个 HTC 盒的 FANSTIC 式动态保证。证书只对**实际交付基底**在指定固定实频移
（如 `s = 0`）成立，与选点规则和频率计划无关。

| 文件 | 用途 |
| --- | --- |
| `deterministic_design.py` | 确定性 HTC 选点（Zolotarev 种子 + 认证贪心）、频率计划、快照与 SVD |
| `certified_box.py` | 交付基底的连续盒固定频移证书（Bernstein 包络 + 分支定界） |
| `certify_extraction.py` | 驱动：确定性基底、stock 基线、整盒证书与全阶对照 |
| `bench_certificate.py` | 证书紧度（`tightness`）与盒分支定界（`bandb`）证据 |
| `bench_pareto_budget.py` | RHS 预算、SVD cutoff 与固定频移缺陷的 Pareto 表 |
| `validate_vendor_step.py` | 采样 step 响应外部对照（实测诊断，非证书） |

支撑模块：`sparse_solve.py`（AMG 预条件 CG）、`zolotarev.py`（有限区间 Zolotarev 规则与每群
谱区间）、`residual_certificate.py`（`A(h_min)`-Riesz 逐点残差证书）、`exact_error.py`
（Woodbury 精确参数映射与误差见证）。本目录不含测试：数学陈述见 `THEORY.md`，算法证据是
`certify_extraction.py` 的整盒审计与 `bench_*` 的原始 JSON，历史测量与失败路线见 `records/`。

`playground/bci_rom_testcase1/reproduce_case1.py` 在固定物理 HTC `(50, 1000)` 上运行 stock
基线，报告**全场**恢复误差（`steady_max_absolute_rise_error_K`、
`steady_max_relative_rise_error`、`transient_final_max_absolute_rise_error_K`）以及 4x4 端口
稳态/瞬态对照；它的 2000 s 瞬态只测名义功率组合。稳态单点测量、采样 step 响应与整盒固定频移
证书必须分别报告，它们都不构成系统范数证书。

从仓库根目录运行（使用仓库已有的 Python 环境与已构建的 C API）：

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/certify_extraction.py 2.5 --steady-cells 8 --steady-order 3 --certificate-blocks 4 --greedy-maximum 3
PYTHONPATH=python python playground/bci_rom_testcase1/reproduce_case1.py
```
