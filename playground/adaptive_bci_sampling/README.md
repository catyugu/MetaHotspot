# BCI 端口传递采样实验

目标是功率端口到结温端口的传递族
`Z(s,h) = G^T(sC + K + Σ h_j H_j)^-1 G`。全场温度精度不是验收目标。
固定实频移下，`Z-Z_V = R^T A^-1 R` 可用于构造最终压缩基底的端口误差上界；
`certified_box.py` 实现连续 HTC 盒的后验证书。它**不**是 Hankel 或冲激响应
系统范数的保证。相关数学命题与未证的动态桥见 [THEORY.md](THEORY.md)。

`python/metahotspot/macromodel/utils.py` 中的 stock 提取器保持原有逐端口裸
`K` 频率计划、随机 HTC 残差探针与 closing SVD。裸 `K` 的正谱端点不保证
覆盖 Robin 族；因此不能据此声称整个 HTC 盒的 FANTASTIC 式动态保证。
本目录的确定性基底也使用 `frequency_plan` 汇总的裸 `K` 计划。
证书只对**实际交付基底**在指定的固定实频移（例如 `s=0`）成立，
不依赖基底选点与频率计划的正确性。

主要脚本：

| 文件 | 用途 |
| --- | --- |
| `deterministic_design.py` | 确定性 HTC 选点、裸 `K` 频率计划、快照与 SVD |
| `certified_box.py` | 最终基底的连续盒固定频移端口证书 |
| `certify_extraction.py` | 确定性基底、stock 基线、证书和采样点验证 |
| `validate_vendor_step.py` | 独立单位输入的采样 step 响应对照 |
| `bench_certificate.py` | 证书紧度与盒细化诊断 |
| `bench_pareto_budget.py` | RHS 预算、SVD cutoff 与固定频移缺陷 |
| `bench_dynamic_bridge_toy.py` | 动态桥的玩具反例 |

原始 `playground/bci_rom_testcase1/reproduce_case1.py` 在固定物理 HTC
`(50, 1000)` 上运行 stock 基线，并在最终 SVD 后报告 4×4 稳态端口传递的
逐项相对误差。它的 2000 s 瞬态只测名义功率组合。稳态单点测量、采样
step 响应与整盒固定频移证书必须分别报告；它们都不构成系统范数证书。

历史测量与盒谱频率计划的探索存放于 `records/`。其中盒计划的数值表格属于
**已撤回实现的历史结果**，不能当作当前生产路径的性能或正确性声明。

从仓库根目录运行（使用仓库已有的 Python 环境与已构建的 C API）：

```text
PYTHONPATH=python python -m unittest discover -s playground/adaptive_bci_sampling -p 'test_certified_sampling.py'
PYTHONPATH=python python playground/adaptive_bci_sampling/certify_extraction.py 2.5 --steady-cells 8 --steady-order 3 --certificate-blocks 4 --greedy-maximum 3
PYTHONPATH=python python playground/bci_rom_testcase1/reproduce_case1.py
```
