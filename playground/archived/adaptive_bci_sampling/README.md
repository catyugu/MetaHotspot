# 自适应 BCI 宏模型提取

本目录探索比 stock Extended BCI FANTASTIC 更高效、并能解释误差保证边界的参数化热模型宏模型提取方法。研究对象是功率输入到**全部温度自由度**的传递算子，而不是仅有部分端口的读出。

目前还没有同时在多类实际模型上稳定优于基线、并对最终动态 ROM 建立所需连续域保证的候选方法。实验路线的记录**只概述问题和失败原因，不保存具体数值结果**；旧实验详情可从 Git 历史及相应原型归档追溯。

生产 `python/metahotspot/macromodel/utils.py` 保持 stock 实现。以下脚本是数学核验和审计工具，并不意味着已替换生产提取器。

## 文档导航

| 文件 | 作用 |
| --- | --- |
| [THEORY.md](THEORY.md) | 误差对象、Riesz–Bernstein 固定实移证书、连续分布风险定理、未证明的动态传递问题 |
| [近期路线梳理](records/RESEARCH_SYNTHESIS_2026-10-17.md) | Project 历史中的研究方案、简要问题与当前状态 |
| [失败路线档案](records/FAILURE_ARCHIVE.md) | 较早路线的失败机制、适用范围及实验实现错误；仅作定性记录 |

## 判定原则

必须区分**原定频移的 SVD 前残差风险**、**固定实移的全场能量误差**和**最终 SVD 后 ROM 的全场动态误差**。前者通过不自动保证后者；任一控制量的 PAC 认证也不等于整个参数盒的逐点认证。

一个候选方法只有在与 stock 采用**相同误差目标、停止规则和数值预算**、计入快照、选点、认证、SVD 等完整成本并在不同模型上重复成立后，才能宣称提取效率改善。负面实验只关闭明确的配方与验收口径，不否决该方法族的全部数学可能性。

## 维护中的工具

| 文件 | 用途 |
| --- | --- |
| `certify_extraction.py` | stock 最终基的 DC 连续盒上界和直接全场对照 |
| `certified_box.py` | 固定实移 Riesz–Bernstein 单元上界与分支定界 |
| `exact_error.py` | 对角 Robin 族的 Woodbury 误差映射 |
| `sparse_solve.py` | 大型实 SPD 系统的 AMG 预条件 CG |
| `test_field_audit.py` | 已有的全输入场误差回归检查 |

旧的确定性种子、网格贪心、预算 sweep 已退役。JFRGE 的历史原型和正式实验原始记录保存在其 [历史提交](https://github.com/catyugu/MetaHotspot/commit/56fbfea98a31b9a07c6553c41a23646f76faad54)，不在当前 `agent/work` 工作树中。

## 核验入口

从仓库根目录运行已有脚本；按本机 Python、原生 API 与依赖环境调整：

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python playground/adaptive_bci_sampling/certify_extraction.py 5 \
    --tolerance 1e-3 --output /tmp/metahotspot-field-audit.json

PYTHONPATH=python python -m pytest \
  python/tests playground/adaptive_bci_sampling/test_field_audit.py \
  -q -p no:cacheprovider
```

这些命令用于重新生成审计数据，不代表此次文档整理重新进行了实验。当前审计不具备浮点区间舍入的严格认证，且固定实移结果不能直接充当动态误差保证。

若后续获得新结果，文档只更新定性的结论和限制；原始测量、机器环境、基底标识、时间与计数应留在可追溯的实验记录中，而非写入路线概览。
