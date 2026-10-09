# 历史方案归档

具体实验数据及结果报告只保存在本地，不随 Git 分发；`records/` 和归档中的同名目录受忽略规则保护。仓库保留代码、证明、复现方法、文献核对及统一失败档案。历史记录中的提交号在历史压缩后可能不再可用。

本目录保存 2026-10-07 至早期 2026-10-08 的证明、实现与实验记录。
**当前目标和默认入口见 [当前 README](../../README.md)。** 归档材料中的“最新”“当前”
仅指其写作时状态，尤其 SVD 后二倍容差不能替代当前 pre-SVD 原始系统验收。

| 路线 | 保留的主要入口 | 归档原因与结论边界 |
|---|---|---|
| 固定实移 Riesz–Bernstein / Woodbury 审计 | `certified_box.py`、`exact_error.py`、`certify_extraction.py`、`THEORY.md` | 固定实移代数和小模型核验仍有效；不直接提供当前全时刻阶跃保证 |
| 有理时间冲激积分 | `rational_dynamic.py`、`run_rational_dynamic.py`、`RATIONAL_DYNAMIC_PROOF.md` | 目标是独立冲激积分范数；Case1 认证成本及宽盒松弛不满足当前需求 |
| Poisson–Loewner 带符号 Gram | `matrix_innovation.py`、`SIGNED_POISSON_PROOF.md`、`confirm_signed_poisson.py` | 保留用户要求的完整独立存档；没有把冲激结果改称阶跃认证 |
| SVD 动态截断与原生核对 | `svd_dynamic_guard.py`、`verify_guarded_svd_cells.py`、`confirm_native_and_mesh.py` | 历史截断/验证工具，保留算法前提与原生验证证据 |
| 稠密谱全时刻阶跃及参数桥 | `steady_step_audit.py`、`affine_step_bridge.py`、相应证明 | 局部证明仍成立，但大模型稠密谱成本和宽域余量不适合当前主线 |
| pre-SVD 稠密认证与完整提取原型 | `pre_svd_audit.py`、`certified_extraction.py`、`benchmark_certified_extraction.py` | 有可运行的局部算法及导出接口；未解决原域、大模型与成本优势 |

历史结果仅在本地保留于本目录的 `records/`，失败判决集中于
[FAILURE_ARCHIVE.md](../../records/FAILURE_ARCHIVE.md)。当前共享 Case1 装配已经独立到
[case1_system.py](../../case1_system.py)；归档驱动复用它，当前原型不依赖本目录。

从仓库根目录显式运行归档，例如：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python
python playground/adaptive_bci_sampling/archive/legacy20261007/run_rational_dynamic.py --help
python playground/adaptive_bci_sampling/archive/legacy20261007/benchmark_certified_extraction.py --help
```

各路线的详细命令见对应证明、记录及 `CERTIFIED_EXTRACTION_REPRODUCE.md`。
本目录采用原来的平面模块布局以保留脚本的导入关系，没有添加通往旧入口的根目录兼容副本。

删除了重复汇总入口 `METHODS_20261007.md`、`REPRODUCE_20261007.md`，以及写回旧比较文档的
`summarize_extraction_benchmark.py`；原文可在清理前远程提交 `9e37684` 中恢复。
具体证明和提取器复现文档随 Git 保留，结果文件仅在本地保留。科学原始档案及候选 NPZ 未删除；仅清除可重建缓存
和已经提交、重复保存的临时 patch。

远程通过 GitHub 接口同步，文件树与本地逐提交校验一致。旧研究记录中的本地提交号
`54d7167` 对应远程 `e034d42`，`f8e6121` 对应远程 `9e37684`；恢复历史材料时使用远程提交号。

当前及归档单元测试均已移除；历史核验数字仅记录当时状态，不再提供测试运行入口。
