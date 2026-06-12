# 切片 8 — 集成测试 + 参考数据

> **Status**: needs-triage
> **依赖**: 切片 0–7
> **阻塞合并**: 是（验证全部算法路径）

## 目标

真实 case XML 跑通 BDF2 / AdaptiveBdf；签入参考数据。

## 新建

- `cases/bdf2_transient_tests/case1.xml` —— 简单瞬态，`<Scheme>Bdf2</Scheme>`
- `cases/adaptive_transient_tests/case1.xml` —— 边界温阶跃，触发自适应加密
- `cases/adaptive_transient_tests/case2.xml` —— 稳态附近，触发自适应放大
- `cases/adaptive_transient_tests/case3.xml` —— `output_dt` 与内部 dt 互不整除
- `cases/adaptive_transient_tests/expected/` 目录存首次运行生成的参考

## 修改

- `scripts/compare_lib.py`（如新 case 需不同容差）
- `run_cases.py`（添加新 case 目录到 `CASE_GROUPS`）

## 测试

- 跑 `python run_cases.py`，所有 case 通过 `compare_lib.py`
- 首次生成参考：`python run_cases.py` + 手动 `git add` 参考
- 第二轮：删 build + 重跑，确认仍通过

## 验证

```bash
conda activate cpp_env
cmake --build build --parallel
python run_cases.py
```
