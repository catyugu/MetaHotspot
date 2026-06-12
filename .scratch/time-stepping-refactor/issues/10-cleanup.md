# 切片 9 — 清理遗留

> **Status**: needs-triage
> **依赖**: 切片 0–8
> **阻塞合并**: 是（必须全删，零兼容）

## 目标

删除 `T_prev`、删除旧 `assemble()`（如果还有）、删除 `nonlinear_solve` 旧签名、删除 deprecated 标记。

## 修改

### `src/data/internal_model.hpp::GlobalState`

- 删除 `T_prev` 字段
- 全代码库 grep `T_prev` 找残留

### `src/assembler/assembler.hpp` / `.cpp`

- 确认 `assemble()` 早已删除（切片 1 完成）
- 清理 `assemble()` 任何残留引用

### `src/nonlinear/nonlinear_solver.hpp` / `.cpp`

- 删除旧 `nonlinear_solve(model, state, solver)` 签名
- 只保留接收 `LinearSystem` 的版本

### 全代码库 grep

- `T_prev` 残留
- `assembler.assemble(` 残留
- 任何 deprecated 注释

## 验证

```bash
grep -rn "T_prev" src/ tests/
grep -rn "assembler\.assemble(" src/ tests/  # 不应有匹配
cmake --build build --parallel
python run_tests.py
python run_cases.py
```

预期：所有测试绿；grep 无残留。
