# 02: BCParamTable pressure 字段

Status: needs-triage · Type: feature · Depends on: 01

## Context

流体解算需要 pressure 边界条件落入 BCParamTable，与现有 dirichlet_T / neumann_q / cauchy_h / cauchy_T_inf 一致。

## Goal

新增 `pressure_p` 字段：

```cpp
struct BCParamTable {
    /* 既有字段 */
    std::vector<CompiledExpression> dirichlet_T;
    std::vector<CompiledExpression> neumann_q;
    std::vector<CompiledExpression> cauchy_h;
    std::vector<CompiledExpression> cauchy_T_inf;
    std::vector<CompiledExpression> pressure_p;     // 新增
};
```

## Scope

- 仅改头文件结构
- 默认空 vector
- 不实现 sidecar 解析路径（属于 issue 04）

## Acceptance

1. `src/data/internal_model.hpp` 含 `pressure_p` 字段
2. 既有 case 跑通
3. `cmake --build` 无 warning

## Notes

- 字段顺序：与 `dirichlet_T` 等并排放在末尾
- 命名沿用 `pressure_p`（"pressure 标量压力值"），与 Python `fields.pressure` 对齐
