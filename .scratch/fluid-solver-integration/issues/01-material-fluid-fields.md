# 01: MaterialProps 流体字段扩展

Status: needs-triage · Type: feature · Depends on: —

## Context

dev 分支 `MaterialProps`（`src/data/internal_model.hpp`）只有 `kx / ky / kz / rho / c` 五个 `CompiledExpression` 字段，没有流体标记和动态粘度。

## Goal

新增两个字段：

```cpp
struct MaterialProps {
    CompiledExpression kx, ky, kz;
    CompiledExpression rho;
    CompiledExpression c;
    bool              is_fluid = false;          // 新增
    CompiledExpression dynamic_viscosity;        // 新增，默认为 make_constant(0.0)
};
```

## Scope

- 仅改头文件结构；不在本 issue 实现 sidecar 注入或流体逻辑
- `is_fluid` 默认 false（保证零回归）
- `dynamic_viscosity` 默认构造为常量 `0.0`，由 sidecar 阶段覆盖

## Acceptance

1. `src/data/internal_model.hpp` 含上述两字段
2. 既有 `simple_steady_tests / case1..4` 跑通（无 sidecar 时 `is_fluid=false` 路径）
3. `cmake --build` 无 warning

## Notes

- 单元测试可能暂时没有（功能未实现）；此 issue 主要是为后续 issue 铺基础结构
- 字段命名遵循 `MaterialProps` 既有风格（中文注释 + 英文标识）
- 不可在 `Rect` / `Block` 加新字段（material 单一来源原则）
