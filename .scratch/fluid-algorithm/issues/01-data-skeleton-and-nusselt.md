# Issue 01: 数据骨架 + Nusselt 函数

## Parent

.fluid-algorithm/PRD.md

## What to build

为流体-固体耦合传热算法铺设数据骨架。新增类型和纯函数,不引入任何运行时行为变更。

具体包含:

1. `types.hpp`: 新增 `FluidBCType` 枚举 (`None = 0, PressureType = 1`)，独立于 `BcType` (热 BC)
2. `internal_model.hpp`: InternalModel 新增流体字段 (is_fluid, pressure, flow_axes, hydroC_x/y/z, dynamic_viscosity, is_pressure_boundary, boundary_pressure, boundary_temperature_fluid),零初始化
3. `io_model.hpp`: 新增 `FluidOverlay`, `FluidMaterial`, `FluidBoundary` 类型
4. `physics_utils.hpp`: 新增 `nusselt_rectangular(double w, double h)` 函数,实现 Shah & London 矩形截面 Nu 公式
5. `cell_fields.hpp` / `material_props.hpp`: CellFields 新增 `fluid_material_id`, MaterialProps 新增 `is_fluid` + `dynamic_viscosity` 字段
6. `BCParamTable`: 新增 `pressure_bc_values` 字段

所有新增字段必须有合理的默认值/零初始化,确保现有测试 100% pass,现有路径完全不受影响。

## Acceptance criteria

- [ ] `FluidBCType` 枚举 (`None=0, PressureType=1`) 添加到 `types.hpp`
- [ ] `InternalModel` 新增所有流体字段,零初始化
- [ ] `FluidOverlay`, `FluidMaterial`, `FluidBoundary` 类型添加到 `io_model.hpp`
- [ ] `CellFields::fluid_material_id` 和 `MaterialProps::is_fluid` / `dynamic_viscosity` 添加
- [ ] `BCParamTable::pressure_bc_values` 添加
- [ ] `nusselt_rectangular()` 实现: `Nu = 8.235 * (1 - 2.0421*AR + 3.0853*AR^2 - 2.4765*AR^3 + 1.0578*AR^4 - 0.1861*AR^5)`, 其中 AR = min(w,h)/max(w,h)
- [ ] `test_nusselt_rectangular` 单元测试: 正方形 AR=1 → Nu≈8.235; 极窄 AR→0 → Nu≈8.235; 典型 AR=0.5 → Nu≈4.5
- [ ] `cmake --build build --parallel && python run_tests.py` 全部通过
- [ ] 无运行时行为变更 — 所有新字段零初始化,现有路径完全不受影响

## Blocked by

None - can start immediately
