# Issue 02: Overlay XML 解析 + Preprocessor 合入

## Parent

.fluid-algorithm/PRD.md

## What to build

让系统能从额外 XML 文件中读取流体 overlay 配置,并在预处理阶段将 overlay 数据合入 InternalModel。

具体包含:

1. **`io.cpp`**: 新增 `read_fluid_overlay_xml(const std::string& xml_path)` 函数,解析 `<FluidOverlay>` 格式的 XML 文件(见 PRD §4.1),返回 `std::optional<FluidOverlay>`。若无 overlay 文件或无流体节点,返回 `std::nullopt`。

2. **`preprocessor.cpp`**: 新增 `apply_fluid_overlay()` 函数,在 `Preprocessor::load()` 末尾调用,若 `FluidOverlay` 存在则:
   - 遍历 `material_table`,`FluidMaterial.name` 匹配的材料标记 `is_fluid = true`
   - 将 `FluidMaterial.dynamic_viscosity` 表达式化简为 `CompiledExpression` 并求值到 double,存入 `model->dynamic_viscosity`
   - 解析 `PressureBoundary` face_keys,匹配暴露面,填充 `model->is_pressure_boundary` 和 `model->boundary_pressure` (独立于 `CellBC`)
   - 初始化 `model->is_fluid`, `model->pressure`, `model->flow_axes`, `model->hydroC_x/y/z` (全零)

## Acceptance criteria

- [ ] `read_fluid_overlay_xml()` 能正确解析 `steady_case1_additional.xml`
- [ ] `Preprocessor::load()` 合入 overlay 后,`model.is_fluid[water_cells] == 1`,`model.dynamic_viscosity[water_cells] == 8.9e-4`
- [ ] 压力 inlet cell 的 `is_pressure_boundary[c]==1`,`boundary_pressure[c]==500`
- [ ] 压力 outlet cell 的 `is_pressure_boundary[c]==1`,`boundary_pressure[c]==0`
- [ ] 现有测试 100% 通过 (overlay 不存在时走无流体路径,行为完全不变)
- [ ] 手动验证: 运行 `steady_case1.xml` (无 overlay) → 结果与之前完全一致

## Blocked by

- Issue 01 (数据骨架)
