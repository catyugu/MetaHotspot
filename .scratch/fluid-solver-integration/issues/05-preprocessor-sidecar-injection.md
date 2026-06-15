# 05: Preprocessor 注入 sidecar 元数据

Status: needs-triage · Type: feature · Depends on: 04

## Context

issue 04 把 sidecar 解析成 `FluidOverlay`。现在需要把 overlay 应用到 `InternalModel`：

1. `MaterialProps.is_fluid = true`（如果 sidecar 声明）
2. `MaterialProps.dynamic_viscosity` 编译为 `CompiledExpression`（覆盖默认 0.0）
3. pressure boundary 编译进 `BCParamTable.pressure_p` + 命中 face_key（与现有 thermal BC 走通一条路径）

## Goal

修改 `Preprocessor::load`，在解析完主 XML / 现有 thermal BC 之后、调用流体解算之前：

```cpp
auto overlay = io::read_additional_xml(derive_sidecar_path(xml_path));

// 1. 注入 material 字段
for (const auto& [name, mat_overlay] : overlay.materials) {
    size_t idx = name_to_idx.at(name);     // 主 XML 已经建好
    if (mat_overlay.is_fluid.value_or(false))
        model->material_table[idx].is_fluid = true;
    model->material_table[idx].dynamic_viscosity
        = mhs::core::parse(substitute_function_args(mat_overlay.dynamic_viscosity, "T", fns));
}

// 2. 注入 pressure BC（与 thermal BC 走同一条 resolve_face_keys 路径）
for (const auto& pb : overlay.pressure_boundaries) {
    uint16_t param_idx = (uint16_t)model->bc_params.pressure_p.size();
    model->bc_params.pressure_p.push_back(
        mhs::core::parse(substitute_function_args(pb.pressure, "T", fns)));
    // 调用 resolve_face_keys 的 pressure 分支（新增）
    // 命中 0 单元 → LOG_WARN
}
```

## Scope

- 修改 `Preprocessor::load`，**不**实现流体解算
- 把"命中 face_key" 的现有 `resolve_face_keys` 函数扩展以支持 pressure 类型
- 重复 material 名 / 不存在的 material 名 应在 `read_additional_xml` 阶段已被捕，此处不再校验

## Acceptance

1. sidecar 存在 → `material_table` 含 `is_fluid=true` + 编译后 μ；`bc_params.pressure_p` 含 1+ 项；`cell_bcs` 中对应 face 已标 pressure BC
2. sidecar 不存在 → 行为与既有完全一致
3. 单元测试覆盖：
   - sidecar 注入 material 后 `material_table[idx].is_fluid == true` 且 `dynamic_viscosity.eval(...) == 0.00089`
   - pressure boundary 的 face_key 命中正确单元
   - pressure boundary 命中 0 单元 → `LOG_WARN` 不报错
4. `cmake --build` 无 warning

## Notes

- 路径派生：`additional_path = replace_ext(xml_path, "_additional.xml")`
- `resolve_face_keys` 函数需要新增 pressure 分支（与 thermal 的 FirstType / SecondType / ThirdType 三分支并列）；不破坏现有签名
- 编译表达式时复用 `substitute_function_args(..., "T", fns)` 与现有 thermal BC 路径一致
