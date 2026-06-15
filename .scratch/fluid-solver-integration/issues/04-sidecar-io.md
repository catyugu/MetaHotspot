# 04: Sidecar I/O 解析

Status: needs-triage · Type: feature · Depends on: 01, 02

## Context

GUI 短期不会给主 XML 加流体字段。需要一种不污染主 XML 的元数据承载方式——sidecar `<name>_additional.xml`。

sidecar 承载两类信息：

1. 哪些 material 是流体（material-level 注入 `is_fluid = true` + μ）
2. pressure 边界条件（与主 XML 的 `<Boundary>` 同构）

## Goal

实现 `io::read_additional_xml(xml_path)`，从主 XML 路径派生 sidecar 路径（`<name>_additional.xml`）并解析：

```cpp
struct FluidOverlay {
    std::unordered_map<std::string, MaterialOverlay> materials;
    std::vector<Boundary> pressure_boundaries;
};

struct MaterialOverlay {
    std::optional<bool> is_fluid;        // 不显式写则保持 material 原值
    std::string dynamic_viscosity;       // μ 表达式，默认 "0.0"
};
```

sidecar XML 结构：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<FluidOverlay>
    <FluidMaterial name="water_25C">
        <DynamicViscosity>0.00089</DynamicViscosity>
    </FluidMaterial>

    <Boundary>
        <BoundaryCategory>ThermalPressure</BoundaryCategory>
        <Name>fluid_inlet</Name>
        <FaceKeys>
            <string>X|E|0|0,50,0,50;50,100,0,50;50,100,50,100</string>
        </FaceKeys>
        <PressureBoundary>
            <Pressure>1.0e5</Pressure>
        </PressureBoundary>
    </Boundary>
</FluidOverlay>
```

错误处理：

| 条件                                                 | 行为                                                  |
|------------------------------------------------------|-------------------------------------------------------|
| 文件不存在                                           | `LOG_INFO "no fluid overlay"` → 返回空 `FluidOverlay` |
| 文件存在但解析失败                                   | `LOG_ERROR` + `std::runtime_error`                    |
| `<FluidMaterial name="X">` 的 X 不在主 XML materials | `LOG_ERROR` + `std::runtime_error`                    |
| 同一 material 重复声明 `<FluidMaterial>`             | `LOG_ERROR` + `std::runtime_error`                    |
| pressure boundary face_key 命中 0 流体单元           | 推迟到 `Preprocessor::load` 阶段报 `LOG_WARN`         |

## Scope

- 新函数 `read_additional_xml`，返回 `FluidOverlay`
- 新类型 `MaterialOverlay`
- 不实现 sidecar 注入到 `material_table` 的逻辑（属于 issue 05）
- 不实现 pressure boundary face_key 解析与命中（属于 issue 06）

## Acceptance

1. `src/io/io.hpp` / `src/io/io.cpp` 含 `read_additional_xml` 函数
2. `src/data/io_model.hpp` 含 `FluidOverlay` / `MaterialOverlay` 类型
3. 单元测试：
   - sidecar 不存在 → 返回空 overlay（`LOG_INFO`）
   - sidecar 存在且引用不存在的 material → 抛 `runtime_error`
   - 同一 material 重复声明 → 抛 `runtime_error`
   - sidecar 存在且完整 → 解析出预期的 materials + boundaries
4. `cmake --build` 无 warning

## Notes

- `BoundaryCategory` 当前只有 `Electrical` 一个值；在 `io_model.hpp` 加 `ThermalPressure` 枚举值
- `<PressureBoundary>` 与 `<ThermalBoundary i:type="…">` 对称；为简洁直接用 `<PressureBoundary>` 标签而非多态
- 不引入新的 XML 解析库，复用现有 tinyxml2 路径
- XML 命名空间与主 XML 保持一致（不强制，可缺省）
