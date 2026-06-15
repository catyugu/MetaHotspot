# 03: FluidFields 数据结构

Status: needs-triage · Type: feature · Depends on: 01

## Context

流体解算的输出（pressure / hydroC / face_velocity / fs_faces）需要一个明确的容器，并且要挂在 InternalModel 上。

## Goal

新建 `src/data/fluid_model.hpp`，定义 `mhs::core::FluidFields` 结构；在 `InternalModel` 上挂 `std::optional<FluidFields> fluid`。

```cpp
struct FluidFields {
    std::vector<double> pressure;                          // [N_active] Pa
    std::array<std::vector<double>, 3> hydroC;             // [N_active][3] m^3·s/kg
    double reference_temperature = 300.0;                  // K, μ 求值时的 T
    std::vector<int> fluid_ids;                            // [n_fluid] c_idx
    std::vector<int> g2f;                                  // [N_active] 默认 -1

    struct FSFace {
        std::array<int, 2> cells;                          // [c0, c1]
        uint8_t axis;                                      // 法向
        double h;
        double D_h;
        double Nu;
    };
    std::vector<FSFace> fs_faces;

    std::vector<std::array<double, 3>> face_velocity;      // [n_internal_face][3] m/s
};
```

InternalModel 增量：

```cpp
struct InternalModel {
    /* 既有字段 */
    std::optional<FluidFields> fluid;                      // 新增
};
```

## Scope

- 仅类型定义 + InternalModel 字段
- 不实现任何填充逻辑（属于后续 issue）
- `fluid` 默认 `std::nullopt`，保证零回归

## Acceptance

1. `src/data/fluid_model.hpp` 存在，含上述定义
2. `src/data/internal_model.hpp` 含 `std::optional<FluidFields> fluid;`
3. 既有 case 跑通
4. `cmake --build` 无 warning（注意 `<optional>` include）

## Notes

- 不可在 `CellFields` 加流体标志（material 单一来源原则）
- 不为流体新建求解器抽象（Q1.11 决策）
- 文件命名 `fluid_model.hpp` 与现有 `internal_model.hpp` / `io_model.hpp` 对齐
