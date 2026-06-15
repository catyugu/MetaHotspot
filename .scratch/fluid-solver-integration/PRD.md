# PRD: Fluid Solver Integration

Status: needs-triage · Owner: TBD · Created: 2026-06-15

> 为 dev 分支 C++ 实现引入基于 Poiseuille 假设的流体解算。流体与热计算相对解耦：流体解算前置一次，结果以"体积对流项 + 流-固边界传热"两种形式进入热装配。设计决策详见 `DECISIONS.md`。

## Background

电子封装微通道冷却场景需要仿真冷却液对芯片热点的对流散热。dev 分支当前是纯扩散 + 边界条件的有限体积热仿真，没有流体解算。

参考实现 `experiment-v1` 分支的 Python 代码 `metahotspot/fluid_preprocessor.py` 已经把 Poiseuille 解析解集成进求解流程：每轴水力传导 `hydroC` + 压力 `p` 一次性算好，然后热装配把它当成"已知速度场"用。

主要约束：

- **GUI 短期无法感知流体字段**（无 `is_fluid` / `μ` / pressure BC 的 UI 入口），所以新功能不能要求 XML schema 扩展
- **流体解算与热计算解耦**：流体解算在 t=0 前置一次；后续时间步只更新 T

## Goals

1. dev 分支 C++ 实现支持微通道冷却的稳态 / 瞬态热仿真
2. Poiseuille 解析解与 Python 参考实现**数值一致**（同 case 误差 < 1e-9 级别）
3. 新功能**不破坏**现有无流体的 case（回归测试字节级一致）
4. 流体解算与热装配的耦合边界清晰、可独立单元测

## Non-Goals

- 不可压缩 Navier-Stokes 一般求解
- 湍流模型（k-ε、k-ω、LES）
- 粘性耗散热源
- inflow/outflow velocity profile 边界
- μ(T)、ρ(T) 强非线性
- 二维 / 轴对称几何
- 重写 GUI 或 XML schema

## Design Summary

完整设计见 `DECISIONS.md`。要点：

### 数据流

```text
主 XML (cases/<name>/<name>.xml)                 sidecar (cases/<name>/<name>_additional.xml)
   │                                                  │
   └────────────────┬─────────────────────────────────┘
                    ▼
        io::read_xml + read_additional_xml
                    │
                    ▼
        Preprocessor::load(IOStructure)
          ├─ resolve_geometry
          ├─ resolve_layers        (material_id 写到 CellFields)
          ├─ merge sidecar         (is_fluid / dynamic_viscosity 注入 material_table)
          ├─ resolve_face_keys     (pressure BC 落 bc_params.pressure_p)
          ├─ compile expressions
          └─ FluidPreprocessor::solve_flow(model)   ← 新增钩子
                ├─ 扫 fluid_ids
                ├─ 算 mu_ref (T_ref 一次求值)
                ├─ 算 hydroC[axis] (per cell)
                ├─ 装配 pressure 系统 C·p = 0
                ├─ SparseLUSolver 解 p
                ├─ 算 face_velocity (per internal face × 3 axis)
                └─ 扫 fs_faces (c0.is_fluid XOR c1.is_fluid)
                      算 Nu · k_fluid / D_h，写 h 到 fs_face
                    │
                    ▼
        InternalModel (含 optional<FluidFields> fluid)
                    │
                    ▼
        Scheduler::run
          ├─ Assembler::assemble(state)   ← 新增体积对流 + 流-固 h 项
          └─ nonlinear_solve / time_scheme
```

### 数据结构增量

**`src/data/internal_model.hpp`**

```cpp
struct MaterialProps {
    CompiledExpression kx, ky, kz;
    CompiledExpression rho;
    CompiledExpression c;
    bool              is_fluid = false;          // 新增
    CompiledExpression dynamic_viscosity;        // 新增，默认 "0.0"
};
```

```cpp
struct InternalModel {
    /* 既有字段 */
    std::optional<FluidFields> fluid;            // 新增
};
```

**`src/data/fluid_model.hpp`** (新文件)

```cpp
struct FluidFields {
    std::vector<double> pressure;                          // [N_active] Pa
    std::array<std::vector<double>, 3> hydroC;             // [N_active][3] m^3·s/kg
    double reference_temperature = 300.0;                  // K, μ 求值时的 T
    std::vector<int> fluid_ids;                            // [n_fluid] 内容是 c_idx
    std::vector<int> g2f;                                  // [N_active] 默认 -1

    struct FSFace {
        std::array<int, 2> cells;                          // [c0, c1]
        uint8_t axis;                                      // 法向 (0/1/2)
        double h;                                          // W/(m²·K)
        double D_h;                                        // m
        double Nu;                                         // 局部 Nusselt
    };
    std::vector<FSFace> fs_faces;                          // 已排序可二分

    std::vector<std::array<double, 3>> face_velocity;      // [n_internal_face][3] m/s
};
```

**`src/data/internal_model.hpp`**

```cpp
struct BCParamTable {
    /* 既有字段 */
    std::vector<CompiledExpression> pressure_p;            // 新增
};
```

### 模块增量

- `src/preprocessor/fluid_preprocessor.hpp` / `.cpp`（新）
    - `class FluidPreprocessor`
    - 公开方法 `void solve_flow(InternalModel& model) const`
    - 内部 inline 匿名 ns：pressure 系统装配（CSR 三元组 → 矩阵）

### 装配器增量

`src/assembler/assembler.cpp` 在面循环开头增加：

```cpp
if (auto it = fs_face_index.find(face_idx); it != fs_face_index.end()) {
    add_fluid_solid_interface(ops, face, it->h, ...);
    continue;   // 跳过扩散 + 边界项
}
if (model.material_table[c0.material_id].is_fluid ||
    model.material_table[c1.material_id].is_fluid) {
    add_convection_contribution(ops, face, model.fluid->face_velocity[face_idx],
                                 rho_face, cp_face, state.T);
}
// 既有：扩散 + cauchy
```

新加两个 helper：

```cpp
namespace {
    void add_fluid_solid_interface(AssemblyResult& ops, const FaceLoopContext& ctx, double h, ...);
    void add_convection_contribution(AssemblyResult& ops, const FaceLoopContext& ctx,
                                      const std::array<double, 3>& u_face,
                                      double rho_face, double cp_face,
                                      const std::vector<double>& T);
}
```

### Sidecar 格式 (`<name>_additional.xml`)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<FluidOverlay>
    <!-- material-level 流体声明 -->
    <FluidMaterial name="water_25C">
        <DynamicViscosity>0.00089</DynamicViscosity>   <!-- Pa·s -->
        <!-- is_fluid 默认为 true；可显式 false 用于"固体材料 + 流体属性"误用纠正 -->
    </FluidMaterial>

    <!-- pressure 边界条件（与主 XML 的 Boundary 同构） -->
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

| 条件                                                           | 行为                               |
|----------------------------------------------------------------|------------------------------------|
| sidecar 不存在                                                 | `LOG_INFO` 跳过                    |
| sidecar 解析失败                                               | `LOG_ERROR` + `std::runtime_error` |
| `<FluidMaterial name="X">` 的 X 不在主 XML materials           | `LOG_ERROR` + `std::runtime_error` |
| 同一 material 重复声明 `<FluidMaterial>`                       | `LOG_ERROR` + `std::runtime_error` |
| `<Boundary><PressureBoundary>` 的 face_key 命中 0 活跃流体单元 | `LOG_WARN` 不报错                  |

### 测试策略

| 套件                             | 目标                                                                                        |
|----------------------------------|---------------------------------------------------------------------------------------------|
| `test_fluid_preprocessor`        | 5×1×1 最小网格：pressure 沿 x 线性、hydroC 与手算 Poiseuille 误差 < 1e-9                    |
| `test_assembler_convection`      | 1×2×1 网格（流体+固体），K 矩阵对流贡献与手算一致                                           |
| `test_fluid_solid_interface`     | 1×2×1 网格（左流体/右固体），fs_face 自动检测、h 由 Nu 公式、K 矩阵对称                     |
| `test_fluid_sidecar_parsing`     | sidecar 存在/不存在/错误引用 三种路径                                                       |
| `test_fluid_pressure_bc_missing` | 调用 `solve_flow` 抛 `std::runtime_error`                                                   |
| `cases/microchannel_steady/`     | 5×1×3 通道，与 Python `experiment-v1/examples/example4` 稳态场比对                          |
| 回归                             | 现有 `simple_steady_tests` / `simple_transient_tests` / `nonlinear_steady_tests` 字节级一致 |

### 文件清单

新增：

- `src/data/fluid_model.hpp`
- `src/preprocessor/fluid_preprocessor.hpp`
- `src/preprocessor/fluid_preprocessor.cpp`
- `tests/test_fluid_preprocessor.cpp`
- `tests/test_assembler_convection.cpp`
- `tests/test_fluid_solid_interface.cpp`
- `tests/test_fluid_sidecar_parsing.cpp`
- `tests/test_fluid_pressure_bc_missing.cpp`
- `cases/microchannel_steady/case.xml`
- `cases/microchannel_steady/case_additional.xml`
- `cases/microchannel_steady/expected/`（Python 参考解，作为 fixture）
- `docs/adr/0006-fluid-solid-interface-override.md`
- `docs/design/fluid-solver.md`

修改：

- `src/data/internal_model.hpp`（加 `is_fluid` / `dynamic_viscosity` / `fluid` optional / `pressure_p`）
- `src/data/io_model.hpp`（加 `<PressureBoundary>` 多态 + `<BoundaryCategory>ThermalPressure>`）
- `src/io/io.cpp`（加 `read_additional_xml` + `<FluidMaterial>` / `<PressureBoundary>` 解析）
- `src/preprocessor/preprocessor.cpp`（load 末尾合并 sidecar、调用 `FluidPreprocessor::solve_flow`）
- `src/preprocessor/CMakeLists.txt`（新文件加进构建）
- `src/assembler/assembler.cpp`（面循环 + helper）
- `src/assembler/assembler.hpp`（如需暴露 helper 给单元测试）
- `CONTEXT.md`（术语表加 FluidMaterial / PressureBC / FluidSolidInterface / FluidFields）
- `tests/CMakeLists.txt`（注册新测试）
- `run_cases.py` / `run_tests.py`（如需新增 fixture 路径）

## Acceptance Criteria

1. `python run_tests.py` 全绿；新增 5 个流体测试 + 既有测试**零回归**
2. `python run_cases.py` 全绿；新增 `microchannel_steady` 与 Python 参考误差 < 1%（或更严，按 fixture 定）
3. 现有 `simple_steady_tests/case1..4` 在 dev 分支上的输出与实现**字节级一致**（验证"无 sidecar 路径"完全不变）
4. 文档：
   - `CONTEXT.md` 术语表更新
   - `docs/adr/0006-fluid-solid-interface-override.md` 存在
   - `docs/design/fluid-solver.md` 存在
5. `cmake -G "Ninja" -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel` 无 warning（项目零 warning 约定）

## Open Questions

无。所有设计岔路已确认。详见 `DECISIONS.md`。

## Out of Scope

- GUI / XML schema 改动
- 流体网格与热网格不同分辨率
- 多相流
- 可压缩流
- 化学反应 / 燃烧
- 微通道以外的冷却拓扑（jet impingement、喷雾冷却等）
