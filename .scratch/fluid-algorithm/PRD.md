# PRD: Fluid-Conjugate Heat Transfer Algorithm Integration

**Feature slug**: `fluid-algorithm`
**Status**: needs-triage → ready-for-agent (after review)
**Author**: catyugu + Claude
**Date**: 2026-06-19

---

## 1. Objective

将 experiment-v1 Python 分支中的微流体耦合传热算法移植到当前 C++ 代码库,使得:

- 稳态求解器在常规传热路径之前先求解流体压力场
- Assembler 在组装传热矩阵时,对流体-固体交界面的导热系数和流体内对流项做修正
- 验证案例 `steady_case1.xml` + `steady_case1_additional.xml` 的结果物理合理(约 300K+,比纯固体参考解略低)

**不做**: 数值与参考解严格一致(参考解本身不考虑流体)、加入 case 列表、瞬态流体支持(本期仅稳态)。

---

## 2. Background & Reference

### 2.1 Python 参考实现 (experiment-v1)

求解流程:

```text
Mesher → PhysicalFields (is_fluid, hydroC, pressure, ...)
FluidPreprocessor.solve_flow():
  1. init_cell_hydro_properties: 每流体 cell 每轴计算 hydroC = hydraulic conductance
     - 矩形截面修正: (1 − 0.63·AR)·w³·h / (12·μ·L)  (AR < 1)
     - 正方形: 0.42229·h⁴ / (12·μ·L)
  2. apply_pressure_bc: 标记压力边界 cell,赋值 pressure
  3. solve_pressure: 组装 Poisson 矩阵 [对角 −ΣC_eff, off-diag +C_eff],
     求解得各流体 cell 的 pressure
FVMAssembler._precompute_flow_axes():
  - 对每个流体 cell,argmax(per-axis |Δp|) 得主导流方向
FVMAssembler._build_conduction_matrix():
  - fluid-solid 交界面: 算 Nusselt 数 → 内部对流换热系数 h_f → 串联热阻
  - 纯 solid-solid / fluid-fluid: 标准串联热阻
FVMAssembler._build_advection_matrix():
  - mass_flux = ΔP · C_eff · ρ_avg
  - 上风格式(upwind): |mass_flux|·cp · (T_up → T_dn)
  - 出口边界温度注入 RHS
```

### 2.2 验证案例

| 文件                          | 内容                                                                                                                                                                                                     |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `steady_case1.xml`            | 8×4×0.7mm, 3 层: 层1 silicon(0.3mm), 层2 含 silicon(块1,块3)+water(块2,0.2mm,热源 2e9 W/m³), 层3 silicon(0.2mm); BC: 底面 ThirdType h=100 T=300K, 顶面 ThirdType h=20 T=320K, 右侧面 FirstType T=298.15K |
| `steady_case1_additional.xml` | Fluid overlay: water μ=8.9e-4 Pa·s; inlet P=500 at `X\|E\|0` (两个水通道入口), outlet P=0 at `X\|E\|8`                                                                                                   |

### 2.3 物理预期

加入流体冷却后,水通道带走热量,最高温度应比纯固体解(~343K)略低,约 300K+。通道入口处 298.15K,出口处略升温。

---

## 3. Design Decisions (confirmed)

| #   | 决策           | 选择                                      | 理由                                                                      |
| --- | -------------- | ----------------------------------------- | ------------------------------------------------------------------------- |
| D1  | 流体配置输入   | `std::optional<FluidOverlay>` + 独立解析  | IOStructure 不被 PTA 不认识的字段污染;overlay 是补充层                    |
| D2  | 耦合方式       | 稳态单次前向: pressure → 整个 T 场 (单向) | 与 Python 参考一致;最简实施;本期仅稳态                                    |
| D3  | 压力求解器     | 复用现有 SparseLU / Pardiso (直接法)      | 流体子域稀疏,直接法稳定;与 Python scipy.spsolve 一致                      |
| D4  | flow_axes 存储 | `std::vector<int8_t>` 沿用 Python 语义    | 每流体 cell [0,2] 选主导轴;与 Python 1:1 映射;便于 TBB 并行               |
| D5  | 流体 BC 独立   | `FluidBCType` 枚举,独立于 `BcType`        | 同面可同时附加热 BC 和流体 BC;不互相影响;`BoundaryCategory::Fluidic` 路由 |

---

## 4. Data Model Extensions

### 4.1 IO 层新增

**文件**: `src/data/io_model.hpp`

```cpp
namespace mhs::core {

// 流体材料属性(overlay 补充 PTA 不支持的字段)
struct FluidMaterial {
    std::string name;           // 与 IOStructure::materials 中已有材料同名
    std::string dynamic_viscosity; // 动力粘度 μ [Pa·s], 表达式字符串
};

// 压力边界条件
struct PressureBoundary {
    double pressure;            // 压力值 [Pa]
};

// 流体边界
struct FluidBoundary {
    std::string name;
    std::vector<std::string> face_keys; // 同格式: X|E|8|...
    PressureBoundary pressure_bc;
};

// 流体 overlay: 从 additional XML 解析,不侵入 IOStructure
struct FluidOverlay {
    std::vector<FluidMaterial> fluid_materials;
    std::vector<FluidBoundary> boundaries;
};

} // namespace mhs::core
```

**新增独立读取函数** (不修改 `read_xml`):

**文件**: `src/io/io.hpp` / `src/io/io.cpp`

```cpp
namespace mhs::io {
    // 现有接口不变
    mhs::core::IOStructure read_xml(const std::string& xml_path);

    // 新增: 读取流体 overlay XML; 无 overlay 文件时返回 std::nullopt
    std::optional<mhs::core::FluidOverlay> read_fluid_overlay_xml(const std::string& xml_path);
}
```

### 4.2 InternalModel 扩展

**文件**: `src/data/internal_model.hpp`

```cpp
struct InternalModel {
    // ... existing fields unchanged ...

    // --- 流体扩展 (本期新增) ---
    std::vector<uint8_t> is_fluid;          // [N_active] 标记流体 cell
    std::vector<double> dynamic_viscosity;  // [N_active] 流体 cell 的 μ; 非 fluid = 0
    std::vector<double> pressure;           // [N_active] 求解后的压力场; 非 fluid = 0
    std::vector<int8_t> flow_axes;          // [N_active] 主导流轴 [0=X,1=Y,2=Z]; 非 fluid = -1
    std::vector<double> hydroC_x;           // [N_active] hydraulic conductance 沿 X; 非 fluid = 0
    std::vector<double> hydroC_y;           // [N_active] ... 沿 Y
    std::vector<double> hydroC_z;           // [N_active] ... 沿 Z

    // 压力边界标记 & 压力值 (preprocessor 填入)
    std::vector<uint8_t> is_pressure_boundary; // [N_active]
    std::vector<double> boundary_pressure;  // [N_active] (仅 is_pressure_boundary=true 有意义)

    // 流体入口温度 (用于 advection RHS 出口注入; 从 FirstType BC 继承)
    std::vector<double> boundary_temperature_fluid; // [N_active], 非流体入口 = NaN
};
```

### 4.3 CellFields 扩展

```cpp
struct CellFields {
    // ... existing fields unchanged ...
    std::vector<uint16_t> fluid_material_id; // 新增: 流体材料索引; max() 时非 fluid
};
```

### 4.4 MaterialProps 扩展

```cpp
struct MaterialProps {
    CompiledExpression kx, ky, kz, rho, c;
    bool is_fluid = false;                // 新增
    CompiledExpression dynamic_viscosity;  // 新增: μ; 非 fluid = make_constant(0)
};
```

### 4.5 BCParamTable 扩展

```cpp
struct BCParamTable {
    // ... existing fields unchanged ...
    // 新增: 压力边界参数(不需要表达式,直接 double)
    std::vector<double> pressure_bc_values; // index by PressureBC idx
};
```

**types.hpp 新增**:

```cpp
// 流体边界条件类型。与 BcType 独立，同一个面可同时有热 BC 和流体 BC。
enum class FluidBCType : uint8_t {
    None = 0, PressureType = 1  // 新增: 压力 BC
};
```

---

## 5. Algorithm Implementation

### 5.1 FluidPreprocessor (新模块)

**文件**: `src/fluid/fluid_preprocessor.hpp` / `.cpp`
**命名空间**: `mhs::sim`

职责: 从 overlay 数据计算 hydroC、标记压力边界、求解压力场、计算 flow_axes。

```cpp
namespace mhs::sim {

class FluidPreprocessor {
public:
    /// 若 InternalModel 无流体 cell (is_fluid 全 false), 直接返回; 不改 model。
    /// 否则:
    ///   1. init_cell_hydro_properties → 填 hydroC_x/y/z
    ///   2. apply_pressure_boundary_conditions → 填 is_pressure_boundary + boundary_pressure
    ///   3. solve_pressure → 填 pressure (Poisson 矩阵 + SparseLU 求解)
    ///   4. precompute_flow_axes → 填 flow_axes
    void solve_flow(mhs::core::InternalModel& model);
private:
    void init_cell_hydro_properties(mhs::core::InternalModel& model);
    void apply_pressure_boundary_conditions(mhs::core::InternalModel& model);
    void solve_pressure(mhs::core::InternalModel& model);
    void precompute_flow_axes(mhs::core::InternalModel& model);
};

} // namespace mhs::sim
```

**核心算法细节**:

#### 5.1.1 hydroC 计算 (矩形截面 Hele-Shaw 修正)

对每个流体 cell,沿 axis (0=X, 1=Y, 2=Z):

```text
ax_w = (axis+1)%3, ax_h = (axis+2)%3
L  = cell_size[axis]   (沿流方向的长度)
w  = cell_size[ax_w]   (截面宽度)
h  = cell_size[ax_h]   (截面高度)
μ  = dynamic_viscosity[cell]

AR = min(w,h)/max(w,h)

if |h − w| < 1e-10 (正方形):
    hydroC = 0.42229 · h⁴ / (12 · μ · L)
elif h > w:
    hydroC = (1 − 0.63·AR) · w³ · h / (12 · μ · L)
elif w > h:
    hydroC = (1 − 0.63·AR) · h³ · w / (12 · μ · L)
```

#### 5.1.2 压力 Poisson 求解

流体子域索引: `fluid_ids = {compact_idx | is_fluid[compact_idx]}`, 映射 `g2f[compact_idx] → fluid_local_idx` (其余 = −1)。

对每个 internal face (c_a, c_b, axis, area):

- 两端都是流体: C_eff = harmonic mean(hydroC_c_a[axis], hydroC_c_b[axis])
    - 若 c_a 不是压力边界: off-diag (g2f[c_a], g2f[c_b]) += C_eff; diag(g2f[c_a]) += C_eff
    - 若 c_b 不是压力边界: off-diag (g2f[c_b], g2f[c_a]) += C_eff; diag(g2f[c_b]) += C_eff
- 一端流体一端固体: face 两侧压力耦合忽略(固体侧不参与压力方程) → 该 face 对压力矩阵贡献为 0

对压力边界 cell: diag = 1, RHS = boundary_pressure[cell]。

对非边界流体 cell: diag = −Σ C_eff(所有邻居)。

**组装**为 Eigen SparseMatrix, 用现有 `SparseLUSolver::solve()` 求解,结果写回 `model.pressure`。

#### 5.1.3 flow_axes 计算

```text
对每个流体 cell c:
  per_axis_pressure_drop[axis] = max(|Δp| across neighbors on axis)
    (遍历所有 internal face,两端都是流体才计)
  flow_axes[c] = argmax(per_axis_pressure_drop)
```

### 5.2 Assembler 修改

**文件**: `src/assembler/assembler.hpp` / `.cpp`

**原则**: 原有纯固体路径完全不变;流体相关逻辑仅在 `is_fluid[c]` 或 `is_fluid[n]` 为 true 时激活。

#### 5.2.1 导热矩阵: 流体-固体交界面修正

当 `BcType::None` 分支中遇到流体-固体交界面:

```text
if is_fluid[c_idx] != is_fluid[n_idx]:
    f_id  = 流体侧 compact idx
    s_id  = 固体侧 compact idx
    f_ax  = flow_axes[f_id]
    ax_w  = (f_ax+1)%3, ax_h = (f_ax+2)%3
    w     = dims[f_id][ax_w]     (cell 尺寸)
    h     = dims[f_id][ax_h]
    AR    = min(w,h)/max(w,h)
    Nu    = Nusselt(AR)           // 8.235·(1 − 2.0421·AR + 3.0853·AR² − ...)
    d_h   = 2·w·h/(w+h)          // hydraulic diameter
    h_f   = Nu·k_fluid / d_h     // 内部对流换热系数
    R     = half_dist_solid / (k_solid · A_f) + 1 / (h_f · A_f)
    cond  = 1 / R
```

其余(solid-solid, fluid-fluid)保持标准串联热阻。

#### 5.2.2 对流矩阵: 上风 advection

新增 advection 组装阶段,仅在流体子域内部 face 上计算:

```text
对 internal face (c_a, c_b, axis, area):
  if !is_fluid[c_a] or !is_fluid[c_b]: skip

  C_eff  = harmonic_mean(hydroC[c_a][axis], hydroC[c_b][axis])
  ρ_avg  = (density[c_a] + density[c_b]) / 2
  mass_flux = (pressure[c_a] − pressure[c_b]) · C_eff · ρ_avg

  // net_outflux 累加: 用于后续边界温度注入
  net_outflux[c_a] += mass_flux
  net_outflux[c_b] -= mass_flux

  if |mass_flux| > tol:
    up = (mass_flux > 0) ? c_a : c_b
    dn = (mass_flux < 0) ? c_a : c_b
    adv = |mass_flux| · cp[up]
    // T_up − T_dn 的上风格式:
    //   row dn, col up: +adv   (dn 从 up 取热)
    //   row up, col up: −adv   (up 失热)
```

**出口边界温度注入 (RHS)**:

```text
对每个流体 cell c:
  if net_outflux[c] > 0 (净流入):
    T_boundary = boundary_temperature_fluid[c]
    if !isnan(T_boundary):
      rhs[c] += net_outflux[c] · cp[c] · T_boundary
  elif net_outflux[c] < 0 (净流出):
    // 对流出 cell 的对角项追加: net_outflux[c] · cp[c]
    // (上风格式中 outflow cell 自己贡献自对角)
```

#### 5.2.3 AssemblyResult 扩展

```cpp
struct AssemblyResult {
    Eigen::SparseMatrix<double> K;     // 导热 + 流体-固体对流 + advection 对角
    Eigen::VectorXd f;                 // 热源 + BC RHS + advection RHS
    Eigen::VectorXd M_diag;            // 不变
};
```

K 和 f 的内容自然扩充;**不改 AssemblyResult 的结构签名** — advection 的贡献合并入 K 和 f,不新增独立字段。Python 中的 `A_total = A_cond + A_bc + A_adv`, `b_total = b_bc + b_adv` 也合并到一个矩阵和一个 RHS。

### 5.3 Scheduler 修改

**文件**: `src/scheduler/scheduler.cpp`

在稳态分支,组装前插入流体求解:

```cpp
if (model_->study_type == mhs::core::StudyType::Steady) {
    // 新增: 流体压力求解(若模型含流体)
    mhs::sim::FluidPreprocessor fluid_prep;
    fluid_prep.solve_flow(*model_);

    // 后续完全不变
    Assembler assembler(*model_);
    ...
}
```

**本期不做**: 瞬态流体支持(但内部 model 字段已预留,未来每步前调用 `solve_flow` 即可)。

### 5.4 Preprocessor 修改

**文件**: `src/preprocessor/preprocessor.cpp`

`Preprocessor::load()` 末尾,新增 overlay 合入:

```text
// 在现有 resolve_layers 完成后:
if (fluid_overlay.has_value()) {
    // 1. 标记 is_fluid: 遍历 material_table, FluidMaterial.name 匹配 → is_fluid=true
    // 2. 填 dynamic_viscosity: 化简 overlay 的表达式 → CompiledExpression → eval 到 double
    // 3. 解析 PressureBoundary face_keys → 标记 is_pressure_boundary + boundary_pressure
    // 4. 填 fluid_material_id
    // 5. 初始化 model.is_fluid, model.pressure, model.flow_axes, model.hydroC_x/y/z (全零)
}
```

---

## 6. Nusselt Number Computation

矩形截面内部流动的 Nusselt 数(Python 参考中的 `compute_nusselt_kernel`):

```text
AR = w/h (w ≤ h)
Nu = 8.235 · (1 − 2.0421·AR + 3.0853·AR² − 2.4765·AR³ + 1.0578·AR⁴ − 0.1861·AR⁵)
```

这是 Shah & London 对矩形截面充分发展层流的理论结果,适用于微通道尺寸下的假设(Re < 2000)。

**实现**: 独立函数 `mhs::utils::nusselt_rectangular(double w, double h)` 在 `src/common/physics_utils.hpp` 中。

---

## 7. File Layout

```text
src/
├── fluid/                          # 新模块
│   ├── fluid_preprocessor.hpp      # FluidPreprocessor 类声明
│   └── fluid_preprocessor.cpp      # hydroC, pressure solve, flow_axes
├── data/
│   ├── types.hpp                   # FluidBCType::PressureType 新增
│   ├── internal_model.hpp          # is_fluid, pressure, hydroC, flow_axes, ...
│   ├── io_model.hpp                # FluidOverlay, FluidMaterial, FluidBoundary
│   ├── linear_system.hpp           # 不变
│   └── solution_history.hpp        # 不变
├── assembler/
│   ├── assembler.hpp               # 不变(签名不变;内部实现修改)
│   ├── assembler.cpp               # 5.2 的流体-固体 + advection 逻辑
├── common/
│   ├── mesh_utils.hpp              # 保留; 核心几何表
│   ├── physics_utils.hpp           # nusselt_rectangular() 新增
├── io/
│   ├── io.hpp                      # read_fluid_overlay_xml() 新增
│   ├── io.cpp                      # overlay XML 解析实现
├── preprocessor/
│   ├── preprocessor.hpp            # 签名不变(内部修改)
│   ├── preprocessor.cpp            # overlay 合入 + is_fluid 标记
│   ├── face_key_processor.hpp/cpp  # PressureType face key 解析新增
├── scheduler/
│   ├── scheduler.cpp               # 稳态分支插入 FluidPreprocessor.solve_flow()

tests/
├── test_fluid_preprocessor.cpp     # 新增: hydroC 计算, Poisson 求解, flow_axes
├── test_assembler.cpp              # 扩展: 流体-固体交界面导热, advection

cases/microfluid_cases/             # 不加入 run_cases.py 列表
├── steady_case1.xml                # 已有
├── steady_case1_additional.xml     # 已有
```

**命名空间**: `mhs::sim` (FluidPreprocessor 与现有 sim 模块同级)。`mhs::core` (FluidOverlay 等数据类型)。`mhs::utils` (Nusselt 计算)。

---

## 8. Test Plan

### 8.1 单元测试 (TDD, red → green → refactor)

| 测试                                | 优先级 | 验证                                                                      |
| ----------------------------------- | ------ | ------------------------------------------------------------------------- |
| `test_nusselt_rectangular`          | P0     | 正方形 AR=1 → Nu≈8.235; 极窄 AR→0 → Nu≈8.235; 典型 AR=0.5 → Nu≈4.5(查表)  |
| `test_hydroC_single_axis`           | P0     | 正方形截面 water μ=8.9e-4, 0.5×0.5×0.2mm → hydroC 值与 Python 一致        |
| `test_pressure_solve_simple`        | P0     | 2-cell 流体串行(1D), inlet P=500, outlet P=0 → pressure = [500, 0] (精度) |
| `test_pressure_solve_2d_channel`    | P1     | 稳态案例的流体子域 → pressure 场全正, outlet 处 ≈0                        |
| `test_flow_axes_dominant_x`         | P0     | ΔP 仅沿 X → flow_axes 全为 0                                              |
| `test_fluid_solid_interface_cond`   | P0     | 单对流面: cond = 1/(R_solid + R_fluid) 与手动计算一致                     |
| `test_advection_upwind_single_face` | P0     | 单 face: mass_flux > 0 → upwind T_up → T_dn 矩阵项正确                    |
| `test_full_steady_case1_fluid`      | P1     | steady_case1 + overlay → T_max < 纯固体解(~343K), T_min ≈ 298.15K         |

### 8.2 集成验证

运行:

```bash
cmake --build build --parallel
python run_tests.py  # 确保所有现有测试仍通过

# 手动验证流体案例:
bin/metahotspot --case cases/microfluid_cases/steady_case1.xml \
    --fluid-overlay cases/microfluid_cases/steady_case1_additional.xml
```

**预期物理合理性检查**:

- 最高温度 < 343K (比纯固体解低)
- 流体入口附近 ≈ 298.15K
- 整体 T 在 300K~340K 范围
- 无负温度、无 NaN

---

## 9. CLI Integration

**文件**: `bin/main.cpp` (当前入口)

新增 `--fluid-overlay` 参数:

```text
Usage: metahotspot --case <path> [--fluid-overlay <path>]

--case           现有参数,XML 案例文件
--fluid-overlay  可选,FluidOverlay XML 文件;无此参数时纯固体求解
```

**实现**:

```cpp
// main.cpp
std::optional<std::string> fluid_overlay_path;
if (args.contains("--fluid-overlay")) {
    fluid_overlay_path = args["--fluid-overlay"];
}

auto io = mhs::io::read_xml(case_path);
auto fluid_overlay = fluid_overlay_path
    ? mhs::io::read_fluid_overlay_xml(*fluid_overlay_path)
    : std::nullopt;

auto model = preprocessor.load(io);
if (fluid_overlay) {
    // overlay 合入已在 Preprocessor::load 内处理
    // (或新增 Preprocessor::apply_fluid_overlay(model, *fluid_overlay))
}
```

---

## 10. Implementation Phases ( tracer-bullet slices )

Issues 已拆分为 6 个垂直切片,存放于 `.scratch/fluid-algorithm/issues/`。

| #   | Issue                                                                                                                   | 类型 | 依赖   |
| --- | ----------------------------------------------------------------------------------------------------------------------- | ---- | ------ |
| 01  | [数据骨架 + Nusselt 函数](issues/01-data-skeleton-and-nusselt.md)                                                       | AFK  | 无     |
| 02  | [Overlay XML 解析 + Preprocessor 合入](issues/02-overlay-io-and-preprocessor-merge.md)                                  | AFK  | 01     |
| 03  | [FluidPreprocessor 求解压力场 + Scheduler 接入](issues/03-fluid-pressure-solver-and-scheduler-integration.md)           | AFK  | 01, 02 |
| 04  | [Assembler 流体-固体交界面导热修正](issues/04-assembler-fluid-solid-interface-conduction.md)                            | AFK  | 03     |
| 05  | [Assembler advection 上风组装 + 出口温度注入](issues/05-assembler-advection-upwind-and-outlet-temperature-injection.md) | AFK  | 03, 04 |
| 06  | [CLI --fluid-overlay + 集成物理合理性验证](issues/06-cli-fluid-overlay-and-integration-validation.md)                   | HITL | 05     |

**依赖图**:

```text
01 ─→ 02 ─→ 03 ─→ 04 ─→ 05 ─→ 06 (HITL)
```

### Phase 1: 数据骨架 + Nusselt 函数 (无行为变更)

对应 Issue 01。

### Phase 2: Overlay 读取 + 预处理合入

对应 Issue 02。

### Phase 3: FluidPreprocessor 核心 (hydroC + pressure + flow_axes)

对应 Issue 03。

### Phase 4: Assembler 流体-固体导热 + 对流

对应 Issue 04 (导热修正) + Issue 05 (advection)。

### Phase 5: CLI + 集成验证

对应 Issue 06。

---

## 11. Out of Scope (本期不做)

- 瞬态流体 (每步刷新 pressure / flow_axes)
- 流体物性随温度变化 (μ(T), ρ(T), k(T))
- Navier-Stokes 全求解 (本期仅 Hele-Shaw / Darcy 类 Poisson)
- 加入 run_cases.py 列表
- 数值与 Python 参考解严格对齐

---

## 12. ADR Considerations

**ADR-0005 (Cell-Level BC)**: `CellBC` 仅存储热 BC (`BcType`)。流体 BC (`FluidBCType`) 独立存储于 `is_pressure_boundary` / `boundary_pressure` 向量，不侵入 `CellBC`。

**ADR-0002 (Cell-Centered DOF)**: 压力场也是 cell-centered DOF,与温度场共用同一网格拓扑,无冲突。

**ADR-0004 (Expression Split)**: 动力粘度目前是 constant(double),未来若需要 μ(T) 则复用 CompiledExpression。

**潜在新 ADR**: "ADR-0006: Hele-Shaw Pressure Poisson for Microfluid" — 记录为什么本期用 Poisson 而非 Navier-Stokes,以及 flow_axes 的存储决策。建议在 Phase 3 完成后撰写。

---

## 13. Risks & Mitigations

| 风险                              | 影响             | 缓解                                                   |
| --------------------------------- | ---------------- | ------------------------------------------------------ |
| 网格 index_map 与流体子域映射复杂 | 压力矩阵组装错误 | 用 `g2f` 映射严格与 Python 一致;单元测试覆盖           |
| 流体子域太小 → Poisson 矩阵奇异   | 求解失败         | 检查边界条件覆盖所有流体边界;降级为无流体              |
| Assembler 流体分支影响 TBB 并行   | 性能回退         | 流体 cell 少时分支开销忽略;路径完全分叉无 shared state |
| overlay XML 格式不稳定            | IO 解析崩溃      | 严格校验;缺 overlay 时 graceful skip                   |
