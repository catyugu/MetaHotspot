# Fluid Solver Integration — Decision Log

> 由 `/grill-with-docs` 流程沉淀。本文件先于 PRD 冻结，后续如有新增决策会追加到末尾并保留修改时间。

**状态**：Draft · 决策日期 2026-06-15 · 范围：dev 分支 C++ 实现

---

## 边界（Q1.1）

- **可改**：dev 分支 C++ 代码、测试、case、CONTEXT.md / ADR / `docs/`、`run_cases.py` / `run_tests.py`、CLI 入口
- **不可改**：外部 GUI（生成 `*.xml` 配置的可视化工具）的字段集与输出 schema
- GUI 短期不感知流体的存在

## Sidecar（Q1.2、Q1.9.a、Q1.14）

- 元数据不进主 XML
- 同目录放 `<name>_additional.xml`（与主 XML 同源，tinyxml2 解析）
- `Preprocessor::load` 末尾读 sidecar 合并进 `InternalModel`；I/O 层完全不知道流体存在
- 错误策略：
    - 文件不存在 → `LOG_INFO` 跳过
    - 解析失败 / 引用了主 XML 不存在的 material → `LOG_ERROR` 抛 `std::runtime_error`
    - pressure BC 命中 0 个活跃单元 → `LOG_WARN` 不报错

## Material 注入（Q1.3、Q1.13）

- `MaterialProps` 新增两字段（默认 false / `"0.0"` 表达式）：
    - `bool is_fluid = false;`
    - `CompiledExpression dynamic_viscosity;`（μ，Pa·s）
- 仅在 sidecar 里被声明时覆盖
- μ **取参考温度 `initial_temperature` 求值一次**，得到标量 `mu_ref` 写入 `FluidFields`（不在装配阶段重算）
- 流体**保留** `k_fluid` 写扩散项（即流体走"扩散 + 体积对流"双贡献路径，不走"无扩散"特殊路径）
- material-level 注入：sidecar 写 `material_name → {is_fluid, dynamic_viscosity, mu_ref, fluid_heat_source}`，不写 per-Block
- 同一 `material_name` 重复声明 / 与既有非流体语义冲突 → `LOG_ERROR` 抛异常
- 流体单元**保留** Block 的 `ti_reyuan_expr`（Q1.12 选 B），按字面意义进入 `heat_source_table`

## 求解顺序（Q1.4）

- 流体解算**前置一次**（Poiseuille 解析解，稳态、不可压、μ 常数）
- 稳态 / 瞬态共用：t=0 前跑一次，整段瞬态复用同一 `FluidFields`
- 钩子位置：`Preprocessor::load` 末尾、`Scheduler::run` 之前
- `Scheduler` 不感知流体是否存在

## 对流项离散（Q1.5、Q1.17、Q1.18）

- 体积对流项在装配器面循环**新增** `convection_contribution(...)`
- `ρ_face = 0.5·(ρ_c0 + ρ_c1)`，`cp_face` 同理（与 k_face 平均风格一致）
- **一阶 upwind** 迎风：`T_upwind` 取上游单元温度
- 稳态用 `state.T` 旧值；瞬态用 `state.history.latest()`（与现有 mass 项处理一致）
- 沿用 `DIR_DX/DY/DZ` 遍历（结构化网格等价于 argmax）
- `fluid.face_velocity[internal_face_idx][axis]` 在 `fluid_preprocessor.cpp` 末尾扫一遍算好（`u = -hydroC_avg · ∇p / dist`）
- 体积对流项**对所有面**调用（c0、c1 均非流体时该项天然为 0；不写 if 分支），代码路径单一
- 每次 Newton 迭代**不**重算对流项（Poiseuille 假设下解不变）

## 流-固边界（Q1.6、Q1.7、Q1.18.b）

- 流-固边界**不**在配置层做 BC（sidecar 也不声明）
- preprocessor 扫 `c0.is_fluid XOR c1.is_fluid` 检测流-固面，存到 `FluidFields.fs_faces`
- 每条 fs_face 算：`h = Nu · k_fluid / D_h`，其中 Nu 用矩形经验式
- 装配器在面循环开头查 `fs_faces` 索引，命中则**只**应用 h·(T_f - T_s) 项，`continue` 跳过扩散 / cauchy
- 流-固面**覆盖**该面在热装配里的一切其它贡献（用户就算写了 cauchy 也忽略）
- `hydroC` 公式与 Python 对齐（矩形 + 0.63 修正项）
- 矩形 Nu 默认式：`Nu = 7.541·(1 - 2.610/α + 4.970/α² - 5.119/α³ + 2.702/α⁴ - 0.548/α⁵)`，α ≥ 1；α<1 取 1/α；α→∞ 退化为 7.54（恒热流）

## Pressure BC（Q1.10）

- sidecar 用 `<BoundaryCategory>ThermalPressure</BoundaryCategory>` + `<PressureBoundary><Pressure>…</Pressure></PressureBoundary>`
- 与现有 `<ThermalBoundary i:type="…">` 多态结构对称
- `face_key` 体系复用现有 `Face|Direction|CoordValue|RectList`
- 落入 `bc_params.pressure_p`（新增字段），preprocessor 末端用 `face_key_processor` 命中
- pressure 方程对 boundary 流体单元：b 行置 1、对角贡献 1，直接给 p（与 Python 一致）

## FluidFields 容器（Q1.8、Q1.15）

- 新文件 `src/data/fluid_model.hpp`，struct `mhs::core::FluidFields`
- `InternalModel` 新增 `std::optional<FluidFields> fluid;`（空 = 无流体）
- `CellFields` 不动；流体与否在装配阶段读 `material_table[mat_id].is_fluid` 判定
- 索引层级：`old_idx → c_idx (active) → f_idx (fluid)`
- `fluid_ids` 存 `std::vector<int>`（内容是 c_idx），长度 n_fluid
- `g2f` 用 `std::vector<int>` 长度 N_active，默认 -1

## 代码归属（Q1.9、Q1.11）

- 新模块 `src/preprocessor/fluid_preprocessor.{hpp,cpp}`，类 `mhs::sim::FluidPreprocessor`
- pressure 系统装配 inline 在 `fluid_preprocessor.cpp` 匿名 namespace（不单列文件）
- pressure 线性系统**默认走 `SparseLUSolver`**（SPD + 一次性，与现有 dev 接口一致）
- `BCParamTable` 新增 `std::vector<CompiledExpression> pressure_p;`

## 测试 / 验证（Q1.19）

三件套全做，按 red → green → refactor：

1. 单元测试：
   - `test_fluid_preprocessor`：最小网格（5×1×1），pressure 沿 x 线性、hydroC 与手算 Poiseuille 误差 < 1e-9
   - `test_assembler_convection`：K 矩阵中流体/邻居面的对流贡献与手算一致
   - `test_fluid_solid_interface`：1×2×1 网格，fs_face 被检测、h 由 Nu 公式算对、K 矩阵对称
2. 集成 case：`cases/microchannel_steady/`（5×1×3 通道），与 `experiment-v1/examples/example4` 参考稳态场比对
3. 回归：`simple_steady_tests / simple_transient_tests / nonlinear_steady_tests` 全跑，确认"无 sidecar / 全 is_fluid=false" 行为**字节级不变**
4. 边界 case：pressure BC 缺失（且无其它压力约束）→ 调用抛 `std::runtime_error`

---

## 待解决 / 显式不做

- 不支持 inflow/outflow velocity profile BC（与 Poiseuille 假设不一致）
- 流体单元 D_h 必须非退化（h × w > 0），否则 `LOG_ERROR` 退出
- μ 不随 T 变（恒定参考温度求值）
- 粘性耗散项**不**实现（Poiseuille 假设下不可压、恒温）
- 不为流体新建求解器抽象（复用 `mhs::sim::LinearSolver`）
- 不在 `Rect` / `Block` 加流体标志（material 单一来源）
- 体积对流项格式限定一阶 upwind（central / TVD 留待后续）
- CLI 不增加新 flag（sidecar 路径从主 XML 派生）
