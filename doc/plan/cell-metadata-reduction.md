# CellFields 契约约简计划（DRY）

## 0. 计划状态

- 状态：已固定，待实施
- 类型：约简重构（DRY）——在已完成的 cell-metadata 迁移（`cell-metadata-migration.md`，已删除）基础上，消除新契约引入的冗余与重复
- 范围：C++ API copy-out、model_compiler、Python `CellFields` 视图、ROM 基础设施（`AffineParametricModel`）、playground 模型
- 纪律：不改变任何数值语义、不改变 Compact Cell Order、不改变公开契约字段集合、不修改 solver/assembler
- 本计划不包含代码实现细节之外的科学算法改动

## 1. 评估结论（refactor/cell-metadata 分支，0a18d97..HEAD）

分支共 3 个提交（`0cc9c16` 迁移计划文档、`88b49b1` 暴露 Cell 字段、`6e7a86e` 统一 copy-out），基线验证：

- `python run_tests.py`：94/94 通过
- `python run_cases.py`：全部案例在阈值内
- `playground/bci_rom_testcase1/reproduce_case1.py`：关键 steady/transient 结果（见下文 Phase 3 基线记录）

已完成的迁移（不再重复）：

1. 单一 typed C API copy-out `mhs_compiled_copy_cell_fields` 取代 3 个零散 copy API（`grid_to_cell` / `layer_ids` / `block_ids`）
2. C++ `CellFields` 新增参考状态物性字段（`conductivity_x/y/z`、`density`、`specific_heat`），compiler 为唯一事实源
3. Python `Compiled.cells` 统一入口；派生属性 `centers` / `half_sizes` / `volumes` / `ijk` / `x/y/z_vertices`
4. ROM `cell_layout` 直接消费 `compiled.cells`，删除 `_axis_vertices` / `_physical_stack` / `_layer_conductivity` 全套 hook

遗留冗余清单（本计划要消除的，R 编号）：

| # | 位置 | 冗余 | 约简手段 |
| - | ---- | ---- | -------- |
| R1 | `src/api/metahotspot.cpp` `mhs_compiled_copy_cell_fields` | 16 段完全相同的 `copy_vector(...); if (status != MHS_OK) return status;` | 表驱动：`std::function` 列表 + 单循环 |
| R2 | `src/compiler/model_compiler.cpp:250-253` 与 `tests/test_preprocessor.cpp:156-158` | grid→(ix,iy,iz) 解码逻辑重复两份 | `mhs::core::decode_grid_index()` 头文件内联 helper，两处共用 |
| R3 | C API struct `heat_source_id`、copy 标签、`types.py`、`compiled.py` vs C++ 核心 `heat_source_idx` | 同一字段跨语言命名漂移 | 统一为 `heat_source_idx`（C++ 核心既有名，见 ADR 0002/0004、assembler.cpp） |
| R4 | `python/metahotspot/compiled.py` `_fetch_metadata`（约 65 行） | 22 个字段被声明 3 次：ctypes struct / 分配 + `data_as` 接线 / `CellFields` 构造 | 单一 `_CELL_FIELD_SPECS` 字段表驱动分配、指针接线与构造 |
| R5 | `CellFields` 派生属性（`centers`/`cell_sizes`/`half_sizes`/`volumes`/`ijk`） | 每次访问都重新解码 `_ijk()`（`cell_layout` 一次要解码 2 次） | `__post_init__` 一次性解码并缓存 |
| R6 | `CellFields.exposed_face_mask`（compiled.py:66-84） | 纯 Python 双重循环（N×6），且与 `surface_exposed_cells` 的邻居判定逻辑重复 | 占用网格 padding + NumPy 向量化邻居比较；成为暴露面判定的唯一事实源 |
| R7 | `_interfaces.py` `surface_exposed_cells`（122-206 行）与 3 个模型的 `boundary_groups` | 邻居判定/面积公式与 R6 重复；模型各自重复 `grid_to_cell.reshape` + 顶点读取；`_chiplet_stack.py:435` 冗余 `np.asarray`；`model_case1.py:327` 用 config 高度而非编译事实 | `surface_exposed_cells` 改写为消费 `CellFields`（mask + ijk + cell_sizes），签名收敛为 `(cells, face, coord, z_range)`；模型侧删除重复局部变量；顶部坐标改用编译 `z[-1]` |
| R8 | `_interfaces.py` `physical_to_effective` / `h_ranges` | 面积加权有效系数公式重复 3 次 | 提取 `_area_weighted(values, areas)` helper |
| R9 | `geometry_compiler.cpp/.hpp`、`_interfaces.py`、`model_case1.py`、`_bci_pop.py`、`_chiplet_stack.py` | 分支删除代码后遗留的多余空行 | 清理 |

注意：`boundary_temperature` 的加权平均（空组回退 0.0）与 R8 模式（空组回退 mean）语义不同，保持原样，不强行合并。

## 2. 目标

1. 每个跨语言字段只声明一次（C++ 核心 / C API struct / ctypes / Python 视图四处各保留其必要声明，接线逻辑单一化）。
2. 每个派生事实只计算一次、只存在于一个位置（ijk、暴露面、面积）。
3. 暴露面判定的唯一实现 = `CellFields.exposed_face_mask`（向量化）；`surface_exposed_cells` 是其薄封装。
4. C++ 与 Python 的 grid→ijk 解码共用同一份逻辑（C++ 侧为唯一实现，Python 侧为对应向量化实现，测试复用 C++ helper）。
5. 全分支零冗余：旧 API、旧属性、旧 hook、无调用者的字段全部消失。

## 3. 实施阶段

### Phase 1：C++ 约简（R1 R2 R3）

任务：

1. R1：`mhs_compiled_copy_cell_fields` 保留前置尺寸校验，将 16 个 copy 段改为 lambda 表 + 单循环（需要 `#include <functional>`）。
2. R2：`src/common/model.hpp` 增加 `GridIndex` + `inline decode_grid_index(Index grid, Index ny, Index nz)`；`model_compiler.cpp` 物性填充循环与 `tests/test_preprocessor.cpp` 新测试改用之。
3. R3：`heat_source_id` → `heat_source_idx`（`metahotspot.h` struct 字段、`metahotspot.cpp` copy 标签、`types.py`、`compiled.py` 后续随 R4 一并完成）。

验证：重新构建；`python run_tests.py` 94/94。

出口条件：C++ 测试全绿；`git grep` 无 `heat_source_id` 残留；解码逻辑无重复。

### Phase 2：Python 绑定约简（R4 R5 R6）

任务：

1. R4：`compiled.py` 新增 `_CELL_FIELD_SPECS`（name, ctype, np.dtype, count 来源），`_fetch_metadata` 改为表驱动（分配 → 指针接线 → `CellFields(**arrays)`），约 65 行 → 约 18 行。
2. R5：`CellFields.__post_init__` 一次性计算并缓存 `_ijk`、`grid3d`、`exposed_face_mask`；`nx/ny/nz` 派生属性保留。
3. R6：`exposed_face_mask` 向量化：`np.full` 占位边界 + 6 方向 `== invalid` 比较 → `(N,6)` uint8 位掩码（bit 0..5 = XM,XP,YM,YP,ZM,ZP，与 `Face` 枚举一致），按 `cell_to_grid` 取活跃行。语义与现循环逐位等价（越界或邻居非活跃即暴露）。

验证：`python run_tests.py` 仍全绿；小模型 smoke（构造 Compiled，比对 mask 与手工循环结果一致）；`reproduce_case1.py` 数值与基线一致。

出口条件：mask 向量化且逐位等价；`_fetch_metadata` 无重复声明；派生属性只解码一次。

### Phase 3：ROM 基础设施约简（R7 R8）

任务：

1. R7：`surface_exposed_cells` 改写为 `(cells: CellFields, face: Face, coord: float, z_range=None)`：
   - 候选 = `exposed_face_mask` 对应 bit 为 1 的活跃 cell（`np.flatnonzero`，升序 = 与现 grid 序一致）；
   - ZM/ZP 仅保留 `iz == 0 / nz-1` 层；侧向面按 `ijk` 做 face 坐标 `inside` 过滤 + `z_range` 过滤（保持现语义：Z 面不应用 z_range）；
   - 面积由 `cell_sizes` 按面法向取两切向分量乘积（与现顶点差公式逐值相同）。
2. 三个模型 `boundary_groups` 改用 `cells = full.cells` + 新签名；删除 `grid`/`x`/`y`/`z` 重复局部变量；`_chiplet_stack.py` 删除冗余 `np.asarray`；`model_case1.py` / `_bci_pop.py` 顶部面坐标改用 `cells.z_vertices[-1]`（编译事实）替代 config 派生高度。
3. R8：提取 `_area_weighted(values, areas)`，`physical_to_effective` 与 `h_ranges` 复用。

基线记录（reproduce_case1，迁移前，2026-08-25）：

```text
model: bci_case1  full cells=134640  ports=4  groups=2
basis order = 45  response err=2.10e-07

=== STEADY junction temperatures (K) ===
port    full FVM   our ROM  Flotherm
S0       316.827   316.827   316.827
S1       320.781   320.781   320.780
S2       324.735   324.735   324.735
S3       328.690   328.690   328.689

steady full-FVM field peak = 328.707 K

=== TRANSIENT max junction error vs full FVM (% of steady rise) ===
port     our ROM  Flotherm
S0        0.0028    0.0088
S1        0.0022    0.0029
S2        0.0018    0.0043
S3        0.0017    0.0026
```

验证：`reproduce_case1.py` 结果与上表逐项一致（面积与暴露集合必须逐位相同，ROM 结果应完全复现）；`python run_cases.py` 全过。

出口条件：`surface_exposed_cells` 不再包含邻居判定逻辑；模型 `boundary_groups` 无重复顶点/reshape 读取；ROM 数值与基线一致。

### Phase 4：清理、文档与完整验证（R9）

任务：

1. R9：清理分支删除代码后遗留的空行。
2. 全仓库 `git grep` 旧 API/旧属性/旧 hook 残留（`layer_ids`、`block_ids`、`grid_to_cell` 旧属性、`_axis_vertices`、`_physical_stack`、`_layer_conductivity`、`heat_source_id`）。
3. 更新 `metahotspot-project` skill 中已过时的描述（`BoundaryGroup(k=, half=)` 与 `_cell_half`/`_z_half`/`_xy_half`/`_cell_z_centers` 已不存在）。
4. 完整验证矩阵：

```text
python run_tests.py
python run_cases.py
playground/bci_rom_testcase1/reproduce_case1.py（结果与基线一致）
```

出口条件：矩阵全绿；无旧契约残留；skill 与代码一致。

## 4. 验收标准

### 契约验收

- `CellFields` 字段集合不变（仅 `heat_source_id` 更名 `heat_source_idx`）；
- `exposed_face_mask` 位布局与 `Face` 枚举一致，逐位等价于旧循环；
- `surface_exposed_cells` 对每个既有调用点返回完全相同的 cells/areas（顺序、dtype、数值）。

### 约简验收

- C++ copy-out 无重复错误检查段；
- grid→ijk 解码在 C++ 侧只有一份实现，测试复用；
- Python `_fetch_metadata` 字段声明单一化；
- 暴露面判定全仓库只有 `exposed_face_mask` 一份实现；
- 派生属性不重复解码。

### 数值验收

- 94/94 C++ 测试通过；
- run_cases 全过；
- reproduce_case1 关键温度、ROM 阶数、瞬态结果与基线一致（本计划只约简不改变数学，结果应完全复现）。

## 5. 明确不做的事情

- 不修改 Cell-centered DOF 数学定义、求解器、assembler、SoA 布局；
- 不修改 Compact Cell Order 与状态向量顺序；
- 不引入新依赖（包括不引入 numpy 新 API 之外的依赖）；
- 不为旧 API 增加兼容层；
- 不改变 `boundary_temperature` 的空组语义；
- 不合并语义不同的加权平均分支；
- 不新增或删除 `CellFields` 公开字段（重命名除外）。

## 6. 实施纪律

- 每个阶段先记录基线，再修改，再跑验证；
- 每个阶段结束时 `git grep` 搜索旧契约残留；
- 不提交半迁移状态；
- 不提交伪造或未经验证的结果；
- 约简后任何数值变化必须解释为 bug，禁止静默接受。
