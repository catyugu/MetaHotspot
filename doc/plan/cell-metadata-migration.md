# Cell 元信息全面迁移实施计划

## 0. 计划状态

- 状态：已固定，待实施
- 类型：破坏性跨语言 API 重构
- 范围：C++ runtime/compiler、C API、ctypes、Python API、ROM 基础设施、playground 实验、测试和设计文档
- 本计划不包含代码实现
- 不保留旧 API，不增加兼容别名，不维护新旧双轨

## 1. 最终目标

建立唯一的 Cell 元信息事实链：

```text
C++ compiler
    → CellMetadata
    → compiled handle
    → typed C API snapshot
    → Python Compiled.cells
    → ROM / experiments / post-processing
```

迁移完成后：

1. C++ compiler 是 Cell 几何、拓扑、归属和参考状态物性的唯一事实源。
2. 所有 Cell 级数组使用同一个 Compact Cell Order。
3. Python 不再从 `grid_to_cell` 反推 `cell_to_grid`、`ijk` 或 Cell 顺序。
4. Python 不再从自身的 vertex、Layer 或材料配置重建 Cell 几何和材料含义。
5. Python 通过 `Compiled.cells` 统一获取 Cell 元信息。
6. 旧 C API 和旧 Python 属性全部删除。
7. solver 的热方程、状态向量顺序、SoA 组装布局和数值语义不改变。

## 2. 固定的目标契约

### 2.1 Compact Cell Order

继续采用已验证的 ix-iy-iz 顺序：

```text
for ix = 0 .. nx-1
  for iy = 0 .. ny-1
    for iz = 0 .. nz-1
```

活跃 Grid Cell 按此顺序分配 Compact Cell ID。该顺序从实现细节提升为跨 C++/C/Python 的公开契约。

### 2.2 CellMetadata 字段

所有字段长度均为 `cell_count`，二维字段使用最后一维表示坐标或面方向。

| 字段 | Shape | 语义 | 单位/类型 |
| --- | ---: | --- | --- |
| `cell_to_grid` | `(N,)` | Compact → Grid 映射 | `size_t` |
| `ijk` | `(N, 3)` | `(ix, iy, iz)` | `uint32` |
| `centers` | `(N, 3)` | Cell 中心坐标 | m |
| `sizes` | `(N, 3)` | x/y/z 边长 | m |
| `half_sizes` | `(N, 3)` | 中心到对应面的距离 | m |
| `volumes` | `(N,)` | Cell 体积 | m³ |
| `layer_id` | `(N,)` | Layer 归属 | `uint32` |
| `block_id` | `(N,)` | Block 归属 | `uint32` |
| `material_id` | `(N,)` | 材料表归属 | `uint32` |
| `heat_source_id` | `(N,)` | 热源表归属 | `uint32` |
| `conductivity` | `(N, 3)` | 参考状态下 kx/ky/kz | W/(m·K) |
| `density` | `(N,)` | 参考状态密度 | kg/m³ |
| `specific_heat` | `(N,)` | 参考状态比热 | J/(kg·K) |
| `exposed_face_mask` | `(N,)` | 无活跃邻居的面 | `uint8` |

参考状态固定为：

```text
T = initial_temperature
 t = 0
```

面方向 bit 固定为：

```text
bit 0 = XM
bit 1 = XP
bit 2 = YM
bit 3 = YP
bit 4 = ZM
bit 5 = ZP
```

`face_bc_type` 与 `face_bc_param_id` 只有在 BC parameter snapshot 同时设计完成后才作为 Python 公共字段暴露，禁止单独暴露半完成的 BC 契约。

### 2.3 Python 入口

最终唯一 Cell 元信息入口：

```python
compiled.cells
```

目标字段：

```python
compiled.cells.cell_to_grid
compiled.cells.ijk
compiled.cells.centers
compiled.cells.sizes
compiled.cells.half_sizes
compiled.cells.volumes
compiled.cells.layer_id
compiled.cells.block_id
compiled.cells.material_id
compiled.cells.heat_source_id
compiled.cells.conductivity
compiled.cells.density
compiled.cells.specific_heat
compiled.cells.exposed_face_mask
```

返回数组必须为 Python-owned、连续、只读 NumPy 数组。

## 3. 实施阶段

### Phase 1：冻结基线和调用清单

目标：在修改代码前锁定当前行为和迁移范围。

任务：

- 运行当前 C++ 测试：`python run_tests.py`；
- 运行当前案例回归：`python run_cases.py`；
- 按现有已验证流程运行 `playground/bci_rom_testcase1/reproduce_case1.py`；
- 记录 full cell count、grid shape、Cell 顺序、关键温度和 ROM 结果；
- 全仓库搜索并列出以下调用者：
    - `grid_to_cell`；
    - `cell_to_grid`；
    - `layer_ids` / `layer_id`；
    - `block_ids` / `block_id`；
    - `_axis_vertices`；
    - `_layer_conductivity`；
    - `surface_exposed_cells`；
    - `mhs_compiled_copy_*` 元信息 API。

出口条件：

- 基线命令和结果已记录；
- 调用者清单完整；
- 没有在基线阶段修改生产代码。

### Phase 2：C++ CellMetadata 内部契约

目标：在 C++ 内部生成完整、对齐、只读的 CellMetadata。

任务：

1. 在 `src/common/` 增加 CellMetadata 运行期类型。
2. 明确所有字段的元素类型、shape、单位和生命周期。
3. 在 Cell 分配完成处同步填充：
   - `cell_to_grid`；
   - `ijk`；
   - centers、sizes、half_sizes、volumes；
   - ownership IDs；
   - 参考状态物性；
   - exposed-face mask。
4. 确保 CellMetadata 与状态向量使用同一 Compact Cell Order。
5. 将 CellMetadata 作为 compiled model 的只读数据发布。
6. 不改变 `CellFields` 的 solver 使用方式和 assembly 热循环。

测试先行：

- 新增 metadata 单元测试；
- 先让测试在旧实现下失败；
- 实现后验证通过。

出口条件：

- C++ 能直接提供完整 CellMetadata；
- 所有字段长度均为 `cell_count`；
- `cell_to_grid` 与 `grid_to_cell` 互逆；
- geometry、ownership 和 material reference tests 通过；
- 现有 93 个测试仍通过。

### Phase 3：C API 破坏性重构

目标：用 typed snapshot 替换零散单数组接口。

任务：

1. 在 `src/api/metahotspot.h` 定义 metadata info 和 typed copy-out API。
2. 按语义分组实现：
   - topology；
   - geometry；
   - ownership/material。
3. 明确 NULL、空数组、错误 count、错误 handle 和错误状态行为。
4. 更新 C API 内部实现和错误消息。
5. 删除以下旧符号及实现：
   - `mhs_compiled_copy_grid_to_cell`；
   - `mhs_compiled_copy_layer_ids`；
   - `mhs_compiled_copy_block_ids`。
6. 不提供兼容别名。
7. 增加 C API buffer、count、dtype 和生命周期测试。

出口条件：

- 公共头文件只包含新契约；
- 旧符号已从头文件、实现、测试和构建产物引用中消失；
- C API 测试覆盖成功和错误路径。

### Phase 4：Python ctypes 与 Compiled API 迁移

目标：Python 只通过 `Compiled.cells` 获取 Cell 元信息。

任务：

1. 更新 `python/metahotspot/types.py` 的 C struct 定义。
2. 更新 `_dll_interface.py` 的函数签名。
3. 新增 Python `CellMetadata` 类型。
4. 在 `Compiled` 中一次性读取并缓存完整 snapshot。
5. 将所有 NumPy 数组转换为 Python-owned contiguous arrays。
6. 设置数组为只读。
7. 删除 `Compiled.grid_to_cell`、`Compiled.layer_ids`、`Compiled.block_ids`。
8. 更新 Python 测试，验证 shape、dtype、长度、只读和 cache。

出口条件：

- `compiled.cells` 可用且字段完整；
- 旧 Python 属性访问失败，不再存在兼容别名；
- Python 不再通过 `flatnonzero(grid_to_cell)` 重建 Compact 顺序。

### Phase 5：ROM 基础设施迁移

目标：删除 Python 侧的 Cell 事实重建逻辑。

任务：

1. 重构 `AffineParametricModel.cell_layout`：
   - centers 直接来自 `compiled.cells.centers`；
   - half_sizes 直接来自 `compiled.cells.half_sizes`；
   - conductivity 直接来自 `compiled.cells.conductivity`；
   - volumes 直接来自 `compiled.cells.volumes`。
2. 删除通过 vertex 数组推导 Cell centers 和 half sizes 的基础逻辑。
3. 删除通过 Layer ID 和 Python material table 选择 Cell conductivity 的基础逻辑。
4. 删除基础类对 `_axis_vertices()` 和 `_layer_conductivity()` 的依赖。
5. 将 source/boundary 分组改为消费 compiled metadata。
6. 保留模型真正拥有的物理配置、功率定义和参数范围，不让模型配置承担编译结果解释职责。

出口条件：

- `CellLayout` 不再重建任何 C++ 已知事实；
- ROM 单元测试和小规模数值测试通过；
- 公开实验 API 不依赖模型私有几何解释 hook。

### Phase 6：Playground 与实验全面迁移

目标：所有实验脚本使用新 API，禁止残留旧访问模式。

任务：

- 迁移 `playground/macromodel/`；
- 迁移 `playground/bci_rom_testcase1/`；
- 迁移所有 `AffineParametricModel` 子类；
- 将 `full.grid_to_cell` 改为 `full.cells.cell_to_grid` 或对应 metadata 字段；
- 将 `full.layer_ids` 改为 `full.cells.layer_id`；
- 将 `full.block_ids` 改为 `full.cells.block_id`；
- 优先使用 `exposed_face_mask` 获取暴露面；
- 删除手工扫描虚拟邻居和网格边界的重复逻辑；
- 保持既有实验数学定义不变，不能同时改变 ROM 算法和 metadata 读取方式。

出口条件：

- 全仓库 Python 代码不再引用旧属性；
- `reproduce_case1.py` 完整运行；
- 关键 steady/transient 结果与基线在原有容差内一致。

### Phase 7：清理、文档和最终验证

目标：完成破坏性迁移并确认没有隐式旧契约。

任务：

1. 删除旧的 C++ metadata 辅助函数和无调用者字段。
2. 删除旧 Python cache 字段和重建代码。
3. 更新设计文档、data flow、project structure 和 ADR。
4. 搜索全仓库确认旧 API、旧属性、旧 hook 无残留。
5. 运行完整验证矩阵。

验证矩阵：

```text
python run_tests.py
python run_cases.py
python -m pytest python/tests
python playground/bci_rom_testcase1/reproduce_case1.py
```

若某个命令因环境或案例依赖无法运行，必须记录真实错误，不得用编译通过替代科学验证。

## 4. 验收标准

### 契约验收

- CellMetadata 字段、shape、dtype、单位和 CellOrder 已固定；
- C++、C API 和 Python 对同一字段使用同一顺序；
- `cell_to_grid`、`ijk`、geometry 和 volumes 一致；
- ownership/material 字段与 compiler 实际归属一致；
- snapshot 生命周期和 buffer 错误行为明确。

### 迁移验收

- 旧 C API 符号不存在；
- 旧 Python metadata 属性不存在；
- 没有兼容别名；
- 没有 Python 侧从 mesh/Layers 重建 Cell 事实的代码；
- ROM 和 playground 全部使用 `Compiled.cells`。

### 数值验收

- 现有 C++ 测试全部通过；
- Python 测试全部通过；
- case 回归通过；
- `reproduce_case1.py` 通过；
- 关键温度、ROM 阶数、瞬态结果在迁移前后既定容差内一致。

## 5. 明确不做的事情

本计划不包含：

- 修改 Cell-centered DOF 数学定义；
- 引入每 Cell 一个 Python 对象；
- 暴露 C++ expression AST；
- 把 Eigen 或 C++ 容器暴露到 C ABI；
- 为旧 API 增加兼容层；
- 同时重写 ROM 数学算法；
- 为展示目的增加额外图片或改变实验输出形式；
- 在没有真实验证结果时更新科学报告中的数值。

## 6. 实施纪律

- 每个阶段先增加失败测试，再实现，再运行回归；
- 每个跨语言字段都必须有 C++、C API、ctypes、Python 测试覆盖；
- 每个阶段结束时搜索旧契约残留；
- 不提交半迁移状态；
- 不提交伪造或未经验证的结果；
- 不修改用户未要求的其他数据结构和数值算法。
