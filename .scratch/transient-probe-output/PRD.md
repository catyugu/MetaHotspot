# PRD: 瞬态求解输出（观察点轨迹 + 末步温度场）

## 目标

为瞬态（`StudyType::Transient`）求解提供两类输出：

1. **任意坐标观察点的温度时间序列** — 用户在 XML 指定若干探针，求解器在每个时间步
   记录该坐标处温度，最终回写到输入 XML 的 `<Results><a:anyType i:type="Result0DTransient">`
   节点中。
2. **最后一个时间步的完整温度场** — VTU（VTK UnstructuredGrid）+ 改写 XML 的
   `Results/a:anyType/Values/Data`（与稳态输出同结构）。

插值算法复用 `src/postprocessor/postprocessor.cpp` 的 `solve_least_squares`
（`T(x,y,z) = T_node + gx·x + gy·y + gz·z`，Tikhonov 正则化 + Householder QR）。
**不发明新算法**。

## XML / 前端格式

输入端（`Structure` 节点下，已存在）：

```xml
<ObservePoints3D>
    <ObservePoint3D>
        <Name>观察点 1</Name>
        <X>5</X>
        <Y>50</Y>
        <Z>5</Z>
    </ObservePoint3D>
    ...
</ObservePoints3D>
```

输出端（与已有 `Result0DTransient` 一致）：

```xml
<Results>
    <a:anyType i:type="Result0DTransient">
        <PhysicsName>温度</PhysicsName>
        <PointName>观察点 1</PointName>
        <TimeUnit>S</TimeUnit>
        <Times>
            <a:double>0</a:double>
            <a:double>1</a:double>
            ...   <!-- N+1 个，t=0, dt, 2dt, ..., N*dt -->
        </Times>
        <UnitName>K</UnitName>
        <Values>
            <a:double>300.0</a:double>
            ...   <!-- N+1 个，与 Times 一一对应 -->
        </Values>
    </a:anyType>
    ...
</Results>
```

> 约定：`Times` / `Values` 元素以 **dt 为步长**，从 0 开始；包含初始值（t=0，对应
> `initial_temperature`）和每个完成时间步的解，共 `N+1` 个条目。
> 错误模式：观察点坐标落在网格外（无任何 cell 包含该点）→ 该点 `Values` 全填 `NaN`，
> 不 panic（与稳态节点外插为 NaN 保持一致）。

## 后端数据流

```text
XML <ObservePoints3D>
  → io::read_xml  → IOStructure.observation_points: vector<ObservationPoint3D>{name, x, y, z}
  → Preprocessor 不动（探针不参与方程）

主流程：
  bin/main.cpp
    scheduler.run()                        // 当前：内部循环中不写
    postprocessor.interpolate_cell_to_node // 当前：仅调用一次（末步）
  → 改为：
    schedule 接收 output callback
      对每个完成时间步：
        node_T = postprocessor.interpolate_cell_to_node(model, state.T)
        对每个观察点 p：
          trace[p].append(t, postprocessor.sample_point(node_T, model, p))
        // VTU / XML 仅末步写

    末步（与稳态相同）：
      write_vtu(output_vtu, model, node_T)
      write_xml(input_path, output_xml, model, node_T, observation_traces)
        · 已有的 Values/Data 节点写入温度（不变）
        · 对每个 observation_point：
            找到/创建 <a:anyType i:type="Result0DTransient"> by PointName
            清空旧 <Times> / <Values> 内的 <a:double>
            按顺序 append 新条目
```

### 新增数据结构

- `src/common/io_model.hpp` — `IOStructure` 加 `observation_points: std::vector<ObservationPoint3D>`，
  其中 `ObservationPoint3D { std::string name; double x, y, z; }`
- `src/common/internal_model.hpp`（按需）— `InternalModel` 复制同样字段
- `src/postprocessor/postprocessor.hpp/cpp` — 新增
  `double sample_point(const std::vector<double>& node_T, const InternalModel& m, const ObservationPoint3D& p) const`
    - 找到含 (px, py, pz) 的 cell（用 `cells.valid_mask` + cell 范围 bbox 命中）；
    - 把该 cell 角点（vertex）作为拟合点集（与 `interpolate_cell_to_node` 同套外推），
    复用 `solve_least_squares(pts, px, py, pz)`；
    - 若点完全在网格外，返回 `NaN`。
- `src/scheduler/scheduler.hpp/cpp` — 新增
  `struct ProbeConfig { std::function<void(double t, const std::vector<double>& cell_T)> on_step_done; }`
    - 在每个 `nonlinear::solve` 完成后、`current_time += dt` 之前调用回调；
    - `on_step_done` 内部做节点插值 + 探针采样（避免 scheduler 依赖 postprocessor）。

## 行为细节

1. **不修改 `interpolate_cell_to_node`** — 它已经是 `const`，复用即可。
2. **不修改 `solve_least_squares` 或 `extrapolate_face_temperature`** — 抽离到匿名命名空间可见性
   不变，由 `sample_point` 通过 cell 角点直接调用（要么把它移到头文件，要么把
   `sample_point` 放进 postprocessor.cpp 同一翻译单元里——选后者，最小改动）。
3. **节点插值 + 探针采样的代价**：`interpolate_cell_to_node` 本身是 O(N_nodes × M=8)，
   每次时间步都跑一次。中等规模瞬态（≤200 步 × 100³）可接受；超大规模用开关
   `<ProbeConfig sample_every_k_steps>` 留作后续 issue（**不在本 PRD 范围**）。
4. **探针坐标系**：(X, Y, Z) 是**用户坐标**（输入 XML 的几何坐标），不是 grid index；
   preprocessor 阶段网格 `vertex_x/y/z` 与之一致，可直接二分 / bbox 定位 cell。
5. **已有 case1.xml 的现成结果**：用例中已有 100 步 × dt=1 的 `Result0DTransient`
   数值（线 2266-2467 段 101 个 Values）。**回归基线**就是这些数。
6. **稳态分支不受影响** — `observation_points` 为空时 `Scheduler` 行为完全不变；
   `write_xml` 写 Values/Data 走原路径。
7. **XML 回写时若观察点未在 input 出现**：在 `Results` 下追加新 `a:anyType` 节点。
   若已存在（按 `PointName` 匹配）则清空 `<Times>`/`<Values>` 重新填充。
8. **time 列表长度 = 探针 trace 长度**：每个 trace 写 `<Times>` 时按 trace 内已有
   时间步生成；不要硬编码 `duration/dt+1`。

## 验收

- 复用现有 `cases/simple_transient_tests/case1.xml`：
    - 末步温度场：与现有 5 个 steady case 的 vtu/xml 写入一致
    - 观察点 1（5, 50, 5）/ 观察点 2（3, 75, 2）：所有时间步 Values 与原 XML 数值
    在 `1e-3` 内一致
- 单元测试：
    - `sample_point` 在简单均匀温度场（cell 中心已知温度）下返回值合理
    - 网格外点返回 NaN
    - `write_xml` 在已有 / 新增观察点两种情况均正确
- 稳态回归：5 个 steady case 数值差异 < 1e-9（不改写路径）
- 失败用例：观察点为 0 个 → 输出 vtu/xml 不变

## 不在范围内

- 改 `interpolate_cell_to_node` / `solve_least_squares` 算法
- 改 `GlobalState` 字段（除增加 trace 容器引用外的内部数据）
- 改 `IOStructure` 现有稳态字段
- 改 FaceDir / CellBC / MaterialProps（这些属于其他 issue）
- 改非线性 / 线性求解器
- schema 文档（如果存在）— 不在代码范围内动
- 大规模时间步采样间隔（`<ProbeConfig sample_every_k_steps>`）

## 涉及文件

- `src/common/io_model.hpp` — `IOStructure` 加 observation_points
- `src/common/internal_model.hpp` — `InternalModel` 加 observation_points
- `src/io/io.cpp` — `read_xml` 解析 `<ObservePoints3D>`；
  `write_xml` 改写以注入 Result0DTransient
- `src/io/io.hpp` — 函数签名（如有）
- `src/postprocessor/postprocessor.hpp/cpp` — `sample_point` 新方法
- `src/scheduler/scheduler.hpp/cpp` — `ProbeConfig` 回调
- `bin/main.cpp` — 装配回调、调用 sample_point、收集 traces
- `tests/...` — sample_point 单测、xml 回写单测
- `cases/...` — 若需新增瞬态 case（验证探针在网格外、多个探针等）

## 关联

- PRD-各向异性 k：本次不引入 FaceDir 改动；探针采样不依赖 k 路径
- ADR-0007（待写）：观察点采样 / 时间步回调决策
