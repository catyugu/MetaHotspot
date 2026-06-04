---
Status: ready-for-agent
---

# 02: Postprocessor 新增 sample_point 方法

## 范围

- `src/postprocessor/postprocessor.hpp`
    - 新增方法签名：

    ```cpp
    double sample_point(const std::vector<double>& node_T,
                        const InternalModel& model,
                        const ObservationPoint3D& point) const;
    ```

- `src/postprocessor/postprocessor.cpp`
    - 实现：
    1. 用点坐标 `(px, py, pz)` 在 mesh 中定位包围 cell
       - 走 bbox/binary-search：找 `(ix, iy, iz)` 使得
         `vertex_x[ix] ≤ px < vertex_x[ix+1]`（对 Y/Z 同理）
       - 若坐标落在网格外 → 返回 NaN
    2. 找到 cell 后，提取该 cell 角点（8 个 vertex）的节点温度
       - 这些 vertex 的 node_T 值已由 `interpolate_cell_to_node` 计算
    3. 用这些角点构造 `DataPoint` 列表喂给 `solve_least_squares(pts, px, py, pz)`
       - 复用现有 `DataPoint` 结构（匿名命名空间内）
       - 权重沿用 k / dist² + 1e-16（与 interpolate_cell_to_node 一致）
       - BC 面外推也复用（对该 cell 的 3 个方向面检查 BC）
    4. Dirichlet 节点附近：若探针恰在 FirstType 边界面上 → 直接返回 Dirichlet 值

## 约束

- **不修改 `solve_least_squares` / `extrapolate_face_temperature` 函数本身**
  — 它们在匿名命名空间里，`sample_point` 在同一翻译单元可直接调用
- `sample_point` 是 `const` 方法（与 `interpolate_cell_to_node` 一致）
- 不引入新算法

## 验收

- 简单均匀温度场（所有 cell T=300K）→ `sample_point` 在网格内部任意点返回 ≈ 300K
- 网格外点 → 返回 NaN
- 线性温度场（T 沿 z 线性梯度）→ 插值误差 < 1e-6

## 不做

- IO 解析（01）
- Scheduler 回调（03）
- write_xml 回写（04）
