# Issue 03: FluidPreprocessor 求解压力场 + Scheduler 接入

## Parent

.fluid-algorithm/PRD.md

## What to build

新建 `src/fluid/` 模块,实现流体压力求解的核心算法,并在 Scheduler 稳态分支中接入调用。

### FluidPreprocessor 类

文件: `src/fluid/fluid_preprocessor.{hpp,cpp}`, 命名空间 `mhs::sim`

```cpp
class FluidPreprocessor {
public:
    void solve_flow(mhs::core::InternalModel& model);
private:
    void init_cell_hydro_properties(mhs::core::InternalModel& model);
    void apply_pressure_boundary_conditions(mhs::core::InternalModel& model);
    void solve_pressure(mhs::core::InternalModel& model);
    void precompute_flow_axes(mhs::core::InternalModel& model);
};
```

### 四个阶段

#### **阶段 1: init_cell_hydro_properties**

- 遍历所有流体 cell,沿 X/Y/Z 三轴计算 `hydroC` (Hele-Shaw 矩形截面修正)
- 正方形: `0.42229 * h^4 / (12 * mu * L)`
- 矩形: `(1 - 0.63*AR) * min^3 * max / (12 * mu * L)`
- 非流体 cell 跳过

#### **阶段 2: apply_pressure_boundary_conditions**

- 遍历 `model.is_pressure_boundary[c]`,找到标记压力边界的 cell
- 将 `model.boundary_pressure[c]` 用于后续求解
- (标记已在 preprocessor 阶段由 `apply_fluid_overlay()` 完成)

#### **阶段 3: solve_pressure**

- 构建流体子域索引: `fluid_ids = {c | is_fluid[c]}`,映射 `g2f[c]`
- 遍历所有 internal faces (通过 `neighbor_grid_index` + `index_map`):
    - 两端都是流体: `C_eff = harmonic_mean(hydroC[c_a][axis], hydroC[c_b][axis])`
    - 非压力边界: off-diag += C_eff, diag += C_eff
    - 压力边界: diag = 1, RHS = boundary_pressure
    - 非边界流体 cell: diag = -Σ C_eff
- 组装为 `Eigen::SparseMatrix`, 用现有 `SparseLUSolver::solve()` 求解
- 结果写回 `model->pressure`

#### **阶段 4: precompute_flow_axes**

- 对每个流体 cell,遍历三轴方向的最大 |Δp|
- `flow_axes[c] = argmax(per_axis_pressure_drop)`

### Scheduler 接入

在 `scheduler.cpp` 的稳态分支中,`Assembler assembler(*model_)` 之前插入:

```cpp
mhs::sim::FluidPreprocessor fluid_prep;
fluid_prep.solve_flow(*model_);
```

若 `is_fluid` 全 false,`FluidPreprocessor::solve_flow()` 直接返回。

## Acceptance criteria

- [ ] `test_hydroC_single_axis`: 正方形截面 water μ=8.9e-4, 0.5×0.5×0.2mm → hydroC 值与 Python 参考一致
- [ ] `test_pressure_solve_simple`: 2-cell 流体串行(1D), inlet P=500, outlet P=0 → pressure = [500, 0]
- [ ] `test_flow_axes_dominant_x`: ΔP 仅沿 X → flow_axes 全为 0
- [ ] steady_case1 + overlay → `model.pressure[fluid_cells]` 全正,outlet 处 ≈ 0
- [ ] `model.flow_axes[fluid_cells]` ∈ {0, 1, 2} (X/Y/Z)
- [ ] 现有测试 100% 通过 (无 overlay 时 FluidPreprocessor 直接返回,不影响)
- [ ] 瞬态路径未受影响 (fluid_prep 仅在稳态分支调用)

## Blocked by

- Issue 01 (数据骨架)
- Issue 02 (overlay IO + Preprocessor 合入,提供 is_fluid + boundary_pressure)
