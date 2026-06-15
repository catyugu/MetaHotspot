# 06: FluidPreprocessor::solve_flow 实现

Status: needs-triage · Type: feature · Depends on: 03, 05

## Context

流体解算的核心。算法参考 `experiment-v1/metaphor_preprocessor.py` 的 `FluidPreprocessor.solve_flow`：

1. 扫 `is_fluid=true` 单元 → `fluid_ids`、`g2f`
2. 取参考温度 `model.initial_temperature` 求值 μ → `mu_ref[c]`
3. 算每轴水力传导 `hydroC[c, axis]`（矩形 + 0.63 修正项）
4. 装配 pressure 系统 C·p = 0（CSR 三元组）
5. `SparseLUSolver` 解 p
6. 算 `face_velocity[face_idx][axis]`
7. 扫 `c0.is_fluid XOR c1.is_fluid` → `fs_faces` + 算 h

## Goal

实现 `mhs::sim::FluidPreprocessor::solve_flow(InternalModel&)`：

```cpp
class FluidPreprocessor {
public:
    void solve_flow(mhs::core::InternalModel& model) const;
};
```

写入 `model.fluid = FluidFields{...}`。

压力方程对 boundary 流体单元：b 行置 1、对角贡献 1、直接给 p（与 Python `apply_pressure_bc` 一致）。

流-固面 h 计算：

```text
D_h = 4·A / P = 2·(w·h) / (w + h)         // 矩形当量直径
α = max(w, h) / min(w, h)
Nu = 7.541·(1 - 2.610/α + 4.970/α² - 5.119/α³ + 2.702/α⁴ - 0.548/α⁵)    // α ≥ 1
h  = Nu · k_fluid / D_h
```

矩形 Poiseuille `hydroC`：

```text
对每个轴 axis：
  ax_w, ax_h = (axis+1)%3, (axis+2)%3
  L, w, h = dims[axis], dims[ax_w], dims[ax_h]
  if h ≈ w:
    hydroC[axis] = (1 - 0.63)·h³·w / (12·μ·L)        // 注：|h-w| < 1e-10 时退化
  else if h > w:
    hydroC[axis] = (1 - 0.63·h/w)·h³·w / (12·μ·L)
  else:
    hydroC[axis] = (1 - 0.63·w/h)·w³·h / (12·μ·L)
```

`face_velocity` 装配：

```text
对每条 internal face (c0, c1)：
  axis = axis_of_dir[dir_of_face]
  u_face = -(hydroC_avg) · (p[c1] - p[c0]) / dist
  face_velocity[face_idx][axis] = u_face
  face_velocity[face_idx][其它两轴] = 0
```

pressure 系统装配（inline 匿名 ns）：

- 矩阵 CSR 行/列/值三 vector + 浮点 b
- 对每条 internal face (c0, c1) 且两端都是流体：
    - `C_eff = 0.5·(hydroC[c0,axis] + hydroC[c1,axis]) · A_face / dist`
    - 非 pressure-boundary 端：行 = i0，列 = i1，值 = +C_eff；行 = i1，列 = i0，值 = +C_eff；对角 += C_eff
- 对每个非 pressure-boundary 流体单元 i：
    - 行 = 列 = i，值 = -diag_C[i]（保 row-sum = 0）
- pressure-boundary 单元：行 = 列 = i，值 = +1；b[i] = p_BC

## Scope

- 新文件 `src/preprocessor/fluid_preprocessor.hpp` / `.cpp`
- 修改 `src/preprocessor/preprocessor.cpp` 在 `load` 末尾调用
- 修改 `src/preprocessor/CMakeLists.txt` 加入新文件
- pressure 系统装配 inline 在 `fluid_preprocessor.cpp` 匿名 ns（Q1.11 决策）

## Acceptance

1. 单元测试 `test_fluid_preprocessor`：
   - 5×1×1 最小网格 → `pressure` 沿 x 线性
   - `hydroC` 与手算 Poiseuille 公式误差 < 1e-9
   - 无 pressure BC + 全封闭 → 抛 `runtime_error`（issue 07）
2. 单元测试 `test_fluid_pressure_bc_missing`：
   - 调用 `solve_flow` 在缺 pressure BC 时抛 `runtime_error`
3. 既有无流体 case 跑通（无 sidecar 路径不变）
4. `cmake --build` 无 warning

## Notes

- 复用现有 `SparseLUSolver`；不在此 issue 创建新求解器
- 复用现有 `LinearSolver` 抽象（不要为流体新建接口）
- "A_face" 沿用热装配器使用的面积计算公式
- "dist" 取两单元中心距（结构化网格 = 0.5·(dx_i + dx_j)）
- 0.63 修正项是矩形管的实验拟合；与 Python 完全对齐
- μ 默认取 1.0 当 `dynamic_viscosity` 表达式求值非正（防御性兜底，可在 issue 评审时讨论）
