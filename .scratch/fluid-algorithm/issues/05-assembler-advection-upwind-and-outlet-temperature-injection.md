# Issue 05: Assembler advection 上风组装 + 出口温度注入

## Parent

.fluid-algorithm/PRD.md

## What to build

在 `assembler.cpp` 中新增 advection 组装阶段,实现流体内部的对流热传递。

### 新增方法

```cpp
// 在 Assembler 类中新增私有方法:
struct AdvectionScratch {
    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::VectorXd rhs;
    Eigen::VectorXd net_outflux; // [N_active]
};

void build_advection_matrix(mhs::core::GlobalState& state, AdvectionScratch& scratch) const;
void inject_outlet_temperature(mhs::core::GlobalState& state, AdvectionScratch& scratch) const;
```

### 算法

#### **阶段 1: 上风 advection 组装**

遍历所有 internal faces (通过 `index_map` + `neighbor_grid_index`):

```text
对 internal face (c_a, c_b, axis):
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

#### **阶段 2: 出口边界温度注入 (RHS)**

```text
对每个流体 cell c:
  if net_outflux[c] > 0 (净流入):
    T_boundary = boundary_temperature_fluid[c]
    if !isnan(T_boundary):
      rhs[c] += net_outflux[c] · cp[c] · T_boundary
  elif net_outflux[c] < 0 (净流出):
    // 对流出 cell 的对角项追加: net_outflux[c] · cp[c]
```

### 边界温度来源

`boundary_temperature_fluid` 需要从 FirstType BC 继承:在 Issue 02 的 overlay 合入阶段,对于 inlet 处的流体 cell,读取其 FirstType BC 的温度值(298.15K)填入 `boundary_temperature_fluid`。

### 集成到 assemble()

在现有导热矩阵组装完成后,将 advection 贡献合并入 `K` 和 `f`:

```cpp
// 现有: K = diffusion + BC
// 新增: K += adv_diagonal, f += adv_rhs
```

不改 `AssemblyResult` 的结构签名。

## Acceptance criteria

- [ ] `test_advection_upwind_single_face`: 单 face, mass_flux > 0 → upwind T_up → T_dn 矩阵项正确
- [ ] 现有测试 100% 通过 (无流体时 advection 阶段直接返回)
- [ ] steady_case1 + overlay → 温度场物理合理:
    - [ ] 最高温度 < 343K (比纯固体解低,流体带走热量)
    - [ ] 流体入口附近 ≈ 298.15K
    - [ ] 整体 T 在 300K~340K 范围
    - [ ] 无负温度、无 NaN
- [ ] 无 overlay 时结果与之前完全一致

## Blocked by

- Issue 03 (需要 pressure 和 flow_axes 已填充)
- Issue 04 (流体-固体导热修正)
