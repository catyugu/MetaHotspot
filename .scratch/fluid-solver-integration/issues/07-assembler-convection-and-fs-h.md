# 07: 装配器体积对流项 + 流-固 h 项

Status: needs-triage · Type: feature · Depends on: 06

## Context

热装配器当前只写扩散项 + 边界条件。流体解算后，要在热装配里加入两类新贡献：

1. **体积对流项**：单元面对流项 `∇·(ρcp·u·T)` 的离散形式
2. **流-固面 h 项**：自动检测流-固边界，叠 h·(T_f - T_s)

## Goal

修改 `src/assembler/assembler.cpp` 的面循环：

```cpp
for each internal face (c0, c1) {
    int face_idx = /* 当前 face 在内部面序列里的索引 */;
    const auto& fluid = model.fluid;

    // (1) 流-固面：跳过一切其它贡献，只应用 h·(T_f - T_s)
    if (fluid.has_value() && fs_face_index.contains(face_idx)) {
        const auto& fsf = fluid->fs_faces[fs_face_index[face_idx]];
        add_fluid_solid_interface(ops, c0, c1, fsf.h, ...);
        continue;
    }

    // (2) 体积对流项（两端任一是流体）：对所有面调用，对流项自然为 0
    if (fluid.has_value()) {
        add_convection_contribution(ops, c0, c1,
                                     fluid->face_velocity[face_idx],
                                     /* rho_face, cp_face */, state.T);
    }

    // (3) 既有：扩散 + cauchy
    add_diffusion(...);
    add_cauchy(...);
}
```

两个 helper（匿名 ns）：

```cpp
namespace {

void add_fluid_solid_interface(AssemblyResult& ops,
                                int cf, int cs,         // 流体单元 / 固体单元
                                double h,
                                double A_face) {
    // g = h * A_face
    // K[cf, cf] -= g,  K[cs, cs] -= g
    // K[cf, cs] += g,  K[cs, cf] += g
    // f[cf] += -g * T_solid_unknown_term   (注意：T_s 是另一侧 DOF，无 rhs)
}

void add_convection_contribution(AssemblyResult& ops,
                                  int c0, int c1,
                                  const std::array<double, 3>& u_face,
                                  double rho_face, double cp_face,
                                  const std::vector<double>& T,
                                  /* dir, A_face */) {
    // 沿 axis_of_dir[dir] 取 u = u_face[axis]
    // ṁ = ρ · u · A_face
    // 一阶 upwind：
    //   if u > 0: 上游 = c0, 下游 = c1
    //   else:     上游 = c1, 下游 = c0
    //   贡献：
    //     K[down, up]   += ṁ · cp
    //     K[down, down] -= ṁ · cp
    //     f[down]       += ṁ · cp · T[up]   (Picard: 用旧值 T)
}

} // namespace
```

## Scope

- 修改 `src/assembler/assembler.cpp` 面循环
- 新增 `add_fluid_solid_interface` / `add_convection_contribution` 匿名 ns helper
- 不修改 face 主循环结构（不重写、不重构）
- 不引入新抽象 / 不抽接口

## Acceptance

1. 单元测试 `test_assembler_convection`：
   - 1×2×1 网格（左流体 / 右固体）、u_face 已知 → K 矩阵对流贡献与手算一致
2. 单元测试 `test_fluid_solid_interface`：
   - 1×2×1 网格、左流体 / 右固体（封闭，无 pressure BC，模拟 Poiseuille 解场手填）→ fs_face 被检测、h 由 Nu 公式算对、K 矩阵对称、`K[fs][fs]` 非零、cauchy 不叠加
3. 既有 case 跑通（无流体路径不变）
4. `cmake --build` 无 warning

## Notes

- 体积对流项格式：**一阶 upwind**（Q1.5.b 决策）
- ρ_face = 0.5·(ρ_c0 + ρ_c1)，cp_face 同理（Q1.5.a）
- 上游判定基于 u_face 在该轴上的符号；与 `face_velocity[face_idx][axis]` 配合
- 流-固面**覆盖**该面的其它贡献（Q1.7 + Q1.18.b）：用户就算在 face_key 上挂了 cauchy / dirichlet，也只应用 h 项
- 注意 helper 的 T 依赖：迎风项的 `f[down] += ṁ·cp·T[up]` 用旧 `state.T`，与现有非线性求解器 Picard 风格一致
- 不需要新抽象：直接在面循环里加 `if` + helper 调用，与既有 `add_diffusion` / `add_cauchy` 风格一致
- thread-local TBB：现有 `Assembler::assemble` 已经有 TLS 模式，新加的 helper 必须线程安全（每个 cell 是独立操作，OK）
