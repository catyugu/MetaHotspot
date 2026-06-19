# Issue 04: Assembler 流体-固体交界面导热修正

## Parent

.fluid-algorithm/PRD.md

## What to build

修改 `assembler.cpp` 的导热矩阵组装逻辑,在 `BcType::None` 分支中识别流体-固体交界面,并用 Nusselt 数修正导热系数。

### 修改点

在 `assembler.cpp` 的 `BcType::None` 分支中,当遇到流体-固体交界面时:

```cpp
// 现有 solid-solid 路径完全不变:
//   cond = A_f / (half_dist / k_face + d_half_neighbor / k_neighbor)

// 新增 fluid-solid 路径:
if (model_->is_fluid[c_idx] != model_->is_fluid[n_idx]) {
    int f_id = model_->is_fluid[c_idx] ? c_idx : n_idx;
    int s_id = model_->is_fluid[c_idx] ? n_idx : c_idx;
    
    // 用 flow_axes 确定流体侧主导流向
    int f_ax = model_->flow_axes[f_id];
    int ax_w = (f_ax + 1) % 3;
    int ax_h = (f_ax + 2) % 3;
    
    // 获取流体 cell 在截面方向的尺寸
    double w = dims[f_id][ax_w];
    double h = dims[f_id][ax_h];
    
    // 计算 Nu
    double Nu = mhs::utils::nusselt_rectangular(w, h);
    
    // 水力直径
    double d_h = 2.0 * w * h / (w + h);
    
    // 内部对流换热系数
    double h_f = Nu * k_fluid / d_h;
    
    // 串联热阻: 固体侧半距离导热 + 流体侧对流
    double R = half_dist_solid / (k_solid * A_f) + 1.0 / (h_f * A_f);
    double cond = 1.0 / R;
    
    // 后续 diag += cond, 写入 off-diagonal triplet 不变
}
```

### 注意事项

- **原则**: 原有纯固体路径完全不变;流体相关逻辑仅在 `is_fluid[c]` 或 `is_fluid[n]` 为 true 时激活
- `dims` 需要从 `mesh.dx/dy/dz` 按 axis 选取
- `k_fluid` / `k_solid` 从 `material_table` 按 `fluid_material_id` 选取
- 此 issue 仅修改导热矩阵,不涉及 advection 矩阵 (见 Issue 05)

## Acceptance criteria

- [ ] `test_fluid_solid_interface_cond`: 单对流面,cond = 1/(R_solid + R_fluid) 与手动计算一致
- [ ] 现有测试 100% 通过 (无流体时走原 solid-solid 路径,完全不变)
- [ ] steady_case1 + overlay → 流体-固体交界面导热增强,T 场出现合理梯度
- [ ] 无 NaN,无负温度

## Blocked by

- Issue 03 (需要 pressure 和 flow_axes 已填充)
