#pragma once

#include "common/internal_model.hpp"
#include <vector>

namespace mhs::post {

    // 从 cell 中心温度场通过各向异性距离加权最小二乘 + 边界外推插值到节点温度场。
    // 返回的向量长度 = (mesh.nx+1) * (mesh.ny+1) * (mesh.nz+1)。
    // Dirichlet 节点具有强约束：返回严格的 Dirichlet 平均值，不参与 LSQ 拟合。
    std::vector<double> interpolate_cell_to_node(
        const mhs::core::InternalModel& model, const std::vector<double>& cell_temperature);

    // 在已完成节点插值的温度场上，采样用户坐标系下任意 (x,y,z) 处的温度。
    // 复用 interpolate_cell_to_node 的 cell 角点 / 中心 / 边界外推 LSQ 拟合。
    // 落在网格外 → 返回 NaN。
    double sample_point(
        const std::vector<double>& node_T, const mhs::core::InternalModel& model, const mhs::core::ProbePoint& point);

    double max_temperature(const std::vector<double>& T);
    double min_temperature(const std::vector<double>& T);

} // namespace mhs::post
