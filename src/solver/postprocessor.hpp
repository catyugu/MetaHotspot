#pragma once

#include "common/model.hpp"
#include <span>
#include <vector>

namespace mhs::post {

    // 从 cell 中心温度场通过各向异性距离加权最小二乘 + 边界外推插值到节点温度场。
    // 返回的向量长度 = (mesh.nx+1) * (mesh.ny+1) * (mesh.nz+1)。
    // Dirichlet 节点具有强约束：返回严格的 Dirichlet 平均值，不参与 LSQ 拟合。
    // `time` 注入到 FieldContext.t，使时间依赖的 BC 表达式（如 "500 + 100*t"）
    // 在正确的时刻被求值。
    std::vector<double> interpolate_cell_to_node(
        const mhs::core::Model& model, std::span<const double> cell_temperature, double time);

    double max_temperature(std::span<const double> T);
    double min_temperature(std::span<const double> T);

} // namespace mhs::post
