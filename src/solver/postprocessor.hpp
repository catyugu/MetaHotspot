#pragma once

#include "core/vector.hpp"
#include "model/model.hpp"
#include <span>

namespace mhs::solver {

    // 从 cell 中心温度场与边界迹通过各向异性距离反比加权插值到节点温度场。
    // Result order follows OutputMesh::points; no Cartesian topology is assumed.
    // `time` 注入到 FieldContext.t，使时间依赖的 BC 表达式（如 "500 + 100*t"）
    // 在正确的时刻被求值。
    mhs::core::Vector<double> interpolate_cell_to_node(
        const mhs::model::Model& model, std::span<const double> cell_temperature, double time);

    double max_temperature(std::span<const double> T);
    double min_temperature(std::span<const double> T);

} // namespace mhs::solver
