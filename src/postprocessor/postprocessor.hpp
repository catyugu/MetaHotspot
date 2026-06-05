#pragma once

#include "common/internal_model.hpp"
#include <vector>

namespace mhs::post {

    class Postprocessor {
    public:
        Postprocessor() = default;
        ~Postprocessor() = default;

        std::vector<double> interpolate_cell_to_node(
            const mhs::core::InternalModel& model, const std::vector<double>& cell_temperature) const;

        // 在已完成节点插值的温度场上，采样用户坐标系下任意 (x,y,z) 处的温度。
        // 复用 interpolate_cell_to_node 的 cell 角点 / 中心 / 边界外推 LSQ 拟合。
        // 落在网格外 → 返回 NaN。
        double sample_point(
            const std::vector<double>& node_T, const mhs::core::InternalModel& model, const mhs::core::ProbePoint& point) const;

        double max_temperature(const std::vector<double>& T) const;
        double min_temperature(const std::vector<double>& T) const;
    };

} // namespace mhs::post
