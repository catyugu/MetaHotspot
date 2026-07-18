#pragma once

#include "runtime/model.hpp"
#include "runtime/solution.hpp"

#include <vector>

namespace mhs::sim {

    // 求解过程内部使用的探针采样与时序记录器，独立于 mhs::post。
    //
    // 设计要点：
    // - **局部采样**：每个时间步只对 (n_probes × 1) 个 cell 做 O(1) 邻域寻址与
    //   局部 LSQ，不再展开到全网格 node_T。
    // - **算法精度对齐 mhs::utils::sampling**：以"邻接 cell 的均值"得到 T_c，
    //   再以各向异性距离权重对 cell 周围 8 cell 中心 + 该 cell 的 Neumann/Cauchy
    //   面中心外推做 LSQ 拟合。Dirichlet 面早返回。
    // - 依赖：仅 mhs::core（Model、FieldContext）。
    class ProbeRecorder {
    public:
        ProbeRecorder() = default;

        // 绑定待记录探针与所属模型。observation_points 为空时 recorder 处于
        // 禁用态，recordStep/recordInitial 都是 no-op。
        void initialize(const mhs::core::Model& model);

        // 记录一个时间点。稳态/瞬态通用：稳态求解场景下调用一次即可。
        void record(double time, const std::vector<double>& cell_T);

        // 对外只读访问。
        const std::vector<mhs::core::ProbeTrace>& traces() const { return traces_; }

    private:
        const mhs::core::Model* model_ = nullptr;
        std::vector<mhs::core::ProbeTrace> traces_;

        // 预解析的探针槽位，避免 record 时重复二分。
        struct ProbeSlot {
            double px = 0.0, py = 0.0, pz = 0.0; // 用户坐标系下的探针坐标（用于 LSQ 锚点）
            mhs::Index ix = mhs::invalidIndex, iy = mhs::invalidIndex, iz = mhs::invalidIndex; // 包围 cell 在网格中的下标
            mhs::Index grid_idx = mhs::invalidIndex; // ix*ny*nz + iy*nz + iz
            bool valid = false; // 网格内 + grid_to_cell[grid_idx] != invalidIndex
        };
        std::vector<ProbeSlot> slots_;

        // 在 cell 邻域内做局部 LSQ 拟合，返回探针点温度；越界或邻域无有效 cell
        // 返回 NaN。`time` 注入 FieldContext.t，让时间依赖的 BC/材料表达式
        // 在正确的时刻被求值。
        double sample_one(const ProbeSlot& slot, const std::vector<double>& cell_T, double time) const;
    };

} // namespace mhs::sim
