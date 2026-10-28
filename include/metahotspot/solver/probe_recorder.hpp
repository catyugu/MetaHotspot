#pragma once

#include "core/model.hpp"
#include "core/solution.hpp"

#include <span>
#include <vector>

namespace mhs::sim {

    // Cache probe locations once; reconstruct local temperatures at each output time.
    class ProbeRecorder {
    public:
        ProbeRecorder() = default;

        // 绑定待记录探针与所属模型。observation_points 为空时 recorder 处于
        // 禁用态，record 不产生记录。
        void initialize(const mhs::core::Model& model);

        // 记录一个时间点。稳态/瞬态通用：稳态求解场景下调用一次即可。
        void record(double time, std::span<const double> cell_T);

        // 对外只读访问。
        const std::vector<mhs::core::ProbeTrace>& traces() const { return traces_; }

    private:
        const mhs::core::Model* model_ = nullptr;
        std::vector<mhs::core::ProbeTrace> traces_;

        // 预解析的探针槽位，避免 record 时重复定位。
        struct ProbeSlot {
            double px = 0.0, py = 0.0, pz = 0.0; // SI coordinates
            mhs::core::CellId cell;
        };
        std::vector<ProbeSlot> slots_;
    };

} // namespace mhs::sim
