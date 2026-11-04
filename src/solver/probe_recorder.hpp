#pragma once

#include "core/model.hpp"
#include "core/solution.hpp"

#include "core/vector.hpp"
#include <span>
#include <utility>

namespace mhs::sim {

    // Cache probe locations once; reconstruct local temperatures at each output time.
    class ProbeRecorder {
    public:
        explicit ProbeRecorder(const mhs::core::Model& model);

        // 记录一个时间点。稳态/瞬态通用：稳态求解场景下调用一次即可。
        void record(double time, std::span<const double> cell_T);

        mhs::core::Vector<mhs::core::ProbeTrace> take_traces() && { return std::move(traces_); }

    private:
        const mhs::core::Model& model_;
        mhs::core::Vector<mhs::core::ProbeTrace> traces_;

        // 预解析的探针槽位，避免 record 时重复定位。
        struct ProbeSlot {
            mhs::core::Vec3 point; // SI coordinates
            mhs::core::CellId cell;
        };
        mhs::core::Vector<ProbeSlot> slots_;
    };

} // namespace mhs::sim
