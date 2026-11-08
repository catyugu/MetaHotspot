#pragma once

#include "model/model.hpp"
#include "solver/solution.hpp"

#include "core/vector.hpp"
#include <span>
#include <utility>

namespace mhs::solver {

    // Cache probe locations once; reconstruct local temperatures at each output time.
    class ProbeRecorder {
    public:
        ProbeRecorder(const mhs::model::FomDomain& model, std::span<const mhs::model::PointProbe> probes);

        // 记录一个时间点。稳态/瞬态通用：稳态求解场景下调用一次即可。
        void record(double time, std::span<const double> cell_T);

        mhs::core::Vector<mhs::solver::ProbeTrace> take_traces() && { return std::move(traces_); }

    private:
        const mhs::model::FomDomain& model_;
        mhs::core::Vector<mhs::solver::ProbeTrace> traces_;

        // 预解析的探针槽位，避免 record 时重复定位。
        struct ProbeSlot {
            mhs::model::Vec3 point; // SI coordinates
            mhs::model::CellId cell = mhs::core::invalid_id;
        };
        mhs::core::Vector<ProbeSlot> slots_;
    };

} // namespace mhs::solver
