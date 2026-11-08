#include "solver/probe_recorder.hpp"
#include "thermal_reconstruction.hpp"

namespace mhs::solver {
    ProbeRecorder::ProbeRecorder(const mhs::model::FomDomain& model, std::span<const mhs::model::PointProbe> probes)
        : model_(model)
    {
        for (const auto& p : probes) {
            traces_.push_back({p.name, {}});
            ProbeSlot slot;
            slot.point = {p.x, p.y, p.z};
            slot.cell = mhs::solver::locate_cell(model.discrete, {p.x, p.y, p.z});
            slots_.push_back(slot);
        }
    }
    void ProbeRecorder::record(
        double time, std::span<const double> temperature, std::span<const SurfaceTrace> interfaces)
    {
        for (size_t i = 0; i < slots_.size(); ++i) {
            const auto& slot = slots_[i];
            traces_[i].values.push_back(sample_point(model_, temperature, time, slot.cell, slot.point, interfaces));
        }
    }
} // namespace mhs::solver
