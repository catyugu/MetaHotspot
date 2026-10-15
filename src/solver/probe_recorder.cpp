#include "solver/probe_recorder.hpp"
#include "solver/interpolation.hpp"
#include <stdexcept>

namespace mhs::sim {
    void ProbeRecorder::initialize(const mhs::core::Model &model) {
        model_ = &model;
        traces_.clear();
        slots_.clear();
        for (const auto &p : model.observation_points) {
            traces_.push_back({p.name, {}, {}});
            ProbeSlot slot;
            slot.px = p.x;
            slot.py = p.y;
            slot.pz = p.z;
            slot.cell = model.discrete.locate({p.x, p.y, p.z});
            slots_.push_back(slot);
        }
    }
    void ProbeRecorder::record(double time, std::span<const double> temperature) {
        if (temperature.size() != model_->dofs.size())
            throw std::invalid_argument("probe state size mismatch");
        for (size_t i = 0; i < slots_.size(); ++i) {
            traces_[i].times.push_back(time);
            traces_[i].values.push_back(sample_one(slots_[i], temperature, time));
        }
    }
    double ProbeRecorder::sample_one(const ProbeSlot &slot, std::span<const double> temperature, double time) const {
        return mhs::utils::sample_point(*model_, temperature, time, slot.cell, {slot.px, slot.py, slot.pz});
    }
} // namespace mhs::sim
