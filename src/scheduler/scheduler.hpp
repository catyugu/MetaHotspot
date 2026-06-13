#pragma once

#include "data/internal_model.hpp"
#include "linear_solver/linear_solver.hpp"
#include "scheduler/probe_recorder.hpp"
#include <memory>
#include <vector>

namespace mhs::sim {

    class Scheduler {
    public:
        Scheduler() = default;
        ~Scheduler() = default;

        void setModel(mhs::core::InternalModel* model);
        void setSolver(std::unique_ptr<LinearSolver> solver);
        void run();

        const std::vector<double>& solution() const { return solution_; }
        double currentTime() const { return state_.current_time; }
        const std::vector<mhs::core::ProbeTrace>& probeTraces() const { return probe_recorder_.traces(); }

    private:
        mhs::core::InternalModel* model_ = nullptr;
        std::unique_ptr<LinearSolver> solver_;
        mhs::core::GlobalState state_;
        std::vector<double> solution_;
        ProbeRecorder probe_recorder_;
    };

} // namespace mhs::sim