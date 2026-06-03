#pragma once

#include "common/internal_model.hpp"
#include "solver/solver.hpp"

namespace mhs {

    struct SchedulerConfig {
        double transient_duration = 0.0;
        double time_step = 1.0;
        int max_nonlinear_iterations = 50;
        double nonlinear_tolerance = 1e-6;
        double underrelaxation = 1.0;
        bool is_steady = false;
    };

    class Scheduler {
    public:
        Scheduler() = default;
        explicit Scheduler(const SchedulerConfig& config) : config_(config) { }
        ~Scheduler() = default;

        void setModel(model::InternalModel* model) { model_ = model; }
        void setSolver(std::unique_ptr<Solver> solver);

        void run();
        const std::vector<double>& solution() const;

    private:
        model::InternalModel* model_ = nullptr;
        std::unique_ptr<Solver> solver_;
        SchedulerConfig config_;
        model::GlobalState state_;
        std::vector<double> solution_;
    };

} // namespace mhs