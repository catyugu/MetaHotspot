#pragma once

#include "solver/assembler.hpp"
#include "solver/linear_system.hpp"
#include "solver/solution_history.hpp"

namespace mhs::sim::time_scheme {

    enum class IntegratorKind { Bdf1, Bdf2 };

    LinearSystem build_system(
        IntegratorKind kind, const Operators& ops, const mhs::core::SolutionHistory& history, double dt);

    struct ErrorControlConfig {
        double abs_tol = 1e-4;
        double safety = 0.9;
    };

    struct ErrorEstimate {
        double error_ratio = 0.0;
        double suggested_factor = 1.0;
    };

    ErrorEstimate estimate_error(const mhs::core::SolutionHistory& accepted, std::span<const double> trial_state,
        double trial_dt, const ErrorControlConfig& config);

    enum class StepStrategy { Adaptive, Fixed };

    class StepController {
    public:
        StepController(StepStrategy strategy, double min_dt, double max_dt, double duration, double output_interval,
            double fixed_dt = 1.0);

        double prepare(double dt_suggested, double current_t);
        bool output_due(double current_t);

        StepStrategy strategy() const noexcept { return strategy_; }
        double min_dt() const noexcept { return min_dt_; }
        double max_dt() const noexcept { return max_dt_; }

    private:
        StepStrategy strategy_ = StepStrategy::Adaptive;
        double duration_ = 0.0;
        double output_interval_ = 0.0;
        double min_dt_ = 1e-8;
        double max_dt_ = 1.0;
        double fixed_dt_ = 1.0;
        double next_output_ = 0.0;
    };

} // namespace mhs::sim::time_scheme
