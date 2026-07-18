#pragma once

#include "solver/assembler.hpp"
#include "solver/linear_system.hpp"
#include "solver/solution_history.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace mhs::sim::time_scheme {

    enum class IntegratorKind { Bdf1, Bdf2 };

    LinearSystem build_system(
        IntegratorKind kind, const AssemblyResult& ops, const mhs::core::SolutionHistory& history, double dt);

    struct ErrorControlConfig {
        double abs_tol = 1e-4;
        double safety = 0.9;
    };

    struct ErrorEstimate {
        double error_ratio = 0.0;
        double suggested_factor = 1.0;
    };

    ErrorEstimate estimate_error(const mhs::core::SolutionHistory& accepted, const std::vector<double>& trial_T,
        double trial_dt, const ErrorControlConfig& config);

    class OutputTimeGrid {
    public:
        OutputTimeGrid() = default;

        explicit OutputTimeGrid(double duration, double output_step)
        {
            if (duration <= 0.0 || output_step <= 0.0)
                return;
            const std::size_t n = static_cast<std::size_t>(duration / output_step);
            times_.reserve(n + 1);
            for (std::size_t i = 0; i <= n; ++i)
                times_.push_back(static_cast<double>(i) * output_step);
        }

        explicit OutputTimeGrid(std::vector<double> times) : times_(std::move(times)) { }

        const std::vector<double>& times() const noexcept { return times_; }
        std::size_t size() const noexcept { return times_.size(); }
        bool empty() const noexcept { return times_.empty(); }

    private:
        std::vector<double> times_;
    };

    enum class StepStrategy {
        Free,
        Strict,
        Intermediate,
        Manual,
    };

    class StepController {
    public:
        StepController(StepStrategy strategy, double min_dt, double max_dt, double fixed_dt = 1.0);

        void rebuild(double duration, double output_dt);
        double prepare(double dt_suggested, double current_t, double duration);
        std::vector<double> flush_outputs(double current_t);

        StepStrategy strategy() const noexcept { return strategy_; }
        double min_dt() const noexcept { return min_dt_; }
        double max_dt() const noexcept { return max_dt_; }

    private:
        StepStrategy strategy_ = StepStrategy::Free;
        OutputTimeGrid grid_;
        double min_dt_ = 1e-8;
        double max_dt_ = 1.0;
        double fixed_dt_ = 1.0;
        std::size_t next_idx_ = 0;
        double last_flushed_t_ = 0.0;
        bool planted_ = false;
    };

} // namespace mhs::sim::time_scheme
