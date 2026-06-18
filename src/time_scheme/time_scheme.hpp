#pragma once

#include "data/internal_model.hpp"
#include "data/linear_system.hpp"
#include "data/solution_history.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace mhs::sim::time_scheme {

    enum class TimeSchemeKind { Bdf1, Bdf2, AdaptiveBdf };

    struct TimeSchemeConfig {
        TimeSchemeKind kind = TimeSchemeKind::Bdf1;
        double initial_dt = 1.0;
        double min_dt = 1e-9;
        double max_dt = 1.0;
        double abs_tol = 1e-4;
        double rel_tol = 1e-6;
        std::size_t max_order = 2;
        double safety = 0.9;
        double output_dt = 0.0;
    };

    struct StepDecision {
        double dt = 0.0;
        std::size_t order = 1;
    };

    struct StepResult {
        bool accepted = true;
    };

    class TimeScheme {
    public:
        virtual ~TimeScheme() = default;

        virtual void initialize(mhs::core::SolutionHistory& accepted, mhs::core::GlobalState& state) const
        {
            accepted.initialize(state.T);
        }

        virtual StepDecision select_step(
            const mhs::core::SolutionHistory& accepted, double current_t, double duration) const
            = 0;

        virtual LinearSystem build_system(const AssemblyResult& ops,
            const mhs::core::SolutionHistory& accepted, std::size_t order, double dt) const
            = 0;

        // 统一处理误差评估、步长接受或拒绝，更新内部的自适应策略参数
        virtual StepResult evaluate_step(
            const mhs::core::SolutionHistory& accepted, const std::vector<double>& trial_T, double trial_dt) const
            = 0;

        virtual bool is_output_boundary(double t) const
        {
            (void)t;
            return true;
        }

        virtual const TimeSchemeConfig& config() const = 0;
    };

    std::unique_ptr<TimeScheme> create_scheme(const TimeSchemeConfig& cfg);

} // namespace mhs::sim::time_scheme