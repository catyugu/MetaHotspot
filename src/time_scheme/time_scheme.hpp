#pragma once

#include "assembler/assembler.hpp"
#include "data/internal_model.hpp"
#include "data/time_step_buffer.hpp"
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
        double abs_tol = 1e-3;
        double rel_tol = 1e-3;
        std::size_t max_order = 2;
        double safety = 0.9;
        double output_dt = 0.0;
    };

    struct StepDecision {
        double dt = 0.0;
        std::size_t order = 1;
    };

    class TimeScheme {
    public:
        virtual ~TimeScheme() = default;

        virtual void initialize(mhs::core::TimeStepBuffer& history, mhs::core::GlobalState& state) const
        {
            history.reset(state.T);
        }

        virtual StepDecision select_step(
            const mhs::core::TimeStepBuffer& history, double current_t, double duration) const
            = 0;
        virtual LinearSystem build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
            const mhs::core::TimeStepBuffer& history, std::size_t order, double dt) const
            = 0;

        virtual void accept_or_reject(const std::vector<double>& error_estimate) const { (void)error_estimate; }
        virtual bool is_output_boundary(double t) const
        {
            (void)t;
            return true;
        }
        virtual const TimeSchemeConfig& config() const = 0;
    };

    std::unique_ptr<TimeScheme> create_scheme(const TimeSchemeConfig& cfg);

} // namespace mhs::sim::time_scheme