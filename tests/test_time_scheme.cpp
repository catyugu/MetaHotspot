#include "core/csr_ops.hpp"
#include "solver/solve.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
    mhs::solver::Operators decay_operators(std::span<const double>, double)
    { return {mhs::core::build_csr(1, 1, {{0, 0, 1.0}}), mhs::core::build_csr(1, 1, {{0, 0, 1.0}}), {0.0}}; }

    TEST(TimeScheme, AdaptiveDecayAndInterpolatedOutput)
    {
        const mhs::solver::Study study {mhs::model::StudyType::Transient, 1.05, 0.2};
        const std::array initial {1.0};
        mhs::solver::SolveOptions options;
        options.time_tolerance = 1e-7;
        options.max_time_step = 0.1;
        mhs::core::Vector<double> observed;
        const auto solution = mhs::solver::solve_system(
            study, decay_operators, initial, options, [&](double time, std::span<const double> value) {
                observed.push_back(time);
                EXPECT_NEAR(value[0], std::exp(-time), 3e-6);
            });
        ASSERT_TRUE(solution.converged);
        EXPECT_DOUBLE_EQ(solution.time, study.duration);
        ASSERT_EQ(solution.snapshot_times.size(), 7u);
        EXPECT_EQ(solution.snapshot_times, observed);
        EXPECT_DOUBLE_EQ(solution.snapshot_states.front(), initial[0]);
        EXPECT_DOUBLE_EQ(solution.snapshot_times.back(), study.duration);
        EXPECT_DOUBLE_EQ(solution.state[0], solution.snapshot_states.back());
    }

    TEST(TimeScheme, MaximumStepAndNoOutputGrid)
    {
        const mhs::solver::Study study {mhs::model::StudyType::Transient, 0.5, 0.0};
        const std::array initial {1.0};
        mhs::solver::SolveOptions options;
        options.max_time_step = 0.02;
        const auto solution = mhs::solver::solve_system(study, decay_operators, initial, options);
        ASSERT_TRUE(solution.converged);
        for (std::size_t i = 1; i < solution.snapshot_times.size(); ++i)
            EXPECT_LE(solution.snapshot_times[i] - solution.snapshot_times[i - 1], 0.02 + 1e-15);
    }

    TEST(TimeScheme, RejectsInvalidTimeControls)
    {
        const mhs::solver::Study study {mhs::model::StudyType::Transient, 1.0, 0.1};
        const std::array initial {1.0};
        mhs::solver::SolveOptions options;
        options.time_tolerance = 0.0;
        EXPECT_THROW(mhs::solver::solve_system(study, decay_operators, initial, options), std::invalid_argument);
        options.time_tolerance = 1e-4;
        for (const double value : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
            options.max_time_step = value;
            EXPECT_THROW(mhs::solver::solve_system(study, decay_operators, initial, options), std::invalid_argument);
        }
    }
} // namespace
