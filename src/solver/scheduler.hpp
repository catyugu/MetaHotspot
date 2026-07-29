#pragma once

#include "runtime/model.hpp"
#include "runtime/solution.hpp"
#include "solver/assembler.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/time_integration.hpp"

#include <functional>
#include <span>

namespace mhs::sim {

    struct SolverOpts {
        // Time integration
        time_scheme::IntegratorKind integrator = time_scheme::IntegratorKind::Bdf1;
        time_scheme::StepStrategy step_strategy = time_scheme::StepStrategy::Adaptive;

        // Error control
        double error_abs_tol = 1e-4;
        double error_safety = 0.9;

        // Step bounds
        double min_dt = 1e-12;
        double max_dt = 1.0;
        double fixed_dt = 1.0;

        // Linear solver
        SolverSpec solver;

        // Non-linear solver
        NonLinearConfig nonlinear;
    };

    /// Whole-system linearization evaluated at the current nonlinear iterate.
    using SystemAssembler = std::function<Operators(std::span<const double> state, double time)>;

    /// Accepted-state observer. Every transient state is an integration result
    /// at an exactly aligned output time; modal coordinates are never interpolated.
    using StateObserver = std::function<void(double time, std::span<const double> state)>;

    struct Study {
        mhs::core::StudyType type = mhs::core::StudyType::Steady;
        double duration = 0.0;
        double output_interval = 1.0;
    };

    /// Run nonlinear iteration and time integration over an externally
    /// assembled system. The scheduler owns when to reassemble; the callback
    /// owns all state partitioning and coupling physics.
    mhs::core::SolveResult solve_system(const Study& study, const SystemAssembler& assemble,
        std::span<const double> initial_state, const SolverOpts& opts = {}, const StateObserver& observe = {});

    /// Solve only the Model's detailed FVM region.
    mhs::core::ThermalSolution solve_thermal(
        const mhs::core::Model& model, const SolverOpts& opts = {}, std::span<const double> initial_state = {});

} // namespace mhs::sim
