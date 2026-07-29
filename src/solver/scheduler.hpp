#pragma once

#include "runtime/model.hpp"
#include "runtime/solution.hpp"
#include "solver/assembler.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/time_integration.hpp"

#include <functional>
#include <optional>
#include <span>

namespace mhs::sim {

    struct SolverOpts {
        // Time integration
        time_scheme::IntegratorKind integrator = time_scheme::IntegratorKind::Bdf1;
        time_scheme::StepStrategy step_strategy = time_scheme::StepStrategy::AdaptiveFree;

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

    /// Four matrix positions contributed by a Model-to-port interface.
    struct CouplingMatrixBlocks {
        Eigen::SparseMatrix<double> model;
        Eigen::SparseMatrix<double> model_to_port;
        Eigen::SparseMatrix<double> port_to_model;
        Eigen::SparseMatrix<double> port;
    };

    /// Additive K/C/f contribution from the Model-to-port interface.
    struct CouplingOperators {
        CouplingMatrixBlocks K;
        CouplingMatrixBlocks C;
        Eigen::VectorXd f_model;
        Eigen::VectorXd f_port;
    };

    /// Interface between the Model and an independently owned macro port.
    struct InterfaceCoupling {
        std::optional<CouplingOperators> fixed;
        using NonlinearCoupling = std::function<CouplingOperators(
            std::span<const double> model_state, std::span<const double> port_state, double time)>;
        NonlinearCoupling nonlinear;
    };

    /// Solve the detailed Model coupled to a port-only macro representation.
    ///
    /// macro_port acts only on retained macro-port DoFs and has no knowledge
    /// of the Model. The interface is supplied as a separate object.
    /// State ordering is [Model FVM DoFs, macro port DoFs]. During nonlinear
    /// iteration the solver reassembles the Model block and evaluates only the
    /// optional nonlinear interface contribution.
    mhs::core::SolveResult solve_coupled(const mhs::core::Model& model, const Operators& macro_port,
        const InterfaceCoupling& interface, std::span<const double> initial_state, const SolverOpts& opts = {});

    /// Solve only the Model's detailed FVM region.
    mhs::core::ThermalSolution solve_thermal(
        const mhs::core::Model& model, const SolverOpts& opts = {}, std::span<const double> initial_state = {});

} // namespace mhs::sim
