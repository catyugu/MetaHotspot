#pragma once

#include "runtime/model.hpp"
#include "solver/assembler.hpp"
#include "solver/scheduler.hpp"

#include <Eigen/Core>
#include <span>
#include <vector>

namespace mhs::macro {

    /// Macro port model expressed in retained coordinates.
    ///
    /// Physical port temperatures are reconstructed as `basis * state_suffix` or,
    /// when basis is empty (unit basis), each macro state IS a physical port temperature.
    ///
    /// Dimension rules:
    ///   macro_state_count = operators.f.size()
    ///   K, C are both macro_state_count × macro_state_count
    ///   When basis is provided: basis is [physical_port_count × macro_state_count]
    ///   When basis is empty  : physical_port_count == macro_state_count (unit/identity)
    struct PortModel {
        mhs::sim::Operators operators;
        Eigen::MatrixXd basis;             // empty = unit basis
        std::size_t physical_port_count = 0;
    };

    /// Geometric connection between exposed FVM faces and physical macro ports.
    ///
    /// `exterior_half_conductance[p]` is the conductance from physical port p
    /// to the shared interface. The FVM-side half conductance is evaluated
    /// from the current material state during every nonlinear linearization.
    struct PortCoupling {
        std::vector<mhs::core::Index> model_cells;
        mhs::core::FaceDir model_face = mhs::core::FaceDir::XP;
        Eigen::VectorXd exterior_half_conductance;
    };

    /// Assemble `C * dx/dt + K * x = f` for `[FVM temperatures, macro states]`.
    mhs::sim::Operators assemble(const mhs::core::Model& model, const PortModel& port,
        const PortCoupling& coupling, std::span<const double> state, double time);

    /// Solve an FVM model coupled to a macro port model.
    /// Internally calls mhs::sim::solve_system with the correct SystemAssembler.
    mhs::core::Solution solve(const mhs::core::Model& model, const PortModel& port,
        const PortCoupling& coupling, std::span<const double> initial_state,
        const mhs::sim::SolveOptions& opts = {});

} // namespace mhs::macro
