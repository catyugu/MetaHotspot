#pragma once

#include "common/model.hpp"
#include "common/solution.hpp"
#include "common/solver.hpp"
#include "solver/assembler.hpp"

#include <Eigen/Core>
#include <span>
#include <vector>

namespace mhs::macro {

    /// Macro port model expressed in retained coordinates.
    struct PortModel {
        mhs::sim::Operators operators;
        Eigen::MatrixXd basis; // empty = unit basis
        std::size_t physical_port_count = 0;
    };

    /// Geometric connection between exposed FVM faces and physical macro ports.
    struct PortCoupling {
        std::vector<mhs::core::Index> model_cells;
        mhs::core::FaceDir model_face = mhs::core::FaceDir::XP;
    };

    /// Assemble `C * dx/dt + K * x = f` for `[FVM temperatures, macro states]`.
    mhs::sim::Operators assemble(const mhs::core::Model& model, const PortModel& port, const PortCoupling& coupling,
        std::span<const double> state, double time);

    /// Solve an FVM model coupled to a macro port model.
    mhs::core::Solution solve(const mhs::core::Model& model, const PortModel& port, const PortCoupling& coupling,
        std::span<const double> initial_state, const mhs::sim::SolveOptions& opts = {});

} // namespace mhs::macro
