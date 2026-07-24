#pragma once

#include "runtime/model.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <span>
#include <vector>

namespace mhs::sim::fluid {

    struct FluidAssemblyIncrement {
        std::vector<Eigen::Triplet<double>> matrix_entries;
        Eigen::VectorXd rhs;
    };

    // Return fluid-only corrections to the base thermal operator. Every matrix
    // coordinate is either a diagonal or an existing direct-neighbor entry.
    FluidAssemblyIncrement assemble_increment(
        const mhs::core::Model& model, std::span<const double> state, double current_time);

} // namespace mhs::sim::fluid
