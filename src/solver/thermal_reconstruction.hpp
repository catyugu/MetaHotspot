#pragma once

// Solver-internal reconstruction of point and node temperatures in a
// cell-centered geometry with patch-level boundary conditions.
//
// The external boundary trace is a local algebraic unknown eliminated by static
// condensation, so nothing here is persistent state: every entry point takes the
// converged cell state and re-evaluates the local law on demand.
#include "core/model.hpp"

#include <optional>
#include <span>

namespace mhs::sim {

    // Condensed contributions of one external boundary patch to its owner cell.
    struct BoundaryTrace {
        double temperature = 0.0;
        double conductance = 0.0;
        double load = 0.0;
    };

    // Solve the local boundary law on an already projected patch and return the
    // trace temperature with its condensed conductance and load.
    BoundaryTrace evaluate_boundary(const mhs::core::BoundaryCondition& condition, const mhs::core::FacePatch& face,
        double cell_temperature, double normal_conductivity, double time);

    // Trace temperature where `point` lies on one of `cell`'s external boundary
    // patches, averaged over coincident patches with Dirichlet precedence.
    // Empty when the point is interior to the cell.
    std::optional<double> boundary_value_at(const mhs::core::Model& model, std::span<const double> temperature,
        double time, mhs::core::CellId cell, const mhs::core::Vec3& point);

    // Temperature at an arbitrary point: the boundary trace on an external
    // patch, otherwise a Green-Gauss gradient extrapolation from the center.
    double sample_point(const mhs::core::Model& model, std::span<const double> temperature, double time,
        mhs::core::CellId cell, const mhs::core::Vec3& point);

} // namespace mhs::sim
