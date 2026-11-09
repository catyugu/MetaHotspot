#pragma once

// Solver-internal reconstruction of point and node temperatures in a
// cell-centered geometry with patch-level boundary conditions.
//
// The external boundary trace is a local algebraic unknown eliminated by static
// condensation, so nothing here is persistent state: every entry point takes the
// converged cell state and re-evaluates the local law on demand.
#include "model/model.hpp"
#include <span>

namespace mhs::solver {

    mhs::model::CellId locate_cell(const mhs::model::DiscreteMesh& mesh, const mhs::model::Vec3& point);

    // Boundary terms sorted by cell in the compiled model.
    std::span<const mhs::model::BoundaryFace> boundary_faces_on_cell(
        const mhs::model::FomDomain& model, mhs::model::CellId cell);

    mhs::model::Vec3 boundary_center(const mhs::model::CellGeometry& cell, const mhs::model::BoundaryFace& term);
    bool boundary_contains(
        const mhs::model::CellGeometry& cell, const mhs::model::BoundaryFace& term, const mhs::model::Vec3& point);

    // Condensed contributions of one external boundary patch to its owner cell.
    struct BoundaryTrace {
        double temperature = 0.0;
        double conductance = 0.0;
        double load = 0.0;
    };

    // Solve the local boundary law on a boundary integration region and return the
    // trace temperature with its condensed conductance and load.
    BoundaryTrace evaluate_boundary(const mhs::model::BoundaryCondition& condition, const mhs::model::Vec3& center,
        double area, double normal_distance, double cell_temperature, double normal_conductivity, double time);

    struct SurfaceTrace {
        model::BoundaryFace face;
        double temperature = 0.0;
    };

    // Temperature at an arbitrary point: the boundary trace on an external
    // patch, otherwise a Green-Gauss gradient extrapolation from the center.
    double sample_point(const mhs::model::FomDomain& model, const mhs::model::CompiledLibrary& library,
        std::span<const double> temperature, double time, mhs::model::CellId cell, const mhs::model::Vec3& point,
        std::span<const SurfaceTrace> interfaces = {});

} // namespace mhs::solver
