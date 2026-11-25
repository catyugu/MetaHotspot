#pragma once

// Solver-internal reconstruction of point temperatures from cell values.
#include "thermal_physics.hpp"
#include <functional>
#include <span>

namespace mhs::solver {

    using BoundarySampler = std::function<BoundaryEvaluation(model::FacetId, const model::Vec3*)>;

    // Temperature at an arbitrary point: the boundary trace on an external
    // patch, otherwise a Green-Gauss gradient extrapolation from the center.
    double sample_point(const mhs::model::Fom& model, std::span<const double> temperature, double time,
        mhs::model::CellId cell, const mhs::model::Vec3& point, const BoundarySampler& sample_boundary = {}, std::optional<double> conductivity_temperature = std::nullopt);

} // namespace mhs::solver
