#pragma once

// Solver-internal reconstruction of point temperatures from cell values.
#include "model/fom.hpp"
#include <functional>
#include <span>

namespace mhs::solver {

    struct BoundarySample {
        double temperature;
        bool dirichlet;
    };

    BoundarySample sample_boundary_temperature(const model::BoundaryCondition& condition, const model::Facet& patch,
        double temperature, double time, double resistance);

    using BoundarySampler = std::function<BoundarySample(model::FacetId, const model::Vec3*)>;

    // Temperature at an arbitrary point: the boundary trace on an external
    // patch, otherwise a Green-Gauss gradient extrapolation from the center.
    double sample_point(const mhs::model::Fom& model, std::span<const double> temperature, double time,
        mhs::model::CellId cell, const mhs::model::Vec3& point, const BoundarySampler& sample_boundary = {});

} // namespace mhs::solver
