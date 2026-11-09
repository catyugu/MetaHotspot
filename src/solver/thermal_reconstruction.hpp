#pragma once

// Solver-internal reconstruction of point temperatures from cell values.
#include "model/model.hpp"
#include <span>

namespace mhs::solver {

    mhs::model::CellId locate_cell(const mhs::model::DiscreteMesh& mesh, const mhs::model::Vec3& point);

    // Temperature at an arbitrary point: the boundary trace on an external
    // patch, otherwise a Green-Gauss gradient extrapolation from the center.
    double sample_point(const mhs::model::Model& model, std::span<const double> temperature, double time,
        mhs::model::CellId cell, const mhs::model::Vec3& point);

} // namespace mhs::solver
