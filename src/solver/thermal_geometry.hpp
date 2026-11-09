#pragma once

#include "model/model.hpp"

namespace mhs::solver {

    inline mhs::model::Vec3 boundary_center(const mhs::model::CellGeometry& cell, const mhs::model::BoundaryFace& term)
    {
        mhs::model::Vec3 point {};
        const auto axis = static_cast<std::size_t>(term.axis);
        point[axis] = term.sign > 0 ? cell.max[axis] : cell.min[axis];
        point[axis == 0 ? 1 : 0] = (term.bounds[0] + term.bounds[1]) / 2;
        point[axis == 2 ? 1 : 2] = (term.bounds[2] + term.bounds[3]) / 2;
        return point;
    }

} // namespace mhs::solver
