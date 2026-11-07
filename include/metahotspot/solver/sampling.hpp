#pragma once

#include "model/model.hpp"
#include <optional>
#include <span>

namespace mhs::solver {
    // Reconstruct the local boundary temperature at a point on the specified
    // cell. Coincident patches use Dirichlet precedence. Interior points have
    // no boundary value. State size and cell ID must match the compiled model.
    std::optional<double> boundary_value_at(const mhs::model::Model& model, std::span<const double> temperature,
        double time, mhs::model::CellId cell, const mhs::model::Vec3& point);
} // namespace mhs::solver
