#pragma once

#include "core/vector.hpp"
#include "model/discrete_mesh.hpp"

namespace mhs::model {
    // Presentation-only connectivity. Cartesian XML sampling shape belongs here,
    // never in the thermal topology or the algebraic state space.
    struct OutputMesh {
        mhs::core::Vector<Vec3> points;
        mhs::core::Vector<std::array<PointId, 8>> cell_vertices;
        std::array<std::size_t, 3> cartesian_shape {};
    };
} // namespace mhs::model
