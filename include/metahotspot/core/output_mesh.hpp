#pragma once

#include "core/discrete_mesh.hpp"
#include "core/vector.hpp"

namespace mhs::core {
    // Presentation-only connectivity. Cartesian XML sampling shape belongs here,
    // never in the thermal topology or the algebraic state space.
    struct OutputMesh {
        mhs::core::Vector<Vec3> points;
        mhs::core::Vector<std::array<Index, 8>> cell_vertices;
        std::array<Index, 3> cartesian_shape {};
    };
} // namespace mhs::core
