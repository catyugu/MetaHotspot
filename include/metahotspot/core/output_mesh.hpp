#pragma once
#include "core/discrete_mesh.hpp"

namespace mhs::core {
    // Presentation-only connectivity. Cartesian XML sampling shape belongs here,
    // never in the thermal topology or the algebraic state space.
    struct OutputMesh {
        std::vector<Vec3> points;
        std::vector<std::array<Index, 8>> cell_vertices;
        std::array<Index, 3> cartesian_shape {};
    };
} // namespace mhs::core
