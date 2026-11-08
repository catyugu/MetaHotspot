#pragma once

#include "core/vector.hpp"
#include "model/discrete_mesh.hpp"
#include "model/model_definition.hpp"

namespace mhs::model {
    // Immutable fluid membership inside a FOM; independent of a particular solve.
    struct FluidTopology {
        core::Vector<CellId> fluid_to_global;
        core::Vector<FluidCellId> global_to_fluid;
        // Connected components of the fluid-cell adjacency graph, indexed by FluidCellId.
        core::Vector<core::EntityIndex> component_by_fluid_cell;
        std::size_t component_count = 0;
        // Active normal-face velocity supports in a staggered Stokes discretization.
        core::Vector<FaceId> velocity_faces;
        // Library fluid-condition IDs by geometric face, invalid on other faces.
        core::Vector<FluidBoundaryId> face_conditions;
    };

    struct FluidBoundaryPatch {
        CellId cell = core::invalid_id;
        Axis axis = Axis::X;
        std::array<double, 4> bounds {};
        FluidBoundaryId condition = core::invalid_id;
    };

} // namespace mhs::model
