#pragma once

#include "core/vector.hpp"
#include "model/discrete_mesh.hpp"
#include "model/model_definition.hpp"

namespace mhs::model {
    // Immutable fluid membership inside a FOM; independent of a particular solve.
    struct FluidTopology {
        core::Vector<CellId> fluid_to_global;
        core::Vector<FluidCellId> global_to_fluid;
    };

    struct FluidBoundaryPatch {
        CellId cell = core::invalid_id;
        Axis axis = Axis::X;
        std::array<double, 4> bounds {};
        std::size_t condition = 0;
    };

} // namespace mhs::model
