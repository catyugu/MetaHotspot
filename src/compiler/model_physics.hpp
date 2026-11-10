#pragma once

#include "compiler/geometry_compiler.hpp"
#include "model/model.hpp"

namespace mhs::compiler {
    // Scratch data for the later boundary and frozen-flow stages. Parsed
    // expressions belong to Model; the temporary symbol table never escapes.
    struct CompiledPhysics {
        mhs::core::Vector<double> initial_viscosity;
        mhs::core::Vector<TaggedSurfaceRegion> thermal_regions, fluid_regions;
        mhs::model::BoundaryConditionId default_thermal_boundary;
    };

    CompiledPhysics compile_model_physics(mhs::model::Model& model, const mhs::model::ModelDefinition& definition);
    void bind_cell_physics(mhs::model::Model& model, std::span<const mhs::model::VolumeSpec> volumes,
        std::span<const RegionId> cell_regions);
} // namespace mhs::compiler
