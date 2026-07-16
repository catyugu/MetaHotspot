#include "preprocessor.hpp"

#include "fluid_preprocessor.hpp"
#include "layer_processor.hpp"
#include "model_builder.hpp"

namespace mhs::sim {

    std::unique_ptr<mhs::core::Model> Preprocessor::load(
        const mhs::core::IOStructure& io_structure, const std::optional<mhs::core::FluidOverlay>& fluid_overlay)
    {
        auto context = detail::make_build_context(io_structure);
        auto model = std::make_unique<mhs::core::Model>();

        detail::copy_study_config(*model, io_structure);
        model->mesh = detail::build_mesh(context);
        model->observation_points = detail::build_observation_points(context);

        auto resolved_layers = resolve_geometry(io_structure.layers, context.si_scale, context.symbols);
        auto materials = detail::build_material_catalog(*model, context, resolved_layers);
        auto heat_sources = detail::build_heat_source_catalog(*model, context, resolved_layers);
        auto boundaries = detail::build_boundary_catalog(*model, context);

        model->cells = assign_cell_layers(resolved_layers, model->mesh, materials.name_to_index, heat_sources);
        resolve_boundary_patches(
            model->mesh, model->cells, boundaries.explicit_patches, boundaries.fallback, model->face_bcs);

        if (fluid_overlay.has_value()) {
            auto fluid_workspace
                = mhs::sim::buildFluidDomain(*model, *fluid_overlay, io_structure, context.symbols, materials.names);
            if (fluid_workspace.has_value())
                mhs::sim::solveFluidFlow(*model, *fluid_workspace);
        }
        return model;
    }

} // namespace mhs::sim
