#include "compiler/model_compiler.hpp"
#include "compiler/fluid_preprocessor.hpp"
#include "compiler/geometry_compiler.hpp"
#include "compiler/model_physics.hpp"
#include "logging/logger.hpp"

#include <chrono>

namespace mhs::compiler {
    namespace {
        void compile_cells(
            mhs::model::Model& model, const mhs::model::ModelDefinition& definition, const mhs::model::MeshSpec& mesh)
        {
            mhs::core::Vector<VolumeGeometry> volumes;
            volumes.reserve(definition.volumes.size());
            for (const auto& volume : definition.volumes)
                volumes.emplace_back(volume.geometry);
            auto geometry = compile_conformal_mesh(mesh, volumes);
            model.discrete = std::move(geometry.mesh);
            bind_cell_physics(model, definition.volumes, geometry.cell_regions);
            for (const auto& probe : definition.point_probes) {
                const mhs::model::Vec3 point {probe.x, probe.y, probe.z};
                model.point_probes.push_back({probe.name, point, locate_conformal_cell(model.discrete, mesh, point)});
            }
        }

        mhs::core::Vector<SurfaceTag> compile_boundaries(mhs::model::Model& model, const CompiledPhysics& physics)
        {
            const std::array<SurfaceRegionGroup, 2> groups {
                {{physics.thermal_regions, physics.default_thermal_boundary}, {physics.fluid_regions}}};
            auto tags = partition_exterior_facets(model.discrete, groups);
            build_cell_facet_incidence(model.discrete);
            model.facet_thermal_condition = std::move(tags[0]);
            return std::move(tags[1]);
        }
    } // namespace

    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition, const mhs::model::MeshSpec& mesh)
    {
        using Clock = std::chrono::steady_clock;
        using Milliseconds [[maybe_unused]] = std::chrono::duration<double, std::milli>;
        [[maybe_unused]] const auto start = Clock::now();
        mhs::model::Model model;
        auto physics = compile_model_physics(model, definition);
        [[maybe_unused]] const auto expressions_done = Clock::now();
        compile_cells(model, definition, mesh);
        [[maybe_unused]] const auto cells_done = Clock::now();
        auto fluid_tags = compile_boundaries(model, physics);
        [[maybe_unused]] const auto boundaries_done = Clock::now();
        fluid::build_domain(model, definition.fluid_boundaries, fluid_tags, physics.initial_viscosity);
        MHS_LOG_DEBUG("Model compilation: expressions={:.3f} ms, cells={:.3f} ms, boundaries={:.3f} ms, flow={:.3f} ms",
            Milliseconds(expressions_done - start).count(), Milliseconds(cells_done - expressions_done).count(),
            Milliseconds(boundaries_done - cells_done).count(), Milliseconds(Clock::now() - boundaries_done).count());
        return model;
    }
} // namespace mhs::compiler
