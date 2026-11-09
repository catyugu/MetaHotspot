#include "compiler/model_compiler.hpp"
#include "compiler/fluid_preprocessor.hpp"
#include "compiler/geometry_compiler.hpp"
#include "compiler/model_functions.hpp"
#include "core/vector.hpp"
#include "expression/expression.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace mhs::compiler {
    namespace {
        template <typename... Ts> struct overloaded : Ts... {
            using Ts::operator()...;
        };
        template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

        mhs::model::BoundaryCondition compile_thermal_boundary(
            const mhs::model::ThermalBoundary& condition, const mhs::expression::SymbolTable& symbols)
        {
            using namespace mhs::core;
            using namespace mhs::model;
            using namespace mhs::expression;
            auto compile = [&](const auto& expression) { return parse(expression, symbols); };
            auto result = std::visit(
                overloaded {
                    [&](const mhs::model::DirichletBoundary& value) -> BoundaryCondition {
                        auto temperature = compile(value.temperature);
                        // A trace that prescribes itself is not a Dirichlet
                        // condition; require an explicit temperature.
                        if (temperature.depends_on_temperature())
                            throw std::invalid_argument("Dirichlet boundary temperature must not depend on T");
                        return Dirichlet {std::move(temperature)};
                    },
                    [&](const mhs::model::NeumannBoundary& value) -> BoundaryCondition {
                        return Neumann {compile(value.heat_flux)};
                    },
                    [&](const mhs::model::ConvectionBoundary& value) -> BoundaryCondition {
                        return Convection {compile(value.coefficient), compile(value.ambient_temperature)};
                    },
                },
                condition);
            return result;
        }

        mhs::core::Vector<TaggedSurfaceRegion> compile_thermal_boundaries(
            const mhs::core::Vector<mhs::model::ThermalBoundarySpec>& boundaries, std::size_t condition_count)
        {
            mhs::core::Vector<TaggedSurfaceRegion> compiled;
            for (const auto& boundary : boundaries) {
                if (boundary.condition >= condition_count)
                    throw std::out_of_range("thermal boundary condition ID out of range");
                for (const auto& region : boundary.regions)
                    compiled.push_back({region, boundary.condition});
            }
            return compiled;
        }

    } // namespace

    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition)
    {
        if (!std::isfinite(definition.reference_temperature))
            throw std::invalid_argument("reference temperature must be finite");

        const auto& library = definition.library();
        mhs::model::Model model;
        model.reference_temperature = definition.reference_temperature;

        auto symbols = build_symbol_table(definition);
        for (const auto& p : definition.point_probes)
            model.point_probes.push_back({p.name, p.x, p.y, p.z});

        // Compile material properties in ModelDefinition::materials order.
        model.material_table.resize(library.materials.size());
        mhs::compiler::fluid::FluidMaterialData fluid_materials;
        fluid_materials.initial_viscosity.assign(library.materials.size(), std::nullopt);

        for (size_t m = 0; m < library.materials.size(); m++) {
            const auto& mat = library.materials[m].value;
            auto compile_mat = [&](const std::string& expr) { return mhs::expression::parse(expr, symbols); };
            auto& props = model.material_table[m];
            props.kx = compile_mat(mat.conductivity_x);
            props.ky = compile_mat(mat.conductivity_y);
            props.kz = compile_mat(mat.conductivity_z);
            props.rho = compile_mat(mat.density);
            props.c = compile_mat(mat.specific_heat);
            if (mat.dynamic_viscosity.has_value()) {
                if (mat.dynamic_viscosity->empty()) {
                    throw std::runtime_error(
                        "fluid material '" + library.materials[m].name + "' is missing DynamicViscosity");
                }
                fluid_materials.initial_viscosity[m]
                    = compile_mat(*mat.dynamic_viscosity).eval({0, 0, 0, model.reference_temperature, 0});
            }
        }

        for (const auto& source : library.sources)
            model.heat_source_table.push_back(mhs::expression::parse(source, symbols));
        for (const auto& volume : definition.volumes) {
            if (volume.material >= library.materials.size())
                throw std::out_of_range("volume material ID out of range");
            if (volume.source != mhs::core::invalid_id && volume.source >= library.sources.size())
                throw std::out_of_range("volume source ID out of range");
            for (const auto& box : volume.geometry)
                for (int a = 0; a < 3; ++a)
                    if (!std::isfinite(box.min[a]) || !std::isfinite(box.max[a]) || box.min[a] >= box.max[a])
                        throw std::invalid_argument("box bounds must be finite and increasing");
        }

        // Compile boundary patches.
        auto& boundary_conditions = model.boundary_conditions;
        for (const auto& condition : library.thermal_boundaries)
            boundary_conditions.push_back(compile_thermal_boundary(condition, symbols));
        auto compiled_boundaries
            = compile_thermal_boundaries(definition.thermal_boundaries, boundary_conditions.size());

        const auto default_thermal_boundary = definition.default_thermal_boundary;
        if (default_thermal_boundary >= boundary_conditions.size())
            throw std::out_of_range("default thermal boundary condition ID out of range");
        for (const auto& boundary : definition.fluid_boundaries)
            if (boundary.condition >= library.fluid_boundaries.size())
                throw std::out_of_range("fluid boundary condition ID out of range");

        compile_cartesian_mesh(model, definition.mesh, definition.volumes);
        project_cell_regions(model, definition.volumes);
        for (const auto material : model.cell_material)
            if (material >= model.material_table.size())
                throw std::out_of_range("cell has no valid material assignment");
        for (const auto& patch :
            project_exterior_surface_regions(model.discrete, compiled_boundaries, default_thermal_boundary))
            model.boundary_faces.push_back({patch.cell, patch.axis, patch.sign,
                static_cast<mhs::model::BoundaryConditionId>(patch.tag), patch.bounds});

        // Fluid coupling.
        const bool has_fluid_material = std::any_of(library.materials.begin(), library.materials.end(),
            [](const mhs::model::NamedMaterial& material) { return material.value.dynamic_viscosity.has_value(); });
        if (has_fluid_material) {
            mhs::compiler::fluid::build_domain(model, definition.fluid_boundaries, library, fluid_materials);
        }

        return model;
    }

} // namespace mhs::compiler
