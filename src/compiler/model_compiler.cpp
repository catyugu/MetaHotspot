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

        inline void copy_scalar_parameters(mhs::model::Model& model, const mhs::model::ModelDefinition& definition)
        {
            model.study_type = definition.settings.study_type;
            model.initial_temperature = definition.settings.initial_temperature;
            model.transient_duration = definition.settings.transient_duration;
            model.transient_output_interval = definition.settings.transient_output_interval;
        }

        mhs::expression::SymbolTable build_symbol_table(const mhs::core::Vector<mhs::model::VariableSpec>& variables,
            const mhs::core::Vector<mhs::model::NamedFunction>& functions)
        {
            mhs::expression::SymbolTable empty_for_geometry_eval;
            mhs::expression::SymbolTable symbols;
            for (const auto& var : variables) {
                double val = mhs::expression::eval_geometry(var.value, empty_for_geometry_eval);
                symbols.variables[var.name] = val;
            }
            register_all_functions(symbols, functions);
            return symbols;
        }

        mhs::model::BoundaryConditionId compile_thermal_boundary(const mhs::model::ThermalBoundary& condition,
            mhs::core::Vector<mhs::model::BoundaryCondition>& conditions, const mhs::expression::SymbolTable& symbols)
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
            const auto index = static_cast<BoundaryConditionId>(conditions.size());
            conditions.push_back(std::move(result));
            return index;
        }

        mhs::core::Vector<CompiledBoundaryRegion> compile_boundary_patches(
            const mhs::core::Vector<mhs::model::BoundaryPatch>& boundaries,
            mhs::core::Vector<mhs::model::BoundaryCondition>& conditions, const mhs::expression::SymbolTable& symbols)
        {
            mhs::core::Vector<CompiledBoundaryRegion> compiled;
            for (const auto& boundary : boundaries) {
                const auto condition = compile_thermal_boundary(boundary.condition, conditions, symbols);
                for (const auto& region : boundary.regions) {
                    CompiledBoundaryRegion item;
                    item.axis = region.axis;
                    item.coordinate = region.coordinate;
                    item.condition_index = condition;
                    item.rectangles.reserve(region.rectangles.size());
                    for (const auto& rectangle : region.rectangles) {
                        item.rectangles.push_back({rectangle.a_min, rectangle.a_max, rectangle.b_min, rectangle.b_max});
                    }
                    compiled.push_back(std::move(item));
                }
            }
            return compiled;
        }

    } // namespace

    mhs::model::Model build_model_impl(
        const mhs::model::ModelDefinition& definition, const mhs::model::DiscreteMesh* external)
    {
        mhs::model::Model model;
        copy_scalar_parameters(model, definition);

        auto symbols = build_symbol_table(definition.variables, definition.functions);
        for (const auto& p : definition.observation_points)
            model.observation_points.push_back({p.name, p.x, p.y, p.z});

        // Compile material properties in ModelDefinition::materials order.
        model.material_table.resize(definition.materials.size());
        mhs::compiler::fluid::FluidMaterialData fluid_materials;
        fluid_materials.initial_viscosity.assign(definition.materials.size(), std::nullopt);

        for (size_t m = 0; m < definition.materials.size(); m++) {
            const auto& mat = definition.materials[m].value;
            auto compile_mat = [&](const std::string& expr) { return mhs::expression::parse(expr, symbols); };
            auto& props = model.material_table[m];
            props.kx = compile_mat(mat.conductivity_x);
            props.ky = compile_mat(mat.conductivity_y);
            props.kz = compile_mat(mat.conductivity_z);
            props.rho = compile_mat(mat.density);
            props.c = compile_mat(mat.specific_heat);
            if (mat.is_fluid) {
                if (!mat.dynamic_viscosity.has_value() || mat.dynamic_viscosity->empty()) {
                    throw std::runtime_error(
                        "fluid material '" + definition.materials[m].name + "' is missing DynamicViscosity");
                }
                fluid_materials.initial_viscosity[m]
                    = compile_mat(*mat.dynamic_viscosity).eval({0, 0, 0, model.initial_temperature, 0});
            }
        }

        for (const auto& source : definition.sources)
            model.heat_source_table.push_back(mhs::expression::parse(source, symbols));
        for (const auto& volume : definition.volumes) {
            if (volume.material >= definition.materials.size())
                throw std::out_of_range("volume material ID out of range");
            if (volume.source != mhs::core::invalid_id && volume.source >= definition.sources.size())
                throw std::out_of_range("volume source ID out of range");
            for (const auto& box : volume.geometry)
                for (int a = 0; a < 3; ++a)
                    if (!std::isfinite(box.min[a]) || !std::isfinite(box.max[a]) || box.min[a] >= box.max[a])
                        throw std::invalid_argument("box bounds must be finite and increasing");
        }

        // Compile boundary patches.
        auto& boundary_conditions = model.boundary_conditions;
        auto compiled_boundaries = compile_boundary_patches(definition.boundaries, boundary_conditions, symbols);

        auto default_boundary = compile_thermal_boundary(definition.default_boundary, boundary_conditions, symbols);

        if (external) {
            model.discrete = *external;
            build_cell_face_incidence(model.discrete);
        }
        else
            compile_cartesian_mesh(model, definition.mesh, definition.volumes);
        project_cell_regions(model, definition.volumes);
        model.boundary_faces = project_surface_regions(model.discrete, compiled_boundaries, default_boundary);

        // Fluid coupling.
        const bool has_fluid_material = std::any_of(definition.materials.begin(), definition.materials.end(),
            [](const mhs::model::NamedMaterial& material) { return material.value.is_fluid; });
        if (has_fluid_material) {
            mhs::compiler::fluid::build_domain(model, definition.fluid_boundaries, fluid_materials);
        }

        return model;
    }

    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition)
    { return build_model_impl(definition, nullptr); }
    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition, const mhs::model::DiscreteMesh& mesh)
    { return build_model_impl(definition, &mesh); }
} // namespace mhs::compiler
