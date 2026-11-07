#include "compiler/model_compiler.hpp"
#include "compiler/fluid_preprocessor.hpp"
#include "compiler/geometry_compiler.hpp"
#include "compiler/model_functions.hpp"
#include "core/vector.hpp"
#include "expression/expression.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace mhs::compiler {
    namespace {
        template <typename... Ts> struct overloaded : Ts... {
            using Ts::operator()...;
        };
        template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

        inline double length_unit_to_si(mhs::model::LengthUnit unit)
        {
            switch (unit) {
            case mhs::model::LengthUnit::Meter:
                return 1.0;
            case mhs::model::LengthUnit::Millimeter:
                return 1e-3;
            case mhs::model::LengthUnit::Micrometer:
                return 1e-6;
            case mhs::model::LengthUnit::Nanometer:
                return 1e-9;
            case mhs::model::LengthUnit::Inch:
                return 0.0254;
            case mhs::model::LengthUnit::Mil:
                return 2.54e-5;
            }
            throw std::invalid_argument("invalid length unit");
        }

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

        mhs::core::Vector<mhs::model::ProbePoint> build_observation_points(
            const mhs::core::Vector<mhs::model::ObservationPointSpec>& src, const mhs::expression::SymbolTable& symbols,
            double si_scale)
        {
            mhs::core::Vector<mhs::model::ProbePoint> out;
            out.reserve(src.size());
            for (const auto& s : src) {
                mhs::model::ProbePoint p;
                p.name = s.name;
                p.x = mhs::expression::eval_geometry(s.x, symbols) * si_scale;
                p.y = mhs::expression::eval_geometry(s.y, symbols) * si_scale;
                p.z = mhs::expression::eval_geometry(s.z, symbols) * si_scale;
                out.push_back(std::move(p));
            }
            return out;
        }

        mhs::model::BoundaryConditionId compile_thermal_boundary(const mhs::model::ThermalBoundary& condition,
            mhs::core::Vector<mhs::model::BoundaryCondition>& conditions,
            const mhs::expression::SymbolTable& symbols)
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
            mhs::core::Vector<mhs::model::BoundaryCondition>& conditions, double si_scale,
            const mhs::expression::SymbolTable& symbols)
        {
            mhs::core::Vector<CompiledBoundaryRegion> compiled;
            for (const auto& boundary : boundaries) {
                const auto condition = compile_thermal_boundary(boundary.condition, conditions, symbols);
                for (const auto& region : boundary.regions) {
                    CompiledBoundaryRegion item;
                    item.axis = region.axis;
                    item.coordinate = region.coordinate * si_scale;
                    item.condition_index = condition;
                    item.rectangles.reserve(region.rectangles.size());
                    for (const auto& rectangle : region.rectangles) {
                        item.rectangles.push_back({rectangle.a_min * si_scale, rectangle.a_max * si_scale,
                            rectangle.b_min * si_scale, rectangle.b_max * si_scale});
                    }
                    compiled.push_back(std::move(item));
                }
            }
            return compiled;
        }

        // Build name-to-index from ModelDefinition::materials (insertion order).
        std::unordered_map<std::string, size_t> build_name_to_idx(
            const mhs::core::Vector<mhs::model::NamedMaterial>& materials)
        {
            std::unordered_map<std::string, size_t> map;
            map.reserve(materials.size());
            for (size_t i = 0; i < materials.size(); ++i)
                map[materials[i].name] = i;
            return map;
        }

    } // namespace

    mhs::model::Model build_model_impl(
        const mhs::model::ModelDefinition& definition, const mhs::model::DiscreteMesh* external)
    {
        mhs::model::Model model;
        copy_scalar_parameters(model, definition);

        auto symbols = build_symbol_table(definition.variables, definition.functions);
        const double si_scale = length_unit_to_si(definition.settings.length_unit);
        model.observation_points = build_observation_points(definition.observation_points, symbols, si_scale);

        auto resolved_layers = resolve_geometry(definition.layers, si_scale, symbols);

        // Build material index from ModelDefinition::materials order.
        auto name_to_idx = build_name_to_idx(definition.materials);

        // Compile material properties in ModelDefinition::materials order.
        model.material_table.resize(definition.materials.size());
        mhs::compiler::fluid::FluidMaterialData fluid_materials;
        fluid_materials.initial_viscosity.assign(definition.materials.size(), std::nullopt);

        for (size_t m = 0; m < definition.materials.size(); m++) {
            const auto& mat = definition.materials[m].value;
            auto compile_mat
                = [&](const std::string& expr) { return mhs::expression::parse(expr, symbols); };
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

        // Assign material_id and heat_source_idx to resolved blocks.
        model.heat_source_table.clear();
        for (size_t l = 0; l < definition.layers.size(); ++l) {
            auto& rl = resolved_layers[l];
            const auto& ls = definition.layers[l];
            for (size_t b = 0; b < ls.blocks.size(); ++b) {
                auto& rb = rl.blocks[b];
                const auto& bs = ls.blocks[b];

                // material_id from name table
                auto it = name_to_idx.find(bs.material);
                if (it == name_to_idx.end())
                    throw std::out_of_range("unknown material: " + bs.material);
                rb.material_id = static_cast<mhs::model::MaterialId>(it->second);

                // heat_source_idx: compile the heat source expression, append to table
                rb.heat_source_idx = static_cast<mhs::model::SourceId>(model.heat_source_table.size());
                model.heat_source_table.push_back(mhs::expression::parse(bs.volumetric_heat_source, symbols));
            }
        }

        // Compile boundary patches.
        auto& boundary_conditions = model.boundary_conditions;
        auto compiled_boundaries
            = compile_boundary_patches(definition.boundaries, boundary_conditions, si_scale, symbols);

        auto default_boundary = compile_thermal_boundary(definition.default_boundary, boundary_conditions, symbols);

        if (external) {
            model.discrete = *external;
            build_cell_face_incidence(model.discrete);
            build_external_output_mesh(model);
        }
        else
            compile_cartesian_mesh(model, definition.mesh, si_scale, resolved_layers);
        project_cell_regions(model, resolved_layers);
        model.boundary_faces = project_surface_regions(model.discrete, compiled_boundaries, default_boundary);

        // Fluid coupling.
        const bool has_fluid_material = std::any_of(definition.materials.begin(), definition.materials.end(),
            [](const mhs::model::NamedMaterial& material) { return material.value.is_fluid; });
        if (has_fluid_material) {
            mhs::compiler::fluid::build_domain(model, definition.fluid_boundaries, si_scale, fluid_materials);
        }

        return model;
    }

    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition)
    { return build_model_impl(definition, nullptr); }
    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition, const mhs::model::DiscreteMesh& mesh)
    { return build_model_impl(definition, &mesh); }
} // namespace mhs::compiler
