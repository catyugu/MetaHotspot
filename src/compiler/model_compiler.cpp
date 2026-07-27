#include "compiler/fluid_preprocessor.hpp"
#include "compiler/geometry_compiler.hpp"
#include "compiler/model_compiler.hpp"
#include "compiler/model_functions.hpp"
#include "numerics/expression/expr.hpp"

#include <algorithm>
#include <cassert>
#include <span>
#include <stdexcept>

namespace mhs::sim {
    namespace {
        template <typename... Ts> struct overloaded : Ts... {
            using Ts::operator()...;
        };
        template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

        double length_unit_to_si(mhs::model::LengthUnit unit)
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
            return 1.0;
        }

        inline void compute_cell_spacing(
            std::span<const double> vertices, std::vector<double>& d, std::vector<double>& c, double si_scale)
        {
            const mhs::core::Index n = static_cast<mhs::core::Index>(d.size());
            for (mhs::core::Index i = 0; i < n; ++i) {
                const double v0 = vertices[i] * si_scale;
                const double v1 = vertices[i + 1] * si_scale;
                d[i] = v1 - v0;
                c[i] = (v0 + v1) * 0.5;
            }
        }

        inline void copy_scalar_parameters(mhs::core::Model& model, const mhs::model::ModelDefinition& definition)
        {
            model.study_type = definition.settings.study_type == mhs::model::StudyType::Transient
                ? mhs::core::StudyType::Transient
                : mhs::core::StudyType::Steady;
            model.initial_temperature = definition.settings.initial_temperature;
            model.transient_duration = definition.settings.transient_duration;
            model.transient_time_step = definition.settings.transient_output_interval;
        }

        mhs::core::SymbolTable build_symbol_table(const std::vector<mhs::model::VariableSpec>& variables,
            const std::vector<mhs::model::NamedFunction>& functions)
        {
            mhs::core::SymbolTable empty_for_geometry_eval;
            mhs::core::SymbolTable symbols;
            for (const auto& var : variables) {
                double val = mhs::core::eval_geometry(var.value, empty_for_geometry_eval);
                symbols.variables[var.name] = val;
            }
            register_all_functions(symbols, functions);
            return symbols;
        }

        std::vector<mhs::core::ProbePoint> build_observation_points(
            const std::vector<mhs::model::ObservationPointSpec>& src, const mhs::core::SymbolTable& symbols,
            double si_scale)
        {
            std::vector<mhs::core::ProbePoint> out;
            out.reserve(src.size());
            for (const auto& s : src) {
                mhs::core::ProbePoint p;
                p.name = s.name;
                p.x = mhs::core::eval_geometry(s.x, symbols) * si_scale;
                p.y = mhs::core::eval_geometry(s.y, symbols) * si_scale;
                p.z = mhs::core::eval_geometry(s.z, symbols) * si_scale;
                out.push_back(std::move(p));
            }
            return out;
        }

        DefaultBoundary compile_thermal_boundary(const mhs::model::ThermalBoundary& condition,
            mhs::core::BCParamTable& parameters, const std::function<std::string(const std::string&)>& rewriter,
            const mhs::core::SymbolTable& symbols)
        {
            DefaultBoundary result;
            std::visit(overloaded {
                           [&](const mhs::model::DirichletBoundary& value) {
                               result.type = mhs::core::BcType::FirstType;
                               result.parameter_index
                                   = static_cast<mhs::core::TableIndex>(parameters.dirichlet_T.size());
                               parameters.dirichlet_T.push_back(mhs::core::parse(rewriter(value.temperature), symbols));
                           },
                           [&](const mhs::model::NeumannBoundary& value) {
                               result.type = mhs::core::BcType::SecondType;
                               result.parameter_index = static_cast<mhs::core::TableIndex>(parameters.neumann_q.size());
                               parameters.neumann_q.push_back(mhs::core::parse(rewriter(value.heat_flux), symbols));
                           },
                           [&](const mhs::model::ConvectionBoundary& value) {
                               result.type = mhs::core::BcType::ThirdType;
                               result.parameter_index = static_cast<mhs::core::TableIndex>(parameters.cauchy_h.size());
                               parameters.cauchy_h.push_back(mhs::core::parse(rewriter(value.coefficient), symbols));
                               parameters.cauchy_T_inf.push_back(
                                   mhs::core::parse(rewriter(value.ambient_temperature), symbols));
                           },
                       },
                condition);
            return result;
        }

        std::vector<CompiledBoundaryRegion> compile_boundary_patches(
            const std::vector<mhs::model::BoundaryPatch>& boundaries, mhs::core::BCParamTable& parameters,
            double si_scale, const std::function<std::string(const std::string&)>& rewriter,
            const mhs::core::SymbolTable& symbols)
        {
            std::vector<CompiledBoundaryRegion> compiled;
            for (const auto& boundary : boundaries) {
                const auto value = compile_thermal_boundary(boundary.condition, parameters, rewriter, symbols);
                for (const auto& region : boundary.regions) {
                    CompiledBoundaryRegion item;
                    item.axis = region.axis;
                    item.coordinate = region.coordinate * si_scale;
                    item.type = value.type;
                    item.parameter_index = value.parameter_index;
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
            const std::vector<mhs::model::NamedMaterial>& materials)
        {
            std::unordered_map<std::string, size_t> map;
            map.reserve(materials.size());
            for (size_t i = 0; i < materials.size(); ++i)
                map[materials[i].name] = i;
            return map;
        }

    } // namespace

    mhs::core::Model build_model(const mhs::model::ModelDefinition& definition)
    {
        mhs::core::Model model;
        copy_scalar_parameters(model, definition);

        auto symbols = build_symbol_table(definition.variables, definition.functions);
        const double si_scale = length_unit_to_si(definition.settings.length_unit);
        model.observation_points = build_observation_points(definition.observation_points, symbols, si_scale);

        auto& mesh = model.mesh;
        mesh.nx = static_cast<mhs::core::Index>(definition.mesh.x_vertices.size()) - 1;
        mesh.ny = static_cast<mhs::core::Index>(definition.mesh.y_vertices.size()) - 1;
        mesh.nz = static_cast<mhs::core::Index>(definition.mesh.z_vertices.size()) - 1;

        assert(mesh.nx > 0 && mesh.ny > 0 && mesh.nz > 0);
        mesh.dx.resize(static_cast<size_t>(mesh.nx));
        mesh.cx.resize(static_cast<size_t>(mesh.nx));
        mesh.dy.resize(static_cast<size_t>(mesh.ny));
        mesh.cy.resize(static_cast<size_t>(mesh.ny));
        mesh.dz.resize(static_cast<size_t>(mesh.nz));
        mesh.cz.resize(static_cast<size_t>(mesh.nz));

        compute_cell_spacing(definition.mesh.x_vertices, mesh.dx, mesh.cx, si_scale);
        compute_cell_spacing(definition.mesh.y_vertices, mesh.dy, mesh.cy, si_scale);
        compute_cell_spacing(definition.mesh.z_vertices, mesh.dz, mesh.cz, si_scale);

        auto resolved_layers = resolve_geometry(definition.layers, si_scale, symbols);

        // Build material index from ModelDefinition::materials order.
        auto name_to_idx = build_name_to_idx(definition.materials);

        // Compile material properties in ModelDefinition::materials order.
        model.material_table.resize(definition.materials.size());
        mhs::sim::fluid::FluidMaterialData fluid_materials;
        fluid_materials.initial_viscosity.assign(definition.materials.size(), std::nullopt);

        for (size_t m = 0; m < definition.materials.size(); m++) {
            const auto& mat = definition.materials[m].value;
            auto compile_mat = [&](const std::string& expr) {
                return mhs::core::parse(substitute_function_args(expr, "T"), symbols);
            };
            auto& props = model.material_table[m];
            props.kx = compile_mat(mat.conductivity_x);
            props.ky = compile_mat(mat.conductivity_y);
            props.kz = compile_mat(mat.conductivity_z);
            props.rho = compile_mat(mat.density);
            props.c = compile_mat(mat.specific_heat);
            if (mat.dynamic_viscosity.has_value()) {
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
                rb.material_id = static_cast<mhs::core::TableIndex>(it->second);

                // heat_source_idx: compile the heat source expression, append to table
                rb.heat_source_idx = static_cast<mhs::core::TableIndex>(model.heat_source_table.size());
                model.heat_source_table.push_back(
                    mhs::core::parse(substitute_function_args(bs.volumetric_heat_source, "t"), symbols));
            }
        }

        // Compile boundary patches.
        auto& bc_params = model.bc_params;
        auto bc_rewriter = [&](const std::string& s) { return substitute_function_args(s, "T"); };
        auto compiled_boundaries
            = compile_boundary_patches(definition.boundaries, bc_params, si_scale, bc_rewriter, symbols);

        auto default_boundary = compile_thermal_boundary(definition.default_boundary, bc_params, bc_rewriter, symbols);

        // Cell assignment and boundary resolution.
        model.cells = assign_cell_layers(resolved_layers, mesh);
        resolve_boundary_patches(mesh, model.cells, compiled_boundaries, default_boundary, model.face_bcs);

        // Fluid coupling.
        const bool has_fluid_material = std::any_of(fluid_materials.initial_viscosity.begin(),
            fluid_materials.initial_viscosity.end(), [](const auto& value) { return value.has_value(); });
        if (has_fluid_material) {
            mhs::sim::fluid::build_domain(model, definition.fluid_boundaries, si_scale, fluid_materials);
        }

        // State layout: thermal-only in the base path.
        const auto cell_count = static_cast<mhs::core::Index>(model.cells.cell_to_grid.size());
        model.layout = mhs::core::StateLayout {mhs::core::DofRange {0, cell_count}, cell_count};

        return model;
    }

} // namespace mhs::sim
