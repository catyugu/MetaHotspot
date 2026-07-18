#include "expr/expr.hpp"
#include "boundary_processor.hpp"
#include "fluid/fluid_preprocessor.hpp"
#include "function_helpers.hpp"
#include "layer_processor.hpp"
#include "preprocessor.hpp"
#include "utils/mesh_utils.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace mhs::sim {
    namespace {
        /// Compute cell widths (d) and centers (c) from vertex coordinates along one axis.
        inline void compute_cell_spacing(
            const std::vector<double>& vertices, std::vector<double>& d, std::vector<double>& c, double si_scale)
        {
            const mhs::Index n = static_cast<mhs::Index>(d.size());
            for (mhs::Index i = 0; i < n; ++i) {
                const double v0 = vertices[i] * si_scale;
                const double v1 = vertices[i + 1] * si_scale;
                d[i] = v1 - v0;
                c[i] = (v0 + v1) * 0.5;
            }
        }

        /// Copy scalar study parameters from the definition to Model.
        inline void copy_scalar_parameters(mhs::core::Model& model, const mhs::model::ModelDefinition& definition)
        {
            model.study_type = definition.settings.study_type == mhs::model::StudyType::Transient
                ? mhs::core::StudyType::Transient
                : mhs::core::StudyType::Steady;
            model.initial_temperature = definition.settings.initial_temperature;
            model.transient_duration = definition.settings.transient_duration;
            model.transient_time_step = definition.settings.transient_output_interval;
        }

        /// Evaluate geometry variables and register all user-defined functions.
        /// Returns a SymbolTable populated with both geometry variables and native functions.
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

        /// Convert observation point expressions to SI-unit ProbePoints.
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

        /// Build the heat source expression table and the per-layer, per-block index map.
        /// Index 0 in the table is reserved as a zero constant (default for blocks without sources).
        /// Returns block_hs_map[l][b] = index into heat_source_table for layer l, block b.
        std::vector<std::vector<uint16_t>> build_heat_source_table(
            std::vector<mhs::core::CompiledExpression>& heat_source_table,
            const std::vector<ResolvedLayerGeometry>& resolved_layers,
            const std::vector<mhs::model::NamedFunction>& functions,
            const mhs::core::SymbolTable& symbols)
        {
            heat_source_table.clear();
            heat_source_table.push_back(mhs::core::CompiledExpression::make_constant(0.0));

            std::vector<std::vector<uint16_t>> block_hs_map(resolved_layers.size());
            for (size_t l = 0; l < resolved_layers.size(); l++) {
                block_hs_map[l].resize(resolved_layers[l].blocks.size(), 0);
                for (size_t b = 0; b < resolved_layers[l].blocks.size(); b++) {
                    const uint16_t hs_idx = static_cast<uint16_t>(heat_source_table.size());
                    const std::string& raw = resolved_layers[l].blocks[b].volumetric_heat_source;
                    heat_source_table.push_back(
                        mhs::core::parse(substitute_function_args(raw, "t", functions), symbols));
                    block_hs_map[l][b] = hs_idx;
                }
            }
            return block_hs_map;
        }

        const mhs::model::MaterialSpec& find_material(
            const std::vector<mhs::model::NamedMaterial>& materials, const std::string& name)
        {
            const auto it = std::find_if(materials.begin(), materials.end(),
                [&](const mhs::model::NamedMaterial& material) { return material.name == name; });
            if (it == materials.end())
                throw std::out_of_range("Unknown material: " + name);
            return it->value;
        }

    } // namespace

    mhs::core::Model build_model(const mhs::model::ModelDefinition& definition)
    {
        mhs::core::Model model;
        copy_scalar_parameters(model, definition);

        auto symbols = build_symbol_table(definition.variables, definition.functions);
        const double si_scale = mhs::utils::length_unit_to_si(definition.settings.length_unit);
        model.observation_points = build_observation_points(definition.observation_points, symbols, si_scale);

        auto& mesh = model.mesh;
        mesh.nx = static_cast<mhs::Index>(definition.mesh.x_vertices.size()) - 1;
        mesh.ny = static_cast<mhs::Index>(definition.mesh.y_vertices.size()) - 1;
        mesh.nz = static_cast<mhs::Index>(definition.mesh.z_vertices.size()) - 1;

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

        // Collect unique material names from resolved blocks.
        std::vector<std::string> material_names;
        std::unordered_map<std::string, size_t> name_to_idx;
        for (const auto& rl : resolved_layers)
            for (const auto& rb : rl.blocks)
                if (name_to_idx.find(rb.material) == name_to_idx.end()) {
                    name_to_idx[rb.material] = material_names.size();
                    material_names.push_back(rb.material);
                }

        // Compile material property expressions.
        model.material_table.resize(material_names.size());
        mhs::sim::fluid::FluidMaterialData fluid_materials;
        fluid_materials.is_fluid.assign(material_names.size(), 0);
        fluid_materials.initial_viscosity.assign(material_names.size(), 0.0);
        for (size_t m = 0; m < material_names.size(); m++) {
            const auto& mat = find_material(definition.materials, material_names[m]);
            auto compile = [&](const std::string& expr) {
                return mhs::core::parse(substitute_function_args(expr, "T", definition.functions), symbols);
            };
            auto& props = model.material_table[m];
            props.kx = compile(mat.conductivity_x);
            props.ky = compile(mat.conductivity_y);
            props.kz = compile(mat.conductivity_z);
            props.rho = compile(mat.density);
            props.c = compile(mat.specific_heat);
            if (mat.dynamic_viscosity.has_value()) {
                fluid_materials.is_fluid[m] = 1;
                fluid_materials.initial_viscosity[m]
                    = compile(*mat.dynamic_viscosity).eval({0, 0, 0, model.initial_temperature, 0});
            }
        }

        // Build heat source table + index map (must precede assign_cell_layers).
        auto block_hs_map
            = build_heat_source_table(model.heat_source_table, resolved_layers, definition.functions, symbols);

        // Compile ordered boundary patches and register BC parameters.
        auto& bc_params = model.bc_params;
        auto bc_rewriter = [&](const std::string& s) { return substitute_function_args(s, "T", definition.functions); };
        auto compiled_boundaries
            = compile_boundary_patches(definition.boundaries, bc_params, si_scale, bc_rewriter, symbols);

        // Compile the default boundary fallback.
        DefaultBoundary default_boundary;
        std::visit(overloaded {
                       [&](const mhs::model::DirichletBoundary& condition) {
                           default_boundary.type = mhs::core::BcType::FirstType;
                           default_boundary.parameter_index = static_cast<uint16_t>(bc_params.dirichlet_T.size());
                           bc_params.dirichlet_T.push_back(
                               mhs::core::parse(bc_rewriter(condition.temperature), symbols));
                       },
                       [&](const mhs::model::NeumannBoundary& condition) {
                           default_boundary.type = mhs::core::BcType::SecondType;
                           default_boundary.parameter_index = static_cast<uint16_t>(bc_params.neumann_q.size());
                           bc_params.neumann_q.push_back(
                               mhs::core::parse(bc_rewriter(condition.heat_flux), symbols));
                       },
                       [&](const mhs::model::ConvectionBoundary& condition) {
                           default_boundary.type = mhs::core::BcType::ThirdType;
                           default_boundary.parameter_index = static_cast<uint16_t>(bc_params.cauchy_h.size());
                           bc_params.cauchy_h.push_back(
                               mhs::core::parse(bc_rewriter(condition.coefficient), symbols));
                           bc_params.cauchy_T_inf.push_back(
                               mhs::core::parse(bc_rewriter(condition.ambient_temperature), symbols));
                       },
                   },
            definition.default_boundary);

        // Cell assignment and boundary resolution.
        model.cells = assign_cell_layers(resolved_layers, mesh, name_to_idx, block_hs_map);
        resolve_boundary_patches(mesh, model.cells, compiled_boundaries, default_boundary, model.face_bcs);

        // Fluid coupling.
        const bool has_fluid_material = std::any_of(
            fluid_materials.is_fluid.begin(), fluid_materials.is_fluid.end(), [](uint8_t value) { return value != 0; });
        if (has_fluid_material) {
            mhs::sim::fluid::build_domain(model, definition.fluid_boundaries, si_scale, fluid_materials);
        }

        return model;
    }

} // namespace mhs::sim
