#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "fluid/fluid_preprocessor.hpp"
#include "function_helpers.hpp"
#include "layer_processor.hpp"
#include "preprocessor.hpp"
#include "utils/mesh_utils.hpp"

#include <algorithm>
#include <cassert>

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
        inline void copy_scalar_parameters(mhs::core::Model& model, const mhs::core::ModelDefinition& definition)
        {
            model.study_type = definition.study_type;
            model.initial_temperature = definition.initial_temperature;
            model.transient_duration = definition.transient_duration;
            model.transient_time_step = definition.transient_time_step;
        }

        /// Evaluate geometry variables and register all user-defined functions.
        /// Returns a SymbolTable populated with both geometry variables and native functions.
        mhs::core::SymbolTable build_symbol_table(const std::vector<mhs::core::Variable>& variables,
            const std::unordered_map<std::string, mhs::core::Function>& functions)
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
            const std::vector<mhs::core::ObservationPoint3D>& src, const mhs::core::SymbolTable& symbols,
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
            const std::unordered_map<std::string, mhs::core::Function>& functions,
            const mhs::core::SymbolTable& symbols)
        {
            heat_source_table.clear();
            heat_source_table.push_back(mhs::core::CompiledExpression::make_constant(0.0));

            std::vector<std::vector<uint16_t>> block_hs_map(resolved_layers.size());
            for (size_t l = 0; l < resolved_layers.size(); l++) {
                block_hs_map[l].resize(resolved_layers[l].blocks.size(), 0);
                for (size_t b = 0; b < resolved_layers[l].blocks.size(); b++) {
                    const uint16_t hs_idx = static_cast<uint16_t>(heat_source_table.size());
                    const std::string& raw = resolved_layers[l].blocks[b].ti_reyuan_expr;
                    heat_source_table.push_back(
                        mhs::core::parse(substitute_function_args(raw, "t", functions), symbols));
                    block_hs_map[l][b] = hs_idx;
                }
            }
            return block_hs_map;
        }

    } // namespace

    mhs::core::Model build_model(const mhs::core::ModelDefinition& definition)
    {
        mhs::core::Model model;
        copy_scalar_parameters(model, definition);

        auto symbols = build_symbol_table(definition.variables, definition.functions);
        const double si_scale = mhs::utils::length_unit_to_si(definition.length_unit);
        model.observation_points = build_observation_points(definition.observation_points, symbols, si_scale);

        auto& mesh = model.mesh;
        mesh.nx = static_cast<mhs::Index>(definition.mesh_vertex_x.size()) - 1;
        mesh.ny = static_cast<mhs::Index>(definition.mesh_vertex_y.size()) - 1;
        mesh.nz = static_cast<mhs::Index>(definition.mesh_vertex_z.size()) - 1;

        assert(mesh.nx > 0 && mesh.ny > 0 && mesh.nz > 0);
        mesh.dx.resize(static_cast<size_t>(mesh.nx));
        mesh.cx.resize(static_cast<size_t>(mesh.nx));
        mesh.dy.resize(static_cast<size_t>(mesh.ny));
        mesh.cy.resize(static_cast<size_t>(mesh.ny));
        mesh.dz.resize(static_cast<size_t>(mesh.nz));
        mesh.cz.resize(static_cast<size_t>(mesh.nz));

        compute_cell_spacing(definition.mesh_vertex_x, mesh.dx, mesh.cx, si_scale);
        compute_cell_spacing(definition.mesh_vertex_y, mesh.dy, mesh.cy, si_scale);
        compute_cell_spacing(definition.mesh_vertex_z, mesh.dz, mesh.cz, si_scale);

        auto resolved_layers = resolve_geometry(definition.layers, si_scale, symbols);

        // Collect unique material names from resolved blocks.
        std::vector<std::string> material_names;
        std::unordered_map<std::string, size_t> name_to_idx;
        for (const auto& rl : resolved_layers)
            for (const auto& rb : rl.blocks)
                if (name_to_idx.find(rb.material_name) == name_to_idx.end()) {
                    name_to_idx[rb.material_name] = material_names.size();
                    material_names.push_back(rb.material_name);
                }

        // Compile material property expressions.
        model.material_table.resize(material_names.size());
        mhs::sim::fluid::FluidMaterialData fluid_materials;
        fluid_materials.is_fluid.assign(material_names.size(), 0);
        fluid_materials.initial_viscosity.assign(material_names.size(), 0.0);
        for (size_t m = 0; m < material_names.size(); m++) {
            const auto& mat = definition.materials.at(material_names[m]);
            auto compile = [&](const std::string& expr) {
                return mhs::core::parse(substitute_function_args(expr, "T", definition.functions), symbols);
            };
            auto& props = model.material_table[m];
            props.kx = compile(mat.kx);
            props.ky = compile(mat.ky);
            props.kz = compile(mat.kz);
            props.rho = compile(mat.midu);
            props.c = compile(mat.bi_rerong);
            if (!mat.dynamic_viscosity.empty()) {
                fluid_materials.is_fluid[m] = 1;
                fluid_materials.initial_viscosity[m]
                    = compile(mat.dynamic_viscosity).eval({0, 0, 0, model.initial_temperature, 0});
            }
        }

        // Build heat source table + index map (must precede assign_cell_layers).
        auto block_hs_map
            = build_heat_source_table(model.heat_source_table, resolved_layers, definition.functions, symbols);

        // Parse boundaries + other_bc; register BC parameters.
        auto& bc_params = model.bc_params;
        auto bc_rewriter = [&](const std::string& s) { return substitute_function_args(s, "T", definition.functions); };
        auto parsed_keys = parse_all_face_keys(definition.boundaries, bc_params, si_scale, bc_rewriter, symbols);

        // Parse other_bc fallback.
        OtherBC other_bc;
        std::visit(overloaded {
                       [&](const mhs::core::FirstTypeThermalBC& b) {
                           other_bc.type = mhs::core::BcType::FirstType;
                           other_bc.param_idx = static_cast<uint16_t>(bc_params.dirichlet_T.size());
                           bc_params.dirichlet_T.push_back(mhs::core::parse(bc_rewriter(b.temperature), symbols));
                       },
                       [&](const mhs::core::SecondTypeThermalBC& b) {
                           other_bc.type = mhs::core::BcType::SecondType;
                           other_bc.param_idx = static_cast<uint16_t>(bc_params.neumann_q.size());
                           bc_params.neumann_q.push_back(mhs::core::parse(bc_rewriter(b.heat_flux), symbols));
                       },
                       [&](const mhs::core::ThirdTypeThermalBC& b) {
                           other_bc.type = mhs::core::BcType::ThirdType;
                           other_bc.param_idx = static_cast<uint16_t>(bc_params.cauchy_h.size());
                           bc_params.cauchy_h.push_back(mhs::core::parse(bc_rewriter(b.convection_coeff), symbols));
                           bc_params.cauchy_T_inf.push_back(mhs::core::parse(bc_rewriter(b.T_inf), symbols));
                       },
                   },
            definition.other_bc);

        // Cell assignment and boundary resolution.
        model.cells = assign_cell_layers(resolved_layers, mesh, name_to_idx, block_hs_map);
        resolve_boundary_patches(mesh, model.cells, parsed_keys, other_bc, model.face_bcs);

        // Fluid coupling.
        const bool has_fluid_material = std::any_of(
            fluid_materials.is_fluid.begin(), fluid_materials.is_fluid.end(), [](uint8_t value) { return value != 0; });
        if (has_fluid_material) {
            mhs::sim::fluid::build_domain(model, definition.fluid_boundaries, si_scale, fluid_materials);
        }

        return model;
    }

} // namespace mhs::sim
