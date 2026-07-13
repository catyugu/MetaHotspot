#include "common/mesh_utils.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "fluid_preprocessor.hpp"
#include "function_helpers.hpp"
#include "layer_processor.hpp"
#include "preprocessor.hpp"

namespace mhs::sim {
    namespace {
        /// Compute cell widths (d) and centers (c) from vertex coordinates along one axis.
        inline void compute_cell_spacing(
            const std::vector<double>& vertices, std::vector<double>& d, std::vector<double>& c, double si_scale)
        {
            const int n = static_cast<int>(d.size());
            for (int i = 0; i < n; ++i) {
                const double v0 = vertices[i] * si_scale;
                const double v1 = vertices[i + 1] * si_scale;
                d[i] = v1 - v0;
                c[i] = (v0 + v1) * 0.5;
            }
        }

        /// Copy scalar study parameters from IO structure to Model.
        inline void copy_scalar_parameters(mhs::core::Model& model, const mhs::core::IOStructure& io)
        {
            model.study_type = io.study_type;
            model.initial_temperature = io.initial_temperature;
            model.transient_duration = io.transient_duration;
            model.transient_time_step = io.transient_time_step;
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

        /// Convert IO-layer observation point expressions to SI-unit ProbePoints.
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

    std::unique_ptr<mhs::core::Model> Preprocessor::load(const mhs::core::IOStructure& ioStructure,
        const std::optional<mhs::core::FluidOverlay>& fluidOverlay,
        const std::vector<mhs::core::SmartMacroModelData>& trained_models)
    {
        auto model = std::make_unique<mhs::core::Model>();
        copy_scalar_parameters(*model, ioStructure);

        auto symbols = build_symbol_table(ioStructure.variables, ioStructure.functions);
        const double si_scale = mhs::utils::length_unit_to_si(ioStructure.length_unit);
        model->observation_points = build_observation_points(ioStructure.observation_points, symbols, si_scale);

        auto& mesh = model->mesh;
        mesh.nx = (int)ioStructure.mesh_vertex_x.size() - 1;
        mesh.ny = (int)ioStructure.mesh_vertex_y.size() - 1;
        mesh.nz = (int)ioStructure.mesh_vertex_z.size() - 1;

        mesh.dx.resize(mesh.nx);
        mesh.cx.resize(mesh.nx);
        mesh.dy.resize(mesh.ny);
        mesh.cy.resize(mesh.ny);
        mesh.dz.resize(mesh.nz);
        mesh.cz.resize(mesh.nz);

        compute_cell_spacing(ioStructure.mesh_vertex_x, mesh.dx, mesh.cx, si_scale);
        compute_cell_spacing(ioStructure.mesh_vertex_y, mesh.dy, mesh.cy, si_scale);
        compute_cell_spacing(ioStructure.mesh_vertex_z, mesh.dz, mesh.cz, si_scale);

        auto resolved_layers = resolve_geometry(ioStructure.layers, si_scale, symbols);

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
        model->material_table.resize(material_names.size());
        for (size_t m = 0; m < material_names.size(); m++) {
            const auto& mat = ioStructure.materials.at(material_names[m]);
            auto compile = [&](const std::string& expr) {
                return mhs::core::parse(substitute_function_args(expr, "T", ioStructure.functions), symbols);
            };
            model->material_table[m].kx = compile(mat.kx);
            model->material_table[m].ky = compile(mat.ky);
            model->material_table[m].kz = compile(mat.kz);
            model->material_table[m].rho = compile(mat.midu);
            model->material_table[m].c = compile(mat.bi_rerong);
        }

        // Build heat source table + index map (must precede assign_cell_layers).
        auto block_hs_map
            = build_heat_source_table(model->heat_source_table, resolved_layers, ioStructure.functions, symbols);

        // Parse boundaries + other_bc; register BC parameters.
        auto& bc_params = model->bc_params;
        auto bc_rewriter
            = [&](const std::string& s) { return substitute_function_args(s, "T", ioStructure.functions); };
        auto parsed_keys = parse_all_face_keys(ioStructure.boundaries, bc_params, si_scale, bc_rewriter, symbols);

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
            ioStructure.other_bc);

        // Cell assignment and boundary resolution.
        model->cells = assign_cell_layers(resolved_layers, mesh, name_to_idx, block_hs_map);
        resolve_boundary_patches(mesh, model->cells, parsed_keys, other_bc, model->face_bcs);

        // SmartMacro coupling.
        if (!trained_models.empty()) {
            build_smart_block_coupling(
                resolved_layers, mesh, model->cells, trained_models, parsed_keys, other_bc, *model);
        }

        // Fluid coupling.
        if (fluidOverlay.has_value()) {
            mhs::sim::applyFluidOverlay(*model, fluidOverlay, ioStructure, symbols);
            mhs::sim::solveFluidFlow(*model);
        }

        return model;
    }

} // namespace mhs::sim