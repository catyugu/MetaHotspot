#include "model_builder.hpp"

#include "function_helpers.hpp"
#include "logger/logger.hpp"
#include "utils/mesh_utils.hpp"

#include <cmath>
#include <limits>

namespace mhs::sim::detail {
    namespace {

        void validate_axis(const std::vector<double>& vertices, const char* field_name)
        {
            if (vertices.size() < 2) {
                MHS_FATAL("{} must contain at least two vertices", field_name);
            }
            for (size_t i = 0; i < vertices.size(); ++i) {
                if (!std::isfinite(vertices[i])) {
                    MHS_FATAL("{} contains a non-finite vertex at index {}", field_name, i);
                }
                if (i > 0 && vertices[i] <= vertices[i - 1]) {
                    MHS_FATAL("{} must be strictly increasing (indices {} and {})", field_name, i - 1, i);
                }
            }
        }

        void compute_cell_spacing(const std::vector<double>& vertices, std::vector<double>& widths,
            std::vector<double>& centers, double si_scale)
        {
            const mhs::Index count = static_cast<mhs::Index>(widths.size());
            for (mhs::Index i = 0; i < count; ++i) {
                const double v0 = vertices[i] * si_scale;
                const double v1 = vertices[i + 1] * si_scale;
                widths[i] = v1 - v0;
                centers[i] = (v0 + v1) * 0.5;
            }
        }

        mhs::core::SymbolTable build_symbol_table(const mhs::core::IOStructure& input)
        {
            mhs::core::SymbolTable geometry_symbols;
            mhs::core::SymbolTable symbols;
            for (const auto& variable : input.variables) {
                symbols.variables[variable.name] = mhs::core::eval_geometry(variable.value, geometry_symbols);
            }
            register_all_functions(symbols, input.functions);
            return symbols;
        }

        mhs::core::CompiledExpression compile_temperature_expression(
            const std::string& expression, const BuildContext& context)
        {
            return mhs::core::parse(
                substitute_function_args(expression, "T", context.input.functions), context.symbols);
        }

    } // namespace

    BuildContext make_build_context(const mhs::core::IOStructure& input)
    { return {input, build_symbol_table(input), mhs::utils::length_unit_to_si(input.length_unit)}; }

    void copy_study_config(mhs::core::Model& model, const mhs::core::IOStructure& input)
    {
        model.study_type = input.study_type;
        model.initial_temperature = input.initial_temperature;
        model.transient_duration = input.transient_duration;
        model.transient_time_step = input.transient_time_step;
    }

    mhs::core::MeshGeometry build_mesh(const BuildContext& context)
    {
        validate_axis(context.input.mesh_vertex_x, "mesh_vertex_x");
        validate_axis(context.input.mesh_vertex_y, "mesh_vertex_y");
        validate_axis(context.input.mesh_vertex_z, "mesh_vertex_z");

        mhs::core::MeshGeometry mesh;
        mesh.nx = context.input.mesh_vertex_x.size() - 1;
        mesh.ny = context.input.mesh_vertex_y.size() - 1;
        mesh.nz = context.input.mesh_vertex_z.size() - 1;

        mesh.dx.resize(mesh.nx);
        mesh.cx.resize(mesh.nx);
        mesh.dy.resize(mesh.ny);
        mesh.cy.resize(mesh.ny);
        mesh.dz.resize(mesh.nz);
        mesh.cz.resize(mesh.nz);

        compute_cell_spacing(context.input.mesh_vertex_x, mesh.dx, mesh.cx, context.si_scale);
        compute_cell_spacing(context.input.mesh_vertex_y, mesh.dy, mesh.cy, context.si_scale);
        compute_cell_spacing(context.input.mesh_vertex_z, mesh.dz, mesh.cz, context.si_scale);
        return mesh;
    }

    std::vector<mhs::core::ProbePoint> build_observation_points(const BuildContext& context)
    {
        std::vector<mhs::core::ProbePoint> result;
        result.reserve(context.input.observation_points.size());
        for (const auto& source : context.input.observation_points) {
            result.push_back({source.name, mhs::core::eval_geometry(source.x, context.symbols) * context.si_scale,
                mhs::core::eval_geometry(source.y, context.symbols) * context.si_scale,
                mhs::core::eval_geometry(source.z, context.symbols) * context.si_scale});
        }
        return result;
    }

    MaterialCatalog build_material_catalog(mhs::core::Model& model, const BuildContext& context,
        const std::vector<mhs::sim::ResolvedLayerGeometry>& resolved_layers)
    {
        MaterialCatalog catalog;
        for (const auto& layer : resolved_layers) {
            for (const auto& block : layer.blocks) {
                if (catalog.name_to_index.find(block.material_name) != catalog.name_to_index.end())
                    continue;
                if (catalog.names.size() >= std::numeric_limits<uint16_t>::max()) {
                    MHS_FATAL("too many materials for the uint16_t material index");
                }
                catalog.name_to_index.emplace(block.material_name, catalog.names.size());
                catalog.names.push_back(block.material_name);
            }
        }

        model.material_table.resize(catalog.names.size());
        for (size_t index = 0; index < catalog.names.size(); ++index) {
            const auto& source = context.input.materials.at(catalog.names[index]);
            auto& target = model.material_table[index];
            target.kx = compile_temperature_expression(source.kx, context);
            target.ky = compile_temperature_expression(source.ky, context);
            target.kz = compile_temperature_expression(source.kz, context);
            target.rho = compile_temperature_expression(source.midu, context);
            target.c = compile_temperature_expression(source.bi_rerong, context);
        }
        return catalog;
    }

    std::vector<std::vector<uint16_t>> build_heat_source_catalog(mhs::core::Model& model, const BuildContext& context,
        const std::vector<mhs::sim::ResolvedLayerGeometry>& resolved_layers)
    {
        model.heat_source_table.clear();
        model.heat_source_table.push_back(mhs::core::CompiledExpression::make_constant(0.0));
        std::unordered_map<std::string, uint16_t> source_by_expression {{"0", 0}};

        std::vector<std::vector<uint16_t>> block_to_source(resolved_layers.size());
        for (size_t layer_index = 0; layer_index < resolved_layers.size(); ++layer_index) {
            const auto& blocks = resolved_layers[layer_index].blocks;
            block_to_source[layer_index].resize(blocks.size(), 0);
            for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
                const std::string expression
                    = substitute_function_args(blocks[block_index].ti_reyuan_expr, "t", context.input.functions);
                const auto existing = source_by_expression.find(expression);
                if (existing != source_by_expression.end()) {
                    block_to_source[layer_index][block_index] = existing->second;
                    continue;
                }
                if (model.heat_source_table.size() >= std::numeric_limits<uint16_t>::max()) {
                    MHS_FATAL("too many heat-source expressions for the uint16_t source index");
                }
                const auto source_index = static_cast<uint16_t>(model.heat_source_table.size());
                model.heat_source_table.push_back(mhs::core::parse(expression, context.symbols));
                source_by_expression.emplace(expression, source_index);
                block_to_source[layer_index][block_index] = source_index;
            }
        }
        return block_to_source;
    }

    BoundaryCatalog build_boundary_catalog(mhs::core::Model& model, const BuildContext& context)
    {
        auto rewrite = [&](const std::string& expression) {
            return substitute_function_args(expression, "T", context.input.functions);
        };

        BoundaryCatalog catalog;
        catalog.explicit_patches = parse_all_face_keys(
            context.input.boundaries, model.bc_params, context.si_scale, rewrite, context.symbols);

        std::visit(
            overloaded {
                [&](const mhs::core::FirstTypeThermalBC& boundary) {
                    catalog.fallback.type = mhs::core::BcType::FirstType;
                    catalog.fallback.param_idx = static_cast<uint16_t>(model.bc_params.dirichlet_T.size());
                    model.bc_params.dirichlet_T.push_back(
                        mhs::core::parse(rewrite(boundary.temperature), context.symbols));
                },
                [&](const mhs::core::SecondTypeThermalBC& boundary) {
                    catalog.fallback.type = mhs::core::BcType::SecondType;
                    catalog.fallback.param_idx = static_cast<uint16_t>(model.bc_params.neumann_q.size());
                    model.bc_params.neumann_q.push_back(mhs::core::parse(rewrite(boundary.heat_flux), context.symbols));
                },
                [&](const mhs::core::ThirdTypeThermalBC& boundary) {
                    catalog.fallback.type = mhs::core::BcType::ThirdType;
                    catalog.fallback.param_idx = static_cast<uint16_t>(model.bc_params.cauchy_h.size());
                    model.bc_params.cauchy_h.push_back(
                        mhs::core::parse(rewrite(boundary.convection_coeff), context.symbols));
                    model.bc_params.cauchy_T_inf.push_back(mhs::core::parse(rewrite(boundary.T_inf), context.symbols));
                },
            },
            context.input.other_bc);
        return catalog;
    }

} // namespace mhs::sim::detail
