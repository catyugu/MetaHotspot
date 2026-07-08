#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "fluid_preprocessor.hpp"
#include "function_helpers.hpp"
#include "layer_processor.hpp"
#include "preprocessor.hpp"

#include "common/mesh_utils.hpp"

namespace mhs::sim {

    std::unique_ptr<mhs::core::InternalModel> Preprocessor::load(
        const mhs::core::IOStructure& ioStructure, const std::optional<mhs::core::FluidOverlay>& fluidOverlay)
    {
        auto model = std::make_unique<mhs::core::InternalModel>();

        model->study_type = ioStructure.study_type;
        model->initial_temperature = ioStructure.initial_temperature;
        model->transient_duration = ioStructure.transient_duration;
        model->transient_time_step = ioStructure.transient_time_step;

        // 解析几何表达式上下文：变量求值后写入本地的 SymbolTable，natives 由 register_all_functions 填充。
        // 必须在 observation_points 求值前完成，因为后者可能引用变量。
        // 几何变量的 value 在当前调用约定下总被解析为纯数字（无前向引用），
        // 所以用一个空表作为临时求值上下文即可。
        mhs::core::SymbolTable empty_for_geometry_eval;
        mhs::core::SymbolTable symbols;
        for (const auto& var : ioStructure.variables) {
            double val = mhs::core::eval_geometry(var.value, empty_for_geometry_eval);
            symbols.variables[var.name] = val;
        }
        register_all_functions(symbols, ioStructure.functions);

        const double si_scale = mhs::utils::length_unit_to_si(ioStructure.length_unit);

        // 探针不参与方程求解：把 IO 层的 muparser 表达式字符串求值到 SI 单位的 double。
        for (const auto& src : ioStructure.observation_points) {
            mhs::core::ProbePoint p;
            p.name = src.name;
            p.x = mhs::core::eval_geometry(src.x, symbols) * si_scale;
            p.y = mhs::core::eval_geometry(src.y, symbols) * si_scale;
            p.z = mhs::core::eval_geometry(src.z, symbols) * si_scale;
            model->observation_points.push_back(std::move(p));
        }

        auto& mesh = model->mesh;

        // IOStructure::mesh_vertex_* 是用户输入的节点坐标；按 SI 单位缩放后直接计算
        // dx/dy/dz 与 cx/cy/cz。MeshGeometry 不再持有 vertex_* 数组——它们在
        // 预处理完成后就是死数据。
        mesh.nx = (int)ioStructure.mesh_vertex_x.size() - 1;
        mesh.ny = (int)ioStructure.mesh_vertex_y.size() - 1;
        mesh.nz = (int)ioStructure.mesh_vertex_z.size() - 1;

        mesh.dx.resize(mesh.nx);
        mesh.dy.resize(mesh.ny);
        mesh.dz.resize(mesh.nz);
        mesh.cx.resize(mesh.nx);
        mesh.cy.resize(mesh.ny);
        mesh.cz.resize(mesh.nz);

        for (int i = 0; i < mesh.nx; i++) {
            double v0 = ioStructure.mesh_vertex_x[i] * si_scale;
            double v1 = ioStructure.mesh_vertex_x[i + 1] * si_scale;
            mesh.dx[i] = v1 - v0;
            mesh.cx[i] = (v0 + v1) * 0.5;
        }
        for (int j = 0; j < mesh.ny; j++) {
            double v0 = ioStructure.mesh_vertex_y[j] * si_scale;
            double v1 = ioStructure.mesh_vertex_y[j + 1] * si_scale;
            mesh.dy[j] = v1 - v0;
            mesh.cy[j] = (v0 + v1) * 0.5;
        }
        for (int k = 0; k < mesh.nz; k++) {
            double v0 = ioStructure.mesh_vertex_z[k] * si_scale;
            double v1 = ioStructure.mesh_vertex_z[k + 1] * si_scale;
            mesh.dz[k] = v1 - v0;
            mesh.cz[k] = (v0 + v1) * 0.5;
        }

        auto resolved_layers = resolve_geometry(ioStructure.layers, si_scale, symbols);

        std::vector<std::string> material_names;
        std::unordered_map<std::string, size_t> name_to_idx;

        for (const auto& layer : ioStructure.layers) {
            for (const auto& block : layer.blocks) {
                if (name_to_idx.find(block.material_name) == name_to_idx.end()) {
                    name_to_idx[block.material_name] = material_names.size();
                    material_names.push_back(block.material_name);
                }
            }
        }

        model->material_table.resize(material_names.size());
        for (size_t m = 0; m < material_names.size(); m++) {
            const auto& mat = ioStructure.materials.at(material_names[m]);
            model->material_table[m].kx
                = mhs::core::parse(substitute_function_args(mat.kx, "T", ioStructure.functions), symbols);
            model->material_table[m].ky
                = mhs::core::parse(substitute_function_args(mat.ky, "T", ioStructure.functions), symbols);
            model->material_table[m].kz
                = mhs::core::parse(substitute_function_args(mat.kz, "T", ioStructure.functions), symbols);
            model->material_table[m].rho
                = mhs::core::parse(substitute_function_args(mat.midu, "T", ioStructure.functions), symbols);
            model->material_table[m].c
                = mhs::core::parse(substitute_function_args(mat.bi_rerong, "T", ioStructure.functions), symbols);
        }

        // --- 热源字典构建（必须在 resolve_layers 之前完成，因 resolve_layers
        //     在层归属遍历中同步消费 block_hs_map 来填 heat_source_idx） ---
        model->heat_source_table.clear();
        model->heat_source_table.push_back(mhs::core::CompiledExpression::make_constant(0.0)); // 索引 0 留空为默认 0.0

        std::vector<std::vector<uint16_t>> block_hs_map(resolved_layers.size());
        for (size_t l = 0; l < resolved_layers.size(); l++) {
            block_hs_map[l].resize(resolved_layers[l].blocks.size(), 0);
            for (size_t b = 0; b < resolved_layers[l].blocks.size(); b++) {
                uint16_t hs_idx = (uint16_t)model->heat_source_table.size();
                const std::string& raw = resolved_layers[l].blocks[b].ti_reyuan_expr;
                model->heat_source_table.push_back(
                    mhs::core::parse(substitute_function_args(raw, "t", ioStructure.functions), symbols));
                block_hs_map[l][b] = hs_idx;
            }
        }

        // 一次遍历完成 index_map + material_id + heat_source_idx + cell_bcs（compact）；
        // index_map 的 invalidIndex 值标记虚拟单元（无 valid_mask 字段）。
        // 不再有第二次全网格遍历去填 heat_source_idx 或 face BCs —— layer/block 归属判定时一并写入
        // cell_bcs 由零初始化保证全 None（不是之前的三重循环）。
        auto& bc_params = model->bc_params;
        auto bc_rewriter
            = [&ioStructure](const std::string& s) { return substitute_function_args(s, "T", ioStructure.functions); };

        // 先展平所有 boundary face keys（不依赖 CellFields）。
        auto parsed_keys = parse_all_face_keys(ioStructure.boundaries, bc_params, si_scale, bc_rewriter, symbols);

        // 构建 other_bc 参数并决定兜底类型的索引。
        mhs::core::BcType other_bc_enum = mhs::core::BcType::None;
        uint16_t other_bc_idx = 0;
        switch (ioStructure.other_bc_type) {
        case mhs::core::ThermalBCType::FirstType: {
            other_bc_enum = mhs::core::BcType::FirstType;
            bc_params.dirichlet_T.push_back(
                mhs::core::parse(bc_rewriter(ioStructure.other_bc_first.temperature), symbols));
            other_bc_idx = (uint16_t)(bc_params.dirichlet_T.size() - 1);
            break;
        }
        case mhs::core::ThermalBCType::SecondType: {
            other_bc_enum = mhs::core::BcType::SecondType;
            bc_params.neumann_q.push_back(
                mhs::core::parse(bc_rewriter(ioStructure.other_bc_second.heat_flux), symbols));
            other_bc_idx = (uint16_t)(bc_params.neumann_q.size() - 1);
            break;
        }
        case mhs::core::ThermalBCType::ThirdType: {
            other_bc_enum = mhs::core::BcType::ThirdType;
            bc_params.cauchy_h.push_back(
                mhs::core::parse(bc_rewriter(ioStructure.other_bc_third.convection_coeff), symbols));
            bc_params.cauchy_T_inf.push_back(mhs::core::parse(bc_rewriter(ioStructure.other_bc_third.T_inf), symbols));
            other_bc_idx = (uint16_t)(bc_params.cauchy_h.size() - 1);
            break;
        }
        }

        model->cells = resolve_layers(
            resolved_layers, mesh, name_to_idx, block_hs_map, parsed_keys, other_bc_enum, other_bc_idx);

        // Apply fluid overlay (if any) using the same SymbolTable so viscosity
        // expressions see the same natives/variables as the rest of the model.
        if (fluidOverlay.has_value()) {
            mhs::sim::applyFluidOverlay(*model, fluidOverlay, ioStructure, symbols);
            mhs::sim::solveFluidFlow(*model);
        }

        return model;
    }

} // namespace mhs::sim