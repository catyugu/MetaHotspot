#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "function_helpers.hpp"
#include "layer_processor.hpp"
#include "preprocessor.hpp"

#include <algorithm>

namespace mhs::sim {

    std::unique_ptr<mhs::core::InternalModel> Preprocessor::load(const mhs::core::IOStructure& ioStructure)
    {
        auto model = std::make_unique<mhs::core::InternalModel>();

        model->study_type = ioStructure.study_type;
        model->initial_temperature = ioStructure.initial_temperature;
        model->transient_duration = ioStructure.transient_duration;
        model->transient_time_step = ioStructure.transient_time_step;

        // 解析几何表达式上下文：变量注册到 expr 全局表，function 在下方 register。
        // 必须在 observation_points 求值前完成，因为后者可能引用变量。
        mhs::core::clear_registry();
        for (const auto& var : ioStructure.variables) {
            double val = mhs::core::eval_geometry(var.value);
            mhs::core::set_variable(var.name, val);
        }
        const auto& fns = ioStructure.functions;
        register_all_functions(fns);

        const double si_scale = length_unit_to_si(ioStructure.length_unit);

        // 探针不参与方程求解：把 IO 层的 muparser 表达式字符串求值到 SI 单位的 double。
        for (const auto& src : ioStructure.observation_points) {
            mhs::core::ProbePoint p;
            p.name = src.name;
            p.x = mhs::core::eval_geometry(src.x) * si_scale;
            p.y = mhs::core::eval_geometry(src.y) * si_scale;
            p.z = mhs::core::eval_geometry(src.z) * si_scale;
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

        auto resolved_layers = resolve_geometry(ioStructure.layers, si_scale);

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
            model->material_table[m].kx = mhs::core::parse(substitute_function_args(mat.kx, "T", fns));
            model->material_table[m].ky = mhs::core::parse(substitute_function_args(mat.ky, "T", fns));
            model->material_table[m].kz = mhs::core::parse(substitute_function_args(mat.kz, "T", fns));
            model->material_table[m].rho = mhs::core::parse(substitute_function_args(mat.midu, "T", fns));
            model->material_table[m].c = mhs::core::parse(substitute_function_args(mat.bi_rerong, "T", fns));
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
                model->heat_source_table.push_back(mhs::core::parse(substitute_function_args(raw, "t", fns)));
                block_hs_map[l][b] = hs_idx;
            }
        }

        // 一次遍历完成 index_map + material_id + heat_source_idx + cell_bcs（compact）；
        // index_map 的 invalidIndex 值标记虚拟单元（无 valid_mask 字段）。
        // 不再有第二次全网格遍历去填 heat_source_idx 或 face BCs —— layer/block 归属判定时一并写入
        // cell_bcs 由零初始化保证全 None（不是之前的三重循环）。
        auto& bc_params = model->bc_params;
        auto bc_rewriter = [&fns](const std::string& s) { return substitute_function_args(s, "T", fns); };

        // 先展平所有 boundary face keys（不依赖 CellFields）。
        auto parsed_keys = parse_all_face_keys(ioStructure.boundaries, bc_params, si_scale, bc_rewriter);

        // 构建 other_bc 参数并决定兜底类型的索引。
        mhs::core::BcType other_bc_enum = mhs::core::BcType::None;
        uint16_t other_bc_idx = 0;
        switch (ioStructure.other_bc_type) {
        case mhs::core::ThermalBCType::FirstType: {
            other_bc_enum = mhs::core::BcType::FirstType;
            bc_params.dirichlet_T.push_back(mhs::core::parse(bc_rewriter(ioStructure.other_bc_first.temperature)));
            other_bc_idx = (uint16_t)(bc_params.dirichlet_T.size() - 1);
            break;
        }
        case mhs::core::ThermalBCType::SecondType: {
            other_bc_enum = mhs::core::BcType::SecondType;
            bc_params.neumann_q.push_back(mhs::core::parse(bc_rewriter(ioStructure.other_bc_second.heat_flux)));
            other_bc_idx = (uint16_t)(bc_params.neumann_q.size() - 1);
            break;
        }
        case mhs::core::ThermalBCType::ThirdType: {
            other_bc_enum = mhs::core::BcType::ThirdType;
            bc_params.cauchy_h.push_back(mhs::core::parse(bc_rewriter(ioStructure.other_bc_third.convection_coeff)));
            bc_params.cauchy_T_inf.push_back(mhs::core::parse(bc_rewriter(ioStructure.other_bc_third.T_inf)));
            other_bc_idx = (uint16_t)(bc_params.cauchy_h.size() - 1);
            break;
        }
        }

        model->cells = resolve_layers(
            resolved_layers, mesh, name_to_idx, block_hs_map, parsed_keys, other_bc_enum, other_bc_idx);

        return model;
    }

    void Preprocessor::applyFluidOverlay(mhs::core::InternalModel& model,
        const std::optional<mhs::core::FluidOverlay>& overlay, const mhs::core::IOStructure& ioStructure)
    {
        if (!overlay.has_value())
            return;

        const auto& fluidOverlay = overlay.value();
        if (fluidOverlay.fluid_materials.empty())
            return;

        const double si_scale = length_unit_to_si(ioStructure.length_unit);
        const int N = static_cast<int>(model.cells.cell_bcs.size());

        // --- Step 1: Build fluid material name → index mapping ---
        std::unordered_map<std::string, uint16_t> fluid_mat_name_to_idx;
        for (const auto& fm : fluidOverlay.fluid_materials) {
            fluid_mat_name_to_idx[fm.name] = 0; // placeholder, filled below
        }

        // Match fluid material names to material_table indices
        // model.material_table indices correspond to material_names populated in load()
        for (uint16_t matIdx = 0; matIdx < static_cast<uint16_t>(model.material_table.size()); ++matIdx) {
            // We need the original name from IOStructure
            for (const auto& fm : fluidOverlay.fluid_materials) {
                // Find the matching material in ioStructure
                auto it = ioStructure.materials.find(fm.name);
                if (it != ioStructure.materials.end()) {
                    // Check if this material name is in our material_table
                    // We reconstruct the name mapping from the load() logic
                    // Since material_table is ordered by first-seen materials from layers,
                    // we need to cross-reference differently.
                }
            }
        }

        // Simpler approach: rebuild the material name → table index mapping
        std::unordered_map<std::string, uint16_t> matNameToTableIdx;
        // Reconstruct from the same logic as load(): iterate layers then blocks
        std::vector<std::string> materialNames;
        for (const auto& layer : ioStructure.layers) {
            for (const auto& block : layer.blocks) {
                if (matNameToTableIdx.find(block.material_name) == matNameToTableIdx.end()) {
                    matNameToTableIdx[block.material_name] = static_cast<uint16_t>(materialNames.size());
                    materialNames.push_back(block.material_name);
                }
            }
        }

        // Mark is_fluid and fill fluid_material_id
        model.is_fluid.assign(N, 0);
        model.cells.fluid_material_id.assign(N, static_cast<uint16_t>(std::numeric_limits<uint16_t>::max()));
        model.dynamic_viscosity.assign(N, 0.0);

        // Build fluid material name → viscosity expression map
        std::unordered_map<std::string, std::string> fluidViscosityMap;
        for (const auto& fm : fluidOverlay.fluid_materials) {
            fluidViscosityMap[fm.name] = fm.dynamic_viscosity;
        }

        for (uint16_t matIdx = 0; matIdx < static_cast<uint16_t>(model.material_table.size()); ++matIdx) {
            const auto& matName = materialNames[matIdx];
            auto visIt = fluidViscosityMap.find(matName);
            if (visIt != fluidViscosityMap.end() && !visIt->second.empty()) {
                model.material_table[matIdx].is_fluid = true;
                model.material_table[matIdx].dynamic_viscosity
                    = mhs::core::CompiledExpression::make_constant(std::stod(visIt->second));
            }
        }

        // Mark fluid cells based on material_table
        for (int c = 0; c < N; ++c) {
            uint16_t matIdx = model.cells.material_id[c];
            if (matIdx < model.material_table.size() && model.material_table[matIdx].is_fluid) {
                model.is_fluid[c] = 1;
                model.cells.fluid_material_id[c] = matIdx;
                model.dynamic_viscosity[c] = model.material_table[matIdx].dynamic_viscosity.constant_value();
            }
        }

        // Check if any fluid cells exist
        bool hasFluid = std::any_of(model.is_fluid.begin(), model.is_fluid.end(), [](uint8_t v) { return v != 0; });
        if (!hasFluid)
            return;

        // --- Step 2: Initialize fluid solver fields ---
        model.pressure.assign(N, 0.0);
        model.flow_axes.assign(N, -1);
        model.hydroC_x.assign(N, 0.0);
        model.hydroC_y.assign(N, 0.0);
        model.hydroC_z.assign(N, 0.0);
        model.is_pressure_boundary.assign(N, 0);
        model.boundary_pressure.assign(N, 0.0);
        model.boundary_temperature_fluid.assign(N, std::numeric_limits<double>::quiet_NaN());

        // --- Step 3: Parse pressure boundaries from overlay ---
        // Scan the full grid to match face keys against fluid cell centers.
        // X-face keys match on (cy, cz), Y-face keys on (cx, cz).
        for (const auto& fb : fluidOverlay.boundaries) {
            for (const auto& keyStr : fb.face_keys) {
                FaceKeyInfo fk = parse_face_key(keyStr, si_scale);

                if (fk.axis == 'X') {
                    // X-face: match on (cy, cz) → tangents are Y, Z
                    for (int ix = 0; ix < model.mesh.nx; ++ix) {
                        for (int iy = 0; iy < model.mesh.ny; ++iy) {
                            for (int iz = 0; iz < model.mesh.nz; ++iz) {
                                int old_idx = ix * model.mesh.ny * model.mesh.nz + iy * model.mesh.nz + iz;
                                int c_idx = static_cast<int>(model.cells.index_map[old_idx]);
                                if (c_idx < 0 || c_idx >= N || !model.is_fluid[c_idx])
                                    continue;

                                double cy = model.mesh.cy[iy];
                                double cz = model.mesh.cz[iz];
                                if (point_in_face_rects(fk, cy, cz)) {
                                    model.is_pressure_boundary[c_idx] = 1;
                                    model.boundary_pressure[c_idx] = fb.pressure_bc.pressure;
                                }
                            }
                        }
                    }
                }
                else if (fk.axis == 'Y') {
                    // Y-face: match on (cx, cz)
                    for (int ix = 0; ix < model.mesh.nx; ++ix) {
                        for (int iy = 0; iy < model.mesh.ny; ++iy) {
                            for (int iz = 0; iz < model.mesh.nz; ++iz) {
                                int old_idx = ix * model.mesh.ny * model.mesh.nz + iy * model.mesh.nz + iz;
                                int c_idx = static_cast<int>(model.cells.index_map[old_idx]);
                                if (c_idx < 0 || c_idx >= N || !model.is_fluid[c_idx])
                                    continue;

                                double cx = model.mesh.cx[ix];
                                double cz = model.mesh.cz[iz];
                                if (point_in_face_rects(fk, cx, cz)) {
                                    model.is_pressure_boundary[c_idx] = 1;
                                    model.boundary_pressure[c_idx] = fb.pressure_bc.pressure;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

} // namespace mhs::sim
