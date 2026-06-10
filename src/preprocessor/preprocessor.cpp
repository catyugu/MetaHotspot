#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "function_helpers.hpp"
#include "layer_processor.hpp"
#include "preprocessor.hpp"

namespace mhs::sim {

    std::unique_ptr<mhs::core::InternalModel> Preprocessor::load(const mhs::core::IOStructure& ioStructure)
    {
        auto model = std::make_unique<mhs::core::InternalModel>();

        model->study_type = ioStructure.study_type;
        model->initial_temperature = ioStructure.initial_temperature;
        model->ambient_temperature = ioStructure.ambient_temperature;
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

        // 探针不参与方程求解：把 IO 层的 exprtk 表达式字符串求值到 SI 单位的 double。
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

        auto layer_result = resolve_layers(resolved_layers, mesh, name_to_idx);
        model->cells = std::move(layer_result.cells);
        // 解引用 layer_id_old：仅在本次预处理中用于查找 block → heat_source，
        // 函数返回时已无其他用途，离开作用域自动释放。
        const std::vector<size_t>& layer_id_old = layer_result.layer_id_old;

        // --- 热源字典构建 ---
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

        auto& cells = model->cells;
        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    if (cells.valid_mask[old_idx] == 1) {
                        int c_idx = (int)cells.index_map[old_idx];
                        int layer_idx = (int)layer_id_old[old_idx];
                        double cx = mesh.cx[ix];
                        double cy = mesh.cy[iy];
                        double cz = mesh.cz[iz];
                        int block_idx = find_block_for_cell(resolved_layers[layer_idx], cx, cy, cz);

                        if (block_idx >= 0) {
                            cells.heat_source_idx[c_idx] = block_hs_map[layer_idx][block_idx];
                        }
                        else {
                            cells.heat_source_idx[c_idx] = 0;
                        }
                    }
                }
            }
        }

        auto& bc_params = model->bc_params;
        auto bc_rewriter = [&fns](const std::string& s) { return substitute_function_args(s, "T", fns); };
        resolve_face_keys(ioStructure.boundaries, ioStructure.other_bc_type, ioStructure.other_bc_first,
            ioStructure.other_bc_second, ioStructure.other_bc_third, mesh, cells, bc_params, si_scale, bc_rewriter);

        return model;
    }

} // namespace mhs::sim
