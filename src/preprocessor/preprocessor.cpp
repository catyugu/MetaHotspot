#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "layer_processor.hpp"
#include "preprocessor.hpp"

namespace mhs {

    std::unique_ptr<InternalModel> Preprocessor::load(const IOStructure& ioStructure)
    {
        auto model = std::make_unique<InternalModel>();

        model->study_type = ioStructure.study_type;
        model->initial_temperature = ioStructure.initial_temperature;
        model->ambient_temperature = ioStructure.ambient_temperature;
        model->transient_duration = ioStructure.transient_duration;
        model->transient_time_step = ioStructure.transient_time_step;

        expr::clear_registry();
        for (const auto& var : ioStructure.variables) {
            double val = expr::eval_geometry(var.value);
            expr::set_variable(var.name, val);
        }

        double si_scale = preprocessor::length_unit_to_si(ioStructure.length_unit);

        auto& mesh = model->mesh;
        mesh.vertex_x = ioStructure.mesh_vertex_x;
        mesh.vertex_y = ioStructure.mesh_vertex_y;
        mesh.vertex_z = ioStructure.mesh_vertex_z;

        for (auto& v : mesh.vertex_x)
            v *= si_scale;
        for (auto& v : mesh.vertex_y)
            v *= si_scale;
        for (auto& v : mesh.vertex_z)
            v *= si_scale;

        mesh.nx = (int)mesh.vertex_x.size() - 1;
        mesh.ny = (int)mesh.vertex_y.size() - 1;
        mesh.nz = (int)mesh.vertex_z.size() - 1;
        mesh.total_cell_count = mesh.nx * mesh.ny * mesh.nz;

        mesh.dx.resize(mesh.nx);
        mesh.dy.resize(mesh.ny);
        mesh.dz.resize(mesh.nz);
        mesh.cx.resize(mesh.nx);
        mesh.cy.resize(mesh.ny);
        mesh.cz.resize(mesh.nz);

        for (int i = 0; i < mesh.nx; i++) {
            mesh.dx[i] = mesh.vertex_x[i + 1] - mesh.vertex_x[i];
            mesh.cx[i] = (mesh.vertex_x[i] + mesh.vertex_x[i + 1]) / 2.0;
        }
        for (int j = 0; j < mesh.ny; j++) {
            mesh.dy[j] = mesh.vertex_y[j + 1] - mesh.vertex_y[j];
            mesh.cy[j] = (mesh.vertex_y[j] + mesh.vertex_y[j + 1]) / 2.0;
        }
        for (int k = 0; k < mesh.nz; k++) {
            mesh.dz[k] = mesh.vertex_z[k + 1] - mesh.vertex_z[k];
            mesh.cz[k] = (mesh.vertex_z[k] + mesh.vertex_z[k + 1]) / 2.0;
        }

        auto resolved_layers = preprocessor::resolve_geometry(ioStructure.layers, si_scale);

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
            model->material_table[m].k = expr::parse(mat.daore_xishu);
            model->material_table[m].rho = expr::parse(mat.midu);
            model->material_table[m].c = expr::parse(mat.bi_rerong);
        }

        auto& cells = model->cells;
        preprocessor::resolve_layers(resolved_layers, mesh, name_to_idx, cells);

        cells.cell_bcs.resize(cells.cell_count);
        cells.heat_source_idx.resize(cells.cell_count, 0);

        // --- 热源字典构建 ---
        model->heat_source_table.clear();
        model->heat_source_table.push_back(
            expr::CompiledExpression::make_constant(0.0)); // 索引 0 留空为默认 0.0

        std::vector<std::vector<uint16_t>> block_hs_map(resolved_layers.size());
        for (size_t l = 0; l < resolved_layers.size(); l++) {
            block_hs_map[l].resize(resolved_layers[l].blocks.size(), 0);
            for (size_t b = 0; b < resolved_layers[l].blocks.size(); b++) {
                uint16_t hs_idx = (uint16_t)model->heat_source_table.size();
                model->heat_source_table.push_back(
                    expr::parse(resolved_layers[l].blocks[b].ti_reyuan_expr));
                block_hs_map[l][b] = hs_idx;
            }
        }

        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    if (cells.valid_mask[old_idx] == 1) {
                        int c_idx = (int)cells.index_map[old_idx];
                        int layer_idx = (int)cells.layer_id[old_idx];
                        double cx = mesh.cx[ix];
                        double cy = mesh.cy[iy];
                        double cz = mesh.cz[iz];
                        int block_idx = preprocessor::find_block_for_cell(
                            resolved_layers[layer_idx], cx, cy, cz);

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
        preprocessor::resolve_face_keys(ioStructure.boundaries, ioStructure.other_bc_type,
            ioStructure.other_bc_first, ioStructure.other_bc_second, ioStructure.other_bc_third,
            mesh, cells, bc_params, si_scale);

        return model;
    }

} // namespace mhs