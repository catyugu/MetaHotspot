#include "layer_processor.hpp"
#include "expr/expr.hpp"

namespace mhs::preprocessor {

    double length_unit_to_si(model::LengthUnit unit)
    {
        switch (unit) {
        case model::LengthUnit::M:
            return 1.0;
        case model::LengthUnit::Mm:
            return 1e-3;
        case model::LengthUnit::Um:
            return 1e-6;
        case model::LengthUnit::Nm:
            return 1e-9;
        case model::LengthUnit::Inch:
            return 0.0254;
        case model::LengthUnit::Mil:
            return 2.54e-5;
        default:
            return 1e-3;
        }
    }

    void compute_layer_z_ranges(const std::vector<model::Layer>& layers,
        double si_scale,
        std::vector<double>& z_start,
        std::vector<double>& z_end)
    {
        int num_layers = (int)layers.size();
        z_start.resize(num_layers);
        z_end.resize(num_layers);

        double z_cursor = 0.0;
        for (int l = 0; l < num_layers; l++) {
            double layer_thick = expr::eval_geometry(layers[l].thickness_expr) * si_scale;
            z_cursor += layer_thick;
        }

        // Top layer starts at the top, going down
        for (int l = 0; l < num_layers; l++) {
            double layer_thick = expr::eval_geometry(layers[l].thickness_expr) * si_scale;
            z_start[l] = z_cursor - layer_thick;
            z_end[l] = z_cursor;
            z_cursor -= layer_thick;
        }
    }

    int find_block_for_cell(const model::Layer& layer,
        double cx, double cy, double cz,
        double si_scale,
        double layer_z_start, double layer_z_end)
    {
        if (cz < layer_z_start || cz >= layer_z_end) {
            return -1;
        }

        // Layer offsets transform rect coords from centered to absolute coordinate system
        double layer_x_offset_orig = expr::eval_geometry(layer.x_offset_expr);
        double layer_y_offset_orig = expr::eval_geometry(layer.y_offset_expr);

        for (int b = 0; b < (int)layer.blocks.size(); b++) {
            const auto& block = layer.blocks[b];
            double block_x_offset_orig = expr::eval_geometry(block.x_offset_expr);
            double block_y_offset_orig = expr::eval_geometry(block.y_offset_expr);

            bool in_add = false;
            bool in_sub = false;

            for (const auto& rect : block.all_rects) {
                double rx = (expr::eval_geometry(rect.x_expr) + block_x_offset_orig + layer_x_offset_orig) * si_scale;
                double ry = (expr::eval_geometry(rect.y_expr) + block_y_offset_orig + layer_y_offset_orig) * si_scale;
                double rw = expr::eval_geometry(rect.width_expr) * si_scale;
                double rh = expr::eval_geometry(rect.height_expr) * si_scale;

                if (cx >= rx && cx < rx + rw && cy >= ry && cy < ry + rh) {
                    if (rect.add_sub) {
                        in_add = true;
                    }
                    else {
                        in_sub = true;
                    }
                }
            }

            if (in_add && !in_sub) {
                return b;
            }
        }

        return -1;
    }

    void resolve_layers(const std::vector<model::Layer>& layers,
        const model::MeshGeometry& mesh,
        double si_scale,
        const std::vector<double>& layer_z_start,
        const std::vector<double>& layer_z_end,
        const std::unordered_map<std::string, size_t>& name_to_idx,
        model::CellFields& cells)
    {
        int num_layers = (int)layers.size();
        int total = mesh.total_cell_count;

        cells.valid_mask.resize(total, 0);
        cells.index_map.resize(total, SIZE_MAX);
        cells.material_id.resize(total, SIZE_MAX);
        cells.layer_id.resize(total, SIZE_MAX);

        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;

                    double cx = mesh.cx[ix];
                    double cy = mesh.cy[iy];
                    double cz = mesh.cz[iz];

                    int layer_idx = -1;
                    int block_idx = -1;

                    for (int l = 0; l < num_layers; l++) {
                        if (cz >= layer_z_start[l] && cz < layer_z_end[l]) {
                            int b = find_block_for_cell(layers[l], cx, cy, cz,
                                si_scale, layer_z_start[l], layer_z_end[l]);
                            if (b >= 0) {
                                layer_idx = l;
                                block_idx = b;
                                break;
                            }
                        }
                    }

                    if (layer_idx >= 0 && block_idx >= 0) {
                        cells.valid_mask[old_idx] = 1;
                        cells.layer_id[old_idx] = layer_idx;
                        const auto& block = layers[layer_idx].blocks[block_idx];
                        cells.material_id[old_idx] = name_to_idx.at(block.material_name);
                    }
                }
            }
        }

        // Build compact layout
        int compact_idx = 0;
        for (int i = 0; i < total; i++) {
            if (cells.valid_mask[i] == 1) {
                cells.index_map[i] = compact_idx++;
            }
        }
        cells.cell_count = compact_idx;
    }

} // namespace mhs::preprocessor