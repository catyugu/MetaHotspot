#include "expr/expr.hpp"
#include "layer_processor.hpp"

namespace mhs::preprocessor {

    constexpr double EPS = 1e-9;

    double length_unit_to_si(LengthUnit unit)
    {
        switch (unit) {
        case LengthUnit::M:
            return 1.0;
        case LengthUnit::Mm:
            return 1e-3;
        case LengthUnit::Um:
            return 1e-6;
        case LengthUnit::Nm:
            return 1e-9;
        case LengthUnit::Inch:
            return 0.0254;
        case LengthUnit::Mil:
            return 2.54e-5;
        default:
            return 1e-3;
        }
    }

    std::vector<ResolvedLayerGeometry> resolve_geometry(const std::vector<Layer>& layers, double si_scale)
    {
        int num_layers = (int)layers.size();
        std::vector<ResolvedLayerGeometry> resolved(num_layers);

        // Compute layer Z ranges (top-down stacking)
        // Evaluate thicknesses once, then assign z_start/z_end directly
        std::vector<double> thickness(num_layers);
        double z_cursor = 0.0;
        for (int l = 0; l < num_layers; l++) {
            thickness[l] = expr::eval_geometry(layers[l].thickness_expr) * si_scale;
            z_cursor += thickness[l];
        }
        for (int l = 0; l < num_layers; l++) {
            resolved[l].z_start = z_cursor - thickness[l];
            resolved[l].z_end = z_cursor;
            z_cursor -= thickness[l];
        }

        for (int l = 0; l < num_layers; l++) {
            const auto& layer = layers[l];
            double layer_x_off_si = expr::eval_geometry(layer.x_offset_expr) * si_scale;
            double layer_y_off_si = expr::eval_geometry(layer.y_offset_expr) * si_scale;

            for (const auto& block : layer.blocks) {
                ResolvedBlock rb;
                double block_x_off_si = expr::eval_geometry(block.x_offset_expr) * si_scale;
                double block_y_off_si = expr::eval_geometry(block.y_offset_expr) * si_scale;
                rb.material_name = block.material_name;
                rb.ti_reyuan_expr = block.ti_reyuan_expr;

                for (const auto& rect : block.all_rects) {
                    ResolvedRect rr;
                    rr.add_sub = rect.add_sub;

                    double x_val = expr::eval_geometry(rect.x_expr);
                    double y_val = expr::eval_geometry(rect.y_expr);
                    double w_val = expr::eval_geometry(rect.width_expr);
                    double h_val = expr::eval_geometry(rect.height_expr);

                    // Normalize negative widths/heights
                    if (w_val < 0) {
                        x_val += w_val;
                        w_val = -w_val;
                    }
                    if (h_val < 0) {
                        y_val += h_val;
                        h_val = -h_val;
                    }

                    // Absolute SI coordinates: rect-local * si_scale + pre-resolved offsets
                    rr.x = x_val * si_scale + block_x_off_si + layer_x_off_si;
                    rr.y = y_val * si_scale + block_y_off_si + layer_y_off_si;
                    rr.width = w_val * si_scale;
                    rr.height = h_val * si_scale;

                    rb.rects.push_back(rr);
                }

                resolved[l].blocks.push_back(rb);
            }
        }

        return resolved;
    }

    int find_block_for_cell(const ResolvedLayerGeometry& resolved_layer, double cx, double cy, double cz)
    {
        if (cz < resolved_layer.z_start - EPS || cz > resolved_layer.z_end + EPS) {
            return -1;
        }

        // Traverse blocks in reverse order: last block wins in overlap regions
        for (int b = (int)resolved_layer.blocks.size() - 1; b >= 0; b--) {
            const auto& block = resolved_layer.blocks[b];

            // ================= 核心布尔逻辑优化 =================
            // 采用单一状态机变量，严格遵循 CAD 特征树的顺序求值
            bool is_inside = false;

            for (const auto& rect : block.rects) {
                // 如果当前网格点落在该矩形内，则此矩形的操作会覆盖之前的状态
                if (cx >= rect.x - EPS && cx <= rect.x + rect.width + EPS && cy >= rect.y - EPS
                    && cy <= rect.y + rect.height + EPS) {
                    // 若是加操作，点变为实心(true)；若是减操作，点变为空洞(false)
                    // 因为是顺次执行，后面的加操作可以完美填补前面减操作挖出来的洞
                    is_inside = rect.add_sub;
                }
            }

            if (is_inside) {
                return b;
            }
            // ====================================================
        }

        return -1;
    }

    void resolve_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers, const MeshGeometry& mesh,
        const std::unordered_map<std::string, size_t>& name_to_idx, CellFields& cells)
    {
        int num_layers = (int)resolved_layers.size();
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
                        if (cz >= resolved_layers[l].z_start - EPS && cz <= resolved_layers[l].z_end + EPS) {
                            int b = find_block_for_cell(resolved_layers[l], cx, cy, cz);
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
                        const auto& block = resolved_layers[layer_idx].blocks[block_idx];
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