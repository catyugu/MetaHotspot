#include "expr/expr.hpp"
#include "layer_processor.hpp"
#include <cstdint>

namespace mhs::sim {

    constexpr double EPS = 1e-9;

    double length_unit_to_si(mhs::core::LengthUnit unit)
    {
        switch (unit) {
        case mhs::core::LengthUnit::M:
            return 1.0;
        case mhs::core::LengthUnit::Mm:
            return 1e-3;
        case mhs::core::LengthUnit::Um:
            return 1e-6;
        case mhs::core::LengthUnit::Nm:
            return 1e-9;
        case mhs::core::LengthUnit::Inch:
            return 0.0254;
        case mhs::core::LengthUnit::Mil:
            return 2.54e-5;
        default:
            return 1e-3;
        }
    }

    std::vector<ResolvedLayerGeometry> resolve_geometry(const std::vector<mhs::core::Layer>& layers, double si_scale)
    {
        int num_layers = (int)layers.size();
        std::vector<ResolvedLayerGeometry> resolved(num_layers);

        // Compute layer Z ranges (top-down stacking)
        // Evaluate thicknesses once, then assign z_start/z_end directly
        std::vector<double> thickness(num_layers);
        double z_cursor = 0.0;
        for (int l = 0; l < num_layers; l++) {
            if (l == 0) {
                double max_t = 0.0;
                for (const auto& b : layers[l].blocks) {
                    if (!b.thickness_expr.empty()) {
                        double t = mhs::core::eval_geometry(b.thickness_expr) * si_scale;
                        if (t > max_t)
                            max_t = t;
                    }
                }
                double layer_t = layers[l].thickness_expr.empty()
                    ? 0.0
                    : mhs::core::eval_geometry(layers[l].thickness_expr) * si_scale;
                thickness[l] = std::max(max_t, layer_t); // 第0层厚度由最大 block 决定
            }
            else {
                thickness[l] = mhs::core::eval_geometry(layers[l].thickness_expr) * si_scale;
            }
            z_cursor += thickness[l];
        }
        for (int l = 0; l < num_layers; l++) {
            resolved[l].z_start = z_cursor - thickness[l];
            resolved[l].z_end = z_cursor;
            z_cursor -= thickness[l];
        }

        for (int l = 0; l < num_layers; l++) {
            const auto& layer = layers[l];
            double layer_x_off_si = mhs::core::eval_geometry(layer.x_offset_expr) * si_scale;
            double layer_y_off_si = mhs::core::eval_geometry(layer.y_offset_expr) * si_scale;

            for (const auto& block : layer.blocks) {
                ResolvedBlock rb;
                double block_x_off_si = mhs::core::eval_geometry(block.x_offset_expr) * si_scale;
                double block_y_off_si = mhs::core::eval_geometry(block.y_offset_expr) * si_scale;
                rb.material_name = block.material_name;
                rb.ti_reyuan_expr = block.ti_reyuan_expr;

                if (l == 0 && !block.thickness_expr.empty()) {
                    double b_thick = mhs::core::eval_geometry(block.thickness_expr) * si_scale;
                    rb.z_start = resolved[l].z_start;
                    rb.z_end = resolved[l].z_start + b_thick;
                }
                else {
                    // 其他层或未指定厚度的 block，默认铺满整层
                    rb.z_start = resolved[l].z_start;
                    rb.z_end = resolved[l].z_end;
                }

                for (const auto& rect : block.all_rects) {
                    ResolvedRect rr;
                    rr.add_sub = rect.add_sub;

                    double x_val = mhs::core::eval_geometry(rect.x_expr);
                    double y_val = mhs::core::eval_geometry(rect.y_expr);
                    double w_val = mhs::core::eval_geometry(rect.width_expr);
                    double h_val = mhs::core::eval_geometry(rect.height_expr);

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

    // 找到 find_block_for_cell 函数，在遍历 block 时引入 Z 的校验
    int find_block_for_cell(const ResolvedLayerGeometry& resolved_layer, double cx, double cy, double cz)
    {
        if (cz < resolved_layer.z_start - EPS || cz > resolved_layer.z_end + EPS) {
            return -1;
        }
        for (int b = (int)resolved_layer.blocks.size() - 1; b >= 0; b--) {
            const auto& block = resolved_layer.blocks[b];
            if (cz < block.z_start - EPS || cz > block.z_end + EPS) {
                continue;
            }
            bool is_inside = false;
            for (const auto& rect : block.rects) {
                if (cx >= rect.x - EPS && cx <= rect.x + rect.width + EPS && cy >= rect.y - EPS
                    && cy <= rect.y + rect.height + EPS) {
                    is_inside = rect.add_sub;
                }
            }
            if (is_inside) {
                return b;
            }
        }
        return -1;
    }

    LayerResolveResult resolve_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const std::unordered_map<std::string, size_t>& name_to_idx)
    {
        int num_layers = (int)resolved_layers.size();
        int total = mesh.nx * mesh.ny * mesh.nz;

        LayerResolveResult result;
        result.cells.valid_mask.resize(total, 0);
        result.cells.index_map.resize(total, SIZE_MAX);
        result.layer_id_old.resize(total, SIZE_MAX);
        // 临时 old_idx 索引的 material 数组：phase 1 写入，phase 2 压缩到 compact 后丢弃
        std::vector<size_t> material_id_temp(total, SIZE_MAX);

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
                        result.cells.valid_mask[old_idx] = 1;
                        result.layer_id_old[old_idx] = layer_idx;
                        const auto& block = resolved_layers[layer_idx].blocks[block_idx];
                        material_id_temp[old_idx] = name_to_idx.at(block.material_name);
                    }
                }
            }
        }

        // Build compact layout: index_map (old → compact) and material_id (compact).
        // cell_bcs / heat_source_idx 也 resize 到 compact 计数，使 cell_bcs.size()
        // 成为活动 cell 计数的唯一来源。
        int compact_idx = 0;
        for (int i = 0; i < total; i++) {
            if (result.cells.valid_mask[i] == 1) {
                result.cells.index_map[i] = compact_idx;
                result.cells.material_id.push_back(static_cast<uint16_t>(material_id_temp[i]));
                compact_idx++;
            }
        }
        result.cells.cell_bcs.resize(compact_idx);
        result.cells.heat_source_idx.resize(compact_idx, 0);

        return result;
    }

} // namespace mhs::sim