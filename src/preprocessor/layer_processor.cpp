#include "common/mesh_utils.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "layer_processor.hpp"
#include <cstdint>

namespace mhs::sim {

    constexpr double EPS = 1e-9;

    std::vector<ResolvedLayerGeometry> resolve_geometry(
        const std::vector<mhs::core::Layer>& layers, double si_scale, const mhs::core::SymbolTable& symbols)
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
                        double t = mhs::core::eval_geometry(b.thickness_expr, symbols) * si_scale;
                        if (t > max_t)
                            max_t = t;
                    }
                }
                double layer_t = layers[l].thickness_expr.empty()
                    ? 0.0
                    : mhs::core::eval_geometry(layers[l].thickness_expr, symbols) * si_scale;
                thickness[l] = std::max(max_t, layer_t); // 第0层厚度由最大 block 决定
            }
            else {
                thickness[l] = mhs::core::eval_geometry(layers[l].thickness_expr, symbols) * si_scale;
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
            double layer_x_off_si = mhs::core::eval_geometry(layer.x_offset_expr, symbols) * si_scale;
            double layer_y_off_si = mhs::core::eval_geometry(layer.y_offset_expr, symbols) * si_scale;

            for (const auto& block : layer.blocks) {
                ResolvedBlock rb;
                double block_x_off_si = mhs::core::eval_geometry(block.x_offset_expr, symbols) * si_scale;
                double block_y_off_si = mhs::core::eval_geometry(block.y_offset_expr, symbols) * si_scale;
                rb.material_name = block.material_name;
                rb.ti_reyuan_expr = block.ti_reyuan_expr;

                if (l == 0 && !block.thickness_expr.empty()) {
                    double b_thick = mhs::core::eval_geometry(block.thickness_expr, symbols) * si_scale;
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

                    double x_val = mhs::core::eval_geometry(rect.x_expr, symbols);
                    double y_val = mhs::core::eval_geometry(rect.y_expr, symbols);
                    double w_val = mhs::core::eval_geometry(rect.width_expr, symbols);
                    double h_val = mhs::core::eval_geometry(rect.height_expr, symbols);

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

    namespace {
        // True iff the cell face at (ix, iy, iz) on dir is "exposed" — i.e., the
        // neighbor is outside the grid or is a virtual cell (via index_map).
        bool is_cell_face_exposed(mhs::core::FaceDir dir, int ix, int iy, int iz, const mhs::core::MeshGeometry& mesh,
            const mhs::core::CellFields& cells)
        {
            int nix = mhs::utils::neighbor_ix(dir, ix);
            int niy = mhs::utils::neighbor_iy(dir, iy);
            int niz = mhs::utils::neighbor_iz(dir, iz);

            if (nix < 0 || nix >= mesh.nx || niy < 0 || niy >= mesh.ny || niz < 0 || niz >= mesh.nz)
                return true;

            int neighbor = nix * mesh.ny * mesh.nz + niy * mesh.nz + niz;
            return cells.index_map[neighbor] == mhs::core::invalidIndex;
        }

        // Apply parsed face keys and other_bc to one cell's faces in place.
        void apply_face_bcs_to_cell(int c_idx, int ix, int iy, int iz, const mhs::core::MeshGeometry& mesh,
            const mhs::core::CellFields& cells, const std::vector<ParsedFaceKey>& parsed_keys,
            mhs::core::BcType other_bc_enum, uint16_t other_bc_idx, mhs::core::CellFields& cells_out)
        {
            for (mhs::core::FaceDir dir : mhs::core::FACE_DIRS) {
                // Axis letter from the direction table — replaces the old face_axis_letter helper.
                static constexpr char AXIS_LETTER[3] = {'X', 'Y', 'Z'};
                const int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];
                const char face_axis = AXIS_LETTER[axis];

                const double face_coord = mhs::utils::face_coord_value(dir, ix, iy, iz, mesh);
                const int ta = mhs::utils::TANGENT_A_OF_DIR[static_cast<size_t>(dir)];
                const int tb = mhs::utils::TANGENT_B_OF_DIR[static_cast<size_t>(dir)];
                const double centers[3] = {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz]};
                const double a_val = centers[ta];
                const double b_val = centers[tb];

                if (!is_cell_face_exposed(dir, ix, iy, iz, mesh, cells))
                    continue;

                // 优先匹配 parsed_keys
                bool matched = false;
                for (const auto& pk : parsed_keys) {
                    if (pk.fk.axis == face_axis && std::abs(face_coord - pk.fk.coord_value) < 1e-10) {
                        if (point_in_face_rects(pk.fk, a_val, b_val)) {
                            cells_out.cell_bcs[c_idx].types[(size_t)dir] = pk.bc_enum;
                            cells_out.cell_bcs[c_idx].param_idxs[(size_t)dir] = pk.param_idx;
                            matched = true;
                            break;
                        }
                    }
                }

                // 未命中 → 兜底 other_bc
                if (!matched && other_bc_enum != mhs::core::BcType::None) {
                    cells_out.cell_bcs[c_idx].types[(size_t)dir] = other_bc_enum;
                    cells_out.cell_bcs[c_idx].param_idxs[(size_t)dir] = other_bc_idx;
                }
            }
        }
    } // anonymous namespace

    mhs::core::CellFields resolve_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const std::unordered_map<std::string, size_t>& name_to_idx,
        const std::vector<std::vector<uint16_t>>& block_hs_map, const std::vector<ParsedFaceKey>& parsed_face_keys,
        mhs::core::BcType other_bc_enum, uint16_t other_bc_idx)
    {
        const int num_layers = (int)resolved_layers.size();
        const int total = mesh.nx * mesh.ny * mesh.nz;

        mhs::core::CellFields cells;
        cells.index_map.resize(total, mhs::core::invalidIndex);

        // Single full-grid traversal: write index_map + push material/heat-source
        // into compact vectors directly.  No full-grid temp arrays needed.
        cells.material_id.reserve(total);
        cells.heat_source_idx.reserve(total);

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
                        const int c_idx = (int)cells.material_id.size(); // compact index grows here
                        cells.index_map[old_idx] = c_idx;
                        const auto& block = resolved_layers[layer_idx].blocks[block_idx];
                        cells.material_id.push_back(static_cast<uint16_t>(name_to_idx.at(block.material_name)));
                        cells.heat_source_idx.push_back(block_hs_map[layer_idx][block_idx]);

                        // BC fill deferred: we need the full cell_bcs array first.
                        // cell_bcs is resized below after we know the count.
                    }
                }
            }
        }

        const int compact_count = (int)cells.material_id.size();
        cells.cell_bcs.resize(compact_count);

        // Second pass: fill BCs for every active cell.
        // Must use index_map to map old grid position back to compact index.
        // This avoids storing an intermediate old_idx→compact map.
        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    uint32_t c_idx = cells.index_map[old_idx];
                    if (c_idx == mhs::core::invalidIndex)
                        continue;
                    apply_face_bcs_to_cell(
                        (int)c_idx, ix, iy, iz, mesh, cells, parsed_face_keys, other_bc_enum, other_bc_idx, cells);
                }
            }
        }

        return cells;
    }

} // namespace mhs::sim
