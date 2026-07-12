#include "common/mesh_utils.hpp"
#include "data/tolerance_config.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "layer_processor.hpp"
#include "smart_block/smart_block_reader.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <unordered_map>

namespace mhs::sim {

    using mhs::core::geometry_eps;
    constexpr double EPS = geometry_eps;

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
                rb.is_smart_macro = (block.block_type == mhs::core::BlockType::SmartMacro);
                rb.model_file = block.model_file;

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
        // Match face keys for the given cell face and return the matching BC.
        // Returns (BcType::None, 0) if no key matches and no valid fallback.
        std::pair<mhs::core::BcType, uint16_t> match_face_bc(mhs::core::FaceDir dir, int ix, int iy, int iz,
            const mhs::core::MeshGeometry& mesh, const std::vector<ParsedFaceKey>& parsed_keys,
            mhs::core::BcType other_bc_enum, uint16_t other_bc_idx)
        {
            static constexpr char AXIS_LETTER[3] = {'X', 'Y', 'Z'};
            const int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];
            const char face_axis_letter = AXIS_LETTER[axis];

            const double face_coord = mhs::utils::face_coord_value(dir, ix, iy, iz, mesh);
            const int ta = mhs::utils::TANGENT_A_OF_DIR[static_cast<size_t>(dir)];
            const int tb = mhs::utils::TANGENT_B_OF_DIR[static_cast<size_t>(dir)];
            const double centers[3] = {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz]};
            const double a_val = centers[ta];
            const double b_val = centers[tb];

            for (const auto& pk : parsed_keys) {
                if (pk.fk.axis == face_axis_letter
                    && std::abs(face_coord - pk.fk.coord_value) < mhs::core::geometry_eps) {
                    if (point_in_face_rects(pk.fk, a_val, b_val)) {
                        return {pk.bc_enum, pk.param_idx};
                    }
                }
            }

            // Fallback to other_bc
            if (other_bc_enum != mhs::core::BcType::None) {
                return {other_bc_enum, other_bc_idx};
            }
            return {mhs::core::BcType::None, 0};
        }
    } // anonymous namespace

    mhs::core::CellFields assign_cell_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const std::unordered_map<std::string, size_t>& name_to_idx,
        const std::vector<std::vector<uint16_t>>& block_hs_map)
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
                        const auto& block = resolved_layers[layer_idx].blocks[block_idx];
                        // Skip SmartMacro blocks — their cells stay virtual (invalidIndex)
                        if (block.is_smart_macro)
                            continue;
                        const int c_idx = (int)cells.material_id.size(); // compact index grows here
                        cells.index_map[old_idx] = c_idx;
                        cells.material_id.push_back(static_cast<uint16_t>(name_to_idx.at(block.material_name)));
                        cells.heat_source_idx.push_back(block_hs_map[layer_idx][block_idx]);
                    }
                }
            }
        }

        return cells;
    }

    void resolve_boundary_patches(const mhs::core::MeshGeometry& mesh, const mhs::core::CellFields& cells,
        const std::vector<ParsedFaceKey>& parsed_face_keys, mhs::core::BcType other_bc_enum, uint16_t other_bc_idx,
        std::vector<mhs::core::FaceBC>& face_bcs)
    {
        const int compact_count = (int)cells.material_id.size();
        face_bcs.assign(compact_count * mhs::core::FACE_COUNT, mhs::core::FaceBC {});

        // Single grid traversal: for each active cell's exposed faces,
        // match the face key and write directly into face_bcs[c*6+dir].
        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    uint32_t c_idx = cells.index_map[old_idx];
                    if (c_idx == mhs::core::invalidIndex)
                        continue;

                    for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
                        mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                        if (mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map)
                            >= 0)
                            continue; // internal face — no BC needed

                        auto [type, param_idx]
                            = match_face_bc(dir, ix, iy, iz, mesh, parsed_face_keys, other_bc_enum, other_bc_idx);
                        face_bcs[c_idx * mhs::core::FACE_COUNT + f] = {type, param_idx};
                    }
                }
            }
        }
    }

    // ── SmartMacro block coupling ───────────────────────────────────────────
    void build_smart_block_coupling(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const mhs::core::CellFields& cells, const std::string& case_dir,
        mhs::core::Model& model)
    {
        // (1) Find SmartMacro blocks in the resolved layers.
        // For each, collect all grid cells inside the block and classify them
        // as interface (adjacent to an active cell) vs interior.
        struct SmartBlockInfo {
            const ResolvedBlock* block = nullptr;
            int layer_index = -1;
            std::vector<int> old_indices; // full-grid indices of cells inside the block
            std::vector<int> port_old_indices; // subset: interface cells only
        };
        std::vector<SmartBlockInfo> info_list;

        for (int l = 0; l < (int)resolved_layers.size(); l++) {
            for (int b = 0; b < (int)resolved_layers[l].blocks.size(); b++) {
                const auto& rb = resolved_layers[l].blocks[b];
                if (!rb.is_smart_macro)
                    continue;

                SmartBlockInfo info;
                info.block = &rb;
                info.layer_index = l;

                // Scan the full grid for cells inside this block
                for (int ix = 0; ix < mesh.nx; ix++) {
                    for (int iy = 0; iy < mesh.ny; iy++) {
                        for (int iz = 0; iz < mesh.nz; iz++) {
                            double cx = mesh.cx[ix];
                            double cy = mesh.cy[iy];
                            double cz = mesh.cz[iz];
                            if (find_block_for_cell(resolved_layers[l], cx, cy, cz) == b) {
                                int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                                info.old_indices.push_back(old_idx);
                            }
                        }
                    }
                }

                if (info.old_indices.empty())
                    continue;

                info_list.push_back(std::move(info));
            }
        }

        if (info_list.empty()) {
            model.smart_blocks.clear();
            return;
        }

        // (2) For each SmartMacro, load the trained model and identify ports.
        // Ports are smart-block cells with at least one face adjacent to an active cell.
        for (auto& si : info_list) {
            const std::string model_path = (std::filesystem::path(case_dir) / si.block->model_file).string();
            auto trained = mhs::core::read_smart_macro_model(model_path);
            const int n_ports = (int)trained.port_ix.size();

            // Build a map: (ix, iy, iz) -> port_index from the trained model.
            struct GridKey {
                int ix, iy, iz;
            };
            auto hash_fn
                = [](const GridKey& k) -> size_t { return (size_t(k.ix) << 32) ^ (size_t(k.iy) << 16) ^ size_t(k.iz); };
            struct KeyEqual {
                bool operator()(const GridKey& a, const GridKey& b) const
                {
                    return a.ix == b.ix && a.iy == b.iy && a.iz == b.iz;
                }
            };
            std::unordered_map<GridKey, int, decltype(hash_fn), KeyEqual> port_map(n_ports, hash_fn, KeyEqual {});
            for (int p = 0; p < n_ports; p++) {
                port_map[{trained.port_ix[p], trained.port_iy[p], trained.port_iz[p]}] = p;
            }

            // (3) Identify all interface ports in the current global grid:
            //     For each smart block cell, check if any neighbor is active.
            //     Also compute C_i = k * A / half_dist for the active neighbor side.
            std::vector<Eigen::Triplet<double>> eff_triplets;
            mhs::core::SmartBlockModel sbm;
            sbm.name = trained.name;
            sbm.K_port = std::move(trained.K_port);
            sbm.ports.resize(n_ports);

            // port -> C_i for each port
            Eigen::VectorXd C_diag = Eigen::VectorXd::Zero(n_ports);

            // Initialize port records with invalid indices
            for (int p = 0; p < n_ports; ++p) {
                sbm.ports[p].port_idx = static_cast<uint16_t>(p);
                sbm.ports[p].active_cell_idx = mhs::core::invalidIndex;
            }

            // For each smart-block cell, check faces
            for (int old_idx : si.old_indices) {
                int ix, iy, iz;
                mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                GridKey key {ix, iy, iz};
                auto pit = port_map.find(key);
                if (pit == port_map.end())
                    continue; // not an interface port in the training model

                int p_idx = pit->second;

                for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
                    mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];

                    int n_old
                        = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                    if (n_old < 0)
                        continue; // no active neighbor → domain boundary, skip

                    uint32_t act_c_idx = cells.index_map[n_old];
                    if (act_c_idx == mhs::core::invalidIndex)
                        continue; // neighbor is also virtual (another smart block)

                    // Active neighbor found — compute C_i
                    int nix = mhs::utils::neighbor_ix(dir, ix);
                    int niy = mhs::utils::neighbor_iy(dir, iy);
                    int niz = mhs::utils::neighbor_iz(dir, iz);

                    // Use the active neighbor's material for conductivity
                    size_t nm_id = model.cells.material_id[act_c_idx];
                    const auto& nmp = model.material_table[nm_id];

                    mhs::core::FieldContext ctx_n {
                        mesh.cx[nix], mesh.cy[niy], mesh.cz[niz], model.initial_temperature, 0.0};
                    double k_active
                        = mhs::utils::k_along(dir, nmp.kx.eval(ctx_n), nmp.ky.eval(ctx_n), nmp.kz.eval(ctx_n));

                    const double A_f = mhs::utils::face_area(dir, mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]);
                    const double half_dist_active
                        = mhs::utils::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);
                    double C_i = k_active * A_f / half_dist_active;

                    C_diag(p_idx) += C_i;

                    sbm.ports[p_idx].active_cell_idx = act_c_idx;
                    sbm.ports[p_idx].dir = dir;
                    sbm.ports[p_idx].C = C_diag(p_idx);

                    // Only count the first active neighbor found (conservative)
                    break;
                }
            }

            // (4) Pre-compute K_eff = C - C * (K_port + C)^(-1) * C
            // where C is the diagonal matrix of coupling conductances
            Eigen::MatrixXd C_mat = C_diag.asDiagonal();
            Eigen::MatrixXd Kpc = sbm.K_port + C_mat; // (K_port + C)
            Eigen::MatrixXd Kpc_inv
                = Kpc.selfadjointView<Eigen::Lower>().llt().solve(Eigen::MatrixXd::Identity(n_ports, n_ports));

            // K_eff = C * (I - Kpc_inv * C)  — more numerically stable factorization
            // K_eff = C - C * Kpc_inv * C
            // Compute: M = C * Kpc_inv * C, then K_eff = C - M
            Eigen::MatrixXd M = C_mat * (Kpc_inv * C_mat);
            Eigen::MatrixXd K_eff_dense = C_mat - M;

            // Convert to sparse for the assembler
            // (Only store non-zeros within tolerance)
            std::vector<Eigen::Triplet<double>> eff_t;
            eff_t.reserve(n_ports * n_ports);
            for (int i = 0; i < n_ports; i++) {
                for (int j = 0; j < n_ports; j++) {
                    double val = K_eff_dense(i, j);
                    if (std::abs(val) > mhs::core::zero_guard) {
                        eff_t.emplace_back(i, j, val);
                    }
                }
            }
            sbm.K_eff.resize(n_ports, n_ports);
            sbm.K_eff.setFromTriplets(eff_t.begin(), eff_t.end());

            // Store the reduced RHS from BCs
            sbm.f_port = std::move(trained.f_port);

            // Pre-compute rhs_eff = C * Kpc_inv * f_port
            // This is the RHS contribution on active-cell DOFs from BCs on the smart block.
            // The interface heat flux is: Q = C * (T_f - T_c)
            // After eliminating T_f: Q = C*(K_port+C)^{-1}*f_port - K_eff*T_c
            // In the assembly: K += K_eff,  b += C*(K_port+C)^{-1}*f_port
            sbm.rhs_eff = C_mat * (Kpc_inv * sbm.f_port);

            model.smart_blocks.push_back(std::move(sbm));
        }
    }

} // namespace mhs::sim
