#include "common/mesh_utils.hpp"
#include "data/tolerance_config.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "layer_processor.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
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

    // ── SmartMacro block coupling (BC-agnostic, face-level) ────────────────
    //
    // For each SmartMacro block, iterate over trained port faces. Each face
    // belongs to a block-interior cell and has a direction `dir`.
    //  1. If the face has an active (non-virtual) neighbor:
    //       C_env = k_n * A_f / h_n     (neighbour conductance)
    //       T_ref = T_neighbor (unknown — goes into the system matrix)
    //       Q_ext = 0
    //     This adds coupling to the neighbour cell DOF.
    //  2. If the face is on the domain boundary:
    //       Match the face against boundary patches (face keys).
    //       Unified (C_env, T_ref, Q_ext) via Thevenin equivalence:
    //         Dirichlet (FirstType):   C = 1e10, T_ref = T_bc, Q_ext = 0
    //         Neumann   (SecondType):  C = 0, T_ref = 0, Q_ext = q*A
    //         Robin     (ThirdType):   C = h*A, T_ref = T_inf, Q_ext = 0
    //  3. Accumulate per-face values for the modal extended system.
    //
    // No BCs are baked into the trained modal data. All external effects enter
    // through the C_env / T_ref / Q_ext parameters.
    void build_smart_block_coupling(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const mhs::core::CellFields& cells,
        const std::vector<mhs::core::SmartMacroModelData>& trained_models,
        const std::vector<ParsedFaceKey>& parsed_face_keys, const mhs::core::BCParamTable& bc_params,
        mhs::core::BcType other_bc_enum, uint16_t other_bc_idx, mhs::core::Model& model)
    {
        model.smart_blocks.clear();
        int cumulative_modes = 0;
        const int N_phys = model.physical_dofs();

        size_t sm_idx = 0;
        for (int l = 0; l < (int)resolved_layers.size(); l++) {
            for (int b = 0; b < (int)resolved_layers[l].blocks.size(); b++) {
                if (!resolved_layers[l].blocks[b].is_smart_macro)
                    continue;
                if (sm_idx >= trained_models.size())
                    continue;

                const auto& trained = trained_models[sm_idx];
                sm_idx++;
                const int n_faces = (int)trained.port_ix.size();
                const int n_modes = trained.n_modes;
                if (n_faces == 0 || n_modes == 0)
                    continue;

                // Wrap POD vectors with Eigen::Map for zero-copy algebra.
                auto phi_map = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                    trained.phi_basis.data(), n_faces, n_modes);
                auto K_modal_map
                    = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                        trained.K_modal.data(), n_modes, n_modes);

                mhs::core::SmartBlockModel sbm;
                sbm.name = trained.name;
                sbm.n_faces = n_faces;
                sbm.n_modes = n_modes;
                sbm.modal_start_idx = N_phys + cumulative_modes;

                sbm.C_env_vec = Eigen::VectorXd::Zero(n_faces);
                sbm.T_ref_vec = Eigen::VectorXd::Zero(n_faces);
                sbm.Q_ext_vec = Eigen::VectorXd::Zero(n_faces);
                sbm.phi_basis = phi_map;

                // Cell-aggregation scratch map
                std::unordered_map<uint32_t, std::pair<double, Eigen::VectorXd>> cell_contrib;
                // cell_contrib[c] = (C_total, phiC_total vector of size n_modes)

                // ── Phase 1: determine environment parameters per port face ──
                for (int p = 0; p < n_faces; p++) {
                    int ix = trained.port_ix[p];
                    int iy = trained.port_iy[p];
                    int iz = trained.port_iz[p];
                    int dir_val = (int)(trained.port_dir.size() > (size_t)p ? trained.port_dir[p] : -1);

                    if (ix < 0 || ix >= mesh.nx || iy < 0 || iy >= mesh.ny || iz < 0 || iz >= mesh.nz)
                        continue;
                    if (dir_val < 0 || (size_t)dir_val >= mhs::core::FACE_COUNT)
                        continue;

                    auto dir = static_cast<mhs::core::FaceDir>(dir_val);

                    // Check if this face has an active neighbor
                    int n_old
                        = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);

                    if (n_old >= 0) {
                        // ── Case 1: Active neighbor cell ──
                        uint32_t act_c_idx = cells.index_map[n_old];
                        size_t nm_id = model.cells.material_id[act_c_idx];
                        const auto& nmp = model.material_table[nm_id];

                        mhs::core::FieldContext ctx_n {
                            mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], model.initial_temperature, 0.0};
                        double k_active
                            = mhs::utils::k_along(dir, nmp.kx.eval(ctx_n), nmp.ky.eval(ctx_n), nmp.kz.eval(ctx_n));

                        const double A_f = mhs::utils::face_area(dir, mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]);
                        const int nix = mhs::utils::neighbor_ix(dir, ix);
                        const int niy = mhs::utils::neighbor_iy(dir, iy);
                        const int niz = mhs::utils::neighbor_iz(dir, iz);
                        const double half_dist_active
                            = mhs::utils::half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);

                        double C_env = k_active * A_f / half_dist_active;

                        sbm.C_env_vec(p) = C_env;
                        sbm.T_ref_vec(p) = 0.0; // T_neighbor is unknown → enters the system matrix
                        sbm.Q_ext_vec(p) = 0.0;

                        // Aggregate C_env and C_env*Φ for this neighbor cell
                        auto& [c_total, phi_total] = cell_contrib[act_c_idx];
                        if (phi_total.size() == 0) {
                            phi_total = Eigen::VectorXd::Zero(n_modes);
                            c_total = C_env;
                        }
                        else {
                            c_total += C_env;
                        }
                        for (int k = 0; k < n_modes; k++) {
                            phi_total(k) += C_env * phi_map(p, k);
                        }
                    }
                    else {
                        // ── Case 2: Domain boundary face → match against BC patches ──
                        const double A_f = mhs::utils::face_area(dir, mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]);
                        mhs::core::FieldContext ctx_c {
                            mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], model.initial_temperature, 0.0};
                        // Match face BC using the same match_face_bc logic
                        auto [bc_type, bc_idx]
                            = match_face_bc(dir, ix, iy, iz, mesh, parsed_face_keys, other_bc_enum, other_bc_idx);

                        double C_env = 0.0, T_ref = 0.0, Q_ext = 0.0;

                        switch (bc_type) {
                        case mhs::core::BcType::None:
                            // Adiabatic — zero flux through this face
                            C_env = 0.0;
                            break;

                        case mhs::core::BcType::FirstType: {
                            // Dirichlet: penalty method
                            double T_bc = bc_params.dirichlet_T[bc_idx].eval(ctx_c);
                            C_env = 1e10; // large penalty
                            T_ref = T_bc;
                            break;
                        }

                        case mhs::core::BcType::SecondType: {
                            // Neumann: prescribed heat flux
                            double q_val = bc_params.neumann_q[bc_idx].eval(ctx_c);
                            C_env = 0.0;
                            T_ref = 0.0;
                            Q_ext = q_val * A_f;
                            break;
                        }

                        case mhs::core::BcType::ThirdType: {
                            // Robin: convection
                            double h_val = bc_params.cauchy_h[bc_idx].eval(ctx_c);
                            double T_inf = bc_params.cauchy_T_inf[bc_idx].eval(ctx_c);
                            C_env = h_val * A_f;
                            T_ref = T_inf;
                            break;
                        }

                        default:
                            break;
                        }

                        sbm.C_env_vec(p) = C_env;
                        sbm.T_ref_vec(p) = T_ref;
                        sbm.Q_ext_vec(p) = Q_ext;
                    }
                }

                // ── Phase 2: Add Φᵀ·C_env·Φ to K_modal_eff for ALL faces ──
                // The variational form Q = C·(T_ref - T_face) + Q_ext
                // expands the modal equation identically for every face:
                //   δE/δa_k = ∫ φ·C·(T_ref - Σφ·a_k) + φ·Q_ext
                //            = φ·C·T_ref - Σ C·φ·φ·a_k + φ·Q_ext
                //   ∂/∂a_kp(bilinear):  K(k,kp) += C · φ(f,k) · φ(f,kp)
                // For neighbor faces, T_ref=T_neighbor enters the system
                // matrix (via coupled_C/coupled_phi);
                // for domain-BC faces, T_ref is known and enters the RHS
                // (handled in the assembler).
                sbm.K_modal_eff = K_modal_map + phi_map.transpose() * sbm.C_env_vec.asDiagonal() * phi_map;

                // ── Phase 3: Aggregate unique coupled cells ──
                sbm.coupled_cells.reserve(cell_contrib.size());
                for (const auto& kv : cell_contrib) {
                    sbm.coupled_cells.push_back(kv.first);
                }
                std::sort(sbm.coupled_cells.begin(), sbm.coupled_cells.end());

                const size_t n_coupled = sbm.coupled_cells.size();
                sbm.coupled_C = Eigen::VectorXd::Zero(n_coupled);
                sbm.coupled_phi = Eigen::MatrixXd::Zero(n_coupled, n_modes);

                for (size_t ci = 0; ci < n_coupled; ci++) {
                    const auto& contrib = cell_contrib[sbm.coupled_cells[ci]];
                    sbm.coupled_C(ci) = contrib.first;
                    sbm.coupled_phi.row(ci) = contrib.second;
                }

                cumulative_modes += n_modes;
                model.smart_blocks.push_back(std::move(sbm));
            }
        }
    }

} // namespace mhs::sim
