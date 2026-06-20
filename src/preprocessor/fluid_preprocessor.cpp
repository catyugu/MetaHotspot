#include "common/logger.hpp"
#include "common/mesh_utils.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "function_helpers.hpp"
#include "linear_solver/linear_solver.hpp"
#include "preprocessor/fluid_preprocessor.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <limits>

namespace mhs::sim {

    // ===========================================================================
    // Anonymous namespace: internal helpers
    // ===========================================================================
    namespace {

        // ── Compact-to-old reverse map ──────────────────────────────────────
        // Build the reverse mapping: compact index → old grid index.
        // O(totalGrid) once, then all phases iterate N_fluid without re-scanning.
        std::vector<int> buildCompactToOld(const mhs::core::CellFields& cells, int totalGrid)
        {
            const int N = static_cast<int>(cells.material_id.size());
            std::vector<int> compact_to_old(N, -1);
            for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
                int c = static_cast<int>(cells.index_map[old_idx]);
                if (c >= 0)
                    compact_to_old[c] = old_idx;
            }
            return compact_to_old;
        }

        // ── Phase 1: Hydraulic conductance (porous-medium permeability) ──────
        void initCellHydroProperties(mhs::core::InternalModel& model, const std::vector<int>& compact_to_old)
        {
            const auto& mesh = model.mesh;

            if (model.n_fluid == 0)
                return;

            for (int fi = 0; fi < model.n_fluid; ++fi) {
                int c = model.fluid_to_global[fi];
                int old_idx = compact_to_old[c];
                int ix, iy, iz;
                mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                double dx_cell = mesh.dx[ix];
                double dy_cell = mesh.dy[iy];
                double dz_cell = mesh.dz[iz];
                double mu = model.dynamic_viscosity[fi];
                double dh = model.hydraulic_diameter[fi];

                double K_perm = (dh * dh) / 32.0;

                double A_xy = dx_cell * dy_cell;
                double A_xz = dx_cell * dz_cell;
                double A_yz = dy_cell * dz_cell;

                model.hydroC_x[fi] = K_perm * A_yz / (mu * dx_cell);
                model.hydroC_y[fi] = K_perm * A_xz / (mu * dy_cell);
                model.hydroC_z[fi] = K_perm * A_xy / (mu * dz_cell);
            }
        }

        // ── Phase 2: Pressure Poisson solve ────────────────────────────────
        void solvePressure(mhs::core::InternalModel& model, const std::vector<int>& compact_to_old)
        {
            const auto& cells = model.cells;
            const auto& mesh = model.mesh;

            if (model.n_fluid == 0)
                return;

            const int nf = model.n_fluid;
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(nf * 7);
            Eigen::VectorXd rhs(nf);
            rhs.setZero();

            for (int fi = 0; fi < nf; ++fi) {
                if (model.is_pressure_boundary[fi]) {
                    triplets.emplace_back(fi, fi, 1.0);
                    rhs(fi) = model.boundary_pressure[fi];
                    continue;
                }

                int c = model.fluid_to_global[fi];
                int old_idx = compact_to_old[c];
                int ix, iy, iz;
                mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                double diagSum = 0.0;

                for (auto dir : mhs::core::FACE_DIRS) {
                    int neighborOld
                        = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                    if (neighborOld < 0)
                        continue;

                    int n = static_cast<int>(cells.index_map[neighborOld]);
                    int fn = (n >= 0 && n < static_cast<int>(model.global_to_fluid.size())) ? model.global_to_fluid[n]
                                                                                            : -1;
                    if (fn < 0)
                        continue;

                    int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];

                    // Get conductance along this axis
                    double hydroC_c, hydroC_n;
                    switch (axis) {
                    case 0:
                        hydroC_c = model.hydroC_x[fi];
                        hydroC_n = model.hydroC_x[fn];
                        break;
                    case 1:
                        hydroC_c = model.hydroC_y[fi];
                        hydroC_n = model.hydroC_y[fn];
                        break;
                    default:
                        hydroC_c = model.hydroC_z[fi];
                        hydroC_n = model.hydroC_z[fn];
                        break;
                    }

                    double C_eff = mhs::utils::harmonicAverage(hydroC_c, hydroC_n);
                    diagSum += C_eff;
                    triplets.emplace_back(fi, fn, -C_eff);
                }
                triplets.emplace_back(fi, fi, diagSum);
            }

            // Assemble and solve
            Eigen::SparseMatrix<double> A(nf, nf);
            A.setFromTriplets(triplets.begin(), triplets.end());

            auto solver = mhs::sim::LinearSolver::create(mhs::sim::SolverType::BiCGSTAB);
            auto result = solver->solve(A, rhs);
            if (!result.success) {
                MHS_LOG_WARN("Fluid pressure solve failed (nf={}, nz={})", nf, static_cast<int>(A.nonZeros()));
                return;
            }

            for (int fi = 0; fi < nf; ++fi) {
                model.pressure[fi] = result.solution(fi);
            }
        }

        // ── Phase 3: Flow axes ──────────────────────────────────────────────
        void precomputeFlowAxes(mhs::core::InternalModel& model, const std::vector<int>& compact_to_old)
        {
            const auto& cells = model.cells;
            const auto& mesh = model.mesh;

            if (model.n_fluid == 0)
                return;

            model.flow_axes.assign(model.n_fluid, -1);

            for (int fi = 0; fi < model.n_fluid; ++fi) {
                int c = model.fluid_to_global[fi];
                int old_idx = compact_to_old[c];
                int ix, iy, iz;
                mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                double maxVal = -1.0;
                int bestAxis = 0;

                for (auto dir : mhs::core::FACE_DIRS) {
                    int neighborOld
                        = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                    if (neighborOld < 0)
                        continue;

                    int n = static_cast<int>(cells.index_map[neighborOld]);
                    int fn = (n >= 0 && n < static_cast<int>(model.global_to_fluid.size())) ? model.global_to_fluid[n]
                                                                                            : -1;
                    if (fn < 0)
                        continue;

                    double dp = std::fabs(model.pressure[fi] - model.pressure[fn]);
                    double hc_a = 0.0, hc_b = 0.0;
                    switch (mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)]) {
                    case 0:
                        hc_a = model.hydroC_x[fi];
                        hc_b = model.hydroC_x[fn];
                        break;
                    case 1:
                        hc_a = model.hydroC_y[fi];
                        hc_b = model.hydroC_y[fn];
                        break;
                    default:
                        hc_a = model.hydroC_z[fi];
                        hc_b = model.hydroC_z[fn];
                        break;
                    }
                    double flux = dp * mhs::utils::harmonicAverage(hc_a, hc_b);
                    int ax = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];
                    if (flux > maxVal) {
                        maxVal = flux;
                        bestAxis = ax;
                    }
                }
                model.flow_axes[fi] = static_cast<int8_t>(bestAxis);
            }
        }

        // ── Channel geometry ────────────────────────────────────────────────
        // Compute per-fluid-cell hydraulic diameter, channel width, and height
        // by finding the contiguous fluid extent in each axis.
        void computeChannelDimensions(mhs::core::InternalModel& model, const std::vector<int>& compact_to_old)
        {
            const auto& mesh = model.mesh;
            const auto& cells = model.cells;

            for (int fi = 0; fi < model.n_fluid; ++fi) {
                int c_idx = model.fluid_to_global[fi];
                int old_idx = compact_to_old[c_idx];
                int ix, iy, iz;
                mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                // 1. X-direction contiguous fluid extent
                int min_ix = ix, max_ix = ix;
                while (min_ix > 0) {
                    int n_old = (min_ix - 1) * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    int n_c = static_cast<int>(cells.index_map[n_old]);
                    if (n_c < 0 || n_c >= static_cast<int>(model.is_fluid.size()) || !model.is_fluid[n_c])
                        break;
                    min_ix--;
                }
                while (max_ix < mesh.nx - 1) {
                    int n_old = (max_ix + 1) * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    int n_c = static_cast<int>(cells.index_map[n_old]);
                    if (n_c < 0 || n_c >= static_cast<int>(model.is_fluid.size()) || !model.is_fluid[n_c])
                        break;
                    max_ix++;
                }
                double len_x = (mesh.cx[max_ix] + mesh.dx[max_ix] * 0.5) - (mesh.cx[min_ix] - mesh.dx[min_ix] * 0.5);

                // 2. Y-direction contiguous fluid extent
                int min_iy = iy, max_iy = iy;
                while (min_iy > 0) {
                    int n_old = ix * mesh.ny * mesh.nz + (min_iy - 1) * mesh.nz + iz;
                    int n_c = static_cast<int>(cells.index_map[n_old]);
                    if (n_c < 0 || n_c >= static_cast<int>(model.is_fluid.size()) || !model.is_fluid[n_c])
                        break;
                    min_iy--;
                }
                while (max_iy < mesh.ny - 1) {
                    int n_old = ix * mesh.ny * mesh.nz + (max_iy + 1) * mesh.nz + iz;
                    int n_c = static_cast<int>(cells.index_map[n_old]);
                    if (n_c < 0 || n_c >= static_cast<int>(model.is_fluid.size()) || !model.is_fluid[n_c])
                        break;
                    max_iy++;
                }
                double len_y = (mesh.cy[max_iy] + mesh.dy[max_iy] * 0.5) - (mesh.cy[min_iy] - mesh.dy[min_iy] * 0.5);

                // 3. Z-direction contiguous fluid extent
                int min_iz = iz, max_iz = iz;
                while (min_iz > 0) {
                    int n_old = ix * mesh.ny * mesh.nz + iy * mesh.nz + (min_iz - 1);
                    int n_c = static_cast<int>(cells.index_map[n_old]);
                    if (n_c < 0 || n_c >= static_cast<int>(model.is_fluid.size()) || !model.is_fluid[n_c])
                        break;
                    min_iz--;
                }
                while (max_iz < mesh.nz - 1) {
                    int n_old = ix * mesh.ny * mesh.nz + iy * mesh.nz + (max_iz + 1);
                    int n_c = static_cast<int>(cells.index_map[n_old]);
                    if (n_c < 0 || n_c >= static_cast<int>(model.is_fluid.size()) || !model.is_fluid[n_c])
                        break;
                    max_iz++;
                }
                double len_z = (mesh.cz[max_iz] + mesh.dz[max_iz] * 0.5) - (mesh.cz[min_iz] - mesh.dz[min_iz] * 0.5);

                // 4. Smallest two dimensions = cross-section width & height
                double lengths[3] = {len_x, len_y, len_z};
                std::sort(lengths, lengths + 3);
                double cross_w = lengths[0];
                double cross_h = lengths[1];

                // 5. Hydraulic diameter Dh = 2 * W * H / (W + H)
                double dh = 0.0;
                if (cross_w + cross_h > 1e-12) {
                    dh = 2.0 * cross_w * cross_h / (cross_w + cross_h);
                }

                model.hydraulic_diameter[fi] = dh;
                model.channel_width[fi] = cross_w;
                model.channel_height[fi] = cross_h;
            }
        }

    } // anonymous namespace

    // ===========================================================================
    // applyFluidOverlay — mark fluid cells, build indirection, parse boundaries,
    //                     compute channel geometry
    // ===========================================================================
    void applyFluidOverlay(mhs::core::InternalModel& model, const std::optional<mhs::core::FluidOverlay>& overlay,
        const mhs::core::IOStructure& ioStructure)
    {
        if (!overlay.has_value())
            return;

        const auto& fluidOverlay = overlay.value();
        if (fluidOverlay.fluid_materials.empty())
            return;

        const double si_scale = mhs::utils::length_unit_to_si(ioStructure.length_unit);
        const int N = static_cast<int>(model.cells.cell_bcs.size());

        // ── Build material name → table index mapping ──────────────────────
        std::unordered_map<std::string, uint16_t> matNameToTableIdx;
        for (const auto& layer : ioStructure.layers) {
            for (const auto& block : layer.blocks) {
                if (matNameToTableIdx.find(block.material_name) == matNameToTableIdx.end()) {
                    matNameToTableIdx[block.material_name] = static_cast<uint16_t>(matNameToTableIdx.size());
                }
            }
        }
        std::vector<std::string> matNamesByTableIdx(matNameToTableIdx.size());
        for (const auto& [name, idx] : matNameToTableIdx) {
            matNamesByTableIdx[idx] = name;
        }

        // ── Mark is_fluid and register viscosity expressions ───────────────
        model.is_fluid.assign(N, 0);
        std::vector<double> visc_temp(N, 0.0);

        std::unordered_map<std::string, std::string> fluidViscosityMap;
        for (const auto& fm : fluidOverlay.fluid_materials) {
            fluidViscosityMap[fm.name] = fm.dynamic_viscosity;
        }

        for (uint16_t matIdx = 0; matIdx < static_cast<uint16_t>(model.material_table.size()); ++matIdx) {
            const auto& matName = matNamesByTableIdx[matIdx];
            auto visIt = fluidViscosityMap.find(matName);
            if (visIt != fluidViscosityMap.end() && !visIt->second.empty()) {
                model.material_table[matIdx].is_fluid = true;
                model.material_table[matIdx].dynamic_viscosity
                    = mhs::core::parse(substitute_function_args(visIt->second, "T", ioStructure.functions));
            }
        }

        for (int c = 0; c < N; ++c) {
            uint16_t matIdx = model.cells.material_id[c];
            if (matIdx < model.material_table.size() && model.material_table[matIdx].is_fluid) {
                model.is_fluid[c] = 1;
                visc_temp[c]
                    = model.material_table[matIdx].dynamic_viscosity.eval({0, 0, 0, model.initial_temperature, 0});
            }
        }

        bool hasFluid = std::any_of(model.is_fluid.begin(), model.is_fluid.end(), [](uint8_t v) { return v != 0; });
        if (!hasFluid)
            return;

        // ── Build fluid indirection mapping ────────────────────────────────
        model.fluid_to_global.clear();
        model.global_to_fluid.assign(N, -1);
        for (int c = 0; c < N; ++c) {
            if (model.is_fluid[c]) {
                model.global_to_fluid[c] = static_cast<int>(model.fluid_to_global.size());
                model.fluid_to_global.push_back(c);
            }
        }
        model.n_fluid = static_cast<int>(model.fluid_to_global.size());

        // ── Compact fluid arrays ───────────────────────────────────────────
        model.dynamic_viscosity.assign(model.n_fluid, 0.0);
        model.pressure.assign(model.n_fluid, 0.0);
        model.flow_axes.assign(model.n_fluid, -1);
        model.hydroC_x.assign(model.n_fluid, 0.0);
        model.hydroC_y.assign(model.n_fluid, 0.0);
        model.hydroC_z.assign(model.n_fluid, 0.0);
        model.is_pressure_boundary.assign(model.n_fluid, 0);
        model.boundary_pressure.assign(model.n_fluid, 0.0);
        model.boundary_temperature_fluid.assign(model.n_fluid, std::numeric_limits<double>::quiet_NaN());
        model.hydraulic_diameter.assign(model.n_fluid, 0.0);
        model.channel_width.assign(model.n_fluid, 0.0);
        model.channel_height.assign(model.n_fluid, 0.0);

        for (int fi = 0; fi < model.n_fluid; ++fi) {
            model.dynamic_viscosity[fi] = visc_temp[model.fluid_to_global[fi]];
        }

        // ── Parse pressure boundaries ──────────────────────────────────────
        // Scan the full grid to match face keys against fluid cell centers.
        auto applyBoundary = [&](int f_idx, const mhs::core::FluidBoundaryOverlay& fb) {
            model.is_pressure_boundary[f_idx] = 1;
            model.boundary_pressure[f_idx] = fb.pressure_bc.pressure;
            if (!std::isnan(fb.inlet_temperature)) {
                model.boundary_temperature_fluid[f_idx] = fb.inlet_temperature;
            }
        };

        for (const auto& fb : fluidOverlay.boundaries) {
            for (const auto& keyStr : fb.face_keys) {
                FaceKeyInfo fk = parse_face_key(keyStr, si_scale);

                if (fk.axis == 'X') {
                    for (int ix = 0; ix < model.mesh.nx; ++ix) {
                        double fx_w = model.mesh.cx[ix] - model.mesh.dx[ix] * 0.5;
                        double fx_e = model.mesh.cx[ix] + model.mesh.dx[ix] * 0.5;
                        if (std::abs(fx_w - fk.coord_value) >= 1e-8 && std::abs(fx_e - fk.coord_value) >= 1e-8)
                            continue;
                        for (int iy = 0; iy < model.mesh.ny; ++iy) {
                            for (int iz = 0; iz < model.mesh.nz; ++iz) {
                                int old_idx = ix * model.mesh.ny * model.mesh.nz + iy * model.mesh.nz + iz;
                                int c_idx = static_cast<int>(model.cells.index_map[old_idx]);
                                if (c_idx < 0 || c_idx >= N || !model.is_fluid[c_idx])
                                    continue;
                                if (point_in_face_rects(fk, model.mesh.cy[iy], model.mesh.cz[iz])) {
                                    applyBoundary(model.global_to_fluid[c_idx], fb);
                                }
                            }
                        }
                    }
                }
                else if (fk.axis == 'Y') {
                    for (int iy = 0; iy < model.mesh.ny; ++iy) {
                        double fy_s = model.mesh.cy[iy] - model.mesh.dy[iy] * 0.5;
                        double fy_n = model.mesh.cy[iy] + model.mesh.dy[iy] * 0.5;
                        if (std::abs(fy_s - fk.coord_value) >= 1e-8 && std::abs(fy_n - fk.coord_value) >= 1e-8)
                            continue;
                        for (int ix = 0; ix < model.mesh.nx; ++ix) {
                            for (int iz = 0; iz < model.mesh.nz; ++iz) {
                                int old_idx = ix * model.mesh.ny * model.mesh.nz + iy * model.mesh.nz + iz;
                                int c_idx = static_cast<int>(model.cells.index_map[old_idx]);
                                if (c_idx < 0 || c_idx >= N || !model.is_fluid[c_idx])
                                    continue;
                                if (point_in_face_rects(fk, model.mesh.cx[ix], model.mesh.cz[iz])) {
                                    applyBoundary(model.global_to_fluid[c_idx], fb);
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Compute channel geometry ───────────────────────────────────────
        computeChannelDimensions(model, buildCompactToOld(model.cells, model.mesh.nx * model.mesh.ny * model.mesh.nz));
    }

    // ===========================================================================
    // solveFluidFlow — hydraulic conductance → pressure → flow axes
    // ===========================================================================
    void solveFluidFlow(mhs::core::InternalModel& model)
    {
        if (std::none_of(model.is_fluid.begin(), model.is_fluid.end(), [](uint8_t v) { return v != 0; }))
            return;

        const int totalGrid = model.mesh.nx * model.mesh.ny * model.mesh.nz;
        auto compact_to_old = buildCompactToOld(model.cells, totalGrid);

        initCellHydroProperties(model, compact_to_old);
        solvePressure(model, compact_to_old);
        precomputeFlowAxes(model, compact_to_old);
    }

} // namespace mhs::sim
