#include "common/logger.hpp"
#include "common/mesh_utils.hpp"
#include "fluid/fluid_preprocessor.hpp"
#include "linear_solver/sparse_lu_solver.hpp"

#include <Eigen/Sparse>

#include <cmath>

namespace mhs::sim {

    void FluidPreprocessor::solveFlow(mhs::core::InternalModel& model)
    {
        const int N = static_cast<int>(model.is_fluid.size());
        bool hasFluid = false;
        for (int i = 0; i < N; ++i) {
            if (model.is_fluid[i]) {
                hasFluid = true;
                break;
            }
        }
        if (!hasFluid)
            return;

        initCellHydroProperties(model);
        // applyPressureBoundaryConditions is already done in preprocessor phase
        solvePressure(model);
        precomputeFlowAxes(model);
    }

    // ---------------------------------------------------------------------------
    // Phase 1: Hydraulic conductance (Hele-Shaw / rectangular duct)
    // ---------------------------------------------------------------------------

    void FluidPreprocessor::initCellHydroProperties(mhs::core::InternalModel& model)
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;
        const int N = static_cast<int>(model.is_fluid.size());

        for (int c = 0; c < N; ++c) {
            if (!model.is_fluid[c])
                continue;

            // Decode compact index to grid coords
            // Build an inverse map: compact_idx -> (ix, iy, iz)
            // For efficiency we scan the grid once below, but for correctness
            // let's build a helper. Since this runs once at startup, O(N*total) is fine.
        }

        // Build compact-index -> grid-coord map
        struct GridPos {
            int ix, iy, iz;
        };
        std::vector<GridPos> compactToGrid(N);
        int totalGrid = mesh.nx * mesh.ny * mesh.nz;
        for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
            int c_idx = static_cast<int>(cells.index_map[old_idx]);
            if (c_idx < 0 || c_idx >= N)
                continue;
            int ix = old_idx / (mesh.ny * mesh.nz);
            int iy = (old_idx % (mesh.ny * mesh.nz)) / mesh.nz;
            int iz = old_idx % mesh.nz;
            compactToGrid[c_idx] = {ix, iy, iz};
        }

        for (int c = 0; c < N; ++c) {
            if (!model.is_fluid[c])
                continue;

            auto [ix, iy, iz] = compactToGrid[c];
            double dx_cell = mesh.dx[ix];
            double dy_cell = mesh.dy[iy];
            double dz_cell = mesh.dz[iz];
            double mu = model.dynamic_viscosity[c];
            if (mu < 1e-30)
                mu = 1e-30; // guard against zero viscosity

            double hydroC[3] = {0.0, 0.0, 0.0};

            // Three axes
            for (int axis = 0; axis < 3; ++axis) {
                double L = (axis == 0) ? dx_cell : ((axis == 1) ? dy_cell : dz_cell);
                int ax_w = (axis + 1) % 3;
                int ax_h = (axis + 2) % 3;
                double w = (ax_w == 0) ? dx_cell : ((ax_w == 1) ? dy_cell : dz_cell);
                double h = (ax_h == 0) ? dx_cell : ((ax_h == 1) ? dy_cell : dz_cell);

                if (L < 1e-30)
                    continue;

                double ar = std::min(w, h) / std::max(w, h);
                double hydroC_val;

                if (std::fabs(h - w) < 1e-10) {
                    // Square
                    hydroC_val = 0.42229 * h * h * h * h / (12.0 * mu * L);
                }
                else if (h > w) {
                    hydroC_val = (1.0 - 0.63 * ar) * w * w * w * h / (12.0 * mu * L);
                }
                else {
                    hydroC_val = (1.0 - 0.63 * ar) * h * h * h * w / (12.0 * mu * L);
                }

                hydroC[axis] = hydroC_val;
            }

            model.hydroC_x[c] = hydroC[0];
            model.hydroC_y[c] = hydroC[1];
            model.hydroC_z[c] = hydroC[2];
        }
    }

    // ---------------------------------------------------------------------------
    // Phase 3: Pressure Poisson solve
    // ---------------------------------------------------------------------------

    void FluidPreprocessor::solvePressure(mhs::core::InternalModel& model)
    {
        const auto& cells = model.cells;
        const auto& mesh = model.mesh;
        const int N = static_cast<int>(model.is_fluid.size());
        int totalGrid = mesh.nx * mesh.ny * mesh.nz;

        // Build fluid subdomain index: g2f[compact_idx] = local fluid index, or -1
        std::vector<int> g2f(N, -1);
        std::vector<int> fluidCompactIdx; // compact indices of fluid cells in order
        int nf = 0;
        for (int c = 0; c < N; ++c) {
            if (model.is_fluid[c]) {
                g2f[c] = nf;
                fluidCompactIdx.push_back(c);
                ++nf;
            }
        }

        if (nf == 0)
            return;

        // Build sparse matrix (CSR via triplets)
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nf * 7); // ~6 neighbors + diagonal
        Eigen::VectorXd rhs(nf);
        rhs.setZero();

        // For each fluid cell, find its internal-face neighbors that are also fluid
        // We iterate over all grid cells and their 6 faces
        for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
            int c = static_cast<int>(cells.index_map[old_idx]);
            if (c < 0 || c >= N || !model.is_fluid[c])
                continue;

            int ix = old_idx / (mesh.ny * mesh.nz);
            int iy = (old_idx % (mesh.ny * mesh.nz)) / mesh.nz;
            int iz = old_idx % mesh.nz;

            int fi = g2f[c];
            double diagSum = 0.0;

            // Iterate 6 faces
            for (size_t f = 0; f < 6; ++f) {
                mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                int neighborOld = mhs::utils::neighbor_grid_index(
                    ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (neighborOld < 0)
                    continue;

                int n = static_cast<int>(cells.index_map[neighborOld]);
                if (n < 0 || n >= N || !model.is_fluid[n])
                    continue; // solid neighbor or virtual cell — skip

                int fn = g2f[n];
                int axis = mhs::utils::AXIS_OF_DIR[f];

                // Get conductance along this axis
                double hydroC_c, hydroC_n;
                switch (axis) {
                case 0:
                    hydroC_c = model.hydroC_x[c];
                    hydroC_n = model.hydroC_x[n];
                    break;
                case 1:
                    hydroC_c = model.hydroC_y[c];
                    hydroC_n = model.hydroC_y[n];
                    break;
                default:
                    hydroC_c = model.hydroC_z[c];
                    hydroC_n = model.hydroC_z[n];
                    break;
                }

                double C_eff = harmonicConductance(hydroC_c, hydroC_n);
                if (C_eff < 1e-30)
                    continue;

                // Both c and n are fluid.
                // FVM for ∇·(K∇P)=0: sum C_eff·(P_n − P_c) = 0 over all neighbor faces.
                // Matrix contribution: diag = −sum(C_eff),  off-diag = +C_eff.
                if (!model.is_pressure_boundary[c]) {
                    triplets.emplace_back(fi, fn, C_eff);
                    diagSum += C_eff;
                }
                if (!model.is_pressure_boundary[n]) {
                    triplets.emplace_back(fn, fi, C_eff);
                    diagSum += C_eff;
                }
            }

            // Diagonal
            if (model.is_pressure_boundary[c]) {
                triplets.emplace_back(fi, fi, 1.0);
                rhs(fi) = model.boundary_pressure[c];
            }
            else {
                triplets.emplace_back(fi, fi, -diagSum);
            }
        }

        // Assemble and solve
        Eigen::SparseMatrix<double> A(nf, nf);
        A.setFromTriplets(triplets.begin(), triplets.end());

        SparseLUSolver solver;
        auto result = solver.solve(A, rhs);
        if (!result.success) {
            MHS_LOG_WARN("Fluid pressure solve failed (nf={}, nz={})", nf, (int)A.nonZeros());
            return;
        }

        // Write back
        for (int i = 0; i < nf; ++i) {
            model.pressure[fluidCompactIdx[i]] = result.solution(i);
        }
    }

    // ---------------------------------------------------------------------------
    // Phase 4: Flow axes
    // ---------------------------------------------------------------------------

    void FluidPreprocessor::precomputeFlowAxes(mhs::core::InternalModel& model)
    {
        const auto& cells = model.cells;
        const auto& mesh = model.mesh;
        const int N = static_cast<int>(model.is_fluid.size());
        int totalGrid = mesh.nx * mesh.ny * mesh.nz;

        // Compute per-axis max |Δp| for each fluid cell
        std::vector<double> maxDeltaP_x(N, 0.0);
        std::vector<double> maxDeltaP_y(N, 0.0);
        std::vector<double> maxDeltaP_z(N, 0.0);

        for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
            int c = static_cast<int>(cells.index_map[old_idx]);
            if (c < 0 || c >= N || !model.is_fluid[c])
                continue;

            int ix = old_idx / (mesh.ny * mesh.nz);
            int iy = (old_idx % (mesh.ny * mesh.nz)) / mesh.nz;
            int iz = old_idx % mesh.nz;

            for (size_t f = 0; f < 6; ++f) {
                mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                int neighborOld = mhs::utils::neighbor_grid_index(
                    ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (neighborOld < 0)
                    continue;

                int n = static_cast<int>(cells.index_map[neighborOld]);
                if (n < 0 || n >= N || !model.is_fluid[n])
                    continue;

                double dp = std::fabs(model.pressure[c] - model.pressure[n]);
                int axis = mhs::utils::AXIS_OF_DIR[f];
                switch (axis) {
                case 0:
                    maxDeltaP_x[c] = std::max(maxDeltaP_x[c], dp);
                    break;
                case 1:
                    maxDeltaP_y[c] = std::max(maxDeltaP_y[c], dp);
                    break;
                default:
                    maxDeltaP_z[c] = std::max(maxDeltaP_z[c], dp);
                    break;
                }
            }
        }

        // argmax
        for (int c = 0; c < N; ++c) {
            if (!model.is_fluid[c]) {
                model.flow_axes[c] = -1;
                continue;
            }
            int axis = 0;
            double maxVal = maxDeltaP_x[c];
            if (maxDeltaP_y[c] > maxVal) {
                maxVal = maxDeltaP_y[c];
                axis = 1;
            }
            if (maxDeltaP_z[c] > maxVal) {
                axis = 2;
            }
            model.flow_axes[c] = static_cast<int8_t>(axis);
        }
    }

} // namespace mhs::sim
