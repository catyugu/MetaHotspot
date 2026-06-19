#include "common/logger.hpp"
#include "common/mesh_utils.hpp"
#include "preprocessor/fluid_preprocessor.hpp"
#include "linear_solver/linear_solver.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>

namespace mhs::sim {

    void FluidPreprocessor::solveFlow(mhs::core::InternalModel& model)
    {
        if (std::none_of(model.is_fluid.begin(), model.is_fluid.end(), [](uint8_t v) { return v != 0; }))
            return;

        initCellHydroProperties(model);
        // applyPressureBoundaryConditions is already done in preprocessor phase
        solvePressure(model);
        precomputeFlowAxes(model);
    }

    // ---------------------------------------------------------------------------
    // Phase 1: Hydraulic conductance (porous-medium permeability)
    // ---------------------------------------------------------------------------

    void FluidPreprocessor::initCellHydroProperties(mhs::core::InternalModel& model)
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;

        if (model.n_fluid == 0)
            return;

        // Build compact → old_idx reverse map once, then iterate N_fluid
        std::vector<int> compact_to_old(static_cast<int>(model.is_fluid.size()), -1);
        int totalGrid = mesh.nx * mesh.ny * mesh.nz;
        for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
            int c = static_cast<int>(cells.index_map[old_idx]);
            if (c >= 0) compact_to_old[c] = old_idx;
        }

        for (int fi = 0; fi < model.n_fluid; ++fi) {
            int c = model.fluid_to_global[fi];
            int old_idx = compact_to_old[c];
            int ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            double dx_cell = mesh.dx[ix];
            double dy_cell = mesh.dy[iy];
            double dz_cell = mesh.dz[iz];
            double mu = model.dynamic_viscosity[fi];
            if (mu < 1e-30)
                mu = 1e-30;

            double dh = model.hydraulic_diameter[fi];
            if (dh < 1e-30) {
                model.hydroC_x[fi] = 0.0;
                model.hydroC_y[fi] = 0.0;
                model.hydroC_z[fi] = 0.0;
                continue;
            }

            double K_perm = (dh * dh) / 32.0;
            if (K_perm < 1e-30) K_perm = 1e-30;

            double A_xy = dx_cell * dy_cell;
            double A_xz = dx_cell * dz_cell;
            double A_yz = dy_cell * dz_cell;

            model.hydroC_x[fi] = K_perm * A_yz / (mu * dx_cell);
            model.hydroC_y[fi] = K_perm * A_xz / (mu * dy_cell);
            model.hydroC_z[fi] = K_perm * A_xy / (mu * dz_cell);
        }
    }

    // ---------------------------------------------------------------------------
    // Phase 3: Pressure Poisson solve
    // ---------------------------------------------------------------------------

    void FluidPreprocessor::solvePressure(mhs::core::InternalModel& model)
    {
        const auto& cells = model.cells;
        const auto& mesh = model.mesh;

        if (model.n_fluid == 0)
            return;

        // Build compact → old_idx reverse map (one-time O(totalGrid), then iterate N_fluid)
        std::vector<int> compact_to_old(static_cast<int>(model.is_fluid.size()), -1);
        int totalGrid = mesh.nx * mesh.ny * mesh.nz;
        for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
            int c = static_cast<int>(cells.index_map[old_idx]);
            if (c >= 0) compact_to_old[c] = old_idx;
        }

        // Build sparse matrix (CSR via triplets) — iterate compact fluid domain
        const int nf = model.n_fluid;
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nf * 7);
        Eigen::VectorXd rhs(nf);
        rhs.setZero();

        for (int fi = 0; fi < nf; ++fi) {
            int c = model.fluid_to_global[fi];
            int old_idx = compact_to_old[c];
            int ix = old_idx / (mesh.ny * mesh.nz);
            int iy = (old_idx % (mesh.ny * mesh.nz)) / mesh.nz;
            int iz = old_idx % mesh.nz;

            double diagSum = 0.0;

            for (size_t f = 0; f < 6; ++f) {
                mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                int neighborOld
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (neighborOld < 0)
                    continue;

                int n = static_cast<int>(cells.index_map[neighborOld]);
                int fn = (n >= 0 && n < static_cast<int>(model.global_to_fluid.size())) ? model.global_to_fluid[n] : -1;
                if (fn < 0)
                    continue; // not fluid

                int axis = mhs::utils::AXIS_OF_DIR[f];

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

                double C_eff = harmonicConductance(hydroC_c, hydroC_n);
                if (C_eff < 1e-30)
                    continue;

                if (!model.is_pressure_boundary[fi]) {
                    triplets.emplace_back(fi, fn, -C_eff);
                    diagSum += C_eff;
                }
            }

            // Diagonal (positive after sign-flip for SPD matrix)
            if (model.is_pressure_boundary[fi]) {
                triplets.emplace_back(fi, fi, 1.0);
                rhs(fi) = model.boundary_pressure[fi];
            }
            else {
                triplets.emplace_back(fi, fi, diagSum);
            }
        }

        // Assemble and solve
        Eigen::SparseMatrix<double> A(nf, nf);
        A.setFromTriplets(triplets.begin(), triplets.end());

        // Use the project's abstract solver factory (BiCGSTAB for SPD Poisson).
        auto solver = mhs::sim::LinearSolver::create(mhs::sim::SolverType::BiCGSTAB);
        auto result = solver->solve(A, rhs);
        if (!result.success) {
            MHS_LOG_WARN("Fluid pressure solve failed (nf={}, nz={})", nf, (int)A.nonZeros());
            return;
        }

        // Write back (pressure is [n_fluid] compact)
        for (int fi = 0; fi < nf; ++fi) {
            model.pressure[fi] = result.solution(fi);
        }
    }

    // ---------------------------------------------------------------------------
    // Phase 4: Flow axes
    // ---------------------------------------------------------------------------

    void FluidPreprocessor::precomputeFlowAxes(mhs::core::InternalModel& model)
    {
        const auto& cells = model.cells;
        const auto& mesh = model.mesh;

        if (model.n_fluid == 0)
            return;

        // Pre-initialize flow axes to -1
        model.flow_axes.assign(model.n_fluid, -1);

        // Build reverse map for compact → old_idx
        std::vector<int> compact_to_old(static_cast<int>(model.is_fluid.size()), -1);
        int totalGrid = mesh.nx * mesh.ny * mesh.nz;
        for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
            int c = static_cast<int>(cells.index_map[old_idx]);
            if (c >= 0) compact_to_old[c] = old_idx;
        }

        // N_fluid iteration: compute dominant flow axis per fluid cell
        for (int fi = 0; fi < model.n_fluid; ++fi) {
            int c = model.fluid_to_global[fi];
            int old_idx = compact_to_old[c];
            int ix = old_idx / (mesh.ny * mesh.nz);
            int iy = (old_idx % (mesh.ny * mesh.nz)) / mesh.nz;
            int iz = old_idx % mesh.nz;

            double maxVal = -1.0;
            int bestAxis = 0;

            for (size_t f = 0; f < 6; ++f) {
                mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                int neighborOld
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (neighborOld < 0)
                    continue;

                int n = static_cast<int>(cells.index_map[neighborOld]);
                int fn = (n >= 0 && n < static_cast<int>(model.global_to_fluid.size())) ? model.global_to_fluid[n] : -1;
                if (fn < 0)
                    continue;

                double dp = std::fabs(model.pressure[fi] - model.pressure[fn]);
                int ax = mhs::utils::AXIS_OF_DIR[f];
                if (dp > maxVal) {
                    maxVal = dp;
                    bestAxis = ax;
                }
            }
            model.flow_axes[fi] = static_cast<int8_t>(bestAxis);
        }
    }

} // namespace mhs::sim
