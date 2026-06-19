#include "common/logger.hpp"
#include "common/mesh_utils.hpp"
#include "fluid/fluid_preprocessor.hpp"
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
        const int N = static_cast<int>(model.is_fluid.size());
        const int totalGrid = mesh.nx * mesh.ny * mesh.nz;

        // Single pass over the grid.
        // Uses the Darcy / porous-medium permeability K = D_h² / 32 derived
        // from the user-provided hydraulic diameter.  The cell's hydraulic
        // conductance along each axis is:
        //
        //     hydroC[axis] = K * A_face / (μ * L_cell)
        //
        // where A_face is the cross-sectional face area perpendicular to the
        // axis and L_cell is the cell's length along the axis.  The existing
        // series combination of two half-cell conductances via
        // harmonicConductance() produces the correct face flux coefficient
        // regardless of mesh resolution, because K is a material property,
        // not a function of cell geometry.
        //
        // This replaces the previous Hele-Shaw / rectangular duct formula
        // that assumed 1 cell = 1 full channel cross-section.
        for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
            int c = static_cast<int>(cells.index_map[old_idx]);
            if (c < 0 || c >= N || !model.is_fluid[c])
                continue;

            int ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            double dx_cell = mesh.dx[ix];
            double dy_cell = mesh.dy[iy];
            double dz_cell = mesh.dz[iz];
            double mu = model.dynamic_viscosity[c];
            if (mu < 1e-30)
                mu = 1e-30;

            double dh = model.hydraulic_diameter[c];
            if (dh < 1e-30) {
                // No hydraulic diameter → zero permeability (stagnant)
                model.hydroC_x[c] = 0.0;
                model.hydroC_y[c] = 0.0;
                model.hydroC_z[c] = 0.0;
                continue;
            }

            // Permeability for laminar duct flow: K = D_h² / 32
            double K_perm = (dh * dh) / 32.0;
            if (K_perm < 1e-30)
                K_perm = 1e-30;

            // Face areas and cell lengths along each axis
            double A_xy = dx_cell * dy_cell; // perpendicular to Z
            double A_xz = dx_cell * dz_cell; // perpendicular to Y
            double A_yz = dy_cell * dz_cell; // perpendicular to X

            // hydroC[axis] = K_perm * A_perp / (μ * L_cell)
            model.hydroC_x[c] = K_perm * A_yz / (mu * dx_cell);
            model.hydroC_y[c] = K_perm * A_xz / (mu * dy_cell);
            model.hydroC_z[c] = K_perm * A_xy / (mu * dz_cell);
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
                int neighborOld
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
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
                // Matrix equation: -diagSum * P_c + Σ C_eff * P_n = 0.
                // Multiply both sides by -1 for positive diagonal:
                //   diagSum * P_c - Σ C_eff * P_n = 0
                // This gives a symmetric positive-definite matrix suitable for
                // iterative solvers (BiCGSTAB).
                if (!model.is_pressure_boundary[c]) {
                    triplets.emplace_back(fi, fn, -C_eff); // off-diag = -C_eff
                    diagSum += C_eff;
                }
            }

            // Diagonal (positive after sign-flip for SPD matrix)
            if (model.is_pressure_boundary[c]) {
                triplets.emplace_back(fi, fi, 1.0);
                rhs(fi) = model.boundary_pressure[c];
            }
            else {
                triplets.emplace_back(fi, fi, diagSum); // positive diagonal
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

        // Single pass: compute dominant flow axis per fluid cell.
        // Pre-initialize all to -1 (solid/virtual — no flow axis).
        for (int c = 0; c < N; ++c) {
            model.flow_axes[c] = -1;
        }

        for (int old_idx = 0; old_idx < totalGrid; ++old_idx) {
            int c = static_cast<int>(cells.index_map[old_idx]);
            if (c < 0 || c >= N || !model.is_fluid[c])
                continue;

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
                if (n < 0 || n >= N || !model.is_fluid[n])
                    continue;

                double dp = std::fabs(model.pressure[c] - model.pressure[n]);
                int ax = mhs::utils::AXIS_OF_DIR[f];
                if (dp > maxVal) {
                    maxVal = dp;
                    bestAxis = ax;
                }
            }
            model.flow_axes[c] = static_cast<int8_t>(bestAxis);
        }
    }

} // namespace mhs::sim
