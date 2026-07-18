#pragma once

// Shared helpers for "evaluate temperature at an arbitrary 3D point in a
// structured cell-centered grid with cell-level BCs".
//
// Extracted from postprocessor/sample_point.hpp to break the cross-module
// dependency scheduler (mhs::sim) → postprocessor (mhs::post).
//
// These are pure data + math — no module-specific logic.
// Used by:
//   - mhs::post::interpolate_cell_to_node  (whole-grid node sampling)
//   - mhs::sim::ProbeRecorder::sample_one  (per-probe local sampling)

#include "compiler/runtime_model.hpp"

#include <cstdint>
#include <vector>

namespace mhs::utils {

    struct SampleDataPoint {
        double x = 0.0, y = 0.0, z = 0.0;
        double T = 0.0;
        double weight = 0.0;
    };

    // Least-squares fit T(x,y,z) ? T_node + gx·x + gy·y + gz·z at the node
    // (node_x, node_y, node_z), with Tikhonov regularization on the gradient.
    // Returns X(0) = interpolated T at the node. NaN if no points.
    double sample_solve_least_squares(
        const std::vector<SampleDataPoint>& pts, double node_x, double node_y, double node_z);

    // World coordinates of the center of the (ix, iy, iz) cell face `dir`.
    void sample_face_center(mhs::core::FaceDir dir, mhs::Index ix, mhs::Index iy, mhs::Index iz, const mhs::core::MeshGeometry& mesh,
        double& fx, double& fy, double& fz);

    // Extrapolate temperature from the cell center to the face center using
    // the BC law (Neumann/Cauchy). Dirichlet and None are caller-handled.
    double sample_extrapolate_face_temperature(mhs::core::FaceDir dir, mhs::core::BcType bc_type, uint16_t param_idx,
        double T_c, double k, const mhs::core::MeshGeometry& mesh, mhs::Index ix, mhs::Index iy, mhs::Index iz,
        const mhs::core::BCParamTable& bc_params, double time);

} // namespace mhs::utils
