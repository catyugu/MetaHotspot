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
#include "mhs/model.hpp"

#include <span>
#include <vector>

namespace mhs::utils {

    struct PointTemperatureSample {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double temperature = 0.0;
        double kx = 1.0;
        double ky = 1.0;
        double kz = 1.0;
    };

    struct TemperatureGradient {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    // Green-Gauss gradient on an orthogonal cell. Internal face temperatures
    // use the same two-point thermal-resistance law as the diffusion assembly.
    TemperatureGradient reconstruct_cell_gradient(const mhs::core::Model& model,
        std::span<const double> cell_temperature, double time, mhs::core::Index ix, mhs::core::Index iy,
        mhs::core::Index iz);

    double extrapolate_cell_temperature(double cell_temperature, const TemperatureGradient& gradient, double cell_x,
        double cell_y, double cell_z, double point_x, double point_y, double point_z);

    // Inverse-distance weighted average (IDW p=2) at the query point.
    // Uses anisotropic thermal distance Δx²/kx + Δy²/ky + Δz²/kz.
    // Convex combination of sample temperatures — naturally bounded,
    // no linear algebra needed.
    double recover_point_temperature(
        const std::vector<PointTemperatureSample>& samples, double point_x, double point_y, double point_z);

    // Extrapolate temperature from the cell center to the face center using
    // the BC law (Neumann/Cauchy). Dirichlet and None are caller-handled.
    double sample_extrapolate_face_temperature(mhs::core::FaceDir dir, mhs::core::BcType bc_type,
        mhs::core::TableIndex param_idx, double T_c, double k, const mhs::core::MeshGeometry& mesh, mhs::core::Index ix,
        mhs::core::Index iy, mhs::core::Index iz, const mhs::core::BCParamTable& bc_params, double time);

} // namespace mhs::utils
