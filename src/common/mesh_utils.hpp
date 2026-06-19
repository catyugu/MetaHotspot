#pragma once

#include "data/internal_model.hpp"
#include "data/types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mhs::utils {

    // ── Per-direction coordinate tables ─────────────────────────────────
    /// Axis index (0=X, 1=Y, 2=Z) for each face direction.
    inline constexpr int AXIS_OF_DIR[mhs::core::FACE_COUNT] = {0, 0, 1, 1, 2, 2};

    /// Neighbor offset along X for each face direction.
    inline constexpr int DIR_DX[mhs::core::FACE_COUNT] = {-1, 1, 0, 0, 0, 0};
    /// Neighbor offset along Y for each face direction.
    inline constexpr int DIR_DY[mhs::core::FACE_COUNT] = {0, 0, -1, 1, 0, 0};
    /// Neighbor offset along Z for each face direction.
    inline constexpr int DIR_DZ[mhs::core::FACE_COUNT] = {0, 0, 0, 0, -1, 1};

    /// Sign of the face: XM/YM/ZM → −1, XP/YP/ZP → +1.
    /// Used to compute face center as `cell_center + sign * half_length_along`.
    inline constexpr int DIR_SIGN[mhs::core::FACE_COUNT] = {-1, +1, -1, +1, -1, +1};

    /// Tangent index "A" (first non-axis direction) for each face.
    /// X-faces (XM/XP) → Y (1), Y-faces → X (0), Z-faces → X (0).
    inline constexpr int TANGENT_A_OF_DIR[mhs::core::FACE_COUNT] = {1, 1, 0, 0, 0, 0};

    /// Tangent index "B" (second non-axis direction) for each face.
    /// X-faces → Z (2), Y-faces → Z (2), Z-faces → Y (1).
    inline constexpr int TANGENT_B_OF_DIR[mhs::core::FACE_COUNT] = {2, 2, 2, 2, 1, 1};

    // ── Neighbor coordinate lookups ─────────────────────────────────────

    /// Neighbor grid-coordinate along each axis — branchless via DIR_DX/DY/DZ.
    inline int neighbor_ix(mhs::core::FaceDir dir, int ix) { return ix + DIR_DX[static_cast<size_t>(dir)]; }
    inline int neighbor_iy(mhs::core::FaceDir dir, int iy) { return iy + DIR_DY[static_cast<size_t>(dir)]; }
    inline int neighbor_iz(mhs::core::FaceDir dir, int iz) { return iz + DIR_DZ[static_cast<size_t>(dir)]; }

    /// Neighbor flat grid index with boundary check; returns -1 if out of bounds or invalid.
    inline int neighbor_grid_index(
        int ix, int iy, int iz, mhs::core::FaceDir dir, int nx, int ny, int nz, const std::vector<uint8_t>& valid_mask)
    {
        int nix = ix + DIR_DX[static_cast<size_t>(dir)];
        int niy = iy + DIR_DY[static_cast<size_t>(dir)];
        int niz = iz + DIR_DZ[static_cast<size_t>(dir)];
        if (nix < 0 || nix >= nx || niy < 0 || niy >= ny || niz < 0 || niz >= nz)
            return -1;
        int idx = nix * ny * nz + niy * nz + niz;
        return valid_mask[idx] ? idx : -1;
    }

    // ── Face-axis-relative geometric lookups ───────────────────────────

    /// Thermal conductivity along the face direction — branchless via AXIS_OF_DIR.
    inline double k_along(mhs::core::FaceDir dir, double kx, double ky, double kz)
    {
        const double k[3] = {kx, ky, kz};
        return k[AXIS_OF_DIR[static_cast<size_t>(dir)]];
    }

    /// Half cell length along the face direction — branchless via AXIS_OF_DIR.
    inline double half_length_along(mhs::core::FaceDir dir, double dx, double dy, double dz)
    {
        const double d[3] = {dx, dy, dz};
        return d[AXIS_OF_DIR[static_cast<size_t>(dir)]] / 2.0;
    }

    /// Cross-sectional face area perpendicular to the face direction — branchless.
    inline double face_area(mhs::core::FaceDir dir, double dx, double dy, double dz)
    {
        const double d[3] = {dx, dy, dz};
        const auto a = AXIS_OF_DIR[static_cast<size_t>(dir)];
        return d[(a + 1) % 3] * d[(a + 2) % 3];
    }

    /// World-space coordinate of the face center along its normal axis.
    /// Returns `cell_center[axis] + sign * sizes[axis] / 2` — i.e., the
    /// midpoint of the cell's `dir` face.
    inline double face_coord_value(mhs::core::FaceDir dir, int ix, int iy, int iz, const mhs::core::MeshGeometry& mesh)
    {
        const int axis = AXIS_OF_DIR[static_cast<size_t>(dir)];
        const int sign = DIR_SIGN[static_cast<size_t>(dir)];
        const double centers[3] = {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz]};
        const double sizes[3] = {mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]};
        return centers[axis] + sign * sizes[axis] * 0.5;
    }

    /// True iff this face lies on the outer hull of the structured grid
    /// (no neighbor slot to step into). Replaces the 6-line `ix==0 || ix==nx-1 || ...`
    /// check scattered across assembler / probe recorder.
    inline bool is_grid_boundary_face(
        mhs::core::FaceDir dir, int ix, int iy, int iz, const mhs::core::MeshGeometry& mesh)
    {
        const int axis = AXIS_OF_DIR[static_cast<size_t>(dir)];
        const int sign = DIR_SIGN[static_cast<size_t>(dir)];
        const int sizes[3] = {mesh.nx, mesh.ny, mesh.nz};
        const int idx[3] = {ix, iy, iz};
        const int i = idx[axis];
        const int n = sizes[axis];
        return (sign < 0 && i == 0) || (sign > 0 && i == n - 1);
    }

} // namespace mhs::utils