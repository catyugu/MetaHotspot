#pragma once

/// Compile-time lookup tables and inline helpers for FaceDir (XM=0, XP=1, YM=2, YP=3, ZM=4, ZP=5).
/// Eliminates switch-case branches in assembler inner loops and preprocessor face logic.

#include "common/types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mhs {

    // ── Axis mapping: XM/XP → 0 (X), YM/YP → 1 (Y), ZM/ZP → 2 (Z) ──
    constexpr size_t AXIS_OF_DIR[6] = {0, 0, 1, 1, 2, 2};

    // ── Neighbor offsets per direction ──
    constexpr int DIR_DX[6] = {-1, 1, 0, 0, 0, 0};
    constexpr int DIR_DY[6] = {0, 0, -1, 1, 0, 0};
    constexpr int DIR_DZ[6] = {0, 0, 0, 0, -1, 1};

    /// Neighbor grid-coordinate along each axis — branchless via DIR_DX/DY/DZ.
    inline int neighbor_ix(FaceDir dir, int ix) { return ix + DIR_DX[static_cast<size_t>(dir)]; }
    inline int neighbor_iy(FaceDir dir, int iy) { return iy + DIR_DY[static_cast<size_t>(dir)]; }
    inline int neighbor_iz(FaceDir dir, int iz) { return iz + DIR_DZ[static_cast<size_t>(dir)]; }

    /// Neighbor flat grid index with boundary check; returns -1 if out of bounds or invalid.
    inline int neighbor_grid_index(
        int ix, int iy, int iz, FaceDir dir, int nx, int ny, int nz, const std::vector<uint8_t>& valid_mask)
    {
        int nix = ix + DIR_DX[static_cast<size_t>(dir)];
        int niy = iy + DIR_DY[static_cast<size_t>(dir)];
        int niz = iz + DIR_DZ[static_cast<size_t>(dir)];
        if (nix < 0 || nix >= nx || niy < 0 || niy >= ny || niz < 0 || niz >= nz)
            return -1;
        int idx = nix * ny * nz + niy * nz + niz;
        return valid_mask[idx] ? idx : -1;
    }

    /// Thermal conductivity along the face direction — branchless via AXIS_OF_DIR.
    inline double k_along(FaceDir dir, double kx, double ky, double kz)
    {
        const double k[3] = {kx, ky, kz};
        return k[AXIS_OF_DIR[static_cast<size_t>(dir)]];
    }

    /// Half cell length along the face direction — branchless via AXIS_OF_DIR.
    inline double half_length_along(FaceDir dir, double dx, double dy, double dz)
    {
        const double d[3] = {dx, dy, dz};
        return d[AXIS_OF_DIR[static_cast<size_t>(dir)]] / 2.0;
    }

    /// Full cell length along the face direction (convenience: half_length_along × 2).
    inline double length_along(FaceDir dir, double dx, double dy, double dz)
    {
        const double d[3] = {dx, dy, dz};
        return d[AXIS_OF_DIR[static_cast<size_t>(dir)]];
    }

    /// Cross-sectional face area perpendicular to the face direction — branchless.
    inline double face_area(FaceDir dir, double dx, double dy, double dz)
    {
        const double d[3] = {dx, dy, dz};
        const auto a = AXIS_OF_DIR[static_cast<size_t>(dir)];
        return d[(a + 1) % 3] * d[(a + 2) % 3];
    }

} // namespace mhs