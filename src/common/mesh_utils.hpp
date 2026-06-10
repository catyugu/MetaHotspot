#pragma once

#include "data/types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mhs::utils {

    /// Neighbor grid-coordinate along each axis — branchless via DIR_DX/DY/DZ.
    inline int neighbor_ix(mhs::core::FaceDir dir, int ix) { return ix + mhs::core::DIR_DX[static_cast<size_t>(dir)]; }
    inline int neighbor_iy(mhs::core::FaceDir dir, int iy) { return iy + mhs::core::DIR_DY[static_cast<size_t>(dir)]; }
    inline int neighbor_iz(mhs::core::FaceDir dir, int iz) { return iz + mhs::core::DIR_DZ[static_cast<size_t>(dir)]; }

    /// Neighbor flat grid index with boundary check; returns -1 if out of bounds or invalid.
    inline int neighbor_grid_index(
        int ix, int iy, int iz, mhs::core::FaceDir dir, int nx, int ny, int nz, const std::vector<uint8_t>& valid_mask)
    {
        int nix = ix + mhs::core::DIR_DX[static_cast<size_t>(dir)];
        int niy = iy + mhs::core::DIR_DY[static_cast<size_t>(dir)];
        int niz = iz + mhs::core::DIR_DZ[static_cast<size_t>(dir)];
        if (nix < 0 || nix >= nx || niy < 0 || niy >= ny || niz < 0 || niz >= nz)
            return -1;
        int idx = nix * ny * nz + niy * nz + niz;
        return valid_mask[idx] ? idx : -1;
    }

    /// Thermal conductivity along the face direction — branchless via AXIS_OF_DIR.
    inline double k_along(mhs::core::FaceDir dir, double kx, double ky, double kz)
    {
        const double k[3] = {kx, ky, kz};
        return k[mhs::core::AXIS_OF_DIR[static_cast<size_t>(dir)]];
    }

    /// Half cell length along the face direction — branchless via AXIS_OF_DIR.
    inline double half_length_along(mhs::core::FaceDir dir, double dx, double dy, double dz)
    {
        const double d[3] = {dx, dy, dz};
        return d[mhs::core::AXIS_OF_DIR[static_cast<size_t>(dir)]] / 2.0;
    }

    /// Cross-sectional face area perpendicular to the face direction — branchless.
    inline double face_area(mhs::core::FaceDir dir, double dx, double dy, double dz)
    {
        const double d[3] = {dx, dy, dz};
        const auto a = mhs::core::AXIS_OF_DIR[static_cast<size_t>(dir)];
        return d[(a + 1) % 3] * d[(a + 2) % 3];
    }

} // namespace mhs::utils
