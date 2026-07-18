#pragma once

#include "compiler/constants.hpp"
#include "compiler/engine_types.hpp"
#include "compiler/runtime_model.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mhs::utils {

    /// Harmonic mean of two conductances.
    inline double harmonicAverage(double a, double b)
    {
        if (a < mhs::core::zero_guard || b < mhs::core::zero_guard)
            return 0.0;
        return (2.0 * a * b) / (a + b);
    }

    // ── Per-direction coordinate tables ─────────────────────────────────
    inline constexpr int AXIS_OF_DIR[mhs::core::FACE_COUNT] = {0, 0, 1, 1, 2, 2};
    inline constexpr int DIR_DX[mhs::core::FACE_COUNT] = {-1, 1, 0, 0, 0, 0};
    inline constexpr int DIR_DY[mhs::core::FACE_COUNT] = {0, 0, -1, 1, 0, 0};
    inline constexpr int DIR_DZ[mhs::core::FACE_COUNT] = {0, 0, 0, 0, -1, 1};
    inline constexpr int DIR_SIGN[mhs::core::FACE_COUNT] = {-1, +1, -1, +1, -1, +1};
    inline constexpr int TANGENT_A_OF_DIR[mhs::core::FACE_COUNT] = {1, 1, 0, 0, 0, 0};
    inline constexpr int TANGENT_B_OF_DIR[mhs::core::FACE_COUNT] = {2, 2, 2, 2, 1, 1};

    /// Decode flat grid index to (ix, iy, iz) coordinates.
    inline void decode_index(mhs::Index old_idx, mhs::Index ny, mhs::Index nz,
        mhs::Index& ix, mhs::Index& iy, mhs::Index& iz)
    {
        ix = old_idx / (ny * nz);
        iy = (old_idx % (ny * nz)) / nz;
        iz = old_idx % nz;
    }

    // ── Neighbor coordinate lookups ─────────────────────────────────────
    inline mhs::Index neighbor_ix(mhs::core::FaceDir dir, mhs::Index ix)
    {
        assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
        return static_cast<mhs::Index>(static_cast<int64_t>(ix) + DIR_DX[static_cast<size_t>(dir)]);
    }
    inline mhs::Index neighbor_iy(mhs::core::FaceDir dir, mhs::Index iy)
    {
        assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
        return static_cast<mhs::Index>(static_cast<int64_t>(iy) + DIR_DY[static_cast<size_t>(dir)]);
    }
    inline mhs::Index neighbor_iz(mhs::core::FaceDir dir, mhs::Index iz)
    {
        assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
        return static_cast<mhs::Index>(static_cast<int64_t>(iz) + DIR_DZ[static_cast<size_t>(dir)]);
    }

    inline mhs::Index neighbor_grid_index(
        mhs::Index ix, mhs::Index iy, mhs::Index iz, mhs::core::FaceDir dir,
        mhs::Index nx, mhs::Index ny, mhs::Index nz,
        const std::vector<mhs::Index>& grid_to_cell)
    {
        assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
        mhs::Index nix = neighbor_ix(dir, ix);
        mhs::Index niy = neighbor_iy(dir, iy);
        mhs::Index niz = neighbor_iz(dir, iz);
        if (nix >= nx || niy >= ny || niz >= nz)
            return mhs::invalidIndex;
        mhs::Index idx = nix * ny * nz + niy * nz + niz;
        return grid_to_cell[idx] != mhs::invalidIndex ? idx : mhs::invalidIndex;
    }

    // ── Face-axis-relative geometric lookups ───────────────────────────
    inline double k_along(mhs::core::FaceDir dir, double kx, double ky, double kz)
    {
        assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
        const double k[3] = {kx, ky, kz};
        return k[AXIS_OF_DIR[static_cast<size_t>(dir)]];
    }

    inline double half_length_along(mhs::core::FaceDir dir, double dx, double dy, double dz)
    {
        assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
        const double d[3] = {dx, dy, dz};
        return d[AXIS_OF_DIR[static_cast<size_t>(dir)]] / 2.0;
    }

    inline double face_area(mhs::core::FaceDir dir, double dx, double dy, double dz)
    {
        assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
        const double d[3] = {dx, dy, dz};
        const auto a = AXIS_OF_DIR[static_cast<size_t>(dir)];
        return d[(a + 1) % 3] * d[(a + 2) % 3];
    }

    inline double face_coord_value(mhs::core::FaceDir dir,
        mhs::Index ix, mhs::Index iy, mhs::Index iz,
        const mhs::core::MeshGeometry& mesh)
    {
        assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
        const int axis = AXIS_OF_DIR[static_cast<size_t>(dir)];
        const int sign = DIR_SIGN[static_cast<size_t>(dir)];
        const double centers[3] = {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz]};
        const double sizes[3] = {mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]};
        return centers[axis] + sign * sizes[axis] * 0.5;
    }

    inline void face_center_3d(mhs::core::FaceDir dir,
        mhs::Index ix, mhs::Index iy, mhs::Index iz,
        const mhs::core::MeshGeometry& mesh,
        double& fx, double& fy, double& fz)
    {
        fx = mesh.cx[ix];
        fy = mesh.cy[iy];
        fz = mesh.cz[iz];
        double half = half_length_along(dir, mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]);
        switch (dir) {
        case mhs::core::FaceDir::XM: fx -= half; break;
        case mhs::core::FaceDir::XP: fx += half; break;
        case mhs::core::FaceDir::YM: fy -= half; break;
        case mhs::core::FaceDir::YP: fy += half; break;
        case mhs::core::FaceDir::ZM: fz -= half; break;
        case mhs::core::FaceDir::ZP: fz += half; break;
        }
    }

    inline bool is_grid_boundary_face(
        mhs::core::FaceDir dir,
        mhs::Index ix, mhs::Index iy, mhs::Index iz,
        const mhs::core::MeshGeometry& mesh)
    {
        assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
        const int axis = AXIS_OF_DIR[static_cast<size_t>(dir)];
        const int sign = DIR_SIGN[static_cast<size_t>(dir)];
        const mhs::Index sizes[3] = {mesh.nx, mesh.ny, mesh.nz};
        const mhs::Index idx[3] = {ix, iy, iz};
        const mhs::Index i = idx[axis];
        const mhs::Index n = sizes[axis];
        return (sign < 0 && i == 0) || (sign > 0 && i == n - 1);
    }

} // namespace mhs::utils
