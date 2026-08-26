#include "solver/interpolation.hpp"
#include "core/mesh.hpp"

#include <cassert>
#include <cstddef>
#include <limits>

namespace mhs::utils {
    namespace {

        // Inverse distance weighting (IDW p=2) with anisotropic thermal
        // distance. Each sample weight w = 1/r² where r² = Δx²/kx + Δy²/ky
        // + Δz²/kz. The result is a convex combination of sample temperatures
        // (naturally bounded between min/max of input, no clamping needed).
        double inverse_distance_weighted_average(
            const std::vector<PointTemperatureSample>& samples, double px, double py, double pz)
        {
            double numerator = 0.0;
            double denominator = 0.0;
            for (const auto& s : samples) {
                const double dx = px - s.x;
                const double dy = py - s.y;
                const double dz = pz - s.z;
                const double r2 = dx * dx / s.kx + dy * dy / s.ky + dz * dz / s.kz;
                const double w = 1.0 / (r2 + std::numeric_limits<double>::epsilon());
                numerator += w * s.temperature;
                denominator += w;
            }
            return numerator / denominator;
        }

        struct Conductivity {
            double x;
            double y;
            double z;
        };

        Conductivity evaluate_conductivity(const mhs::core::Model& model, std::span<const double> temperature,
            mhs::core::Index compact, mhs::core::Index ix, mhs::core::Index iy, mhs::core::Index iz, double time)
        {
            const auto& material = model.material_table[model.cells.material_id[compact]];
            const mhs::core::FieldContext context {
                model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz], temperature[compact], time};
            const Conductivity conductivity {
                material.kx.eval(context), material.ky.eval(context), material.kz.eval(context)};
            assert(conductivity.x > 0.0 && std::isfinite(conductivity.x));
            assert(conductivity.y > 0.0 && std::isfinite(conductivity.y));
            assert(conductivity.z > 0.0 && std::isfinite(conductivity.z));
            return conductivity;
        }

        double internal_face_temperature(const mhs::core::Model& model, std::span<const double> temperature,
            double time, mhs::core::Index compact, mhs::core::Index neighbor, mhs::core::Index ix, mhs::core::Index iy,
            mhs::core::Index iz, mhs::core::Index nix, mhs::core::Index niy, mhs::core::Index niz,
            mhs::core::FaceDir dir, const Conductivity& conductivity)
        {
            const auto& mesh = model.mesh;
            const auto neighbor_conductivity = evaluate_conductivity(model, temperature, neighbor, nix, niy, niz, time);
            const double cell_k = k_along(dir, conductivity.x, conductivity.y, conductivity.z);
            const double neighbor_k
                = k_along(dir, neighbor_conductivity.x, neighbor_conductivity.y, neighbor_conductivity.z);
            const double cell_distance = half_length_along(dir, mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]);
            const double neighbor_distance = half_length_along(dir, mesh.dx[nix], mesh.dy[niy], mesh.dz[niz]);
            const double cell_resistance = cell_distance / cell_k;
            const double neighbor_resistance = neighbor_distance / neighbor_k;
            const double resistance_sum = cell_resistance + neighbor_resistance;
            assert(resistance_sum > 0.0 && std::isfinite(resistance_sum));

            // The face temperature is the point at which the two half-cell
            // thermal resistances carry the same normal heat flux.
            return (neighbor_resistance * temperature[compact] + cell_resistance * temperature[neighbor])
                / resistance_sum;
        }

        double boundary_face_temperature(const mhs::core::Model& model, std::span<const double> temperature,
            double time, mhs::core::Index compact, mhs::core::Index ix, mhs::core::Index iy, mhs::core::Index iz,
            mhs::core::FaceDir dir, const Conductivity& conductivity)
        {
            const auto& boundary = model.face_bcs[compact * mhs::core::FACE_COUNT + static_cast<std::size_t>(dir)];
            const double cell_temperature = temperature[compact];
            const double face_k = k_along(dir, conductivity.x, conductivity.y, conductivity.z);
            if (boundary.type == mhs::core::BcType::FirstType) {
                double fx, fy, fz;
                face_center_3d(dir, ix, iy, iz, model.mesh, fx, fy, fz);
                return model.bc_params.dirichlet_T[boundary.param_idx].eval({fx, fy, fz, cell_temperature, time});
            }
            if (boundary.type == mhs::core::BcType::SecondType || boundary.type == mhs::core::BcType::ThirdType) {
                return sample_extrapolate_face_temperature(dir, boundary.type, boundary.param_idx, cell_temperature,
                    face_k, model.mesh, ix, iy, iz, model.bc_params, time);
            }
            // BcType::None denotes an adiabatic active-domain surface.
            return cell_temperature;
        }

    } // namespace

    TemperatureGradient reconstruct_cell_gradient(const mhs::core::Model& model,
        std::span<const double> cell_temperature, double time, mhs::core::Index ix, mhs::core::Index iy,
        mhs::core::Index iz)
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;
        const mhs::core::Index grid = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
        const mhs::core::Index compact = cells.grid_to_cell[grid];
        assert(compact != mhs::core::invalidIndex);
        assert(compact < cell_temperature.size());

        const auto conductivity = evaluate_conductivity(model, cell_temperature, compact, ix, iy, iz, time);
        std::array<double, mhs::core::FACE_COUNT> face_temperature {};
        for (std::size_t face = 0; face < mhs::core::FACE_COUNT; ++face) {
            const auto dir = mhs::core::FACE_DIRS[face];
            const auto neighbor_grid
                = neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.grid_to_cell);
            if (neighbor_grid == mhs::core::invalidIndex) {
                face_temperature[face]
                    = boundary_face_temperature(model, cell_temperature, time, compact, ix, iy, iz, dir, conductivity);
                continue;
            }

            const mhs::core::Index nix = neighbor_ix(dir, ix);
            const mhs::core::Index niy = neighbor_iy(dir, iy);
            const mhs::core::Index niz = neighbor_iz(dir, iz);
            face_temperature[face] = internal_face_temperature(model, cell_temperature, time, compact,
                cells.grid_to_cell[neighbor_grid], ix, iy, iz, nix, niy, niz, dir, conductivity);
        }

        return {(face_temperature[static_cast<std::size_t>(mhs::core::FaceDir::XP)]
                    - face_temperature[static_cast<std::size_t>(mhs::core::FaceDir::XM)])
                / mesh.dx[ix],
            (face_temperature[static_cast<std::size_t>(mhs::core::FaceDir::YP)]
                - face_temperature[static_cast<std::size_t>(mhs::core::FaceDir::YM)])
                / mesh.dy[iy],
            (face_temperature[static_cast<std::size_t>(mhs::core::FaceDir::ZP)]
                - face_temperature[static_cast<std::size_t>(mhs::core::FaceDir::ZM)])
                / mesh.dz[iz]};
    }

    double extrapolate_cell_temperature(double cell_temperature, const TemperatureGradient& gradient, double cell_x,
        double cell_y, double cell_z, double point_x, double point_y, double point_z)
    {
        return cell_temperature + gradient.x * (point_x - cell_x) + gradient.y * (point_y - cell_y)
            + gradient.z * (point_z - cell_z);
    }

    double recover_point_temperature(
        const std::vector<PointTemperatureSample>& samples, double point_x, double point_y, double point_z)
    {
        assert(!samples.empty());
        assert(std::isfinite(point_x) && std::isfinite(point_y) && std::isfinite(point_z));
        return inverse_distance_weighted_average(samples, point_x, point_y, point_z);
    }

    double sample_extrapolate_face_temperature(mhs::core::FaceDir dir, mhs::core::BcType bc_type,
        mhs::core::TableIndex param_idx, double T_c, double k, const mhs::core::MeshGeometry& mesh, mhs::core::Index ix,
        mhs::core::Index iy, mhs::core::Index iz, const mhs::core::BCParamTable& bc_params, double time)
    {
        assert(k > 0.0 && std::isfinite(k));
        double fx, fy, fz;
        face_center_3d(dir, ix, iy, iz, mesh, fx, fy, fz);
        mhs::core::FieldContext ctx {fx, fy, fz, T_c, time};
        const double half_dist = half_length_along(dir, mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]);
        assert(half_dist > 0.0 && std::isfinite(half_dist));

        if (bc_type == mhs::core::BcType::SecondType) {
            const double q = bc_params.neumann_q[param_idx].eval(ctx);
            return T_c + (q * half_dist) / k;
        }
        if (bc_type == mhs::core::BcType::ThirdType) {
            const double h = bc_params.cauchy_h[param_idx].eval(ctx);
            const double T_inf = bc_params.cauchy_T_inf[param_idx].eval(ctx);
            const double conductance = k / half_dist;
            return (h * T_inf + conductance * T_c) / (h + conductance);
        }
        return T_c;
    }

} // namespace mhs::utils
