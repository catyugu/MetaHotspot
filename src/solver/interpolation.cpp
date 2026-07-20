#include "runtime/mesh.hpp"
#include "solver/interpolation.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

namespace mhs::utils {
    namespace {

        struct Conductivity {
            double x;
            double y;
            double z;
        };

        using Coordinate = std::array<double, 3>;

        struct RecoveryGeometry {
            std::vector<Coordinate> offsets;
            Coordinate scale {};
            std::array<int, 3> active_axes {};
            int active_axis_count = 0;
            double maximum_weight = 0.0;
        };

        std::array<bool, 3> detect_resistance_axes(
            const std::vector<PointTemperatureSample>& samples, const Coordinate& query)
        {
            std::array<bool, 3> result {};
            for (int axis = 0; axis < 3; ++axis) {
                bool have_negative = false;
                bool have_positive = false;
                bool planar = true;
                mhs::core::TableIndex negative_material = 0;
                mhs::core::TableIndex positive_material = 0;
                for (const auto& sample : samples) {
                    if (!sample.is_cell_center)
                        continue;
                    const Coordinate position {sample.x, sample.y, sample.z};
                    const double offset = position[axis] - query[axis];
                    if (offset < 0.0) {
                        if (!have_negative) {
                            negative_material = sample.material;
                            have_negative = true;
                        }
                        else if (negative_material != sample.material) {
                            planar = false;
                        }
                    }
                    else if (offset > 0.0) {
                        if (!have_positive) {
                            positive_material = sample.material;
                            have_positive = true;
                        }
                        else if (positive_material != sample.material) {
                            planar = false;
                        }
                    }
                }
                result[axis] = planar && have_negative && have_positive && negative_material != positive_material;
            }
            return result;
        }

        RecoveryGeometry analyze_recovery_geometry(
            const std::vector<PointTemperatureSample>& samples, const Coordinate& query)
        {
            const auto resistance_axes = detect_resistance_axes(samples, query);
            RecoveryGeometry result;
            result.offsets.resize(samples.size());
            Coordinate coordinate_min {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
            Coordinate coordinate_max {-std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};

            for (std::size_t row = 0; row < samples.size(); ++row) {
                const auto& sample = samples[row];
                assert(std::isfinite(sample.x) && std::isfinite(sample.y) && std::isfinite(sample.z));
                assert(std::isfinite(sample.temperature));
                assert(sample.weight > 0.0 && std::isfinite(sample.weight));
                assert(sample.kx > 0.0 && std::isfinite(sample.kx));
                assert(sample.ky > 0.0 && std::isfinite(sample.ky));
                assert(sample.kz > 0.0 && std::isfinite(sample.kz));
                const Coordinate position {sample.x, sample.y, sample.z};
                const Coordinate conductivity {sample.kx, sample.ky, sample.kz};
                for (int axis = 0; axis < 3; ++axis) {
                    const double metric = resistance_axes[axis] ? conductivity[axis] : 1.0;
                    const double offset = (position[axis] - query[axis]) / metric;
                    result.offsets[row][axis] = offset;
                    result.scale[axis] = std::max(result.scale[axis], std::abs(offset));
                    coordinate_min[axis] = std::min(coordinate_min[axis], offset);
                    coordinate_max[axis] = std::max(coordinate_max[axis], offset);
                }
                result.maximum_weight = std::max(result.maximum_weight, sample.weight);
            }

            constexpr double dimension_tolerance = 64.0 * std::numeric_limits<double>::epsilon();
            for (int axis = 0; axis < 3; ++axis) {
                const double range = coordinate_max[axis] - coordinate_min[axis];
                if (result.scale[axis] > 0.0 && range > dimension_tolerance * result.scale[axis])
                    result.active_axes[result.active_axis_count++] = axis;
            }
            return result;
        }

        Conductivity evaluate_conductivity(const mhs::core::Model& model, const std::vector<double>& temperature,
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

        double internal_face_temperature(const mhs::core::Model& model, const std::vector<double>& temperature,
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

        double boundary_face_temperature(const mhs::core::Model& model, const std::vector<double>& temperature,
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
        const std::vector<double>& cell_temperature, double time, mhs::core::Index ix, mhs::core::Index iy,
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

        const auto geometry = analyze_recovery_geometry(samples, {point_x, point_y, point_z});
        const Eigen::Index column_count = 1 + geometry.active_axis_count;
        Eigen::MatrixXd design(samples.size(), column_count);
        Eigen::VectorXd values(samples.size());
        for (std::size_t row = 0; row < samples.size(); ++row) {
            const double normalized_weight = samples[row].weight / geometry.maximum_weight;
            const double square_root_weight = std::sqrt(normalized_weight);
            design(static_cast<Eigen::Index>(row), 0) = square_root_weight;
            for (int column = 0; column < geometry.active_axis_count; ++column) {
                const int axis = geometry.active_axes[column];
                design(static_cast<Eigen::Index>(row), column + 1)
                    = square_root_weight * geometry.offsets[row][axis] / geometry.scale[axis];
            }
            values[static_cast<Eigen::Index>(row)] = square_root_weight * samples[row].temperature;
        }

        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> decomposition(design);
        return decomposition.solve(values)[0];
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
