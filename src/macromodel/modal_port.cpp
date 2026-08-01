#include "macromodel/modal_port.hpp"

#include "common/mesh.hpp"
#include "solver/solve.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mhs::macro {

    namespace {

        void append_sparse_block(std::vector<Eigen::Triplet<double>>& entries, const Eigen::SparseMatrix<double>& block,
            Eigen::Index row_offset, Eigen::Index column_offset)
        {
            for (Eigen::Index outer = 0; outer < block.outerSize(); ++outer) {
                for (Eigen::SparseMatrix<double>::InnerIterator entry(block, outer); entry; ++entry) {
                    entries.emplace_back(entry.row() + row_offset, entry.col() + column_offset, entry.value());
                }
            }
        }

        void validate_sparse_finite(const Eigen::SparseMatrix<double>& matrix, const char* name)
        {
            for (Eigen::Index outer = 0; outer < matrix.outerSize(); ++outer) {
                for (Eigen::SparseMatrix<double>::InnerIterator entry(matrix, outer); entry; ++entry) {
                    if (!std::isfinite(entry.value()))
                        throw std::invalid_argument(std::string("non-finite value in DtN ") + name);
                }
            }
        }

        void validate_operators(const mhs::sim::Operators& operators, Eigen::Index size)
        {
            if (operators.K.rows() != size || operators.K.cols() != size || operators.C.rows() != size
                || operators.C.cols() != size || operators.f.size() != size) {
                throw std::invalid_argument("DtN K/C/f dimensions do not match");
            }
            validate_sparse_finite(operators.K, "K");
            validate_sparse_finite(operators.C, "C");
            if (!operators.f.allFinite())
                throw std::invalid_argument("non-finite value in DtN f");
        }

        std::optional<mhs::core::Index> active_neighbor(const mhs::core::Model& model, mhs::core::Index ix,
            mhs::core::Index iy, mhs::core::Index iz, mhs::core::FaceDir face)
        {
            using mhs::core::FaceDir;
            if (face == FaceDir::XM) {
                if (ix == 0)
                    return std::nullopt;
                --ix;
            }
            else if (face == FaceDir::XP) {
                if (ix + 1 >= model.mesh.nx)
                    return std::nullopt;
                ++ix;
            }
            else if (face == FaceDir::YM) {
                if (iy == 0)
                    return std::nullopt;
                --iy;
            }
            else if (face == FaceDir::YP) {
                if (iy + 1 >= model.mesh.ny)
                    return std::nullopt;
                ++iy;
            }
            else if (face == FaceDir::ZM) {
                if (iz == 0)
                    return std::nullopt;
                --iz;
            }
            else {
                if (iz + 1 >= model.mesh.nz)
                    return std::nullopt;
                ++iz;
            }
            const auto grid = (ix * model.mesh.ny + iy) * model.mesh.nz + iz;
            const auto cell = model.cells.grid_to_cell[grid];
            if (cell == mhs::core::invalidIndex)
                return std::nullopt;
            return cell;
        }

        double face_coordinate(const mhs::core::Model& model, mhs::core::Index ix, mhs::core::Index iy,
            mhs::core::Index iz, mhs::core::FaceDir face)
        {
            using mhs::core::FaceDir;
            if (face == FaceDir::XM || face == FaceDir::XP)
                return model.mesh.cx[ix] + (face == FaceDir::XP ? 0.5 : -0.5) * model.mesh.dx[ix];
            if (face == FaceDir::YM || face == FaceDir::YP)
                return model.mesh.cy[iy] + (face == FaceDir::YP ? 0.5 : -0.5) * model.mesh.dy[iy];
            return model.mesh.cz[iz] + (face == FaceDir::ZP ? 0.5 : -0.5) * model.mesh.dz[iz];
        }

        std::pair<double, double> tangential_center(const mhs::core::Model& model, mhs::core::Index ix,
            mhs::core::Index iy, mhs::core::Index iz, mhs::core::FaceDir face)
        {
            using mhs::core::FaceDir;
            if (face == FaceDir::XM || face == FaceDir::XP)
                return {model.mesh.cy[iy], model.mesh.cz[iz]};
            if (face == FaceDir::YM || face == FaceDir::YP)
                return {model.mesh.cx[ix], model.mesh.cz[iz]};
            return {model.mesh.cx[ix], model.mesh.cy[iy]};
        }

        bool inside(double value, double lower, double upper, double tolerance)
        { return value >= std::min(lower, upper) - tolerance && value <= std::max(lower, upper) + tolerance; }

        double interface_conductance(
            const mhs::core::Model& model, const PortFace& port_face, double temperature, double time)
        {
            const auto grid = model.cells.cell_to_grid[port_face.cell];
            mhs::core::Index ix, iy, iz;
            mhs::utils::decode_index(grid, model.mesh.ny, model.mesh.nz, ix, iy, iz);
            if (active_neighbor(model, ix, iy, iz, port_face.face).has_value())
                throw std::invalid_argument("port patch selected a face with an active FVM neighbor");
            const auto& material = model.material_table[model.cells.material_id[port_face.cell]];
            const mhs::core::FieldContext context {
                model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz], temperature, time};
            const double conductivity = mhs::utils::k_along(
                port_face.face, material.kx.eval(context), material.ky.eval(context), material.kz.eval(context));
            const double area
                = mhs::utils::face_area(port_face.face, model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]);
            const double half_length = mhs::utils::half_length_along(
                port_face.face, model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]);
            if (!std::isfinite(conductivity) || conductivity < 0.0 || !std::isfinite(area) || area <= 0.0
                || !std::isfinite(half_length) || half_length <= 0.0) {
                throw std::invalid_argument("invalid material or geometry while evaluating port conductance");
            }
            return conductivity * area / half_length;
        }

        void validate_port_map(const mhs::core::Model& model, const PortMap& ports)
        {
            if (ports.port_count == 0 || ports.faces.empty())
                throw std::invalid_argument("port map must contain at least one patch and one exposed face");
            std::vector<bool> seen(ports.port_count, false);
            for (const auto& face : ports.faces) {
                if (face.cell >= model.cells.cell_to_grid.size() || face.port >= ports.port_count)
                    throw std::invalid_argument("port map contains an out-of-range entry");
                seen[face.port] = true;
            }
            if (std::find(seen.begin(), seen.end(), false) != seen.end())
                throw std::invalid_argument("every physical port patch must select at least one face");
        }

    } // namespace

    PortMap compile_port_map(const mhs::core::Model& model, std::span<const PortPatch> patches)
    {
        if (patches.empty())
            throw std::invalid_argument("compile_port_map: patches must not be empty");
        PortMap result;
        result.port_count = patches.size();
        std::unordered_set<std::size_t> claimed;
        const double tolerance = 1.0e-10;

        for (std::size_t port = 0; port < patches.size(); ++port) {
            const auto& patch = patches[port];
            std::size_t matches = 0;
            for (mhs::core::Index cell = 0; cell < model.cells.cell_to_grid.size(); ++cell) {
                const auto grid = model.cells.cell_to_grid[cell];
                mhs::core::Index ix, iy, iz;
                mhs::utils::decode_index(grid, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                if (active_neighbor(model, ix, iy, iz, patch.face).has_value())
                    continue;
                if (std::abs(face_coordinate(model, ix, iy, iz, patch.face) - patch.coordinate) > tolerance)
                    continue;
                const auto [a, b] = tangential_center(model, ix, iy, iz, patch.face);
                if (!inside(a, patch.a_min, patch.a_max, tolerance) || !inside(b, patch.b_min, patch.b_max, tolerance))
                    continue;

                const auto key = cell * mhs::core::FACE_COUNT + static_cast<std::size_t>(patch.face);
                if (!claimed.insert(key).second)
                    throw std::invalid_argument("port patches overlap on the same exposed face");
                result.faces.push_back({cell, port, patch.face});
                ++matches;
            }
            if (matches == 0)
                throw std::invalid_argument("port patch does not select any exposed FVM face");
        }
        return result;
    }

    mhs::sim::Operators assemble_dtn(
        const mhs::core::Model& model, const PortMap& ports, std::span<const double> cell_state, double time)
    {
        validate_port_map(model, ports);
        const auto cell_count = model.cells.cell_to_grid.size();
        if (cell_state.size() != cell_count)
            throw std::invalid_argument("assemble_dtn: state size must equal model cell count");

        const auto base = mhs::sim::assemble_thermal(model, cell_state, time);
        const Eigen::Index port_count = static_cast<Eigen::Index>(ports.port_count);
        const Eigen::Index fvm_count = static_cast<Eigen::Index>(cell_count);
        const Eigen::Index total = port_count + fvm_count;
        std::vector<Eigen::Triplet<double>> k_entries;
        std::vector<Eigen::Triplet<double>> c_entries;
        k_entries.reserve(static_cast<std::size_t>(base.K.nonZeros()) + 4 * ports.faces.size());
        c_entries.reserve(static_cast<std::size_t>(base.C.nonZeros()));
        append_sparse_block(k_entries, base.K, port_count, port_count);
        append_sparse_block(c_entries, base.C, port_count, port_count);

        for (const auto& face : ports.faces) {
            const auto port = static_cast<Eigen::Index>(face.port);
            const auto cell = port_count + static_cast<Eigen::Index>(face.cell);
            const double g = interface_conductance(model, face, cell_state[face.cell], time);
            k_entries.emplace_back(port, port, g);
            k_entries.emplace_back(port, cell, -g);
            k_entries.emplace_back(cell, port, -g);
            k_entries.emplace_back(cell, cell, g);
        }

        mhs::sim::Operators result;
        result.K.resize(total, total);
        result.C.resize(total, total);
        result.K.setFromTriplets(k_entries.begin(), k_entries.end());
        result.C.setFromTriplets(c_entries.begin(), c_entries.end());
        result.f = Eigen::VectorXd::Zero(total);
        result.f.tail(fvm_count) = base.f;
        return result;
    }

    mhs::sim::Operators assemble_coupled(const mhs::core::Model& model, const DtNModel& dtn, const PortMap& ports,
        std::span<const double> state, double time)
    {
        validate_port_map(model, ports);
        const Eigen::Index macro_count = dtn.operators.f.size();
        validate_operators(dtn.operators, macro_count);
        if (macro_count < static_cast<Eigen::Index>(ports.port_count))
            throw std::invalid_argument("DtN states must begin with one state per physical port");
        const Eigen::Index fvm_count = static_cast<Eigen::Index>(model.cells.cell_to_grid.size());
        if (state.size() != static_cast<std::size_t>(fvm_count + macro_count))
            throw std::invalid_argument("coupled state size must equal FVM cells + DtN states");

        const auto temperatures = state.first(static_cast<std::size_t>(fvm_count));
        const auto base = mhs::sim::assemble_thermal(model, temperatures, time);
        const Eigen::Index total = fvm_count + macro_count;
        std::vector<Eigen::Triplet<double>> k_entries;
        std::vector<Eigen::Triplet<double>> c_entries;
        k_entries.reserve(
            static_cast<std::size_t>(base.K.nonZeros() + dtn.operators.K.nonZeros()) + 4 * ports.faces.size());
        c_entries.reserve(static_cast<std::size_t>(base.C.nonZeros() + dtn.operators.C.nonZeros()));
        append_sparse_block(k_entries, base.K, 0, 0);
        append_sparse_block(k_entries, dtn.operators.K, fvm_count, fvm_count);
        append_sparse_block(c_entries, base.C, 0, 0);
        append_sparse_block(c_entries, dtn.operators.C, fvm_count, fvm_count);

        for (const auto& face : ports.faces) {
            const Eigen::Index cell = static_cast<Eigen::Index>(face.cell);
            const Eigen::Index port_state = fvm_count + static_cast<Eigen::Index>(face.port);
            const double g = interface_conductance(model, face, temperatures[face.cell], time);
            k_entries.emplace_back(cell, cell, g);
            k_entries.emplace_back(cell, port_state, -g);
            k_entries.emplace_back(port_state, cell, -g);
            k_entries.emplace_back(port_state, port_state, g);
        }

        mhs::sim::Operators result;
        result.K.resize(total, total);
        result.C.resize(total, total);
        result.K.setFromTriplets(k_entries.begin(), k_entries.end());
        result.C.setFromTriplets(c_entries.begin(), c_entries.end());
        result.f.resize(total);
        result.f.head(fvm_count) = base.f;
        result.f.tail(macro_count) = dtn.operators.f;
        return result;
    }

    mhs::core::Solution solve(const mhs::core::Model& model, const DtNModel& dtn, const PortMap& ports,
        std::span<const double> initial_state, const mhs::sim::SolveOptions& opts)
    {
        mhs::sim::Study study {model.study_type, model.transient_duration, model.transient_time_step};
        mhs::sim::SystemAssembler assembler = [&model, &dtn, &ports](std::span<const double> state, double time) {
            return assemble_coupled(model, dtn, ports, state, time);
        };
        return mhs::sim::solve_system(study, assembler, initial_state, opts);
    }

} // namespace mhs::macro
