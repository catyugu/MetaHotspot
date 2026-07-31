#include "macromodel/modal_port.hpp"

#include "common/constants.hpp"
#include "common/mesh.hpp"
#include "solver/solve.hpp"

#include <Eigen/Sparse>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mhs::macro {

    namespace {

        void append_block(std::vector<Eigen::Triplet<double>>& entries, const Eigen::SparseMatrix<double>& block,
            Eigen::Index row_offset, Eigen::Index column_offset)
        {
            for (Eigen::Index outer = 0; outer < block.outerSize(); ++outer) {
                for (Eigen::SparseMatrix<double>::InnerIterator entry(block, outer); entry; ++entry) {
                    entries.emplace_back(entry.row() + row_offset, entry.col() + column_offset, entry.value());
                }
            }
        }

        void validate_operators(const mhs::sim::Operators& operators, Eigen::Index size)
        {
            if (operators.K.rows() != size || operators.K.cols() != size || operators.C.rows() != size
                || operators.C.cols() != size || operators.f.size() != size) {
                throw std::invalid_argument("assemble: macro K/C/f dimensions do not match");
            }
        }

        void validate(const mhs::core::Model& model, const PortModel& port, const PortCoupling& coupling,
            std::span<const double> state)
        {
            const auto model_count = model.cells.cell_to_grid.size();
            const auto macro_state_count = static_cast<std::size_t>(port.operators.f.size());
            const auto physical_port_count = port.physical_port_count;

            if (macro_state_count == 0) {
                throw std::invalid_argument("assemble: macro has zero states");
            }
            if (physical_port_count == 0) {
                throw std::invalid_argument("assemble: physical_port_count is zero");
            }

            const bool has_basis = (port.basis.rows() > 0 && port.basis.cols() > 0);
            if (has_basis) {
                if (static_cast<std::size_t>(port.basis.rows()) != physical_port_count) {
                    throw std::invalid_argument("assemble: basis rows must equal physical_port_count");
                }
                if (static_cast<std::size_t>(port.basis.cols()) != macro_state_count) {
                    throw std::invalid_argument("assemble: basis cols must equal macro_state_count");
                }
            }
            else {
                if (physical_port_count != macro_state_count) {
                    throw std::invalid_argument(
                        "assemble: unit-basis requires physical_port_count == macro_state_count");
                }
            }

            validate_operators(port.operators, static_cast<Eigen::Index>(macro_state_count));

            if (coupling.model_cells.size() != physical_port_count) {
                throw std::invalid_argument("assemble: coupling.model_cells size must match physical port count");
            }

            if (state.size() != model_count + macro_state_count) {
                throw std::invalid_argument("assemble: state must contain FVM temperatures followed by macro states");
            }

            std::unordered_set<mhs::core::Index> unique_cells;
            for (const auto cell : coupling.model_cells) {
                if (cell >= model_count || !unique_cells.insert(cell).second) {
                    throw std::invalid_argument("assemble: interface cells must be unique valid FVM cells");
                }
            }
        }

        /// FVM-side half-conductance k * A / (dx/2) at the interface face.
        /// The macro side is on the face itself — no series combination needed.
        double interface_conductance(const mhs::core::Model& model, mhs::core::Index cell, mhs::core::FaceDir face,
            std::span<const double> temperature, double time)
        {
            const auto grid = model.cells.cell_to_grid[cell];
            mhs::core::Index ix, iy, iz;
            mhs::utils::decode_index(grid, model.mesh.ny, model.mesh.nz, ix, iy, iz);
            const auto neighbor_grid = mhs::utils::neighbor_grid_index(
                ix, iy, iz, face, model.mesh.nx, model.mesh.ny, model.mesh.nz, model.cells.grid_to_cell);
            if (neighbor_grid != mhs::core::invalidIndex) {
                throw std::invalid_argument("assemble: an interface face has an active FVM neighbor");
            }

            const auto& material = model.material_table[model.cells.material_id[cell]];
            const mhs::core::FieldContext ctx {
                model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz], temperature[cell], time};
            const double k
                = mhs::utils::k_along(face, material.kx.eval(ctx), material.ky.eval(ctx), material.kz.eval(ctx));
            const double area = mhs::utils::face_area(face, model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]);
            const double half_length
                = mhs::utils::half_length_along(face, model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]);
            if (k <= mhs::core::zero_guard || half_length <= mhs::core::zero_guard)
                return 0.0;
            return k * area / half_length;
        }

        /// Identity-projection coefficients: for unit basis, (port, mode) == (port == mode ? 1.0 : 0.0).
        inline double basis_coeff(const PortModel& port, std::size_t physical_port, Eigen::Index mode)
        {
            if (port.basis.rows() > 0 && port.basis.cols() > 0) {
                return port.basis(static_cast<Eigen::Index>(physical_port), mode);
            }
            // Unit basis — identity
            return (static_cast<Eigen::Index>(physical_port) == mode) ? 1.0 : 0.0;
        }

    } // namespace

    mhs::sim::Operators assemble(const mhs::core::Model& model, const PortModel& port, const PortCoupling& coupling,
        std::span<const double> state, double time)
    {
        validate(model, port, coupling, state);

        const auto model_count = model.cells.cell_to_grid.size();
        const auto macro_state_count = static_cast<std::size_t>(port.operators.f.size());
        const auto state_count = model_count + macro_state_count;
        const auto eigen_model_count = static_cast<Eigen::Index>(model_count);
        const auto eigen_macro_count = static_cast<Eigen::Index>(macro_state_count);
        const auto eigen_state_count = static_cast<Eigen::Index>(state_count);

        const auto temperature = state.first(model_count);
        auto model_operators = mhs::sim::assemble_thermal(model, temperature, time);

        std::vector<Eigen::Triplet<double>> stiffness;
        std::vector<Eigen::Triplet<double>> capacity;
        stiffness.reserve(static_cast<std::size_t>(model_operators.K.nonZeros() + port.operators.K.nonZeros()
            + coupling.model_cells.size() * (2 * macro_state_count + macro_state_count * macro_state_count + 1)));
        capacity.reserve(static_cast<std::size_t>(model_operators.C.nonZeros() + port.operators.C.nonZeros()));
        append_block(stiffness, model_operators.K, 0, 0);
        append_block(stiffness, port.operators.K, eigen_model_count, eigen_model_count);
        append_block(capacity, model_operators.C, 0, 0);
        append_block(capacity, port.operators.C, eigen_model_count, eigen_model_count);

        for (std::size_t physical_port = 0; physical_port < coupling.model_cells.size(); ++physical_port) {
            const auto cell = coupling.model_cells[physical_port];
            const double conductance = interface_conductance(model, cell, coupling.model_face, temperature, time);
            const auto cell_row = static_cast<Eigen::Index>(cell);
            stiffness.emplace_back(cell_row, cell_row, conductance);

            for (Eigen::Index mode = 0; mode < eigen_macro_count; ++mode) {
                const double projected = conductance * basis_coeff(port, physical_port, mode);
                const auto mode_row = eigen_model_count + mode;
                stiffness.emplace_back(cell_row, mode_row, -projected);
                stiffness.emplace_back(mode_row, cell_row, -projected);

                for (Eigen::Index other_mode = 0; other_mode < eigen_macro_count; ++other_mode) {
                    const double modal_conductance = projected * basis_coeff(port, physical_port, other_mode);
                    stiffness.emplace_back(mode_row, eigen_model_count + other_mode, modal_conductance);
                }
            }
        }

        mhs::sim::Operators combined;
        combined.K.resize(eigen_state_count, eigen_state_count);
        combined.K.setFromTriplets(stiffness.begin(), stiffness.end());
        combined.C.resize(eigen_state_count, eigen_state_count);
        combined.C.setFromTriplets(capacity.begin(), capacity.end());
        combined.f.resize(eigen_state_count);
        combined.f.head(eigen_model_count) = model_operators.f;
        combined.f.tail(eigen_macro_count) = port.operators.f;
        return combined;
    }

    mhs::core::Solution solve(const mhs::core::Model& model, const PortModel& port, const PortCoupling& coupling,
        std::span<const double> initial_state, const mhs::sim::SolveOptions& opts)
    {
        mhs::sim::Study study {model.study_type, model.transient_duration, model.transient_time_step};
        mhs::sim::SystemAssembler asm_fn
            = [&](std::span<const double> state, double time) { return assemble(model, port, coupling, state, time); };

        auto result = mhs::sim::solve_system(study, asm_fn, initial_state, opts);
        result.fvm_count = model.cells.cell_to_grid.size();
        return result;
    }

} // namespace mhs::macro
