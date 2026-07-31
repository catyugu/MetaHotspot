#include "macromodel/modal_port.hpp"

#include "common/constants.hpp"
#include "common/mesh.hpp"
#include "solver/solve.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace mhs::macro {

    namespace {

        void append_sparse_block(std::vector<Eigen::Triplet<double>>& entries,
            const Eigen::SparseMatrix<double>& block, Eigen::Index row_offset, Eigen::Index column_offset)
        {
            for (Eigen::Index outer = 0; outer < block.outerSize(); ++outer) {
                for (Eigen::SparseMatrix<double>::InnerIterator entry(block, outer); entry; ++entry) {
                    entries.emplace_back(entry.row() + row_offset, entry.col() + column_offset, entry.value());
                }
            }
        }

        void append_dense_block(std::vector<Eigen::Triplet<double>>& entries, const Eigen::MatrixXd& block,
            Eigen::Index row_offset, Eigen::Index column_offset)
        {
            for (Eigen::Index row = 0; row < block.rows(); ++row) {
                for (Eigen::Index column = 0; column < block.cols(); ++column) {
                    const double value = block(row, column);
                    if (std::abs(value) > mhs::core::zero_guard) {
                        entries.emplace_back(row + row_offset, column + column_offset, value);
                    }
                }
            }
        }

        void validate_sparse_finite(const Eigen::SparseMatrix<double>& matrix, const char* name)
        {
            for (Eigen::Index outer = 0; outer < matrix.outerSize(); ++outer) {
                for (Eigen::SparseMatrix<double>::InnerIterator entry(matrix, outer); entry; ++entry) {
                    if (!std::isfinite(entry.value())) {
                        throw std::invalid_argument(std::string("assemble: non-finite value in macro ") + name);
                    }
                }
            }
        }

        void validate_operators(const mhs::sim::Operators& operators, Eigen::Index size)
        {
            if (operators.K.rows() != size || operators.K.cols() != size || operators.C.rows() != size
                || operators.C.cols() != size || operators.f.size() != size) {
                throw std::invalid_argument("assemble: macro K/C/f dimensions do not match");
            }
            validate_sparse_finite(operators.K, "K");
            validate_sparse_finite(operators.C, "C");
            if (!operators.f.allFinite()) {
                throw std::invalid_argument("assemble: non-finite value in macro f");
            }
        }

        bool has_explicit_basis(const PortModel& port)
        {
            const bool has_rows = port.basis.rows() > 0;
            const bool has_columns = port.basis.cols() > 0;
            if (has_rows != has_columns) {
                throw std::invalid_argument("assemble: basis must be either empty or a non-empty matrix");
            }
            return has_rows;
        }

        Eigen::MatrixXd materialize_basis(const PortModel& port, Eigen::Index macro_state_count)
        {
            if (has_explicit_basis(port)) {
                return port.basis;
            }
            return Eigen::MatrixXd::Identity(
                static_cast<Eigen::Index>(port.physical_port_count), macro_state_count);
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

            const bool has_basis = has_explicit_basis(port);
            if (has_basis) {
                if (static_cast<std::size_t>(port.basis.rows()) != physical_port_count) {
                    throw std::invalid_argument("assemble: basis rows must equal physical_port_count");
                }
                if (static_cast<std::size_t>(port.basis.cols()) != macro_state_count) {
                    throw std::invalid_argument("assemble: basis cols must equal macro_state_count");
                }
                if (!port.basis.allFinite()) {
                    throw std::invalid_argument("assemble: basis contains non-finite values");
                }
            }
            else if (physical_port_count != macro_state_count) {
                throw std::invalid_argument("assemble: unit-basis requires physical_port_count == macro_state_count");
            }

            validate_operators(port.operators, static_cast<Eigen::Index>(macro_state_count));

            if (coupling.model_cells.size() != physical_port_count) {
                throw std::invalid_argument("assemble: coupling.model_cells size must match physical port count");
            }

            if (state.size() != model_count + macro_state_count) {
                throw std::invalid_argument("assemble: state must contain FVM temperatures followed by macro states");
            }
            if (!std::all_of(state.begin(), state.end(), [](double value) { return std::isfinite(value); })) {
                throw std::invalid_argument("assemble: state contains non-finite values");
            }

            std::unordered_set<mhs::core::Index> unique_cells;
            for (const auto cell : coupling.model_cells) {
                if (cell >= model_count || !unique_cells.insert(cell).second) {
                    throw std::invalid_argument("assemble: interface cells must be unique valid FVM cells");
                }
            }
        }

        /// FVM-side half-conductance k * A / (dx/2) at the interface face.
        /// The macro side is represented at the interface face, so no series
        /// combination is performed in this assembly layer.
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
        const auto physical_port_count = static_cast<Eigen::Index>(coupling.model_cells.size());

        const auto temperature = state.first(model_count);
        auto model_operators = mhs::sim::assemble_thermal(model, temperature, time);
        const Eigen::MatrixXd basis = materialize_basis(port, eigen_macro_count);

        Eigen::VectorXd conductance(physical_port_count);
        for (Eigen::Index physical_port = 0; physical_port < physical_port_count; ++physical_port) {
            const auto cell = coupling.model_cells[static_cast<std::size_t>(physical_port)];
            conductance[physical_port]
                = interface_conductance(model, cell, coupling.model_face, temperature, time);
        }

        // Assemble B^T G B once.  The old implementation inserted a full
        // r-by-r block for every physical port and relied on duplicate-triplet
        // summation, creating O(n_port * r^2) triplets.  This formulation keeps
        // the same arithmetic but stores only O(r^2 + n_port * r) entries.
        const Eigen::MatrixXd weighted_basis = conductance.asDiagonal() * basis;
        const Eigen::MatrixXd modal_interface = basis.transpose() * weighted_basis;

        std::vector<Eigen::Triplet<double>> stiffness;
        std::vector<Eigen::Triplet<double>> capacity;
        const auto cross_entry_bound = static_cast<std::size_t>(physical_port_count * eigen_macro_count * 2);
        const auto modal_entry_bound = static_cast<std::size_t>(eigen_macro_count * eigen_macro_count);
        stiffness.reserve(static_cast<std::size_t>(model_operators.K.nonZeros() + port.operators.K.nonZeros())
            + coupling.model_cells.size() + cross_entry_bound + modal_entry_bound);
        capacity.reserve(static_cast<std::size_t>(model_operators.C.nonZeros() + port.operators.C.nonZeros()));
        append_sparse_block(stiffness, model_operators.K, 0, 0);
        append_sparse_block(stiffness, port.operators.K, eigen_model_count, eigen_model_count);
        append_sparse_block(capacity, model_operators.C, 0, 0);
        append_sparse_block(capacity, port.operators.C, eigen_model_count, eigen_model_count);

        for (Eigen::Index physical_port = 0; physical_port < physical_port_count; ++physical_port) {
            const auto cell = coupling.model_cells[static_cast<std::size_t>(physical_port)];
            const auto cell_row = static_cast<Eigen::Index>(cell);
            const double g = conductance[physical_port];
            if (std::abs(g) > mhs::core::zero_guard) {
                stiffness.emplace_back(cell_row, cell_row, g);
            }

            for (Eigen::Index mode = 0; mode < eigen_macro_count; ++mode) {
                const double projected = -g * basis(physical_port, mode);
                if (std::abs(projected) <= mhs::core::zero_guard) {
                    continue;
                }
                const auto mode_row = eigen_model_count + mode;
                stiffness.emplace_back(cell_row, mode_row, projected);
                stiffness.emplace_back(mode_row, cell_row, projected);
            }
        }
        append_dense_block(stiffness, modal_interface, eigen_model_count, eigen_model_count);

        mhs::sim::Operators combined;
        combined.K.resize(eigen_state_count, eigen_state_count);
        combined.K.setFromTriplets(stiffness.begin(), stiffness.end());
        combined.K.makeCompressed();
        combined.C.resize(eigen_state_count, eigen_state_count);
        combined.C.setFromTriplets(capacity.begin(), capacity.end());
        combined.C.makeCompressed();
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
