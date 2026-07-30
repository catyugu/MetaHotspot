#include "macromodel/modal_port.hpp"

#include "runtime/constants.hpp"
#include "runtime/mesh.hpp"

#include <Eigen/Sparse>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mhs::sim {

    namespace {

        void append_block(std::vector<Eigen::Triplet<double>>& entries,
            const Eigen::SparseMatrix<double>& block, Eigen::Index row_offset, Eigen::Index column_offset)
        {
            for (Eigen::Index outer = 0; outer < block.outerSize(); ++outer) {
                for (Eigen::SparseMatrix<double>::InnerIterator entry(block, outer); entry; ++entry) {
                    entries.emplace_back(
                        entry.row() + row_offset, entry.col() + column_offset, entry.value());
                }
            }
        }

        void validate_operators(const Operators& operators, Eigen::Index size)
        {
            if (operators.K.rows() != size || operators.K.cols() != size || operators.C.rows() != size
                || operators.C.cols() != size || operators.f.size() != size) {
                throw std::invalid_argument("assemble_modal_port_system: modal K/C/f dimensions do not match");
            }
        }

        void validate_interface(const mhs::core::Model& model, const ModalPort& macro,
            const ThermalPortInterface& interface, std::span<const double> state)
        {
            const auto model_count = model.cells.cell_to_grid.size();
            const auto physical_port_count = static_cast<std::size_t>(macro.basis.rows());
            const auto mode_count = static_cast<std::size_t>(macro.basis.cols());
            validate_operators(macro.operators, static_cast<Eigen::Index>(mode_count));

            if (physical_port_count == 0 || mode_count == 0) {
                throw std::invalid_argument("assemble_modal_port_system: port basis is empty");
            }
            if (interface.model_cells.size() != physical_port_count
                || static_cast<std::size_t>(interface.exterior_half_conductance.size()) != physical_port_count) {
                throw std::invalid_argument(
                    "assemble_modal_port_system: interface data does not match physical port count");
            }
            if (state.size() != model_count + mode_count) {
                throw std::invalid_argument(
                    "assemble_modal_port_system: state must contain FVM temperatures followed by port modes");
            }

            std::unordered_set<mhs::core::Index> unique_cells;
            for (const auto cell : interface.model_cells) {
                if (cell >= model_count || !unique_cells.insert(cell).second) {
                    throw std::invalid_argument(
                        "assemble_modal_port_system: interface cells must be unique valid FVM cells");
                }
            }
        }

        double interface_conductance(const mhs::core::Model& model, mhs::core::Index cell,
            mhs::core::FaceDir face, double exterior_half_conductance, std::span<const double> temperature, double time)
        {
            if (exterior_half_conductance <= mhs::core::zero_guard)
                return 0.0;

            const auto grid = model.cells.cell_to_grid[cell];
            mhs::core::Index ix, iy, iz;
            mhs::utils::decode_index(grid, model.mesh.ny, model.mesh.nz, ix, iy, iz);
            const auto neighbor_grid = mhs::utils::neighbor_grid_index(
                ix, iy, iz, face, model.mesh.nx, model.mesh.ny, model.mesh.nz, model.cells.grid_to_cell);
            if (neighbor_grid != mhs::core::invalidIndex) {
                throw std::invalid_argument(
                    "assemble_modal_port_system: an interface face has an active FVM neighbor");
            }

            const auto& material = model.material_table[model.cells.material_id[cell]];
            const mhs::core::FieldContext context {
                model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz], temperature[cell], time};
            const double conductivity = mhs::utils::k_along(
                face, material.kx.eval(context), material.ky.eval(context), material.kz.eval(context));
            const double area = mhs::utils::face_area(
                face, model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]);
            const double half_distance = mhs::utils::half_length_along(
                face, model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]);
            const double model_half_conductance = conductivity * area / half_distance;
            if (model_half_conductance <= mhs::core::zero_guard)
                return 0.0;
            return model_half_conductance * exterior_half_conductance
                / (model_half_conductance + exterior_half_conductance);
        }

    } // namespace

    Operators assemble_modal_port_system(const mhs::core::Model& model, const ModalPort& macro,
        const ThermalPortInterface& interface, std::span<const double> state, double time)
    {
        validate_interface(model, macro, interface, state);

        const auto model_count = model.cells.cell_to_grid.size();
        const auto mode_count = static_cast<std::size_t>(macro.basis.cols());
        const auto state_count = model_count + mode_count;
        const auto eigen_model_count = static_cast<Eigen::Index>(model_count);
        const auto eigen_mode_count = static_cast<Eigen::Index>(mode_count);
        const auto eigen_state_count = static_cast<Eigen::Index>(state_count);

        const auto temperature = state.first(model_count);
        auto model_operators = assemble_thermal(model, temperature, time);

        std::vector<Eigen::Triplet<double>> stiffness;
        std::vector<Eigen::Triplet<double>> capacity;
        stiffness.reserve(static_cast<std::size_t>(
            model_operators.K.nonZeros() + macro.operators.K.nonZeros()
            + interface.model_cells.size() * (2 * mode_count + mode_count * mode_count + 1)));
        capacity.reserve(
            static_cast<std::size_t>(model_operators.C.nonZeros() + macro.operators.C.nonZeros()));
        append_block(stiffness, model_operators.K, 0, 0);
        append_block(stiffness, macro.operators.K, eigen_model_count, eigen_model_count);
        append_block(capacity, model_operators.C, 0, 0);
        append_block(capacity, macro.operators.C, eigen_model_count, eigen_model_count);

        for (std::size_t physical_port = 0; physical_port < interface.model_cells.size(); ++physical_port) {
            const auto cell = interface.model_cells[physical_port];
            const double conductance = interface_conductance(model, cell, interface.model_face,
                interface.exterior_half_conductance[static_cast<Eigen::Index>(physical_port)], temperature, time);
            const auto cell_row = static_cast<Eigen::Index>(cell);
            stiffness.emplace_back(cell_row, cell_row, conductance);

            for (Eigen::Index mode = 0; mode < eigen_mode_count; ++mode) {
                const double projected = conductance * macro.basis(
                    static_cast<Eigen::Index>(physical_port), mode);
                const auto mode_row = eigen_model_count + mode;
                stiffness.emplace_back(cell_row, mode_row, -projected);
                stiffness.emplace_back(mode_row, cell_row, -projected);

                for (Eigen::Index other_mode = 0; other_mode < eigen_mode_count; ++other_mode) {
                    const double modal_conductance = projected
                        * macro.basis(static_cast<Eigen::Index>(physical_port), other_mode);
                    stiffness.emplace_back(mode_row, eigen_model_count + other_mode, modal_conductance);
                }
            }
        }

        Operators combined;
        combined.K.resize(eigen_state_count, eigen_state_count);
        combined.K.setFromTriplets(stiffness.begin(), stiffness.end());
        combined.C.resize(eigen_state_count, eigen_state_count);
        combined.C.setFromTriplets(capacity.begin(), capacity.end());
        combined.f.resize(eigen_state_count);
        combined.f.head(eigen_model_count) = model_operators.f;
        combined.f.tail(eigen_mode_count) = macro.operators.f;
        return combined;
    }

} // namespace mhs::sim
