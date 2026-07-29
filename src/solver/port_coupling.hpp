#pragma once

#include "runtime/model.hpp"
#include "solver/assembler.hpp"

#include <Eigen/Core>
#include <span>
#include <vector>

namespace mhs::sim {

    /// Linear macro model expressed in retained modal coordinates.
    ///
    /// Physical port temperatures are reconstructed as `basis * q`, where
    /// `q` is the modal suffix of the global state.
    struct ModalPort {
        Operators operators;
        Eigen::MatrixXd basis;
    };

    /// Geometric connection between exposed FVM faces and physical macro ports.
    ///
    /// `exterior_half_conductance[p]` is the conductance from physical port p
    /// to the shared interface. The FVM-side half conductance is evaluated
    /// from the current material state during every nonlinear linearization.
    struct ThermalPortInterface {
        std::vector<mhs::core::Index> model_cells;
        mhs::core::FaceDir model_face = mhs::core::FaceDir::XP;
        Eigen::VectorXd exterior_half_conductance;
    };

    /// Assemble `C * dx/dt + K * x = f` for `[FVM temperatures, port modes]`.
    Operators assemble_modal_port_system(const mhs::core::Model& model, const ModalPort& macro,
        const ThermalPortInterface& interface, std::span<const double> state, double time);

} // namespace mhs::sim
