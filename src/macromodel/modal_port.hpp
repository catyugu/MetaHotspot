#pragma once

#include "common/model.hpp"
#include "common/solution.hpp"
#include "common/solver.hpp"
#include "solver/assembler.hpp"

#include <Eigen/Core>
#include <span>
#include <vector>

namespace mhs::macro {

    /// A geometric boundary patch representing one physical DtN port.
    /// Rectangle coordinates are the two tangential coordinates of the face.
    struct PortPatch {
        mhs::core::FaceDir face = mhs::core::FaceDir::XP;
        double coordinate = 0.0;
        double a_min = 0.0;
        double a_max = 0.0;
        double b_min = 0.0;
        double b_max = 0.0;
    };

    struct PortFace {
        mhs::core::Index cell = mhs::core::invalidIndex;
        std::size_t port = 0;
        mhs::core::FaceDir face = mhs::core::FaceDir::XP;
    };

    /// Compiled geometric mapping from physical ports to exposed FVM faces.
    struct PortMap {
        std::size_t port_count = 0;
        std::vector<PortFace> faces;
    };

    /// Reduced Dirichlet-to-Neumann model expressed in retained coordinates.
    struct DtNModel {
        mhs::sim::Operators operators;
        /// [physical ports x retained states]. When empty, the physical port
        /// temperatures are the leading physical_port_count retained states,
        /// i.e. the implicit basis is [I, 0].
        Eigen::MatrixXd port_basis;
        std::size_t physical_port_count = 0;
    };

    /// Resolve geometric patches once against a compiled model.
    PortMap compile_port_map(const mhs::core::Model& model, std::span<const PortPatch> patches);

    /// Assemble an isolated component as [physical face ports, FVM cell states].
    /// The port rows have zero capacity and expose the component's DtN relation.
    mhs::sim::Operators assemble_dtn(
        const mhs::core::Model& model, const PortMap& ports, std::span<const double> cell_state, double time);

    /// Assemble C*dx/dt + K*x = f for [FVM temperatures, retained DtN states].
    mhs::sim::Operators assemble_coupled(const mhs::core::Model& model, const DtNModel& dtn, const PortMap& ports,
        std::span<const double> state, double time);

    /// Solve an FVM model coupled to a reduced DtN model.
    mhs::core::Solution solve(const mhs::core::Model& model, const DtNModel& dtn, const PortMap& ports,
        std::span<const double> initial_state, const mhs::sim::SolveOptions& opts = {});

} // namespace mhs::macro
