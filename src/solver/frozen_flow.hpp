#pragma once
#include "model/model.hpp"
#include <span>

namespace mhs::solver {
    using mhs::model::CellId;
    struct FluidInflow {
        double mass_flow_rate;
        double temperature;
    };
    struct FluidOutflow {
        double mass_flow_rate;
    };
    using FluidTransportCondition = std::variant<FluidInflow, FluidOutflow>;
    struct FluidBoundaryTerm {
        CellId cell = core::invalid_id;
        FluidTransportCondition condition;
    };

    // Owned by one solve/assembly workspace, never written back to CompiledModel.
    struct FrozenFlow {
        core::Vector<double> face_mass_flow;
        core::Vector<double> interface_heat_transfer_factor;
        core::Vector<FluidBoundaryTerm> boundary_terms;
    };

    FrozenFlow initialize_frozen_flow(const mhs::model::FomDomain& model, const mhs::model::CompiledLibrary& library,
        std::span<const double> temperature, double time = 0.0);
}
