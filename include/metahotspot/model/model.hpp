#pragma once

#include "expression/expression.hpp"
#include "model/discrete_mesh.hpp"
#include "model/fluid_model.hpp"
#include "model/model_definition.hpp"
#include "model/state_layout.hpp"
#include "model/types.hpp"

#include "core/vector.hpp"
#include <string>
#include <variant>

namespace mhs::model {

    // Projected external boundary integration term; independent of mesh face IDs.
    struct BoundaryFace {
        CellId cell = mhs::core::invalid_id;
        Axis axis = Axis::X;
        std::int8_t sign = 1;
        ThermalBoundaryIndex condition = mhs::core::invalid_id;
        std::array<double, 4> bounds;
    };

    struct MaterialProps {
        mhs::expression::CompiledExpression kx;
        mhs::expression::CompiledExpression ky;
        mhs::expression::CompiledExpression kz;
        mhs::expression::CompiledExpression rho;
        mhs::expression::CompiledExpression c;
        std::optional<mhs::expression::CompiledExpression> dynamic_viscosity;
    };

    struct Dirichlet {
        mhs::expression::CompiledExpression temperature;
    };
    struct Neumann {
        mhs::expression::CompiledExpression heat_flux;
    };
    struct Convection {
        mhs::expression::CompiledExpression coefficient;
        mhs::expression::CompiledExpression ambient_temperature;
    };
    using BoundaryCondition = std::variant<Dirichlet, Neumann, Convection>;

    struct SourceTerm {
        CellId cell = mhs::core::invalid_id;
        SourceIndex source_index = mhs::core::invalid_id;
        double weight;
        Vec3 point;
    };

    struct CompiledFluidAssignment {
        FluidBoundaryIndex condition = core::invalid_id;
        double area = 0.0;
    };

    struct FomDomain {
        DiscreteMesh discrete;
        FlowMode flow_mode = FlowMode::Frozen;
        mhs::core::Vector<MaterialIndex> cell_material;
        mhs::core::Vector<SourceTerm> source_terms;
        mhs::core::Vector<BoundaryFace> boundary_faces;
        FluidTopology fluid;
        mhs::core::Vector<FluidBoundaryPatch> fluid_boundaries;
        core::Vector<CompiledFluidAssignment> fluid_assignments;
    };

    struct CompiledLibrary {
        mhs::core::Vector<MaterialProps> material_table;
        mhs::core::Vector<mhs::expression::CompiledExpression> heat_source_table;
        mhs::core::Vector<BoundaryCondition> boundary_conditions;
        mhs::core::Vector<FluidBoundaryCondition> fluid_conditions;
    };

    struct RomTracePatch {
        double half_resistance = 0.0;
        std::array<double, 2> center {};
        core::Vector<double> trace;
        double offset = 0.0;
        std::array<core::Vector<double>, 2> tangential_gradient;
    };

    struct RomDomain {
        core::Vector<double> K, C, f, initial_state;
        core::Vector<core::Vector<RomTracePatch>> ports;
    };

    struct ConnectionPatch {
        Id a = core::invalid_id, b = core::invalid_id;
        std::array<double, 4> bounds {};
    };

    struct ConnectionEndpoint {
        DomainId domain = core::invalid_id;
        PortId port = core::invalid_id;
    };

    struct ThermalConnection {
        ConnectionEndpoint a, b;
        Axis axis = Axis::X;
        bool nonmatching = false;
        core::Vector<ConnectionPatch> patches;
        std::int8_t sign_a = 1;
    };

    struct RomBoundaryTerm {
        DomainId domain = core::invalid_id;
        PortId port = core::invalid_id;
        std::size_t patch = 0;
        ThermalBoundaryIndex condition = core::invalid_id;
        Vec3 center {};
        double area = 0.0;
    };

    struct CompiledObservation {
        std::string name;
        DomainId domain = core::invalid_id;
        CellId cell = core::invalid_id;
        Vec3 point {};
        core::Vector<double> weights;
        double offset = 0.0;
    };

    struct CompiledComponent {
        std::variant<FomDomain, RomDomain> physics;
        FieldId state_field = core::invalid_id;
    };

    struct CompiledModel {
        mhs::core::Vector<CompiledComponent> components;
        CompiledLibrary library_data;

        FomDomain& fom(DomainId id) { return std::get<FomDomain>(components.at(id).physics); }
        const FomDomain& fom(DomainId id) const { return std::get<FomDomain>(components.at(id).physics); }
        const RomDomain& rom(DomainId id) const { return std::get<RomDomain>(components.at(id).physics); }
        CompiledLibrary& library() { return library_data; }
        const CompiledLibrary& library() const { return library_data; }

        StateLayout layout;
        core::Vector<ThermalConnection> connections;
        core::Vector<RomBoundaryTerm> rom_boundary_terms;
        core::Vector<CompiledObservation> observations;
    };

} // namespace mhs::model
