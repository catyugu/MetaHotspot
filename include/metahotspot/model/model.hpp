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
        BoundaryConditionId condition = mhs::core::invalid_id; // index in the condition table used for this projection
        std::array<double, 4> bounds; // tangential min/max pairs
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

    // Shared compiled laws; per-face BoundaryFace records store only an index.
    using BoundaryCondition = std::variant<Dirichlet, Neumann, Convection>;

    // Midpoint quadrature on a physical source subvolume.
    struct SourceTerm {
        CellId cell = mhs::core::invalid_id;
        SourceId source = mhs::core::invalid_id;
        double weight;
        Vec3 point;
    };

    struct FomDomain {
        std::shared_ptr<const struct CompiledLibrary> library;
        DiscreteMesh discrete;
        FlowMode flow_mode = FlowMode::Frozen;

        mhs::core::Vector<MaterialId> cell_material;
        mhs::core::Vector<SourceTerm> source_terms;
        mhs::core::Vector<BoundaryFace> boundary_faces; // sorted by CellId
        FluidTopology fluid;
        mhs::core::Vector<FluidBoundaryPatch> fluid_boundaries;
    };

    struct CompiledLibrary {
        mhs::core::Vector<MaterialProps> material_table;
        mhs::core::Vector<mhs::expression::CompiledExpression> heat_source_table;
        mhs::core::Vector<BoundaryCondition> boundary_conditions;
        mhs::core::Vector<FluidBoundaryCondition> fluid_conditions;
    };

    struct ConnectionPatch {
        CellId a = core::invalid_id, b = core::invalid_id;
        std::array<double, 4> bounds {};
    };

    struct ThermalConnection {
        PortReference a, b;
        Axis axis = Axis::X;
        bool nonmatching = false;
        core::Vector<ConnectionPatch> patches;
        std::int8_t sign_a = 1; // Geometric outward normal on side A.
    };

    // Exterior ROM subface after intersecting port facets and thermal BCs.
    // Its physical trace and half-resistance come from the owning port patch.
    struct RomBoundaryTerm {
        DomainId domain = core::invalid_id;
        PortId port = core::invalid_id;
        std::size_t patch = 0;
        BoundaryConditionId condition = core::invalid_id;
        std::array<double, 4> bounds {};
    };

    struct CompiledComponent {
        std::variant<FomDomain, RomDomain> physics;
        core::Vector<PortDefinition> ports;
        core::Vector<ObservationDefinition> observations;
        std::shared_ptr<const CompiledLibrary> library;
    };

    // Published as a const snapshot by the API; solve workspaces own all flow state.
    struct CompiledModel {
        mhs::core::Vector<CompiledComponent> components;

        FomDomain& fom(DomainId id) { return std::get<FomDomain>(components.at(id).physics); }
        const FomDomain& fom(DomainId id) const { return std::get<FomDomain>(components.at(id).physics); }
        const RomDomain& rom(DomainId id) const { return std::get<RomDomain>(components.at(id).physics); }
        const CompiledLibrary& library() const { return *components.front().library; }

        StateLayout layout;
        core::Vector<ThermalConnection> connections;
        core::Vector<RomBoundaryTerm> rom_boundary_terms;
    };

} // namespace mhs::model
