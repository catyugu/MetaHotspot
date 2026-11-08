#pragma once

#include "core/vector.hpp"
#include "model/types.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <utility>
#include <variant>

namespace mhs::model {

    using Expression = std::string;

    enum class GeometryOperation : uint8_t { Add, Subtract };

    struct MeshSpec {
        mhs::core::Vector<double> x_vertices;
        mhs::core::Vector<double> y_vertices;
        mhs::core::Vector<double> z_vertices;
    };

    struct VariableSpec {
        std::string name;
        Expression value;
    };

    // SI geometry. Later volumes override earlier volumes; operations are ordered.
    struct BoxOperation {
        GeometryOperation operation = GeometryOperation::Add;
        std::array<double, 3> min {};
        std::array<double, 3> max {};
    };
    struct VolumeSpec {
        MaterialId material = mhs::core::invalid_id;
        SourceId source = mhs::core::invalid_id;
        mhs::core::Vector<BoxOperation> geometry;
    };

    struct DirichletBoundary {
        Expression temperature = "300.0";
    };

    struct NeumannBoundary {
        Expression heat_flux = "0.0";
    };

    struct ConvectionBoundary {
        Expression coefficient = "0.0";
        Expression ambient_temperature = "300.0";
    };

    using ThermalBoundary = std::variant<DirichletBoundary, NeumannBoundary, ConvectionBoundary>;

    struct RegionRect {
        double a_min = 0.0;
        double a_max = 0.0;
        double b_min = 0.0;
        double b_max = 0.0;
    };

    struct SurfaceRegion {
        Axis axis = Axis::Z;
        double coordinate = 0.0;
        RegionRect rectangle;
    };

    struct ThermalBoundarySpec {
        mhs::core::Vector<SurfaceRegion> regions;
        ThermalBoundaryId condition;
    };

    struct MaterialSpec {
        Expression conductivity_x = "0.0";
        Expression conductivity_y = "0.0";
        Expression conductivity_z = "0.0";
        Expression density = "0.0";
        Expression specific_heat = "0.0";
        std::optional<Expression> dynamic_viscosity;
    };

    struct NamedMaterial {
        std::string name;
        MaterialSpec value;
    };

    struct ExpressionFunctionSpec {
        Expression expression;
    };

    struct DoubleExponentialFunctionSpec {
        double amplitude = 0.0;
        double alpha = 0.0;
        double beta = 0.0;
    };

    struct GaussFunctionSpec {
        double amplitude = 0.0;
        double tau = 0.0;
        double center = 0.0;
    };

    struct SineFunctionSpec {
        double amplitude = 0.0;
        double angular_frequency = 0.0;
        double phase = 0.0;
    };

    struct PiecewiseFunctionSpec {
        struct Point {
            double x = 0.0;
            double y = 0.0;
        };
        mhs::core::Vector<Point> points;
    };

    struct PeriodicPiecewiseConstantFunctionSpec {
        double period = 0.0;
        mhs::core::Vector<double> values;
    };

    using FunctionSpec = std::variant<ExpressionFunctionSpec, DoubleExponentialFunctionSpec, GaussFunctionSpec,
        SineFunctionSpec, PiecewiseFunctionSpec, PeriodicPiecewiseConstantFunctionSpec>;

    struct NamedFunction {
        std::string name;
        FunctionSpec value;
    };

    struct PointProbe {
        std::string name;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    // Incoming flow requires an explicit temperature, including pressure-boundary backflow.
    struct PressureInlet {
        double pressure = 0.0;
        std::optional<double> temperature = std::nullopt;
    };
    struct MassFlowInlet {
        double mass_flow_rate = 0.0;
        std::optional<double> temperature = std::nullopt;
    };
    struct PressureOutlet {
        double pressure = 0.0;
        std::optional<double> temperature = std::nullopt;
    };
    using FluidBoundaryCondition = std::variant<PressureInlet, MassFlowInlet, PressureOutlet>;
    struct FluidBoundarySpec {
        mhs::core::Vector<SurfaceRegion> regions;
        FluidBoundaryId condition;
    };

    // Reusable attributes shared by any number of models.
    struct LibraryDefinition {
        mhs::core::Vector<VariableSpec> variables;
        mhs::core::Vector<NamedFunction> functions;
        mhs::core::Vector<NamedMaterial> materials;
        mhs::core::Vector<Expression> sources;
        mhs::core::Vector<ThermalBoundary> thermal_boundaries;
        mhs::core::Vector<FluidBoundaryCondition> fluid_boundaries;

        ThermalBoundaryId add_thermal_boundary(ThermalBoundary value)
        {
            const auto id = static_cast<BoundaryConditionId>(thermal_boundaries.size());
            thermal_boundaries.push_back(std::move(value));
            return id;
        }
        FluidBoundaryId add_fluid_boundary(FluidBoundaryCondition value)
        {
            const auto id = static_cast<BoundaryConditionId>(fluid_boundaries.size());
            fluid_boundaries.push_back(std::move(value));
            return id;
        }
    };

    struct PortReference {
        DomainId domain = mhs::core::invalid_id;
        PortId port = mhs::core::invalid_id;
    };

    // Perfect thermal contact between two independent domains.
    struct ConnectionDefinition {
        PortReference a, b;
    };

    enum class FlowMode : std::uint8_t { Frozen, Stokes };

    struct FomDomainDefinition {
        MeshSpec mesh;
        FlowMode flow_mode = FlowMode::Frozen;
        mhs::core::Vector<VolumeSpec> volumes;
        ThermalBoundaryId default_thermal_boundary = mhs::core::invalid_id;
        mhs::core::Vector<FluidBoundarySpec> fluid_boundaries;
    };

    // Meshless modal domain. Port patches carry physical boundary geometry,
    // normal half-resistance and affine maps from modal state to cell traces.
    struct RomPortPatch {
        RegionRect rectangle;
        double half_resistance = 0.0; // m^2 K/W
        std::array<double, 2> center {};
        mhs::core::Vector<double> trace;
        double offset = 0.0;
        // Optional for conforming ports; required by nonmatching NCC.
        std::array<mhs::core::Vector<double>, 2> tangential_gradient;
    };

    struct PortDefinition {
        std::string name;
        SurfaceRegion region;
        PortDefinition() = default;
        PortDefinition(std::string n, SurfaceRegion r) : name(std::move(n)), region(r) { }
        // FVM derives the normal from geometry; reduced ports provide
        // their normal and physical trace patches.
        std::int8_t outward_sign = 1;
        mhs::core::Vector<RomPortPatch> patches;
    };

    // A physical temperature observation belongs to a component.
    // Cartesian observations use the point only; reduced observations carry
    // a modal trace map and an affine offset.
    struct ObservationDefinition {
        PointProbe point;
        mhs::core::Vector<double> weights;
        double offset = 0.0;
    };

    // Optional intrinsic affine operator parameters: K(mu) = K0 + sum mu_i * K_i,
    // f(mu) = f0 + sum mu_i * f_i. Physical exterior HTC belongs to the
    // ordinary boundary/port coupling, not to this parameter family.
    struct RomAffineTerm {
        std::string name;
        double value = 0.0;
        mhs::core::Vector<double> K;
        mhs::core::Vector<double> f;
    };

    // Dense row-major projected matrices and physical port/observation maps.
    struct RomDomainDefinition {
        std::size_t order = 0;
        mhs::core::Vector<double> K, C, f, initial_state;
        mhs::core::Vector<RomAffineTerm> affine_terms;
    };

    using RomDomain = RomDomainDefinition;

    // Numerical formulation and physical exterior are independent concepts.
    struct ComponentDefinition {
        std::variant<FomDomainDefinition, RomDomainDefinition> physics;
        mhs::core::Vector<PortDefinition> ports;
        mhs::core::Vector<ThermalBoundarySpec> thermal_boundaries;
        mhs::core::Vector<ObservationDefinition> observations;
    };

    struct ModelDefinition {
        std::shared_ptr<LibraryDefinition> library = std::make_shared<LibraryDefinition>();
        mhs::core::Vector<ComponentDefinition> components;

        ComponentDefinition& component(DomainId id) { return components.at(id); }
        const ComponentDefinition& component(DomainId id) const { return components.at(id); }
        FomDomainDefinition& fom(DomainId id) { return std::get<FomDomainDefinition>(component(id).physics); }
        const auto& fom(DomainId id) const { return std::get<FomDomainDefinition>(component(id).physics); }
        RomDomainDefinition& rom(DomainId id) { return std::get<RomDomainDefinition>(component(id).physics); }

        mhs::core::Vector<ConnectionDefinition> connections;
    };

} // namespace mhs::model
