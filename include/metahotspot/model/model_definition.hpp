#pragma once

#include "core/vector.hpp"
#include "model/types.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <string>
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

    template <typename Condition> struct SurfaceConditionSpec {
        mhs::core::Vector<SurfaceRegion> regions;
        Condition condition;
    };

    using ThermalBoundarySpec = SurfaceConditionSpec<ThermalBoundary>;

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
    using FluidBoundarySpec = SurfaceConditionSpec<FluidBoundaryCondition>;

    struct PortDefinition {
        std::string name;
        SurfaceRegion region;
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
        mhs::core::Vector<PortDefinition> ports;
        mhs::core::Vector<VolumeSpec> volumes;
        mhs::core::Vector<ThermalBoundarySpec> thermal_boundaries;
        ThermalBoundary default_thermal_boundary = NeumannBoundary {};
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

    struct RomPort {
        std::string name;
        SurfaceRegion region;
        std::int8_t outward_sign = 1;
        mhs::core::Vector<RomPortPatch> patches;
    };

    struct RomObservation {
        PointProbe point;
        mhs::core::Vector<double> weights;
        double offset = 0.0;
    };

    // Affine HTC-like parameter contribution: K(mu) = K0 + sum mu_i * K_i,
    // f(mu) = f0 + sum mu_i * f_i. C is parameter-independent in this form.
    // One compiled snapshot fixes parameter values; the same extracted matrices
    // and port maps can be reused when constructing another snapshot.
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
        mhs::core::Vector<RomPort> ports;
        mhs::core::Vector<RomObservation> observations;
        mhs::core::Vector<RomAffineTerm> affine_terms;
    };

    struct ModelDefinition {
        Study study;
        mhs::core::Vector<FomDomainDefinition> domains {1};
        // Global DomainId order: all FOM domains followed by ROM domains.
        mhs::core::Vector<RomDomainDefinition> rom_domains;
        mhs::core::Vector<ConnectionDefinition> connections;
        mhs::core::Vector<VariableSpec> variables;
        mhs::core::Vector<NamedFunction> functions;
        mhs::core::Vector<NamedMaterial> materials;
        mhs::core::Vector<Expression> sources;
        mhs::core::Vector<PointProbe> point_probes;
    };

} // namespace mhs::model
