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

    // SI geometry; operations within each region are ordered.
    struct BoxOperation {
        GeometryOperation operation = GeometryOperation::Add;
        std::array<double, 3> min {};
        std::array<double, 3> max {};
    };
    // Reusable continuous geometry; IDs are indices in ModelDefinition::regions.
    struct RegionSpec {
        mhs::core::Vector<BoxOperation> geometry;
    };

    // The union of material regions defines the computational domain.
    // Later assignments override earlier ones where their regions overlap.
    struct MaterialAssignment {
        RegionId region = mhs::core::invalid_id;
        MaterialId material = mhs::core::invalid_id;
    };

    enum class SourceOperation : uint8_t { Add, Replace };

    struct SourceAssignment {
        RegionId region = mhs::core::invalid_id;
        SourceId source = mhs::core::invalid_id;
        // Replace also permits invalid_id to clear preceding sources locally.
        SourceOperation operation = SourceOperation::Add;
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
        MaterialKind kind = MaterialKind::Solid;
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

    struct PointProbeSpec {
        std::string name;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    // Omitted incoming temperature uses the model reference temperature.
    struct PressureInlet {
        double pressure = 0.0;
        std::optional<double> temperature;
    };
    struct MassFlowInlet {
        double mass_flow_rate = 0.0;
        std::optional<double> temperature;
    };
    struct PressureOutlet {
        double pressure = 0.0;
    };
    using FluidBoundaryCondition = std::variant<PressureInlet, MassFlowInlet, PressureOutlet>;
    using FluidBoundarySpec = SurfaceConditionSpec<FluidBoundaryCondition>;

    struct ModelDefinition {
        double reference_temperature = 300.0;
        MeshSpec mesh;
        mhs::core::Vector<NamedFunction> functions;
        mhs::core::Vector<NamedMaterial> materials;
        mhs::core::Vector<Expression> sources;
        mhs::core::Vector<RegionSpec> regions;
        mhs::core::Vector<MaterialAssignment> material_assignments;
        mhs::core::Vector<SourceAssignment> source_assignments;
        mhs::core::Vector<ThermalBoundarySpec> thermal_boundaries;
        ThermalBoundary default_thermal_boundary = NeumannBoundary {};
        mhs::core::Vector<PointProbeSpec> point_probes;
        mhs::core::Vector<FluidBoundarySpec> fluid_boundaries;
    };

} // namespace mhs::model
