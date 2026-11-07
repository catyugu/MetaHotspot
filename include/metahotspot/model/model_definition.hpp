#pragma once

#include "core/vector.hpp"
#include "model/types.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace mhs::model {

    using Expression = std::string;

    enum class LengthUnit { Meter, Millimeter, Micrometer, Nanometer, Inch, Mil };
    enum class GeometryOperation : uint8_t { Add, Subtract };

    struct ModelSettings {
        StudyType study_type = StudyType::Steady;
        LengthUnit length_unit = LengthUnit::Meter;
        double initial_temperature = 300.0;
        double transient_duration = 0.0;
        double transient_output_interval = 1.0;
    };

    struct MeshSpec {
        mhs::core::Vector<double> x_vertices;
        mhs::core::Vector<double> y_vertices;
        mhs::core::Vector<double> z_vertices;
    };

    struct VariableSpec {
        std::string name;
        Expression value;
    };

    struct RectSpec {
        Expression x;
        Expression y;
        Expression width;
        Expression height;
    };

    struct RectOperation {
        GeometryOperation operation = GeometryOperation::Add;
        RectSpec rect;
    };

    struct BlockSpec {
        std::string material;
        Expression volumetric_heat_source;
        Expression x_offset;
        Expression y_offset;
        std::optional<Expression> thickness;
        mhs::core::Vector<RectOperation> geometry;
    };

    struct LayerSpec {
        Expression thickness;
        Expression x_offset;
        Expression y_offset;
        mhs::core::Vector<BlockSpec> blocks;
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

    struct FaceRegion {
        Axis axis = Axis::Z;
        double coordinate = 0.0;
        mhs::core::Vector<RegionRect> rectangles;
    };

    struct BoundaryPatch {
        mhs::core::Vector<FaceRegion> regions;
        ThermalBoundary condition;
    };

    struct MaterialSpec {
        Expression conductivity_x = "0.0";
        Expression conductivity_y = "0.0";
        Expression conductivity_z = "0.0";
        Expression density = "0.0";
        Expression specific_heat = "0.0";
        std::optional<Expression> dynamic_viscosity;
        bool is_fluid = false;
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

    struct ObservationPointSpec {
        std::string name;
        Expression x;
        Expression y;
        Expression z;
    };

    // Omitted incoming temperature uses the model's initial temperature.
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
    struct FluidBoundarySpec {
        mhs::core::Vector<FaceRegion> regions;
        FluidBoundaryCondition condition;
    };

    struct ModelDefinition {
        ModelSettings settings;
        MeshSpec mesh;
        mhs::core::Vector<VariableSpec> variables;
        mhs::core::Vector<NamedFunction> functions;
        mhs::core::Vector<NamedMaterial> materials;
        mhs::core::Vector<LayerSpec> layers;
        mhs::core::Vector<BoundaryPatch> boundaries;
        ThermalBoundary default_boundary = NeumannBoundary {};
        mhs::core::Vector<ObservationPointSpec> observation_points;
        mhs::core::Vector<FluidBoundarySpec> fluid_boundaries;
    };

} // namespace mhs::model
