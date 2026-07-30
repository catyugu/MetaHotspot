#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mhs::model {

    using Expression = std::string;

    enum class StudyType : uint8_t { Steady, Transient };
    enum class LengthUnit { Meter, Millimeter, Micrometer, Nanometer, Inch, Mil };
    enum class GeometryOperation : uint8_t { Add, Subtract };
    enum class Axis : uint8_t { X, Y, Z };
    enum class FluidBoundaryKind : uint8_t { None, Pressure, MassFlowRate, Velocity };

    struct ModelSettings {
        StudyType study_type = StudyType::Steady;
        LengthUnit length_unit = LengthUnit::Meter;
        double initial_temperature = 300.0;
        double transient_duration = 0.0;
        double transient_output_interval = 1.0;
    };

    struct MeshSpec {
        std::vector<double> x_vertices;
        std::vector<double> y_vertices;
        std::vector<double> z_vertices;
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
        std::vector<RectOperation> geometry;
    };

    struct LayerSpec {
        Expression thickness;
        Expression x_offset;
        Expression y_offset;
        std::vector<BlockSpec> blocks;
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
        std::vector<RegionRect> rectangles;
    };

    struct BoundaryPatch {
        std::vector<FaceRegion> regions;
        ThermalBoundary condition;
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
        std::vector<Point> points;
    };

    struct PeriodicPiecewiseConstantFunctionSpec {
        double period = 0.0;
        std::vector<double> values;
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

    struct FluidBoundarySpec {
        std::vector<FaceRegion> regions;
        FluidBoundaryKind kind = FluidBoundaryKind::None;
        double value = 0.0;
        double inlet_temperature = std::numeric_limits<double>::quiet_NaN();
    };

    struct ModelDefinition {
        ModelSettings settings;
        MeshSpec mesh;
        std::vector<VariableSpec> variables;
        std::vector<NamedFunction> functions;
        std::vector<NamedMaterial> materials;
        std::vector<LayerSpec> layers;
        std::vector<BoundaryPatch> boundaries;
        ThermalBoundary default_boundary = NeumannBoundary {};
        std::vector<ObservationPointSpec> observation_points;
        std::vector<FluidBoundarySpec> fluid_boundaries;
    };

} // namespace mhs::model
