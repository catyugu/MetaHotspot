#pragma once

#include "core/vector.hpp"
#include "model/types.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
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

    // A spatial assignment refers to one reusable condition of the appropriate kind.
    struct SurfaceAssignment {
        mhs::core::Vector<SurfaceRegion> regions;
        BoundaryConditionId condition = mhs::core::invalid_id;
    };

    using ThermalBoundarySpec = SurfaceAssignment;
    using FluidBoundarySpec = SurfaceAssignment;

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

    // Definitions are reusable across models. Each category owns a separate, append-only ID space.
    struct LibraryDefinition {
        mhs::core::Vector<NamedFunction> functions;
        mhs::core::Vector<NamedMaterial> materials;
        mhs::core::Vector<Expression> sources;
        mhs::core::Vector<ThermalBoundary> thermal_boundaries;
        mhs::core::Vector<FluidBoundaryCondition> fluid_boundaries;

        void add_function(NamedFunction value)
        {
            const auto same_name = [&](const auto& entry) { return entry.name == value.name; };
            if (std::any_of(functions.begin(), functions.end(), same_name))
                throw std::invalid_argument("duplicate function name: " + value.name);
            checked_id(functions.size());
            functions.push_back(std::move(value));
        }
        MaterialId add_material(NamedMaterial value)
        {
            const auto same_name = [&](const auto& entry) { return entry.name == value.name; };
            if (std::any_of(materials.begin(), materials.end(), same_name))
                throw std::invalid_argument("duplicate material name: " + value.name);
            const auto id = checked_id(materials.size());
            materials.push_back(std::move(value));
            return id;
        }
        SourceId add_source(Expression value)
        {
            const auto id = checked_id(sources.size());
            sources.push_back(std::move(value));
            return id;
        }
        BoundaryConditionId add_thermal_boundary(ThermalBoundary value)
        {
            const auto id = checked_id(thermal_boundaries.size());
            thermal_boundaries.push_back(std::move(value));
            return id;
        }
        BoundaryConditionId add_fluid_boundary(FluidBoundaryCondition value)
        {
            const auto id = checked_id(fluid_boundaries.size());
            fluid_boundaries.push_back(std::move(value));
            return id;
        }

    private:
        static BoundaryConditionId checked_id(std::size_t size)
        {
            if (size >= mhs::core::invalid_id)
                throw std::length_error("library resource ID space exhausted");
            return static_cast<BoundaryConditionId>(size);
        }
    };

    struct ModelDefinition {
        explicit ModelDefinition(std::shared_ptr<LibraryDefinition> shared = std::make_shared<LibraryDefinition>())
            : library_handle(std::move(shared))
        {
            if (!library_handle)
                throw std::invalid_argument("model requires a Library");
        }

        LibraryDefinition& library() { return *library_handle; }
        const LibraryDefinition& library() const { return *library_handle; }
        void detach_library() { library_handle = std::make_shared<LibraryDefinition>(*library_handle); }

        double reference_temperature = 300.0;
        MeshSpec mesh;
        mhs::core::Vector<VariableSpec> variables;
        mhs::core::Vector<VolumeSpec> volumes;
        mhs::core::Vector<ThermalBoundarySpec> thermal_boundaries;
        BoundaryConditionId default_thermal_boundary = mhs::core::invalid_id;
        mhs::core::Vector<PointProbeSpec> point_probes;
        mhs::core::Vector<FluidBoundarySpec> fluid_boundaries;

    private:
        std::shared_ptr<LibraryDefinition> library_handle;
    };

} // namespace mhs::model
