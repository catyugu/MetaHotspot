#pragma once

#include "core/vector.hpp"
#include "model/types.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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

    struct SurfaceAssignment {
        mhs::core::Vector<SurfaceRegion> regions;
        Id condition = core::invalid_id;
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
    struct LibraryDefinition {
    private:
        mhs::core::Vector<VariableSpec> variables_;
        mhs::core::Vector<NamedFunction> functions_;
        mhs::core::Vector<NamedMaterial> materials_;
        mhs::core::Vector<Expression> sources_;
        mhs::core::Vector<ThermalBoundary> thermal_boundaries_;
        mhs::core::Vector<FluidBoundaryCondition> fluid_boundaries_;

        template <class T, class U> static Id insert(mhs::core::Vector<T>& values, U&& value)
        {
            if (values.size() >= core::invalid_id)
                throw std::length_error("library ID space exhausted");
            const auto id = static_cast<Id>(values.size());
            values.emplace_back(std::forward<U>(value));
            return id;
        }

    public:
        const mhs::core::Vector<VariableSpec>& variables() const { return variables_; }
        const mhs::core::Vector<NamedFunction>& functions() const { return functions_; }
        const mhs::core::Vector<NamedMaterial>& materials() const { return materials_; }
        const mhs::core::Vector<Expression>& sources() const { return sources_; }
        const mhs::core::Vector<ThermalBoundary>& thermal_boundaries() const { return thermal_boundaries_; }
        const mhs::core::Vector<FluidBoundaryCondition>& fluid_boundaries() const { return fluid_boundaries_; }

        // These views allow editing registered payloads but cannot grow the registry.
        std::span<NamedMaterial> edit_materials() { return materials_; }
        std::span<ThermalBoundary> edit_thermal_boundaries() { return thermal_boundaries_; }

        const VariableSpec& variable(VariableId id) const { return variables_.at(id); }
        VariableSpec& variable(VariableId id) { return variables_.at(id); }
        const NamedFunction& function(FunctionId id) const { return functions_.at(id); }
        NamedFunction& function(FunctionId id) { return functions_.at(id); }
        const NamedMaterial& material(MaterialId id) const { return materials_.at(id); }
        NamedMaterial& material(MaterialId id) { return materials_.at(id); }
        const Expression& source(SourceId id) const { return sources_.at(id); }
        Expression& source(SourceId id) { return sources_.at(id); }
        const ThermalBoundary& thermal_boundary(ThermalBoundaryId id) const { return thermal_boundaries_.at(id); }
        ThermalBoundary& thermal_boundary(ThermalBoundaryId id) { return thermal_boundaries_.at(id); }
        const FluidBoundaryCondition& fluid_boundary(FluidBoundaryId id) const { return fluid_boundaries_.at(id); }
        FluidBoundaryCondition& fluid_boundary(FluidBoundaryId id) { return fluid_boundaries_.at(id); }

        MaterialId material_id(std::string_view name) const
        {
            for (std::size_t i = 0; i < materials_.size(); ++i)
                if (materials_[i].name == name)
                    return static_cast<MaterialId>(i);
            throw std::invalid_argument("unknown material: " + std::string(name));
        }

        VariableId add_variable(VariableSpec v) { return insert(variables_, std::move(v)); }
        FunctionId add_function(NamedFunction v) { return insert(functions_, std::move(v)); }
        MaterialId add_material(NamedMaterial v) { return insert(materials_, std::move(v)); }
        SourceId add_source(Expression v) { return insert(sources_, std::move(v)); }
        ThermalBoundaryId add_thermal_boundary(ThermalBoundary v) { return insert(thermal_boundaries_, std::move(v)); }
        FluidBoundaryId add_fluid_boundary(FluidBoundaryCondition v) { return insert(fluid_boundaries_, std::move(v)); }
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
        mhs::core::Vector<SurfaceAssignment> fluid_boundaries;
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
        mhs::core::Vector<SurfaceAssignment> thermal_boundaries;
        mhs::core::Vector<ObservationDefinition> observations;
    };

    struct ModelDefinition {
        std::shared_ptr<LibraryDefinition> library_handle = std::make_shared<LibraryDefinition>();
        mhs::core::Vector<ComponentDefinition> components;

        ModelDefinition() = default;
        explicit ModelDefinition(std::shared_ptr<LibraryDefinition> shared) : library_handle(std::move(shared))
        {
            if (!library_handle)
                throw std::invalid_argument("model requires a Library");
        }

        ComponentDefinition& add_component() { return components.emplace_back(); }
        ComponentDefinition& component(DomainId id) { return components.at(id); }
        const ComponentDefinition& component(DomainId id) const { return components.at(id); }
        LibraryDefinition& library() { return *library_handle; }
        const LibraryDefinition& library() const { return *library_handle; }
        FomDomainDefinition& fom(DomainId id) { return std::get<FomDomainDefinition>(component(id).physics); }
        const auto& fom(DomainId id) const { return std::get<FomDomainDefinition>(component(id).physics); }
        RomDomainDefinition& rom(DomainId id) { return std::get<RomDomainDefinition>(component(id).physics); }

        mhs::core::Vector<ConnectionDefinition> connections;
    };

} // namespace mhs::model
