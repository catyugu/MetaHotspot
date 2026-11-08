#pragma once

#include "core/vector.hpp"
#include "model/types.hpp"
#include <array>
#include <cstdint>
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
        mhs::core::Vector<RomPort> ports;
        // Ordinary thermal BCs on exposed port regions; connected ports
        // receive their thermal exchange from Connection instead.
        mhs::core::Vector<ThermalBoundarySpec> thermal_boundaries;
        mhs::core::Vector<RomObservation> observations;
        mhs::core::Vector<RomAffineTerm> affine_terms;
    };

    // Stable public DomainId -> typed local storage index. The FOM and ROM
    // payloads stay in homogeneous arrays; creation order controls DomainId.
    enum class DomainKind : std::uint8_t { Fom, Rom };

    class DomainRegistry {
    public:
        DomainId add_fom() { return add(DomainKind::Fom, fom_ids_); }
        DomainId add_rom() { return add(DomainKind::Rom, rom_ids_); }
        void clear()
        {
            entries_.clear();
            fom_ids_.clear();
            rom_ids_.clear();
        }
        std::size_t size() const { return entries_.size(); }
        std::size_t fom_count() const { return fom_ids_.size(); }
        std::size_t rom_count() const { return rom_ids_.size(); }
        bool is_fom(DomainId domain) const { return entries_.at(domain).kind == DomainKind::Fom; }
        bool is_rom(DomainId domain) const { return !is_fom(domain); }
        std::size_t fom(DomainId domain) const { return index(domain, DomainKind::Fom); }
        std::size_t rom(DomainId domain) const { return index(domain, DomainKind::Rom); }
        DomainId fom_id(std::size_t local) const { return fom_ids_.at(local); }
        DomainId rom_id(std::size_t local) const { return rom_ids_.at(local); }

    private:
        struct Entry {
            DomainKind kind;
            std::size_t index;
        };
        mhs::core::Vector<Entry> entries_;
        mhs::core::Vector<DomainId> fom_ids_, rom_ids_;

        DomainId add(DomainKind kind, mhs::core::Vector<DomainId>& ids)
        {
            if (entries_.size() >= static_cast<std::size_t>(mhs::core::invalid_id))
                throw std::length_error("too many domains");
            const auto id = static_cast<DomainId>(entries_.size());
            entries_.push_back({kind, ids.size()});
            try {
                ids.push_back(id);
            }
            catch (...) {
                entries_.pop_back();
                throw;
            }
            return id;
        }
        std::size_t index(DomainId domain, DomainKind expected) const
        {
            const auto& entry = entries_.at(domain);
            if (entry.kind != expected)
                throw std::invalid_argument("domain has the wrong physical representation");
            return entry.index;
        }
    };

    struct ModelDefinition {
        mhs::core::Vector<FomDomainDefinition> domains {1};
        mhs::core::Vector<RomDomainDefinition> rom_domains;
        DomainRegistry registry = [] {
            DomainRegistry ids;
            ids.add_fom();
            return ids;
        }();

        DomainId add_fom_domain(FomDomainDefinition domain = {})
        {
            domains.push_back(std::move(domain));
            return registry.add_fom();
        }
        DomainId add_rom_domain(RomDomainDefinition domain)
        {
            rom_domains.push_back(std::move(domain));
            return registry.add_rom();
        }
        bool has_untouched_default_fom() const
        {
            if (registry.size() != 1 || !registry.is_fom(0))
                return false;
            const auto& domain = domains.front();
            return domain.mesh.x_vertices.empty() && domain.mesh.y_vertices.empty()
                && domain.mesh.z_vertices.empty() && domain.volumes.empty() && domain.ports.empty()
                && domain.thermal_boundaries.empty() && domain.fluid_boundaries.empty();
        }
        void clear_domains()
        {
            domains.clear();
            rom_domains.clear();
            registry.clear();
        }
        FomDomainDefinition& fom(DomainId id) { return domains.at(registry.fom(id)); }
        const FomDomainDefinition& fom(DomainId id) const { return domains.at(registry.fom(id)); }
        RomDomainDefinition& rom(DomainId id) { return rom_domains.at(registry.rom(id)); }
        const RomDomainDefinition& rom(DomainId id) const { return rom_domains.at(registry.rom(id)); }
        mhs::core::Vector<ConnectionDefinition> connections;
        mhs::core::Vector<VariableSpec> variables;
        mhs::core::Vector<NamedFunction> functions;
        mhs::core::Vector<NamedMaterial> materials;
        mhs::core::Vector<Expression> sources;
        mhs::core::Vector<PointProbe> point_probes;
    };

} // namespace mhs::model
