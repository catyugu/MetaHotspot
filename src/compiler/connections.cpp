#include "compiler/connections.hpp"
#include "compiler/geometry_compiler.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace mhs::compiler {
    namespace {
        using namespace model;
        struct Facet {
            CellId index = core::invalid_id; // FOM cell or ROM port-patch index
            SurfaceBounds bounds {};
            std::int8_t sign = 1;
        };
        using Facets = core::Vector<Facet>;

        bool intersects(const SurfaceRegion& a, const SurfaceRegion& b)
        {
            return a.axis == b.axis && a.coordinate == b.coordinate
                && positive_bounds(intersect_bounds(surface_bounds(a), surface_bounds(b)));
        }
    } // namespace

    void compile_connections(const model::ModelDefinition& definition, model::CompiledModel& compiled)
    {
        using namespace model;
        const auto ndom = definition.components.size();
        core::Vector<core::Vector<Facets>> ports(ndom);
        core::Vector<core::Vector<bool>> connected(ndom);
        for (std::size_t d = 0; d < ndom; ++d) {
            const auto count = definition.components[d].ports.size();
            ports[d].resize(count);
            connected[d].resize(count, false);
        }
        for (const auto& spec : definition.connections) {
            if (spec.a.domain == spec.b.domain || spec.a.domain >= ndom || spec.b.domain >= ndom)
                throw std::invalid_argument("connection must join two different domains");
            if (connected.at(spec.a.domain).at(spec.a.port) || connected.at(spec.b.domain).at(spec.b.port))
                throw std::invalid_argument("port is connected more than once");
            connected[spec.a.domain][spec.a.port] = true;
            connected[spec.b.domain][spec.b.port] = true;
        }

        // Every component validates the same named physical-port contract.
        // Cartesian faces are projected once for both boundary laws and ports;
        // reduced ports instead carry explicit trace facets.
        for (std::size_t d = 0; d < ndom; ++d) {
            const auto& component = definition.component(static_cast<DomainId>(d));
            const auto* rom = std::get_if<RomDomain>(&compiled.components[d].physics);
            auto* fom = rom ? nullptr : &compiled.fom(static_cast<DomainId>(d));
            std::set<std::string> names;
            core::Vector<TaggedSurfaceRegion> regions;
            const auto port_offset = static_cast<core::EntityIndex>(compiled.library().boundary_conditions.size());
            if (fom)
                for (const auto& boundary : component.thermal_boundaries)
                    for (const auto& region : boundary.regions)
                        regions.push_back({region, boundary.condition});
            for (std::size_t p = 0; p < component.ports.size(); ++p) {
                const auto& port = component.ports[p];
                const auto& r = port.region;
                validate_surface_region(r);
                if (port.name.empty() || !names.insert(port.name).second)
                    throw std::invalid_argument("port names must be nonempty and unique within a component");
                for (std::size_t previous = 0; previous < p; ++previous)
                    if (intersects(r, component.ports[previous].region))
                        throw std::invalid_argument("component ports overlap");
                if (fom) {
                    for (const auto& bc : component.thermal_boundaries)
                        for (const auto& region : bc.regions)
                            if (intersects(r, region))
                                throw std::invalid_argument("port overlaps an explicit thermal boundary");
                    for (const auto& bc : definition.fom(static_cast<DomainId>(d)).fluid_boundaries)
                        for (const auto& region : bc.regions)
                            if (intersects(r, region))
                                throw std::invalid_argument("port overlaps a fluid boundary");
                    regions.push_back({r, port_offset + static_cast<core::EntityIndex>(p)});
                    continue;
                }
                if (port.outward_sign != 1 && port.outward_sign != -1)
                    throw std::invalid_argument("ROM outward port sign must be +1 or -1");
                double total = 0;
                const auto order = rom->f.size();
                const auto finite = [](double x) { return std::isfinite(x); };
                for (std::size_t j = 0; j < port.patches.size(); ++j) {
                    const auto& patch = port.patches[j];
                    const auto rect = surface_bounds(patch.rectangle);
                    const auto common = intersect_bounds(rect, surface_bounds(r));
                    if (!positive_bounds(rect) || rect != common || !std::isfinite(patch.half_resistance)
                        || patch.half_resistance <= 0 || !std::isfinite(patch.offset) || patch.trace.size() != order
                        || !std::all_of(patch.trace.begin(), patch.trace.end(), finite)
                        || !std::all_of(patch.center.begin(), patch.center.end(), finite))
                        throw std::invalid_argument("invalid ROM port patch geometry or trace");
                    for (const auto& row : patch.tangential_gradient)
                        if (!row.empty() && (row.size() != order || !std::all_of(row.begin(), row.end(), finite)))
                            throw std::invalid_argument("ROM tangential gradient has invalid dimensions");
                    for (std::size_t prev = 0; prev < j; ++prev)
                        if (positive_bounds(intersect_bounds(rect, surface_bounds(port.patches[prev].rectangle))))
                            throw std::invalid_argument("ROM port patches overlap");
                    total += surface_area(rect);
                    ports[d][p].push_back({static_cast<CellId>(j), rect, port.outward_sign});
                }
                const double required = surface_area(surface_bounds(r));
                if (port.patches.empty() || std::abs(total - required) > 1e-10 * required)
                    throw std::invalid_argument("ROM port patches must exactly cover the port");
            }
            if (!fom) {
                const auto domain = static_cast<DomainId>(d);
                core::Vector<core::Vector<TaggedBounds>> per_port(component.ports.size());
                for (const auto& boundary : component.thermal_boundaries) {
                    (void)definition.library().thermal_boundary(boundary.condition);
                    for (const auto& region : boundary.regions) {
                        validate_surface_region(region);
                        const auto requested = surface_bounds(region);
                        const double requested_area = surface_area(requested);
                        double covered = 0.0;
                        for (std::size_t p = 0; p < component.ports.size(); ++p) {
                            const auto& candidate = component.ports[p];
                            if (region.axis != candidate.region.axis
                                || region.coordinate != candidate.region.coordinate)
                                continue;
                            const auto intersection = intersect_bounds(requested, surface_bounds(candidate.region));
                            if (!positive_bounds(intersection))
                                continue;
                            if (connected[d][p])
                                throw std::invalid_argument("ROM thermal boundary overlaps a connected port");
                            per_port[p].push_back({intersection, boundary.condition});
                            covered += surface_area(intersection);
                        }
                        if (std::abs(covered - requested_area) > 1e-10 * requested_area)
                            throw std::invalid_argument("ROM thermal boundary must be covered by exposed ports");
                    }
                }
                for (std::size_t p = 0; p < component.ports.size(); ++p) {
                    const auto& port = component.ports[p];
                    for (std::size_t j = 0; j < port.patches.size(); ++j)
                        for (const auto& tagged :
                            partition_surface_bounds(surface_bounds(port.patches[j].rectangle), per_port[p])) {
                            Vec3 center {};
                            const auto axis = static_cast<std::size_t>(port.region.axis);
                            center[axis] = port.region.coordinate;
                            std::size_t tangent = 0;
                            for (std::size_t a = 0; a < 3; ++a)
                                if (a != axis) {
                                    center[a] = (tagged.bounds[2 * tangent] + tagged.bounds[2 * tangent + 1]) / 2;
                                    ++tangent;
                                }
                            compiled.rom_boundary_terms.push_back(
                                {domain, static_cast<PortId>(p), j, tagged.tag, center, surface_area(tagged.bounds)});
                        }
                }
                continue;
            }
            const auto id = definition.fom(static_cast<DomainId>(d)).default_thermal_boundary;
            const auto fallback = id;
            const auto projection = project_exterior_surface_regions(fom->discrete, regions, fallback);
            for (const auto& patch : projection) {
                if (patch.tag == core::invalid_id || patch.tag < port_offset) {
                    fom->boundary_faces.push_back({patch.cell, patch.axis, patch.sign, patch.tag, patch.bounds});
                    continue;
                }
                const auto p = patch.tag - port_offset;
                if (!fom->fluid.global_to_fluid.empty() && fom->fluid.global_to_fluid[patch.cell] != core::invalid_id)
                    throw std::invalid_argument("thermal connections currently require solid interface cells");
                ports[d][p].push_back({patch.cell, patch.bounds, patch.sign});
            }
            for (std::size_t p = 0; p < component.ports.size(); ++p) {
                const auto& facets = ports[d][p];
                double covered = 0;
                for (const auto& facet : facets)
                    covered += surface_area(facet.bounds);
                const double required = surface_area(surface_bounds(component.ports[p].region));
                if (facets.empty() || std::abs(covered - required) > 1e-10 * required)
                    throw std::invalid_argument("port is not fully covered by exterior domain faces");
                if (!std::all_of(facets.begin(), facets.end(),
                        [&](const auto& facet) { return facet.sign == facets.front().sign; }))
                    throw std::invalid_argument("port must have a single outward normal");
            }
        }

        auto region = [&](PortReference x) { return definition.component(x.domain).ports.at(x.port).region; };
        for (const auto& spec : definition.connections) {
            const auto& pa = ports.at(spec.a.domain).at(spec.a.port);
            const auto& pb = ports.at(spec.b.domain).at(spec.b.port);
            const auto a = region(spec.a), b = region(spec.b);
            if (a.axis != b.axis || a.coordinate != b.coordinate || surface_bounds(a) != surface_bounds(b)
                || pa.front().sign == pb.front().sign)
                throw std::invalid_argument("connection ports must coincide and have opposing outward normals");
            ThermalConnection conn {{spec.a.domain, spec.a.port}, {spec.b.domain, spec.b.port}, a.axis, false, {}};
            for (const auto& left : pa)
                for (const auto& right : pb) {
                    const auto rect = intersect_bounds(left.bounds, right.bounds);
                    if (!positive_bounds(rect))
                        continue;
                    conn.patches.push_back({left.index, right.index, rect});
                    conn.nonmatching = conn.nonmatching || left.bounds != right.bounds;
                }
            if (conn.patches.empty())
                throw std::invalid_argument("connection has no common interface area");
            conn.sign_a = pa.front().sign;
            compiled.connections.push_back(std::move(conn));
        }
        // Only FOM ports require Connection. A ROM port can instead carry an
        // exterior boundary law; an otherwise unused ROM port is adiabatic.
        for (const auto& field : compiled.layout.fields)
            if (field.quantity == FieldQuantity::Temperature
                && std::find(connected[field.domain].begin(), connected[field.domain].end(), false)
                    != connected[field.domain].end())
                throw std::invalid_argument("every declared FOM port must be connected");

        // A nonmatching ROM connection needs a projected tangential gradient
        // on every participating ROM facet: zeroing NCC is not permissible.
        for (const auto& conn : compiled.connections) {
            if (!conn.nonmatching)
                continue;
            for (std::size_t side = 0; side < 2; ++side) {
                const auto ref = side == 0 ? conn.a : conn.b;
                const auto* rom = std::get_if<RomDomain>(&compiled.components.at(ref.domain).physics);
                if (!rom)
                    continue;
                const auto& port = rom->ports.at(ref.port);
                for (const auto& facet : conn.patches) {
                    const auto index = side == 0 ? facet.a : facet.b;
                    for (const auto& row : port.at(index).tangential_gradient)
                        if (row.size() != rom->f.size())
                            throw std::invalid_argument("nonmatching ROM port requires NCC gradient maps");
                }
            }
        }
    }
} // namespace mhs::compiler
