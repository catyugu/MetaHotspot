#include "compiler/connections.hpp"
#include "compiler/geometry_compiler.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace mhs::compiler {
    namespace {
        using namespace model;
        using Bounds = std::array<double, 4>;
        struct Facet {
            CellId index = core::invalid_id; // FOM cell or ROM port-patch index
            Bounds bounds {};
            std::int8_t sign = 1;
        };
        using Facets = core::Vector<Facet>;

        double area(const Bounds& r) { return (r[1] - r[0]) * (r[3] - r[2]); }
        Bounds overlap(const Bounds& a, const Bounds& b)
        { return {std::max(a[0], b[0]), std::min(a[1], b[1]), std::max(a[2], b[2]), std::min(a[3], b[3])}; }
        bool positive(const Bounds& r) { return r[1] > r[0] && r[3] > r[2]; }
        Bounds bounds(const SurfaceRegion& r)
        { return {r.rectangle.a_min, r.rectangle.a_max, r.rectangle.b_min, r.rectangle.b_max}; }
        Bounds bounds(const RegionRect& r) { return {r.a_min, r.a_max, r.b_min, r.b_max}; }
        bool intersects(const SurfaceRegion& a, const SurfaceRegion& b)
        { return a.axis == b.axis && a.coordinate == b.coordinate && positive(overlap(bounds(a), bounds(b))); }
        void validate_region(const SurfaceRegion& r)
        {
            const auto rect = bounds(r);
            if (static_cast<unsigned>(r.axis) > 2 || !std::isfinite(r.coordinate) || !positive(rect)
                || !std::all_of(rect.begin(), rect.end(), [](double x) { return std::isfinite(x); }))
                throw std::invalid_argument("invalid port surface region");
        }
        bool same_bounds(const Bounds& a, const Bounds& b) { return a == b; }
    } // namespace

    void compile_connections(const model::ModelDefinition& definition, model::CompiledModel& compiled)
    {
        using namespace model;
        const auto ndom = definition.domains.size();
        core::Vector<core::Vector<Facets>> ports(ndom);
        core::Vector<core::Vector<bool>> used(ndom);

        // Full-order port projection and exterior-boundary exclusion.
        for (std::size_t d = 0; d < ndom; ++d) {
            const auto* value = std::get_if<FomDomainDefinition>(&definition.domains[d]);
            if (!value)
                continue;
            const auto& domain = *value;
            auto& fom = compiled.fom(static_cast<DomainId>(d));
            used[d].resize(domain.ports.size(), false);
            ports[d].resize(domain.ports.size());
            std::set<std::string> names;
            for (std::size_t p = 0; p < domain.ports.size(); ++p) {
                const auto& port = domain.ports[p];
                const auto& r = port.region;
                validate_region(r);
                if (port.name.empty() || !names.insert(port.name).second)
                    throw std::invalid_argument("port names must be nonempty and unique within a domain");
                for (std::size_t previous = 0; previous < p; ++previous)
                    if (intersects(r, domain.ports[previous].region))
                        throw std::invalid_argument("domain ports overlap");
                for (const auto& bc : domain.thermal_boundaries)
                    for (const auto& region : bc.regions)
                        if (intersects(r, region))
                            throw std::invalid_argument("port overlaps an explicit thermal boundary");
                for (const auto& bc : domain.fluid_boundaries)
                    for (const auto& region : bc.regions)
                        if (intersects(r, region))
                            throw std::invalid_argument("port overlaps a fluid boundary");
                const auto projection = project_exterior_surface_regions(fom.discrete, {{r, 0}});
                double covered = 0;
                for (const auto& patch : projection) {
                    covered += area(patch.bounds);
                    const auto& fluid = fom.fluid;
                    if (!fluid.global_to_fluid.empty() && fluid.global_to_fluid[patch.cell] != core::invalid_id)
                        throw std::invalid_argument("thermal connections currently require solid interface cells");
                    ports[d].at(p).push_back({patch.cell, patch.bounds, patch.sign});
                }
                if (projection.empty() || std::abs(covered - area(bounds(r))) > 1e-10 * area(bounds(r)))
                    throw std::invalid_argument("port is not fully covered by exterior domain faces");
                if (!std::all_of(projection.begin(), projection.end(),
                        [&](const auto& x) { return x.sign == projection.front().sign; }))
                    throw std::invalid_argument("port must have a single outward normal");
            }
            if (!domain.ports.empty()) {
                core::Vector<TaggedSurfaceRegion> regions;
                for (std::size_t bc = 0; bc < domain.thermal_boundaries.size(); ++bc)
                    for (const auto& region : domain.thermal_boundaries[bc].regions)
                        regions.push_back({region, bc});
                for (const auto& port : domain.ports)
                    regions.push_back({port.region, core::invalid_id});
                fom.boundary_faces.clear();
                for (const auto& patch :
                    project_exterior_surface_regions(fom.discrete, regions, fom.boundary_conditions.size() - 1))
                    if (patch.tag != core::invalid_id)
                        fom.boundary_faces.push_back({patch.cell, patch.axis, patch.sign,
                            static_cast<BoundaryConditionId>(patch.tag), patch.bounds});
            }
        }

        // ROM ports describe only their physical boundary; their interior is
        // represented exclusively by the reduced operators.
        for (std::size_t d = 0; d < ndom; ++d) {
            const auto* value = std::get_if<RomDomain>(&compiled.domains[d]);
            if (!value)
                continue;
            const auto& rom = *value;
            used[d].resize(rom.ports.size(), false);
            ports[d].resize(rom.ports.size());
            std::set<std::string> names;
            for (std::size_t p = 0; p < rom.ports.size(); ++p) {
                const auto& port = rom.ports[p];
                validate_region(port.region);
                if (port.name.empty() || !names.insert(port.name).second)
                    throw std::invalid_argument("ROM port names must be nonempty and unique");
                if (port.outward_sign != 1 && port.outward_sign != -1)
                    throw std::invalid_argument("ROM outward port sign must be +1 or -1");
                for (std::size_t prev = 0; prev < p; ++prev)
                    if (intersects(port.region, rom.ports[prev].region))
                        throw std::invalid_argument("ROM ports overlap");
                double total = 0;
                for (std::size_t j = 0; j < port.patches.size(); ++j) {
                    const auto& patch = port.patches[j];
                    const auto rect = bounds(patch.rectangle);
                    const auto common = overlap(rect, bounds(port.region));
                    if (!positive(rect) || !same_bounds(rect, common) || !std::isfinite(patch.half_resistance)
                        || patch.half_resistance <= 0 || !std::isfinite(patch.offset) || patch.trace.size() != rom.order
                        || !std::all_of(
                            patch.trace.begin(), patch.trace.end(), [](double x) { return std::isfinite(x); })
                        || !std::all_of(
                            patch.center.begin(), patch.center.end(), [](double x) { return std::isfinite(x); }))
                        throw std::invalid_argument("invalid ROM port patch geometry or trace");
                    for (const auto& row : patch.tangential_gradient)
                        if (!row.empty()
                            && (row.size() != rom.order
                                || !std::all_of(row.begin(), row.end(), [](double x) { return std::isfinite(x); })))
                            throw std::invalid_argument("ROM tangential gradient has invalid dimensions");
                    for (std::size_t prev = 0; prev < j; ++prev)
                        if (positive(overlap(rect, bounds(port.patches[prev].rectangle))))
                            throw std::invalid_argument("ROM port patches overlap");
                    total += area(rect);
                    ports[d].at(p).push_back({static_cast<CellId>(j), rect, port.outward_sign});
                }
                if (port.patches.empty()
                    || std::abs(total - area(bounds(port.region))) > 1e-10 * area(bounds(port.region)))
                    throw std::invalid_argument("ROM port patches must exactly cover the port");
            }
        }

        auto region = [&](PortReference x) -> SurfaceRegion {
            return std::visit(
                [&](const auto& domain) { return domain.ports.at(x.port).region; }, definition.domains.at(x.domain));
        };
        for (const auto& spec : definition.connections) {
            if (spec.a.domain == spec.b.domain || spec.a.domain >= ndom || spec.b.domain >= ndom)
                throw std::invalid_argument("connection must join two different domains");
            const auto& pa = ports.at(spec.a.domain).at(spec.a.port);
            const auto& pb = ports.at(spec.b.domain).at(spec.b.port);
            if (used.at(spec.a.domain).at(spec.a.port) || used.at(spec.b.domain).at(spec.b.port))
                throw std::invalid_argument("port is connected more than once");
            const auto a = region(spec.a), b = region(spec.b);
            if (a.axis != b.axis || a.coordinate != b.coordinate || bounds(a) != bounds(b)
                || pa.front().sign == pb.front().sign)
                throw std::invalid_argument("connection ports must coincide and have opposing outward normals");
            ThermalConnection conn {spec.a, spec.b, a.axis, false, {}};
            for (const auto& left : pa)
                for (const auto& right : pb) {
                    const auto rect = overlap(left.bounds, right.bounds);
                    if (!positive(rect))
                        continue;
                    conn.patches.push_back({left.index, right.index, rect});
                    conn.nonmatching = conn.nonmatching || left.bounds != right.bounds;
                }
            if (conn.patches.empty())
                throw std::invalid_argument("connection has no common interface area");
            conn.sign_a = pa.front().sign;
            used[spec.a.domain][spec.a.port] = used[spec.b.domain][spec.b.port] = true;
            compiled.connections.push_back(std::move(conn));
        }
        // Only FOM ports require Connection. A ROM port can instead carry an
        // exterior boundary law; an otherwise unused ROM port is adiabatic.
        for (std::size_t d = 0; d < ndom; ++d)
            if (std::holds_alternative<FomDomainDefinition>(definition.domains[d])
                && std::find(used[d].begin(), used[d].end(), false) != used[d].end())
                throw std::invalid_argument("every declared FOM port must be connected");

        // A nonmatching ROM connection needs a projected tangential gradient
        // on every participating ROM facet: zeroing NCC is not permissible.
        for (const auto& conn : compiled.connections) {
            if (!conn.nonmatching)
                continue;
            for (std::size_t side = 0; side < 2; ++side) {
                const auto ref = side == 0 ? conn.a : conn.b;
                const auto* rom = std::get_if<RomDomain>(&compiled.domains.at(ref.domain));
                if (!rom)
                    continue;
                const auto& port = rom->ports.at(ref.port);
                for (const auto& facet : conn.patches) {
                    const auto index = side == 0 ? facet.a : facet.b;
                    for (const auto& row : port.patches.at(index).tangential_gradient)
                        if (row.size() != rom->order)
                            throw std::invalid_argument("nonmatching ROM port requires NCC gradient maps");
                }
            }
        }
    }
} // namespace mhs::compiler
