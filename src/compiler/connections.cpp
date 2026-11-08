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
        double area(const Bounds& r) { return (r[1] - r[0]) * (r[3] - r[2]); }
        Bounds overlap(const Bounds& a, const Bounds& b)
        { return {std::max(a[0], b[0]), std::min(a[1], b[1]), std::max(a[2], b[2]), std::min(a[3], b[3])}; }
        bool positive(const Bounds& r) { return r[1] > r[0] && r[3] > r[2]; }
        Bounds bounds(const SurfaceRegion& r)
        { return {r.rectangle.a_min, r.rectangle.a_max, r.rectangle.b_min, r.rectangle.b_max}; }
        bool intersects(const SurfaceRegion& a, const SurfaceRegion& b)
        { return a.axis == b.axis && a.coordinate == b.coordinate && positive(overlap(bounds(a), bounds(b))); }
    }

    void compile_connections(const model::ModelDefinition& definition, model::CompiledModel& compiled)
    {
        using namespace model;
        using Projection = core::Vector<ProjectedSurfacePatch>;
        core::Vector<core::Vector<Projection>> ports(definition.domains.size());
        core::Vector<core::Vector<bool>> used(definition.domains.size());
        for (std::size_t d = 0; d < definition.domains.size(); ++d) {
            const auto& domain = definition.domains[d];
            used[d].resize(domain.ports.size(), false);
            std::set<std::string> names;
            for (std::size_t p = 0; p < domain.ports.size(); ++p) {
                const auto& port = domain.ports[p];
                const auto& r = port.region;
                const auto rect = bounds(r);
                if (port.name.empty() || !names.insert(port.name).second)
                    throw std::invalid_argument("port names must be nonempty and unique within a domain");
                if (static_cast<unsigned>(r.axis) > 2 || !std::isfinite(r.coordinate) || !positive(rect)
                    || !std::all_of(rect.begin(), rect.end(), [](double x) { return std::isfinite(x); }))
                    throw std::invalid_argument("invalid port surface region");
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
                auto projection = project_exterior_surface_regions(compiled.domains[d].discrete, {{r, 0}});
                double covered = 0;
                for (const auto& patch : projection) {
                    covered += area(patch.bounds);
                    const auto& fluid = compiled.domains[d].fluid;
                    if (!fluid.global_to_fluid.empty() && fluid.global_to_fluid[patch.cell] != core::invalid_id)
                        throw std::invalid_argument("thermal connections currently require solid interface cells");
                }
                if (projection.empty() || std::abs(covered - area(rect)) > 1e-10 * area(rect))
                    throw std::invalid_argument("port is not fully covered by exterior domain faces");
                if (!std::all_of(projection.begin(), projection.end(),
                        [&](const auto& x) { return x.sign == projection.front().sign; }))
                    throw std::invalid_argument("port must have a single outward normal");
                ports[d].push_back(std::move(projection));
            }
            if (!domain.ports.empty()) {
                core::Vector<TaggedSurfaceRegion> regions;
                for (std::size_t bc = 0; bc < domain.thermal_boundaries.size(); ++bc)
                    for (const auto& region : domain.thermal_boundaries[bc].regions)
                        regions.push_back({region, bc});
                for (const auto& port : domain.ports)
                    regions.push_back({port.region, core::invalid_id});
                auto& fom = compiled.domains[d];
                fom.boundary_faces.clear();
                for (const auto& patch :
                    project_exterior_surface_regions(fom.discrete, regions, fom.boundary_conditions.size() - 1))
                    if (patch.tag != core::invalid_id)
                        fom.boundary_faces.push_back({patch.cell, patch.axis, patch.sign,
                            static_cast<BoundaryConditionId>(patch.tag), patch.bounds});
            }
        }
        for (const auto& spec : definition.connections) {
            if (spec.a.domain == spec.b.domain)
                throw std::invalid_argument("connection must join distinct domains");
            const auto& a = definition.domains.at(spec.a.domain).ports.at(spec.a.port).region;
            const auto& b = definition.domains.at(spec.b.domain).ports.at(spec.b.port).region;
            if (used.at(spec.a.domain).at(spec.a.port) || used.at(spec.b.domain).at(spec.b.port))
                throw std::invalid_argument("port is connected more than once");
            const auto& pa = ports[spec.a.domain][spec.a.port];
            const auto& pb = ports[spec.b.domain][spec.b.port];
            if (a.axis != b.axis || a.coordinate != b.coordinate || bounds(a) != bounds(b)
                || pa.front().sign == pb.front().sign)
                throw std::invalid_argument("connection ports must coincide and have opposing outward normals");
            ThermalConnection connection {spec.a, spec.b, a.axis, false, {}};
            // Common integration patches are independent of either side's geometric face IDs.
            for (const auto& left : pa)
                for (const auto& right : pb) {
                    const auto rect = overlap(left.bounds, right.bounds);
                    if (!positive(rect))
                        continue;
                    connection.patches.push_back({left.cell, right.cell, rect});
                    connection.nonmatching = connection.nonmatching || left.bounds != right.bounds;
                }
            used[spec.a.domain][spec.a.port] = used[spec.b.domain][spec.b.port] = true;
            compiled.connections.push_back(std::move(connection));
        }
        for (const auto& domain : used)
            if (std::find(domain.begin(), domain.end(), false) != domain.end())
                throw std::invalid_argument("every declared port must be connected");
    }
}
