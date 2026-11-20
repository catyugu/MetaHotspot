#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace mhs::compiler {

    constexpr double EPS = mhs::model::geometry_eps;

    std::optional<std::size_t> classify_point(
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes, const mhs::model::Vec3& p)
    {
        for (std::size_t i = volumes.size(); i-- > 0;) {
            bool inside = false;
            for (const auto& box : volumes[i].geometry) {
                bool contains = true;
                for (int a = 0; a < 3; ++a)
                    contains = contains && p[a] >= box.min[a] - EPS && p[a] <= box.max[a] + EPS;
                if (contains)
                    inside = box.operation == mhs::model::GeometryOperation::Add;
            }
            if (inside)
                return i;
        }
        return std::nullopt;
    }

    std::array<mhs::core::Vector<SurfaceTag>, 2> compile_exterior_facets(mhs::model::ConformalMesh& mesh,
        std::span<const TaggedSurfaceRegion> thermal, std::span<const TaggedSurfaceRegion> fluid,
        SurfaceTag thermal_fallback)
    {
        using namespace mhs::model;
        std::array<mhs::core::Vector<SurfaceTag>, 2> assigned;
        const std::array regions {thermal, fluid};
        const std::array fallback {thermal_fallback, mhs::core::invalid_id};
        mhs::core::Vector<Facet> refined;
        refined.reserve(mesh.facets.size());
        mhs::core::Vector<CellFacet> topology;
        topology.reserve(mesh.facets.size());
        for (std::size_t id = 0; id < mesh.facets.size(); ++id) {
            const auto& facet = mesh.facets[id];
            const auto& cells = mesh.cell_facets[id];
            std::array<mhs::core::Vector<const TaggedSurfaceRegion*>, 2> candidates;
            auto append = [&](const Facet& piece) {
                refined.push_back(piece);
                topology.push_back(cells);
                const auto a = std::midpoint(piece.bounds[0], piece.bounds[1]);
                const auto b = std::midpoint(piece.bounds[2], piece.bounds[3]);
                for (std::size_t physics = 0; physics < assigned.size(); ++physics) {
                    auto tag = cells.exterior() ? fallback[physics] : mhs::core::invalid_id;
                    for (auto it = candidates[physics].rbegin(); it != candidates[physics].rend(); ++it)
                        if (rect_contains((*it)->region.bounds, a, b)) {
                            tag = (*it)->tag;
                            break;
                        }
                    assigned[physics].push_back(tag);
                }
            };
            if (!cells.exterior()) {
                append(facet);
                continue;
            }
            mhs::core::Vector<double> a, b;
            for (std::size_t physics = 0; physics < regions.size(); ++physics)
                for (const auto& tagged : regions[physics]) {
                    const auto& region = tagged.region;
                    if (region.axis != facet.axis || std::abs(region.coordinate - facet.coordinate) >= EPS)
                        continue;
                    const auto hit = rect_intersection(region.bounds, facet.bounds);
                    if (!hit)
                        continue;
                    candidates[physics].push_back(&tagged);
                    if (a.empty()) {
                        a = {facet.bounds[0], facet.bounds[1]};
                        b = {facet.bounds[2], facet.bounds[3]};
                    }
                    a.insert(a.end(), {(*hit)[0], (*hit)[1]});
                    b.insert(b.end(), {(*hit)[2], (*hit)[3]});
                }
            if (a.empty()) {
                append(facet);
                continue;
            }
            auto unique = [](auto& cuts) {
                std::sort(cuts.begin(), cuts.end());
                cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
            };
            unique(a);
            unique(b);
            for (std::size_t u = 1; u < a.size(); ++u)
                for (std::size_t v = 1; v < b.size(); ++v) {
                    if (a[u] <= a[u - 1] || b[v] <= b[v - 1])
                        continue;
                    auto piece = facet;
                    piece.bounds = {a[u - 1], a[u], b[v - 1], b[v]};
                    append(piece);
                }
        }
        mesh.facets = std::move(refined);
        mesh.cell_facets = std::move(topology);
        build_cell_facet_incidence(mesh);
        return assigned;
    }

} // namespace mhs::compiler
