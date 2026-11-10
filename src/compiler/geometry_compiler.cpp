#include "compiler/geometry_compiler.hpp"
#include "model/constants.hpp"

#include <algorithm>
#include <cmath>

namespace mhs::compiler {
    namespace {
        using namespace mhs::core;
        using namespace mhs::model;
        constexpr double EPS = geometry_eps;

        struct IndexedRegion {
            const TaggedSurfaceRegion* tagged;
            std::size_t group, order;
        };

        bool point_in_region(const SurfaceRegion& region, double a, double b)
        {
            const auto& r = region.rectangle;
            return a >= r.a_min - EPS && a <= r.a_max + EPS && b >= r.b_min - EPS && b <= r.b_max + EPS;
        }

        void sort_unique(Vector<double>& cuts)
        {
            std::sort(cuts.begin(), cuts.end());
            cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        }
    } // namespace

    mhs::core::Vector<mhs::core::Vector<SurfaceTag>> partition_exterior_facets(
        mhs::model::DiscreteMesh& mesh, std::span<const SurfaceRegionGroup> groups)
    {
        std::array<Vector<IndexedRegion>, 3> planes;
        std::size_t order = 0;
        for (std::size_t g = 0; g < groups.size(); ++g)
            for (const auto& region : groups[g].regions)
                planes[static_cast<std::size_t>(region.region.axis)].push_back({&region, g, order++});
        for (auto& regions : planes)
            std::sort(regions.begin(), regions.end(),
                [](const auto& a, const auto& b) { return a.tagged->region.coordinate < b.tagged->region.coordinate; });

        Vector<Vector<SurfaceTag>> assigned(groups.size());
        for (auto& tags : assigned)
            tags.reserve(mesh.facets.size());
        Vector<Facet> refined;
        refined.reserve(mesh.facets.size());
        // Reuse per-facet scratch storage, including when a model has many planes.
        Vector<const IndexedRegion*> candidates;
        Vector<double> a, b;
        for (const auto& facet : mesh.facets) {
            if (!facet.exterior()) {
                refined.push_back(facet);
                for (auto& tags : assigned)
                    tags.push_back(invalid_id);
                continue;
            }
            candidates.clear();
            a.clear();
            b.clear();
            const auto& regions = planes[static_cast<std::size_t>(facet.axis)];
            auto it = std::lower_bound(regions.begin(), regions.end(), facet.coordinate - EPS,
                [](const auto& indexed, double coordinate) { return indexed.tagged->region.coordinate < coordinate; });
            for (; it != regions.end() && it->tagged->region.coordinate <= facet.coordinate + EPS; ++it) {
                const auto& region = it->tagged->region;
                if (std::abs(region.coordinate - facet.coordinate) >= EPS)
                    continue;
                const auto& r = region.rectangle;
                // Keep tolerance-only matches for tagging even when they add no
                // geometric cuts. This preserves the existing closed-region rule.
                if (r.a_max < facet.bounds[0] - EPS || r.a_min > facet.bounds[1] + EPS
                    || r.b_max < facet.bounds[2] - EPS || r.b_min > facet.bounds[3] + EPS)
                    continue;
                candidates.push_back(&*it);
                if (r.a_max <= facet.bounds[0] || r.a_min >= facet.bounds[1] || r.b_max <= facet.bounds[2]
                    || r.b_min >= facet.bounds[3])
                    continue;
                if (a.empty()) {
                    a = {facet.bounds[0], facet.bounds[1]};
                    b = {facet.bounds[2], facet.bounds[3]};
                }
                a.push_back(std::clamp(r.a_min, facet.bounds[0], facet.bounds[1]));
                a.push_back(std::clamp(r.a_max, facet.bounds[0], facet.bounds[1]));
                b.push_back(std::clamp(r.b_min, facet.bounds[2], facet.bounds[3]));
                b.push_back(std::clamp(r.b_max, facet.bounds[2], facet.bounds[3]));
            }
            // Plane sorting must not change the caller's precedence order.
            std::sort(candidates.begin(), candidates.end(),
                [](const auto* lhs, const auto* rhs) { return lhs->order < rhs->order; });
            auto append = [&](const Facet& piece) {
                refined.push_back(piece);
                for (std::size_t g = 0; g < groups.size(); ++g)
                    assigned[g].push_back(groups[g].fallback);
                const double u = (piece.bounds[0] + piece.bounds[1]) / 2;
                const double v = (piece.bounds[2] + piece.bounds[3]) / 2;
                for (const auto* candidate : candidates)
                    if (point_in_region(candidate->tagged->region, u, v))
                        assigned[candidate->group].back() = candidate->tagged->tag;
            };
            if (a.empty()) {
                append(facet);
                continue;
            }
            sort_unique(a);
            sort_unique(b);
            for (std::size_t u = 1; u < a.size(); ++u)
                for (std::size_t v = 1; v < b.size(); ++v) {
                    auto piece = facet;
                    piece.bounds = {a[u - 1], a[u], b[v - 1], b[v]};
                    append(piece);
                }
        }
        mesh.facets = std::move(refined);
        return assigned;
    }
} // namespace mhs::compiler
