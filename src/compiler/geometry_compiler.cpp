#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mhs::compiler {

    constexpr double EPS = mhs::model::geometry_eps;

    mhs::core::Vector<SurfaceTag> compile_exterior_facets(mhs::model::ConformalMesh& mesh,
        std::span<const TaggedSurfaceRegion> thermal, std::span<const TaggedSurfaceRegion> fluid,
        SurfaceTag thermal_fallback)
    {
        using namespace mhs::model;
        mhs::core::Vector<SurfaceTag> fluid_tags;
        const std::array regions {thermal, fluid};
        mhs::core::Vector<ExteriorFacet> refined;
        refined.reserve(mesh.exterior_facets.size());
        mesh.boundary_offsets.assign(mesh.cell_count() + 1, 0);
        for (const auto& exterior : mesh.exterior_facets) {
            const auto& facet = exterior.geometry;
            std::array<mhs::core::Vector<const TaggedSurfaceRegion*>, 2> candidates;
            auto append = [&](const Facet& piece) {
                const auto a = std::midpoint(piece.bounds[0], piece.bounds[1]);
                const auto b = std::midpoint(piece.bounds[2], piece.bounds[3]);
                auto tag = [&](std::size_t physics, SurfaceTag fallback) {
                    for (const auto* candidate : candidates[physics] | std::views::reverse)
                        if (rect_contains(candidate->region.bounds, a, b))
                            return candidate->tag;
                    return fallback;
                };
                if (refined.size() == mhs::core::invalid_id)
                    throw std::length_error("too many exterior facets");
                refined.push_back({piece, exterior.owner, tag(0, thermal_fallback)});
                fluid_tags.push_back(tag(1, mhs::core::invalid_id));
                ++mesh.boundary_offsets[exterior.owner + std::size_t {1}];
            };
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
                    auto piece = facet;
                    piece.bounds = {a[u - 1], a[u], b[v - 1], b[v]};
                    append(piece);
                }
        }
        std::partial_sum(mesh.boundary_offsets.begin(), mesh.boundary_offsets.end(), mesh.boundary_offsets.begin());
        mesh.exterior_facets = std::move(refined);
        return fluid_tags;
    }

} // namespace mhs::compiler
