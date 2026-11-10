#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"

#include <algorithm>
#include <cmath>

namespace mhs::compiler {

    constexpr double EPS = mhs::model::geometry_eps;

    namespace {

        bool point_in_region(const mhs::model::SurfaceRegion& region, double a, double b)
        {
            const auto& r = region.rectangle;
            return a >= r.a_min - EPS && a <= r.a_max + EPS && b >= r.b_min - EPS && b <= r.b_max + EPS;
        }

    } // namespace

    bool geometry_contains(const mhs::model::VolumeGeometry& geometry, const mhs::model::Vec3& point)
    {
        bool inside = false;
        for (const auto& box : geometry.operations) {
            bool contains = true;
            for (int a = 0; a < 3; ++a)
                contains = contains && point[a] >= box.min[a] - EPS && point[a] <= box.max[a] + EPS;
            if (contains)
                inside = box.operation == mhs::model::GeometryOperation::Add;
        }
        return inside;
    }

    void project_volume_sources(
        mhs::model::Model& model, const mhs::core::Vector<mhs::model::VolumeSourceSpec>& sources)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        model.source_terms.clear();
        for (std::size_t id = 0; id < sources.size(); ++id) {
            const auto& geometry = sources[id].geometry;
            // An additive box is required for nonempty source geometry.
            Vec3 lo {}, hi {};
            bool has_add = false;
            for (const auto& box : geometry.operations) {
                if (box.operation != GeometryOperation::Add)
                    continue;
                if (!has_add) {
                    lo = box.min;
                    hi = box.max;
                }
                else
                    for (int a = 0; a < 3; ++a) {
                        lo[a] = std::min(lo[a], box.min[a]);
                        hi[a] = std::max(hi[a], box.max[a]);
                    }
                has_add = true;
            }
            if (!has_add)
                continue;

            for (std::size_t c = 0; c < model.discrete.cells.size(); ++c) {
                const auto& cell = model.discrete.cells[c];
                bool overlaps = true;
                for (int a = 0; a < 3; ++a)
                    overlaps = overlaps && hi[a] > cell.min[a] && lo[a] < cell.max[a];
                if (!overlaps)
                    continue;

                std::array<Vector<double>, 3> cuts;
                for (const auto& box : geometry.operations) {
                    bool intersects = true;
                    for (int a = 0; a < 3; ++a)
                        intersects = intersects && box.max[a] > cell.min[a] && box.min[a] < cell.max[a];
                    if (intersects)
                        for (int a = 0; a < 3; ++a)
                            for (double v : {box.min[a], box.max[a]})
                                if (v > cell.min[a] + EPS && v < cell.max[a] - EPS)
                                    cuts[a].push_back(v);
                }

                if (std::all_of(cuts.begin(), cuts.end(), [](const auto& v) { return v.empty(); })) {
                    if (geometry_contains(geometry, cell.center()))
                        model.source_terms.push_back(
                            {static_cast<CellId>(c), static_cast<SourceId>(id), cell.volume(), cell.center()});
                    continue;
                }
                for (int a = 0; a < 3; ++a) {
                    cuts[a].push_back(cell.min[a]);
                    cuts[a].push_back(cell.max[a]);
                    std::sort(cuts[a].begin(), cuts[a].end());
                    cuts[a].erase(std::unique(cuts[a].begin(), cuts[a].end()), cuts[a].end());
                }
                for (std::size_t x = 1; x < cuts[0].size(); ++x)
                    for (std::size_t y = 1; y < cuts[1].size(); ++y)
                        for (std::size_t z = 1; z < cuts[2].size(); ++z) {
                            const Vec3 p {(cuts[0][x - 1] + cuts[0][x]) / 2,
                                (cuts[1][y - 1] + cuts[1][y]) / 2, (cuts[2][z - 1] + cuts[2][z]) / 2};
                            if (!geometry_contains(geometry, p))
                                continue;
                            const double weight = (cuts[0][x] - cuts[0][x - 1]) * (cuts[1][y] - cuts[1][y - 1])
                                * (cuts[2][z] - cuts[2][z - 1]);
                            model.source_terms.push_back({static_cast<CellId>(c), static_cast<SourceId>(id), weight, p});
                        }
            }
        }
    }

    void refine_exterior_facets(
        mhs::model::DiscreteMesh& mesh, const mhs::core::Vector<mhs::model::SurfaceRegion>& regions)
    {
        using namespace mhs::model;
        mhs::core::Vector<Facet> refined;
        refined.reserve(mesh.facets.size());
        for (const auto& facet : mesh.facets) {
            if (!facet.exterior()) {
                refined.push_back(facet);
                continue;
            }
            mhs::core::Vector<double> a, b;
            for (const auto& region : regions) {
                if (region.axis != facet.axis || std::abs(region.coordinate - facet.coordinate) >= EPS)
                    continue;
                const auto& r = region.rectangle;
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
            if (a.empty()) {
                refined.push_back(facet);
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
                    refined.push_back(piece);
                }
        }
        mesh.facets = std::move(refined);
    }

    mhs::core::Vector<SurfaceTag> assign_exterior_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<TaggedSurfaceRegion>& regions, SurfaceTag fallback)
    {
        mhs::core::Vector<SurfaceTag> assigned(mesh.facets.size(), mhs::core::invalid_id);
        for (std::size_t id = 0; id < mesh.facets.size(); ++id) {
            const auto& facet = mesh.facets[id];
            if (!facet.exterior())
                continue;
            assigned[id] = fallback;
            const auto a = (facet.bounds[0] + facet.bounds[1]) / 2;
            const auto b = (facet.bounds[2] + facet.bounds[3]) / 2;
            for (auto it = regions.rbegin(); it != regions.rend(); ++it) {
                if (it->region.axis == facet.axis && std::abs(it->region.coordinate - facet.coordinate) < EPS
                    && point_in_region(it->region, a, b)) {
                    assigned[id] = it->tag;
                    break;
                }
            }
        }
        return assigned;
    }

} // namespace mhs::compiler
