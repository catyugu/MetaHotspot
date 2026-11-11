#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mhs::compiler {

    constexpr double EPS = mhs::model::geometry_eps;

    namespace {

        bool point_in_region(const mhs::model::SurfaceRegion& region, double a, double b)
        {
            const auto& r = region.rectangle;
            return a >= r.a_min - EPS && a <= r.a_max + EPS && b >= r.b_min - EPS && b <= r.b_max + EPS;
        }

    } // namespace

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

    bool compile_cell(mhs::model::Model& model, const mhs::model::CellGeometry& cell,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        std::array<Vector<double>, 3> cuts;
        for (int a = 0; a < 3; ++a)
            cuts[a] = {cell.min[a], cell.max[a]};
        for (const auto& volume : volumes)
            for (const auto& box : volume.geometry) {
                bool intersects = true;
                for (int a = 0; a < 3; ++a)
                    intersects = intersects && box.max[a] > cell.min[a] && box.min[a] < cell.max[a];
                if (!intersects)
                    continue;
                for (int a = 0; a < 3; ++a)
                    for (double p : {box.min[a], box.max[a]})
                        if (p > cell.min[a] + geometry_eps && p < cell.max[a] - geometry_eps)
                            cuts[a].push_back(p);
            }
        for (auto& axis : cuts) {
            std::sort(axis.begin(), axis.end());
            axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
        }
        MaterialId material = invalid_id;
        bool active = false, inactive = false;
        Vector<SourceTerm> sources;
        const auto id = static_cast<CellId>(model.cells.size());
        for (std::size_t x = 1; x < cuts[0].size(); ++x)
            for (std::size_t y = 1; y < cuts[1].size(); ++y)
                for (std::size_t z = 1; z < cuts[2].size(); ++z) {
                    const CellGeometry piece {
                        {cuts[0][x - 1], cuts[1][y - 1], cuts[2][z - 1]}, {cuts[0][x], cuts[1][y], cuts[2][z]}};
                    const auto point = piece.center();
                    const auto owner = classify_point(volumes, point);
                    if (!owner) {
                        inactive = true;
                        continue;
                    }
                    const auto& volume = volumes[*owner];
                    if (active && material != volume.material)
                        throw std::invalid_argument("mesh cell crosses an effective material interface");
                    active = true;
                    material = volume.material;
                    if (volume.source != invalid_id)
                        sources.push_back({id, volume.source, piece.volume(), point});
                }
        if (active && inactive)
            throw std::invalid_argument("mesh cell crosses an effective active-domain boundary");
        if (!active)
            return false;
        model.cells.push_back({material, model.material_table[material].kind == MaterialKind::Fluid});
        model.source_terms.insert(model.source_terms.end(), sources.begin(), sources.end());
        return true;
    }

    void refine_boundary_facets(mhs::core::Vector<mhs::model::BoundaryFacet>& facets,
        const mhs::core::Vector<mhs::model::SurfaceRegion>& regions)
    {
        using namespace mhs::model;
        mhs::core::Vector<BoundaryFacet> refined;
        refined.reserve(facets.size());
        for (const auto& facet : facets) {
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
        facets = std::move(refined);
    }

    mhs::core::Vector<SurfaceTag> assign_boundary_regions(const mhs::core::Vector<mhs::model::BoundaryFacet>& facets,
        const mhs::core::Vector<TaggedSurfaceRegion>& regions, SurfaceTag fallback)
    {
        mhs::core::Vector<SurfaceTag> assigned(facets.size(), mhs::core::invalid_id);
        for (std::size_t id = 0; id < facets.size(); ++id) {
            const auto& facet = facets[id];
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
