#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
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

    bool project_cell_regions(mhs::model::Model& model, const mhs::model::CellGeometry& cell,
        const mhs::core::Vector<mhs::model::RegionSpec>& regions,
        const mhs::core::Vector<mhs::model::MaterialAssignment>& materials,
        const mhs::core::Vector<mhs::model::SourceAssignment>& sources)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        std::array<mhs::core::Vector<double>, 3> cuts;
        for (int a = 0; a < 3; ++a)
            cuts[a] = {cell.min[a], cell.max[a]};
        mhs::core::Vector<RegionId> candidates;
        // Geometry is queried once per cell, shared by material and source assignments.
        for (std::size_t r = 0; r < regions.size(); ++r) {
            bool relevant = false;
            for (const auto& box : regions[r].geometry) {
                bool overlaps = true;
                for (int a = 0; a < 3; ++a)
                    overlaps = overlaps && box.max[a] > cell.min[a] && box.min[a] < cell.max[a];
                if (!overlaps)
                    continue;
                relevant = true;
                for (int a = 0; a < 3; ++a)
                    for (const auto v : {box.min[a], box.max[a]})
                        if (v > cell.min[a] + EPS && v < cell.max[a] - EPS)
                            cuts[a].push_back(v);
            }
            if (relevant)
                candidates.push_back(static_cast<RegionId>(r));
        }
        if (candidates.empty())
            return false;
        for (auto& axis : cuts) {
            std::sort(axis.begin(), axis.end());
            axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
        }
        mhs::core::Vector<bool> inside(regions.size(), false);
        std::optional<MaterialId> cell_material;
        const auto cell_id = static_cast<CellId>(model.discrete.cells.size());
        for (std::size_t x = 1; x < cuts[0].size(); ++x)
            for (std::size_t y = 1; y < cuts[1].size(); ++y)
                for (std::size_t z = 1; z < cuts[2].size(); ++z) {
                    const Vec3 p {(cuts[0][x - 1] + cuts[0][x]) / 2, (cuts[1][y - 1] + cuts[1][y]) / 2,
                        (cuts[2][z - 1] + cuts[2][z]) / 2};
                    for (const auto r : candidates) {
                        inside[r] = false;
                        for (const auto& box : regions[r].geometry) {
                            bool contains = true;
                            for (int a = 0; a < 3; ++a)
                                contains = contains && p[a] >= box.min[a] - EPS && p[a] <= box.max[a] + EPS;
                            if (contains)
                                inside[r] = box.operation == GeometryOperation::Add;
                        }
                    }
                    MaterialId material = invalid_id;
                    for (const auto& assignment : materials)
                        if (inside[assignment.region])
                            material = assignment.material;
                    if (cell_material && *cell_material != material)
                        throw std::invalid_argument("cell " + std::to_string(cell_id)
                            + " crosses a material or domain boundary; align the mesh with the physical regions");
                    cell_material = material;
                    if (material == invalid_id)
                        continue;
                    const double volume
                        = (cuts[0][x] - cuts[0][x - 1]) * (cuts[1][y] - cuts[1][y - 1]) * (cuts[2][z] - cuts[2][z - 1]);
                    // Replace is an explicit ordered field operation used by block adapters.
                    // Independent sources use Add and may share the same region.
                    std::size_t begin = 0;
                    for (std::size_t i = 0; i < sources.size(); ++i)
                        if (inside[sources[i].region] && sources[i].operation == SourceOperation::Replace)
                            begin = i;
                    for (std::size_t i = begin; i < sources.size(); ++i)
                        if (inside[sources[i].region] && sources[i].source != invalid_id)
                            model.source_terms.push_back({cell_id, sources[i].source, volume, p});
                }
        if (!cell_material || *cell_material == invalid_id)
            return false;
        model.cell_material.push_back(*cell_material);
        return true;
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
