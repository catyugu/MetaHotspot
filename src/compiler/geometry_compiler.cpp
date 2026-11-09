#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"

#include "expression/expression.hpp"
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

    void project_cell_regions(mhs::model::Model& model, const mhs::core::Vector<mhs::model::VolumeSpec>& volumes)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        model.cell_material.clear();
        model.source_terms.clear();
        for (std::size_t c = 0; c < model.discrete.cells.size(); ++c) {
            const auto& cell = model.discrete.cells[c];
            std::array<mhs::core::Vector<double>, 3> cuts;
            for (int a = 0; a < 3; ++a)
                cuts[a] = {cell.min[a], cell.max[a]};
            auto cut = [&](int a, double v) {
                if (v > cell.min[a] + EPS && v < cell.max[a] - EPS)
                    cuts[a].push_back(v);
            };
            for (const auto& volume : volumes)
                for (const auto& box : volume.geometry) {
                    bool overlaps = true;
                    for (int a = 0; a < 3; ++a)
                        overlaps = overlaps && box.max[a] > cell.min[a] && box.min[a] < cell.max[a];
                    if (overlaps)
                        for (int a = 0; a < 3; ++a) {
                            cut(a, box.min[a]);
                            cut(a, box.max[a]);
                        }
                }
            for (auto& v : cuts) {
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
            }
            MaterialId material = mhs::core::invalid_id;
            for (std::size_t x = 1; x < cuts[0].size(); ++x)
                for (std::size_t y = 1; y < cuts[1].size(); ++y)
                    for (std::size_t z = 1; z < cuts[2].size(); ++z) {
                        Vec3 p {(cuts[0][x - 1] + cuts[0][x]) / 2, (cuts[1][y - 1] + cuts[1][y]) / 2,
                            (cuts[2][z - 1] + cuts[2][z]) / 2};
                        const auto location = classify_point(volumes, p);
                        // A block edge can cut a control volume and leave a piece
                        // outside every block. Such a piece carries no source.
                        if (!location)
                            continue;
                        const auto& volume_spec = volumes[*location];
                        material = volume_spec.material;
                        const double volume = (cuts[0][x] - cuts[0][x - 1]) * (cuts[1][y] - cuts[1][y - 1])
                            * (cuts[2][z] - cuts[2][z - 1]);
                        if (volume_spec.source != invalid_id)
                            model.source_terms.push_back({static_cast<CellId>(c), volume_spec.source, volume, p});
                    }
            model.cell_material.push_back(material);
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
            mhs::core::Vector<double> a {facet.bounds[0], facet.bounds[1]};
            mhs::core::Vector<double> b {facet.bounds[2], facet.bounds[3]};
            for (const auto& region : regions) {
                if (region.axis != facet.axis || std::abs(region.coordinate - facet.coordinate) >= EPS)
                    continue;
                const auto& r = region.rectangle;
                if (r.a_max <= facet.bounds[0] || r.a_min >= facet.bounds[1] || r.b_max <= facet.bounds[2]
                    || r.b_min >= facet.bounds[3])
                    continue;
                a.push_back(std::clamp(r.a_min, facet.bounds[0], facet.bounds[1]));
                a.push_back(std::clamp(r.a_max, facet.bounds[0], facet.bounds[1]));
                b.push_back(std::clamp(r.b_min, facet.bounds[2], facet.bounds[3]));
                b.push_back(std::clamp(r.b_max, facet.bounds[2], facet.bounds[3]));
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

    mhs::core::Vector<std::size_t> assign_exterior_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<TaggedSurfaceRegion>& regions, std::size_t fallback)
    {
        mhs::core::Vector<std::size_t> assigned(mesh.facets.size(), std::numeric_limits<std::size_t>::max());
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
