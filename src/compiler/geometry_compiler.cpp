#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"

#include "expression/expression.hpp"
#include <algorithm>
#include <cmath>

namespace mhs::compiler {

    constexpr double EPS = mhs::model::geometry_eps;

    SurfaceBounds surface_bounds(const mhs::model::RegionRect& r) { return {r.a_min, r.a_max, r.b_min, r.b_max}; }
    SurfaceBounds intersect_bounds(const SurfaceBounds& a, const SurfaceBounds& b)
    { return {std::max(a[0], b[0]), std::min(a[1], b[1]), std::max(a[2], b[2]), std::min(a[3], b[3])}; }
    bool positive_bounds(const SurfaceBounds& r) { return r[1] > r[0] && r[3] > r[2]; }
    double surface_area(const SurfaceBounds& r) { return (r[1] - r[0]) * (r[3] - r[2]); }

    mhs::core::Vector<TaggedBounds> partition_surface_bounds(const SurfaceBounds& outer,
        std::span<const TaggedBounds> regions, std::optional<mhs::core::EntityIndex> fallback)
    {
        mhs::core::Vector<double> a {outer[0], outer[1]}, b {outer[2], outer[3]};
        for (const auto& tagged : regions) {
            const auto overlap = intersect_bounds(outer, tagged.first);
            if (!positive_bounds(overlap))
                continue;
            a.insert(a.end(), {overlap[0], overlap[1]});
            b.insert(b.end(), {overlap[2], overlap[3]});
        }
        auto unique = [](auto& values) {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        };
        unique(a);
        unique(b);
        mhs::core::Vector<TaggedBounds> result;
        for (std::size_t i = 1; i < a.size(); ++i)
            for (std::size_t j = 1; j < b.size(); ++j) {
                SurfaceBounds bounds {a[i - 1], a[i], b[j - 1], b[j]};
                const double x = (bounds[0] + bounds[1]) / 2, y = (bounds[2] + bounds[3]) / 2;
                auto tag = fallback;
                for (const auto& region : regions)
                    if (x >= region.bounds[0] && x <= region.bounds[1] && y >= region.bounds[2]
                        && y <= region.bounds[3])
                        tag = region.tag;
                if (tag)
                    result.emplace_back(bounds, *tag)
            }
        return result;
    }

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

    void project_cell_regions(mhs::model::FomDomain& model, const mhs::core::Vector<mhs::model::VolumeSpec>& volumes)
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

    mhs::core::Vector<ProjectedSurfacePatch> project_exterior_surface_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<TaggedSurfaceRegion>& boundaries, std::optional<mhs::core::EntityIndex> fallback)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        mhs::core::Vector<ProjectedSurfacePatch> faces;
        for (const auto& face : mesh.faces) {
            if (!face.exterior())
                continue;
            const auto cell = face.minus != invalid_id ? face.minus : face.plus;
            const auto axis = static_cast<std::size_t>(face.axis);
            const std::int8_t sign = face.minus != invalid_id ? 1 : -1;
            const double coordinate = sign > 0 ? mesh.cells[cell].max[axis] : mesh.cells[cell].min[axis];
            mhs::core::Vector<TaggedBounds> regions;
            for (const auto& tagged : boundaries)
                if (tagged.region.axis == face.axis && std::abs(tagged.region.coordinate - coordinate) < EPS)
                    regions.push_back({surface_bounds(tagged.region.rectangle), tagged.second});
            for (const auto& patch : partition_surface_bounds(face.bounds, regions, fallback))
                faces.push_back({cell, face.axis, sign, patch.second, patch.first});
        }
        std::stable_sort(faces.begin(), faces.end(), [](const auto& a, const auto& b) { return a.cell < b.cell; });
        return faces;
    }
} // namespace mhs::compiler
