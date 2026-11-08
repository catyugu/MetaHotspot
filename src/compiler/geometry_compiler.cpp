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

    mhs::core::Vector<ProjectedSurfacePatch> project_exterior_surface_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<TaggedSurfaceRegion>& boundaries, std::optional<std::size_t> fallback)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        mhs::core::Vector<ProjectedSurfacePatch> faces;
        for (std::size_t id = 0; id < mesh.faces.size(); ++id) {
            const auto& face = mesh.faces[id];
            if (!face.exterior())
                continue;
            const auto cell = face.minus != invalid_id ? face.minus : face.plus;
            const auto axis = static_cast<std::size_t>(face.axis);
            const std::int8_t sign = face.minus != invalid_id ? 1 : -1;
            const double coordinate = sign > 0 ? mesh.cells[cell].max[axis] : mesh.cells[cell].min[axis];
            mhs::core::Vector<double> ca {face.bounds[0], face.bounds[1]}, cb {face.bounds[2], face.bounds[3]};
            for (const auto& tagged : boundaries) {
                const auto& region = tagged.region;
                if (region.axis != face.axis || std::abs(region.coordinate - coordinate) >= EPS)
                    continue;
                const auto& r = region.rectangle;
                if (r.a_max <= face.bounds[0] || r.a_min >= face.bounds[1] || r.b_max <= face.bounds[2]
                    || r.b_min >= face.bounds[3])
                    continue;
                ca.push_back(std::clamp(r.a_min, face.bounds[0], face.bounds[1]));
                ca.push_back(std::clamp(r.a_max, face.bounds[0], face.bounds[1]));
                cb.push_back(std::clamp(r.b_min, face.bounds[2], face.bounds[3]));
                cb.push_back(std::clamp(r.b_max, face.bounds[2], face.bounds[3]));
            }
            auto unique = [](auto& cuts) {
                std::sort(cuts.begin(), cuts.end());
                cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
            };
            unique(ca);
            unique(cb);
            for (std::size_t i = 1; i < ca.size(); ++i)
                for (std::size_t j = 1; j < cb.size(); ++j) {
                    const std::array<double, 4> bounds {ca[i - 1], ca[i], cb[j - 1], cb[j]};
                    const double center_a = (bounds[0] + bounds[1]) / 2;
                    const double center_b = (bounds[2] + bounds[3]) / 2;
                    auto tag = fallback;
                    for (auto it = boundaries.rbegin(); it != boundaries.rend(); ++it) {
                        if (it->region.axis == face.axis && std::abs(it->region.coordinate - coordinate) < EPS
                            && point_in_region(it->region, center_a, center_b)) {
                            tag = it->tag;
                            break;
                        }
                    }
                    if (tag)
                        faces.push_back({cell, face.axis, sign, *tag, bounds});
                }
        }
        std::stable_sort(faces.begin(), faces.end(), [](const auto& a, const auto& b) { return a.cell < b.cell; });
        return faces;
    }

} // namespace mhs::compiler
