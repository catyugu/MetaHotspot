#include "compiler/volume_rasterizer.hpp"
#include "model/constants.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mhs::compiler {
    namespace {
        using namespace mhs::model;
        using namespace mhs::core;
        using Axes = std::array<std::span<const double>, 3>;
        constexpr double EPS = geometry_eps;

        struct Range {
            std::size_t begin, end;
        };

        Range intersected_cells(std::span<const double> vertices, double lo, double hi)
        {
            if (lo >= vertices.back() || hi <= vertices.front())
                return {0, 0};
            const auto begin = std::upper_bound(vertices.begin(), vertices.end(), lo);
            const auto end = std::lower_bound(vertices.begin(), vertices.end(), hi);
            return {begin == vertices.begin() ? 0 : static_cast<std::size_t>(begin - vertices.begin() - 1),
                std::min(static_cast<std::size_t>(end - vertices.begin()), vertices.size() - 1)};
        }

        RegionId classify_point(std::span<const VolumeGeometry> volumes, const Vec3& p)
        {
            for (std::size_t i = volumes.size(); i-- > 0;) {
                bool inside = false;
                for (const auto& box : volumes[i]) {
                    bool contains = true;
                    for (int a = 0; a < 3; ++a)
                        contains = contains && p[a] >= box.min[a] - EPS && p[a] <= box.max[a] + EPS;
                    if (contains)
                        inside = box.operation == GeometryOperation::Add;
                }
                if (inside)
                    return static_cast<RegionId>(i);
            }
            return invalid_id;
        }

        void sort_unique(Vector<double>& values)
        {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }

        // The usual aligned path never enters this check. For a raw face that
        // crosses a cell, partition only that cell by intersecting box faces and
        // verify that every open piece has the same effective region owner.
        void validate_effective_boundaries(const Axes& axes, std::span<const VolumeGeometry> volumes,
            std::span<const RegionId> owners, const std::array<std::size_t, 3>& n)
        {
            Vector<std::size_t> ambiguous;
            for (const auto& volume : volumes)
                for (const auto& box : volume) {
                    const std::array<Range, 3> overlap {intersected_cells(axes[0], box.min[0], box.max[0]),
                        intersected_cells(axes[1], box.min[1], box.max[1]),
                        intersected_cells(axes[2], box.min[2], box.max[2])};
                    if (overlap[0].begin == overlap[0].end || overlap[1].begin == overlap[1].end
                        || overlap[2].begin == overlap[2].end)
                        continue;
                    for (int a = 0; a < 3; ++a)
                        for (const double boundary : {box.min[a], box.max[a]}) {
                            const auto vertices = axes[a];
                            if (boundary <= vertices.front() || boundary >= vertices.back())
                                continue;
                            const auto it = std::lower_bound(vertices.begin(), vertices.end(), boundary);
                            if (std::abs(*it - boundary) <= EPS || std::abs(*(it - 1) - boundary) <= EPS)
                                continue;
                            auto ranges = overlap;
                            const auto index = static_cast<std::size_t>(it - vertices.begin() - 1);
                            ranges[a] = {index, index + 1};
                            for (auto x = ranges[0].begin; x < ranges[0].end; ++x)
                                for (auto y = ranges[1].begin; y < ranges[1].end; ++y)
                                    for (auto z = ranges[2].begin; z < ranges[2].end; ++z)
                                        ambiguous.push_back((x * n[1] + y) * n[2] + z);
                        }
                }
            std::sort(ambiguous.begin(), ambiguous.end());
            ambiguous.erase(std::unique(ambiguous.begin(), ambiguous.end()), ambiguous.end());
            for (const auto flat : ambiguous) {
                const std::array<std::size_t, 3> index {flat / n[2] / n[1], flat / n[2] % n[1], flat % n[2]};
                std::array<Vector<double>, 3> cuts;
                for (int a = 0; a < 3; ++a)
                    cuts[a] = {axes[a][index[a]], axes[a][index[a] + 1]};
                for (const auto& volume : volumes)
                    for (const auto& box : volume) {
                        bool intersects = true;
                        for (int a = 0; a < 3; ++a)
                            intersects
                                = intersects && box.max[a] > axes[a][index[a]] && box.min[a] < axes[a][index[a] + 1];
                        if (!intersects)
                            continue;
                        for (int a = 0; a < 3; ++a)
                            for (const auto boundary : {box.min[a], box.max[a]})
                                if (boundary > axes[a][index[a]] + EPS && boundary < axes[a][index[a] + 1] - EPS)
                                    cuts[a].push_back(boundary);
                    }
                for (auto& values : cuts)
                    sort_unique(values);
                for (std::size_t x = 1; x < cuts[0].size(); ++x)
                    for (std::size_t y = 1; y < cuts[1].size(); ++y)
                        for (std::size_t z = 1; z < cuts[2].size(); ++z) {
                            const Vec3 p {(cuts[0][x - 1] + cuts[0][x]) / 2, (cuts[1][y - 1] + cuts[1][y]) / 2,
                                (cuts[2][z - 1] + cuts[2][z]) / 2};
                            if (classify_point(volumes, p) != owners[flat])
                                throw std::invalid_argument(
                                    "effective volume geometry must align with Cartesian mesh vertices");
                        }
            }
        }
    } // namespace

    mhs::core::Vector<RegionId> rasterize_volumes(
        const std::array<std::span<const double>, 3>& axes, std::span<const VolumeGeometry> volumes)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        if (volumes.size() >= invalid_id)
            throw std::length_error("too many volume regions");
        std::array<std::size_t, 3> n {};
        std::array<Vector<double>, 3> centers;
        std::size_t count = 1;
        for (int a = 0; a < 3; ++a) {
            n[a] = axes[a].size() - 1;
            if (n[a] > std::numeric_limits<std::size_t>::max() / count)
                throw std::length_error("Cartesian grid size overflow");
            count *= n[a];
            centers[a].reserve(n[a]);
            for (std::size_t i = 0; i < n[a]; ++i)
                centers[a].push_back((axes[a][i] + axes[a][i + 1]) / 2);
        }
        Vector<RegionId> owners(count, invalid_id);
        auto remaining = count;
        const auto excluded = static_cast<RegionId>(volumes.size());
        Vector<std::size_t> rejected;
        // Visit highest-priority decisions first. Subtract masks only the current
        // volume and is cleared before the next, lower-priority volume is visited.
        for (std::size_t id = volumes.size(); id-- > 0 && remaining != 0;) {
            const auto& volume = volumes[id];
            for (auto it = volume.rbegin(); it != volume.rend() && remaining != 0; ++it) {
                std::array<Range, 3> ranges;
                for (int a = 0; a < 3; ++a) {
                    const auto& c = centers[a];
                    ranges[a] = {static_cast<std::size_t>(
                                     std::lower_bound(c.begin(), c.end(), it->min[a] - geometry_eps) - c.begin()),
                        static_cast<std::size_t>(
                            std::upper_bound(c.begin(), c.end(), it->max[a] + geometry_eps) - c.begin())};
                }
                for (auto x = ranges[0].begin; x < ranges[0].end; ++x)
                    for (auto y = ranges[1].begin; y < ranges[1].end; ++y)
                        for (auto z = ranges[2].begin; z < ranges[2].end; ++z) {
                            const auto flat = (x * n[1] + y) * n[2] + z;
                            if (owners[flat] != invalid_id)
                                continue;
                            if (it->operation == GeometryOperation::Add) {
                                owners[flat] = static_cast<RegionId>(id);
                                --remaining;
                            }
                            else {
                                owners[flat] = excluded;
                                rejected.push_back(flat);
                            }
                        }
            }
            for (const auto flat : rejected)
                owners[flat] = invalid_id;
            rejected.clear();
        }
        validate_effective_boundaries(axes, volumes, owners, n);
        return owners;
    }
} // namespace mhs::compiler
