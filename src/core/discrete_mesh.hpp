#pragma once
#include "core/types.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace mhs::core {
    template <class Tag> struct EntityId {
        Index value = invalidIndex;
        bool valid() const { return value != invalidIndex; }
        auto operator<=>(const EntityId&) const = default;
    };
    using CellId = EntityId<struct CellTag>;
    using FaceId = EntityId<struct FaceTag>;
    using DofId = EntityId<struct DofTag>;
    using RegionId = EntityId<struct RegionTag>;
    using Vec3 = std::array<double, 3>;

    struct CellGeometry {
        Vec3 min {}, max {};
        Vec3 center() const { return {(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2}; }
        Vec3 size() const { return {max[0] - min[0], max[1] - min[1], max[2] - min[2]}; }
        double volume() const
        {
            auto d = size();
            return d[0] * d[1] * d[2];
        }
        bool contains(const Vec3& p, double eps = 0) const
        {
            for (int a = 0; a < 3; ++a)
                if (p[a] < min[a] - eps || p[a] > max[a] + eps)
                    return false;
            return true;
        }
    };
    // Rectangular axis-aligned patches. The sign is the outward owner normal.
    // Geometry only: coefficients and boundary assignments live in Model.
    struct FacePatch {
        CellId owner, neighbor;
        int axis = 0, sign = 1;
        Vec3 min {}, max {};
        double area = 0, owner_distance = 0, neighbor_distance = 0;
        Vec3 center() const { return {(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2}; }
        CellId other(CellId c) const { return c == owner ? neighbor : owner; }
        bool contains(const Vec3& p, double eps = 1e-12) const
        {
            for (int a = 0; a < 3; ++a)
                if (p[a] < min[a] - eps || p[a] > max[a] + eps)
                    return false;
            return true;
        }
    };
    struct DiscreteMesh {
        std::vector<CellGeometry> cells;
        std::vector<FacePatch> faces;
        std::vector<Index> cell_face_offsets;
        std::vector<FaceId> cell_face_ids;
        std::span<const FaceId> incident_faces(CellId c) const
        {
            return std::span(cell_face_ids)
                .subspan(
                    cell_face_offsets.at(c.value), cell_face_offsets.at(c.value + 1) - cell_face_offsets.at(c.value));
        }
        void build_incidence()
        {
            cell_face_offsets.assign(cells.size() + 1, 0);
            for (const auto& f : faces) {
                if (f.owner.value >= cells.size() || (f.neighbor.valid() && f.neighbor.value >= cells.size()))
                    throw std::invalid_argument("face references unknown cell");
                ++cell_face_offsets[f.owner.value + 1];
                if (f.neighbor.valid())
                    ++cell_face_offsets[f.neighbor.value + 1];
            }
            std::partial_sum(cell_face_offsets.begin(), cell_face_offsets.end(), cell_face_offsets.begin());
            cell_face_ids.resize(cell_face_offsets.back());
            auto next = cell_face_offsets;
            for (Index i = 0; i < faces.size(); ++i) {
                const auto& f = faces[i];
                cell_face_ids[next[f.owner.value]++] = {i};
                if (f.neighbor.valid())
                    cell_face_ids[next[f.neighbor.value]++] = {i};
            }
        }
        CellId locate(const Vec3& p) const
        {
            // Cache once per probe. Shared-face ties select the lexicographically
            // greatest lower corner, independent of cell storage/DOF ordering.
            CellId result;
            for (Index i = 0; i < cells.size(); ++i)
                if (cells[i].contains(p) && (!result.valid() || cells[i].min > cells[result.value].min))
                    result = {i};
            return result;
        }
        void validate() const
        {
            if (cell_face_offsets.size() != cells.size() + 1 || cell_face_offsets.front() != 0
                || cell_face_offsets.back() != cell_face_ids.size())
                throw std::invalid_argument("invalid cell-face CSR");
            for (const auto& c : cells)
                for (int a = 0; a < 3; ++a)
                    if (!std::isfinite(c.min[a]) || !std::isfinite(c.max[a]) || c.max[a] <= c.min[a])
                        throw std::invalid_argument("invalid cell bounds");
            for (const auto& f : faces) {
                if (f.owner.value >= cells.size()
                    || (f.neighbor.valid() && (f.neighbor.value >= cells.size() || f.neighbor == f.owner)) || f.axis < 0
                    || f.axis > 2 || (f.sign != 1 && f.sign != -1) || !std::isfinite(f.area) || f.area <= 0
                    || !std::isfinite(f.owner_distance) || f.owner_distance <= 0
                    || (f.neighbor.valid() && (!std::isfinite(f.neighbor_distance) || f.neighbor_distance <= 0)))
                    throw std::invalid_argument("invalid face patch");
                for (int a = 0; a < 3; ++a)
                    if (!std::isfinite(f.min[a]) || !std::isfinite(f.max[a]))
                        throw std::invalid_argument("nonfinite face bounds");
                if (f.min[f.axis] != f.max[f.axis])
                    throw std::invalid_argument("face patch is not planar");
                const int a = (f.axis + 1) % 3, b = (f.axis + 2) % 3;
                const double area = (f.max[a] - f.min[a]) * (f.max[b] - f.min[b]);
                if (f.max[a] <= f.min[a] || f.max[b] <= f.min[b] || std::abs(area - f.area) > 1e-10 * f.area)
                    throw std::invalid_argument("face area disagrees with bounds");
                auto check_side = [&](CellId id, int sign, double distance) {
                    const auto& c = cells[id.value];
                    const auto size = c.size();
                    for (int axis = 0; axis < 3; ++axis) {
                        const double tol = 64 * std::numeric_limits<double>::epsilon()
                                * std::max(std::abs(c.min[axis]), std::abs(c.max[axis]))
                            + 1e-10 * size[axis];
                        if (axis == f.axis) {
                            const double plane = sign < 0 ? c.min[axis] : c.max[axis];
                            if (std::abs(f.min[axis] - plane) > tol || std::abs(distance - size[axis] / 2) > tol)
                                throw std::invalid_argument("face plane or distance disagrees with cell");
                        }
                        else if (f.min[axis] < c.min[axis] - tol || f.max[axis] > c.max[axis] + tol)
                            throw std::invalid_argument("face extends beyond cell");
                    }
                };
                check_side(f.owner, f.sign, f.owner_distance);
                if (f.neighbor.valid())
                    check_side(f.neighbor, -f.sign, f.neighbor_distance);
            }
            std::vector<Index> uses(faces.size(), 0);
            for (Index c = 0; c < cells.size(); ++c) {
                if (cell_face_offsets[c] > cell_face_offsets[c + 1])
                    throw std::invalid_argument("nonmonotone CSR");
                std::array<double, 6> area {};
                const auto incidence = incident_faces(CellId {c});
                for (auto id : incidence) {
                    if (id.value >= faces.size())
                        throw std::invalid_argument("invalid face incidence");
                    const auto& f = faces[id.value];
                    if (f.owner.value != c && f.neighbor.value != c)
                        throw std::invalid_argument("unrelated face incidence");
                    const int sign = f.owner.value == c ? f.sign : -f.sign;
                    area[2 * f.axis + (sign > 0)] += f.area;
                    ++uses[id.value];
                }
                const auto d = cells[c].size();
                for (int axis = 0; axis < 3; ++axis)
                    for (int sign = 0; sign < 2; ++sign) {
                        const auto expected = d[(axis + 1) % 3] * d[(axis + 2) % 3];
                        if (std::abs(area[2 * axis + sign] - expected) > 1e-10 * expected)
                            throw std::invalid_argument("cell surface coverage is incomplete or duplicated");
                    }
                for (Index i = 0; i < incidence.size(); ++i)
                    for (Index j = i + 1; j < incidence.size(); ++j) {
                        const auto& f = faces[incidence[i].value];
                        const auto& g = faces[incidence[j].value];
                        if (f.axis != g.axis
                            || (f.owner.value == c ? f.sign : -f.sign) != (g.owner.value == c ? g.sign : -g.sign))
                            continue;
                        const int a = (f.axis + 1) % 3, b = (f.axis + 2) % 3;
                        if (std::min(f.max[a], g.max[a]) - std::max(f.min[a], g.min[a]) > 1e-10 * d[a]
                            && std::min(f.max[b], g.max[b]) - std::max(f.min[b], g.min[b]) > 1e-10 * d[b])
                            throw std::invalid_argument("overlapping face patches");
                    }
            }
            for (Index f = 0; f < faces.size(); ++f)
                if (uses[f] != (faces[f].neighbor.valid() ? 2u : 1u))
                    throw std::invalid_argument("duplicate or missing face incidence");
        }
        void require_orthogonal_tpfa() const
        {
            for (const auto& f : faces)
                if (f.neighbor.valid()) {
                    const auto a = cells[f.owner.value].center(), b = cells[f.neighbor.value].center();
                    for (int axis = 0; axis < 3; ++axis)
                        if (axis != f.axis
                            && std::abs(a[axis] - b[axis]) > 1e-10
                                    * std::max(cells[f.owner.value].size()[axis], cells[f.neighbor.value].size()[axis]))
                            throw std::invalid_argument("nonorthogonal mesh requires an unimplemented flux scheme; "
                                                        "current TPFA supports aligned cell centers only");
                }
        }
    };
    struct DofMap {
        std::vector<DofId> cell_to_dof;
        std::vector<CellId> dof_to_cell;
        DofId temperature(CellId cell) const { return cell_to_dof.at(cell.value); }
        CellId cell(DofId dof) const { return dof_to_cell.at(dof.value); }
        Index size() const { return dof_to_cell.size(); }
        void set_permutation(const std::vector<Index>& permutation)
        {
            std::vector<DofId> forward(permutation.size());
            std::vector<CellId> inverse(permutation.size());
            for (Index c = 0; c < permutation.size(); ++c) {
                auto d = permutation[c];
                if (d >= permutation.size() || inverse[d].valid())
                    throw std::invalid_argument("invalid DOF permutation");
                forward[c] = {d};
                inverse[d] = {c};
            }
            cell_to_dof = std::move(forward);
            dof_to_cell = std::move(inverse);
        }
        void identity(Index n)
        {
            std::vector<Index> p(n);
            std::iota(p.begin(), p.end(), 0);
            set_permutation(p);
        }
    };
} // namespace mhs::core
