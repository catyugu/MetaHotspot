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
    using SurfaceId = EntityId<struct SurfaceTag>;
    using PortId = EntityId<struct PortTag>;
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
            // Initialization-only query; probes cache CellId for all time steps.
            for (Index i = 0; i < cells.size(); ++i)
                if (cells[i].contains(p))
                    return {i};
            return {};
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
                    || (f.neighbor.valid() && (f.neighbor.value >= cells.size() || f.owner == f.neighbor)) || f.axis < 0
                    || f.axis > 2 || (f.sign != 1 && f.sign != -1) || !std::isfinite(f.area) || f.area <= 0
                    || !std::isfinite(f.owner_distance) || f.owner_distance <= 0
                    || (f.neighbor.valid() && (!std::isfinite(f.neighbor_distance) || f.neighbor_distance <= 0)))
                    throw std::invalid_argument("invalid face patch");
            }
            std::vector<Index> uses(faces.size(), 0);
            for (Index c = 0; c < cells.size(); ++c) {
                if (cell_face_offsets[c] > cell_face_offsets[c + 1])
                    throw std::invalid_argument("nonmonotone CSR");
                Vec3 closure {};
                for (auto id : incident_faces(CellId {c})) {
                    if (id.value >= faces.size())
                        throw std::invalid_argument("invalid face incidence");
                    const auto& f = faces[id.value];
                    if (f.owner.value != c && f.neighbor.value != c)
                        throw std::invalid_argument("unrelated face incidence");
                    closure[f.axis] += (f.owner.value == c ? f.sign : -f.sign) * f.area;
                    ++uses[id.value];
                }
                auto d = cells[c].size();
                const double scale = 2 * (d[0] * d[1] + d[1] * d[2] + d[0] * d[2]);
                for (auto x : closure)
                    if (std::abs(x) > 1e-10 * scale)
                        throw std::invalid_argument("cell surface is not closed");
            }
            for (Index f = 0; f < faces.size(); ++f)
                if (uses[f] != (faces[f].neighbor.valid() ? 2u : 1u))
                    throw std::invalid_argument("duplicate or missing face incidence");
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
}
