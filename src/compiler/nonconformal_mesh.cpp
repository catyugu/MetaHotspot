#include "compiler/geometry_compiler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace mhs::compiler {
    namespace {
        using namespace mhs::model;
        using namespace mhs::core;

        // Compile-only BVH: validate overlaps and omitted contacts without an all-cell pair scan.
        struct Node {
            CellGeometry box;
            std::size_t begin, end;
            std::size_t left = 0, right = 0;
        };
        class CellTree {
            const Vector<CellGeometry>& cells_;
            Vector<CellId> order_;
            Vector<Node> nodes_;
            std::size_t build(std::size_t begin, std::size_t end)
            {
                CellGeometry box = cells_[order_[begin]];
                for (auto i = begin + 1; i < end; ++i)
                    for (int a = 0; a < 3; ++a) {
                        box.min[a] = std::min(box.min[a], cells_[order_[i]].min[a]);
                        box.max[a] = std::max(box.max[a], cells_[order_[i]].max[a]);
                    }
                const auto id = nodes_.size();
                nodes_.push_back({box, begin, end});
                if (end - begin > 8) {
                    const auto size = box.size();
                    const auto axis = std::max_element(size.begin(), size.end()) - size.begin();
                    const auto mid = begin + (end - begin) / 2;
                    std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
                        [&](CellId a, CellId b) { return cells_[a].center()[axis] < cells_[b].center()[axis]; });
                    const auto left = build(begin, mid), right = build(mid, end);
                    nodes_[id].left = left;
                    nodes_[id].right = right;
                }
                return id;
            }
            template <class Callback> void query(std::size_t node, const CellGeometry& box, Callback& callback) const
            {
                const auto& n = nodes_[node];
                for (int a = 0; a < 3; ++a)
                    if (box.max[a] < n.box.min[a] - geometry_eps || n.box.max[a] < box.min[a] - geometry_eps)
                        return;
                if (n.left) {
                    query(n.left, box, callback);
                    query(n.right, box, callback);
                }
                else
                    for (auto i = n.begin; i < n.end; ++i)
                        callback(order_[i]);
            }

        public:
            explicit CellTree(const Vector<CellGeometry>& cells) : cells_(cells), order_(cells.size())
            {
                std::iota(order_.begin(), order_.end(), CellId {0});
                if (!cells.empty())
                    build(0, cells.size());
            }
            template <class Callback> void query(const CellGeometry& box, Callback callback) const
            {
                if (!nodes_.empty())
                    query(0, box, callback);
            }
        };
        std::uint64_t key(CellId a, CellId b)
        { return (static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b); }
        void subtract(Vector<BoundaryFacet>& pieces, const FaceGeometry& cut)
        {
            Vector<BoundaryFacet> result;
            for (const auto& f : pieces) {
                const double u0 = std::max(f.bounds[0], cut.bounds[0]), u1 = std::min(f.bounds[1], cut.bounds[1]);
                const double v0 = std::max(f.bounds[2], cut.bounds[2]), v1 = std::min(f.bounds[3], cut.bounds[3]);
                if (u1 <= u0 || v1 <= v0) {
                    result.push_back(f);
                    continue;
                }
                for (const auto& bounds : {std::array {f.bounds[0], u0, f.bounds[2], f.bounds[3]},
                         std::array {u1, f.bounds[1], f.bounds[2], f.bounds[3]}, std::array {u0, u1, f.bounds[2], v0},
                         std::array {u0, u1, v1, f.bounds[3]}}) {
                    if (bounds[1] <= bounds[0] || bounds[3] <= bounds[2])
                        continue;
                    auto piece = f;
                    piece.bounds = bounds;
                    result.push_back(piece);
                }
            }
            pieces = std::move(result);
        }
    }

    void compile_nonconformal_mesh(Model& model, const NonConformalMesh& input, const Vector<VolumeSpec>& volumes)
    {
        if (input.cells.size() >= invalid_id)
            throw std::invalid_argument("NonConformalMesh cell count exceeds CellId capacity");
        for (const auto& c : input.cells) {
            for (int a = 0; a < 3; ++a)
                if (!std::isfinite(c.min[a]) || !std::isfinite(c.max[a]) || c.min[a] >= c.max[a])
                    throw std::invalid_argument("NonConformalMesh cells need finite, increasing bounds");
            if (!compile_cell(model, c, volumes))
                throw std::invalid_argument("NonConformalMesh cell lies outside the active physical geometry");
        }
        std::unordered_set<std::uint64_t> connections;
        Vector<Vector<std::size_t>> incident(input.cells.size()); // compile-only reverse lookup
        for (std::size_t i = 0; i < input.connections.size(); ++i) {
            const auto [a, b] = input.connections[i];
            if (a >= input.cells.size() || b >= input.cells.size() || a == b)
                throw std::invalid_argument("NonConformalMesh Connection has invalid CellIds");
            (void)connection_geometry(input.cells[a], input.cells[b]);
            if (!connections.insert(key(a, b)).second)
                throw std::invalid_argument("duplicate NonConformalMesh Connection");
            incident[a].push_back(i);
            incident[b].push_back(i);
        }
        const CellTree tree(input.cells);
        for (std::size_t i = 0; i < input.cells.size(); ++i)
            tree.query(input.cells[i], [&](CellId j) {
                if (j <= i)
                    return;
                const auto& a = input.cells[i];
                const auto& b = input.cells[j];
                std::array<double, 3> overlap;
                for (int axis = 0; axis < 3; ++axis)
                    overlap[axis] = std::min(a.max[axis], b.max[axis]) - std::max(a.min[axis], b.min[axis]);
                if (std::all_of(overlap.begin(), overlap.end(), [](double x) { return x > geometry_eps; }))
                    throw std::invalid_argument("NonConformalMesh cells overlap in volume");
                for (int axis = 0; axis < 3; ++axis) {
                    const int u = axis == 0 ? 1 : 0, v = axis == 2 ? 1 : 2;
                    if (std::abs(overlap[axis]) <= geometry_eps && overlap[u] > 0 && overlap[v] > 0
                        && !connections.contains(key(static_cast<CellId>(i), j)))
                        throw std::invalid_argument("NonConformalMesh omits a face Connection");
                }
            });
        for (std::size_t i = 0; i < input.cells.size(); ++i) {
            const auto& cell = input.cells[i];
            for (int a = 0; a < 3; ++a)
                for (int sign : {-1, 1}) {
                    BoundaryFacet f;
                    f.cell = static_cast<CellId>(i);
                    f.side = sign;
                    f.axis = static_cast<Axis>(a);
                    f.coordinate = sign > 0 ? cell.max[a] : cell.min[a];
                    const int u = a == 0 ? 1 : 0, v = a == 2 ? 1 : 2;
                    f.bounds = {cell.min[u], cell.max[u], cell.min[v], cell.max[v]};
                    Vector<BoundaryFacet> remaining {f};
                    for (const auto id : incident[i]) {
                        const auto [ca, cb] = input.connections[id];
                        const auto g = connection_geometry(input.cells[ca], input.cells[cb]);
                        const auto other = ca == i ? cb : ca;
                        const int side = input.cells[other].center()[a] > cell.center()[a] ? 1 : -1;
                        if (g.axis == f.axis && side == sign && std::abs(g.coordinate - f.coordinate) <= geometry_eps)
                            subtract(remaining, g);
                    }
                    model.boundary_facets.insert(model.boundary_facets.end(), remaining.begin(), remaining.end());
                }
        }
        model.mesh = input;
    }
} // namespace mhs::compiler
