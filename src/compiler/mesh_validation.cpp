#include "compiler/geometry_compiler.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mhs::sim {
    void validate_external_cell_coverage(const mhs::core::DiscreteMesh &mesh) {
        using namespace mhs::core;
        // A bounding-volume hierarchy is used only for overlap validation.
        // It never generates or changes the supplied discretization.
        struct Node {
            Vec3 min{}, max{};
            Index begin = 0, end = 0, left = invalidIndex, right = invalidIndex;
            bool leaf() const { return left == invalidIndex; }
        };
        std::vector<Index> order(mesh.cells.size());
        std::iota(order.begin(), order.end(), 0);
        std::vector<Node> nodes;
        nodes.reserve(2 * order.size());
        auto build = [&](auto &&self, Index begin, Index end) -> Index {
            const auto id = nodes.size();
            nodes.push_back({});
            Node node;
            node.begin = begin;
            node.end = end;
            node.min = mesh.cells[order[begin]].min;
            node.max = mesh.cells[order[begin]].max;
            for (Index i = begin + 1; i < end; ++i)
                for (int a = 0; a < 3; ++a) {
                    node.min[a] = std::min(node.min[a], mesh.cells[order[i]].min[a]);
                    node.max[a] = std::max(node.max[a], mesh.cells[order[i]].max[a]);
                }
            if (end - begin > 8) {
                int axis = 0;
                for (int a = 1; a < 3; ++a)
                    if (node.max[a] - node.min[a] > node.max[axis] - node.min[axis])
                        axis = a;
                const auto mid = begin + (end - begin) / 2;
                std::nth_element(
                    order.begin() + begin, order.begin() + mid, order.begin() + end,
                    [&](Index i, Index j) { return mesh.cells[i].center()[axis] < mesh.cells[j].center()[axis]; });
                node.left = self(self, begin, mid);
                node.right = self(self, mid, end);
            }
            nodes[id] = node;
            return id;
        };
        if (order.empty())
            return;
        const auto root = build(build, 0, order.size());
        auto overlap = [](const auto &a, const auto &b) {
            for (int axis = 0; axis < 3; ++axis) {
                const double scale = std::max(
                    {std::abs(a.min[axis]), std::abs(a.max[axis]), std::abs(b.min[axis]), std::abs(b.max[axis])});
                if (std::min(a.max[axis], b.max[axis]) - std::max(a.min[axis], b.min[axis]) <=
                    64 * std::numeric_limits<double>::epsilon() * scale)
                    return false;
            }
            return true;
        };
        auto visit = [&](auto &&self, Index i, Index j) -> void {
            const auto &a = nodes[i];
            const auto &b = nodes[j];
            if (i == j) {
                if (a.leaf()) {
                    for (Index u = a.begin; u < a.end; ++u)
                        for (Index v = u + 1; v < a.end; ++v)
                            if (overlap(mesh.cells[order[u]], mesh.cells[order[v]]))
                                throw std::invalid_argument("overlapping external control volumes");
                } else {
                    self(self, a.left, a.left);
                    self(self, a.left, a.right);
                    self(self, a.right, a.right);
                }
            } else {
                if (!overlap(a, b))
                    return;
                if (a.leaf() && b.leaf()) {
                    for (Index u = a.begin; u < a.end; ++u)
                        for (Index v = b.begin; v < b.end; ++v)
                            if (overlap(mesh.cells[order[u]], mesh.cells[order[v]]))
                                throw std::invalid_argument("overlapping external control volumes");
                } else if (!a.leaf() && (b.leaf() || a.end - a.begin >= b.end - b.begin)) {
                    self(self, a.left, j);
                    self(self, a.right, j);
                } else {
                    self(self, i, b.left);
                    self(self, i, b.right);
                }
            }
        };
        visit(visit, root, root);
    }
} // namespace mhs::sim
