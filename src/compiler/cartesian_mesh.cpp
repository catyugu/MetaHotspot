#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace mhs::compiler {

    namespace {
        using namespace mhs::model;
        using mhs::core::Vector;
        using Plane = std::pair<int, double>;

        struct RectangleOwner {
            CellId cell;
            std::array<double, 4> bounds;
        };
        struct PlaneOwners {
            Vector<RectangleOwner> minus, plus;
        };

        // Refinement is a local topological operation. A plane's projected
        // rectangles are partitioned into disjoint subfaces with one flux per
        // tile, including T-junctions. All coordinates are Cartesian.
        void compile_refined_mesh(FomDomain& domain, const std::array<Vector<double>, 3>& axes,
            const Vector<VolumeSpec>& volumes, const Vector<CartesianRefinement>& refinements)
        {
            for (const auto& region : refinements) {
                if (region.levels == 0 || region.levels > 3)
                    throw std::invalid_argument("refinement levels must be between 1 and 3");
                for (int axis = 0; axis < 3; ++axis) {
                    if (!std::isfinite(region.min[axis]) || !std::isfinite(region.max[axis])
                        || region.min[axis] >= region.max[axis])
                        throw std::invalid_argument("refinement box bounds must be finite and increasing");
                    const auto aligned = [&](double v) {
                        const auto& points = axes[axis];
                        const auto it = std::lower_bound(points.begin(), points.end(), v);
                        return it != points.end() && *it == v;
                    };
                    if (!aligned(region.min[axis]) || !aligned(region.max[axis]))
                        throw std::invalid_argument("refinement box must align with parent Cartesian vertices");
                }
            }
            const std::array<std::size_t, 3> n {
                axes[0].size() - 1, axes[1].size() - 1, axes[2].size() - 1};
            for (std::size_t i = 0; i < n[0]; ++i)
                for (std::size_t j = 0; j < n[1]; ++j)
                    for (std::size_t k = 0; k < n[2]; ++k) {
                        const CellGeometry parent {
                            {axes[0][i], axes[1][j], axes[2][k]},
                            {axes[0][i + 1], axes[1][j + 1], axes[2][k + 1]}};
                        if (!classify_point(volumes, parent.center()))
                            continue;
                        std::uint32_t levels = 0;
                        for (const auto& r : refinements) {
                            const auto center = parent.center();
                            bool within = true;
                            for (int a = 0; a < 3; ++a)
                                within = within && center[a] > r.min[a] && center[a] < r.max[a];
                            if (within)
                                levels = std::max(levels, r.levels);
                        }
                        const unsigned pieces = 1u << levels;
                        const auto bound = [&](int a, unsigned index) {
                            if (!index)
                                return parent.min[a];
                            if (index == pieces)
                                return parent.max[a];
                            return parent.min[a] + (parent.max[a] - parent.min[a])
                                * (static_cast<double>(index) / pieces);
                        };
                        for (unsigned a = 0; a < pieces; ++a)
                            for (unsigned b = 0; b < pieces; ++b)
                                for (unsigned c = 0; c < pieces; ++c) {
                                    const CellGeometry child {
                                        {bound(0, a), bound(1, b), bound(2, c)},
                                        {bound(0, a + 1), bound(1, b + 1), bound(2, c + 1)}};
                                    mhs::core::detail::checked_sparse_size(domain.discrete.cells.size() + 1);
                                    domain.discrete.cells.push_back(child);
                                }
                    }

            std::map<Plane, PlaneOwners> planes;
            for (std::size_t id = 0; id < domain.discrete.cells.size(); ++id) {
                const auto& cell = domain.discrete.cells[id];
                for (int a = 0; a < 3; ++a) {
                    const int t = a == 0 ? 1 : 0, u = a == 2 ? 1 : 2;
                    const std::array<double, 4> bounds {
                        cell.min[t], cell.max[t], cell.min[u], cell.max[u]};
                    planes[{a, cell.min[a]}].plus.push_back({static_cast<CellId>(id), bounds});
                    planes[{a, cell.max[a]}].minus.push_back({static_cast<CellId>(id), bounds});
                }
            }
            for (const auto& [plane, owners] : planes) {
                Vector<double> a, b;
                auto cuts = [&](const Vector<RectangleOwner>& rectangles) {
                    for (const auto& r : rectangles) {
                        a.push_back(r.bounds[0]);
                        a.push_back(r.bounds[1]);
                        b.push_back(r.bounds[2]);
                        b.push_back(r.bounds[3]);
                    }
                };
                cuts(owners.minus);
                cuts(owners.plus);
                auto unique = [](Vector<double>& points) {
                    std::sort(points.begin(), points.end());
                    points.erase(std::unique(points.begin(), points.end()), points.end());
                };
                unique(a);
                unique(b);
                const auto nu = a.size() - 1, nv = b.size() - 1;
                mhs::core::detail::checked_sparse_size(nu * nv);
                Vector<CellId> minus(nu * nv, mhs::core::invalid_id);
                Vector<CellId> plus(nu * nv, mhs::core::invalid_id);
                auto fill = [&](const Vector<RectangleOwner>& rectangles, Vector<CellId>& target) {
                    for (const auto& r : rectangles) {
                        const auto a0 = std::lower_bound(a.begin(), a.end(), r.bounds[0]) - a.begin();
                        const auto a1 = std::lower_bound(a.begin(), a.end(), r.bounds[1]) - a.begin();
                        const auto b0 = std::lower_bound(b.begin(), b.end(), r.bounds[2]) - b.begin();
                        const auto b1 = std::lower_bound(b.begin(), b.end(), r.bounds[3]) - b.begin();
                        for (auto i = a0; i < a1; ++i)
                            for (auto j = b0; j < b1; ++j) {
                                auto& owner = target[i * nv + j];
                                if (owner != mhs::core::invalid_id)
                                    throw std::invalid_argument("overlapping Cartesian cells on one mesh face");
                                owner = r.cell;
                            }
                    }
                };
                fill(owners.minus, minus);
                fill(owners.plus, plus);
                for (std::size_t i = 0; i < nu; ++i)
                    for (std::size_t j = 0; j < nv; ++j) {
                        const auto offset = i * nv + j;
                        if (minus[offset] == mhs::core::invalid_id && plus[offset] == mhs::core::invalid_id)
                            continue;
                        mhs::core::detail::checked_sparse_size(domain.discrete.faces.size() + 1);
                        domain.discrete.faces.push_back({
                            minus[offset], plus[offset], static_cast<Axis>(plane.first),
                            {a[i], a[i + 1], b[j], b[j + 1]}});
                    }
            }
            build_cell_face_incidence(domain.discrete);
        }
    } // namespace

    // The only structured-index arithmetic in the compiled model pipeline.
    // This adapter consumes supplied axes; it does not choose/refine a mesh.
    void compile_cartesian_mesh(mhs::model::FomDomain& model, const mhs::model::MeshSpec& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes,
        const mhs::core::Vector<mhs::model::CartesianRefinement>& refinements)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        std::array<mhs::core::Vector<double>, 3> axes {input.x_vertices, input.y_vertices, input.z_vertices};
        std::array<std::size_t, 3> n {};
        for (int a = 0; a < 3; ++a) {
            if (axes[a].size() < 2)
                throw std::invalid_argument("Cartesian axes require at least two vertices");
            for (std::size_t i = 0; i < axes[a].size(); ++i)
                if (!std::isfinite(axes[a][i]) || (i && axes[a][i] <= axes[a][i - 1]))
                    throw std::invalid_argument("Cartesian axes must be finite and strictly increasing");
            n[a] = axes[a].size() - 1;
        }
        if (!refinements.empty()) {
            compile_refined_mesh(model, axes, volumes, refinements);
            return;
        }
        auto flatten = [&](std::size_t x, std::size_t y, std::size_t z) { return (x * n[1] + y) * n[2] + z; };
        mhs::core::Vector<CellId> active(n[0] * n[1] * n[2], invalid_id);
        for (std::size_t x = 0; x < n[0]; ++x)
            for (std::size_t y = 0; y < n[1]; ++y)
                for (std::size_t z = 0; z < n[2]; ++z) {
                    CellGeometry cell {
                        {axes[0][x], axes[1][y], axes[2][z]}, {axes[0][x + 1], axes[1][y + 1], axes[2][z + 1]}};
                    if (!classify_point(volumes, cell.center()))
                        continue;
                    active[flatten(x, y, z)] = static_cast<CellId>(model.discrete.cells.size());
                    model.discrete.cells.push_back(cell);
                }
        for (std::size_t x = 0; x < n[0]; ++x)
            for (std::size_t y = 0; y < n[1]; ++y)
                for (std::size_t z = 0; z < n[2]; ++z) {
                    const auto c = active[flatten(x, y, z)];
                    if (c == mhs::core::invalid_id)
                        continue;
                    for (int a = 0; a < 3; ++a)
                        for (int sign : {-1, 1}) {
                            std::array<std::size_t, 3> v {x, y, z};
                            CellId neighbor = mhs::core::invalid_id;
                            if ((sign < 0 && v[a] > 0) || (sign > 0 && v[a] + 1 < n[a])) {
                                if (sign < 0)
                                    --v[a];
                                else
                                    ++v[a];
                                neighbor = active[flatten(v[0], v[1], v[2])];
                            }
                            if ((neighbor != mhs::core::invalid_id) && neighbor < c)
                                continue;
                            const auto& cell = model.discrete.cells[c];
                            FacePatch f;
                            f.minus = sign > 0 ? c : neighbor;
                            f.plus = sign > 0 ? neighbor : c;
                            f.axis = static_cast<Axis>(a);
                            const int t = a == 0 ? 1 : 0, u = a == 2 ? 1 : 2;
                            f.bounds = {cell.min[t], cell.max[t], cell.min[u], cell.max[u]};
                            model.discrete.faces.push_back(f);
                        }
                }
        build_cell_face_incidence(model.discrete);
    }
} // namespace mhs::compiler
