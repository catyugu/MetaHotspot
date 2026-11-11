#include "compiler/geometry_compiler.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mhs::compiler {
    void compile_conformal_mesh(mhs::model::Model& model, const mhs::model::ConformalMeshSpec& input,
        const mhs::core::Vector<mhs::model::VolumeSpec>& volumes)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        ConformalMesh mesh {input.x_vertices, input.y_vertices, input.z_vertices, {}, {}};
        std::size_t count = 1;
        for (int a = 0; a < 3; ++a) {
            const auto& axis = mesh.axis(a);
            if (axis.size() < 2)
                throw std::invalid_argument("ConformalMesh axes require at least two vertices");
            for (std::size_t i = 0; i < axis.size(); ++i)
                if (!std::isfinite(axis[i]) || (i && axis[i] <= axis[i - 1]))
                    throw std::invalid_argument("ConformalMesh axes must be finite and strictly increasing");
            if (count > (static_cast<std::size_t>(invalid_id) - 1) / (axis.size() - 1))
                throw std::invalid_argument("ConformalMesh cell count exceeds CellId capacity");
            count *= axis.size() - 1;
        }
        Vector<CellId> active(count, invalid_id);
        Vector<CellId> logical;
        for (std::size_t i = 0; i < count; ++i) {
            const auto cell = cell_geometry(mesh, static_cast<CellId>(i));
            const auto id = static_cast<CellId>(model.cells.size());
            if (compile_cell(model, cell, volumes)) {
                active[i] = id;
                logical.push_back(static_cast<CellId>(i));
            }
        }
        if (model.cells.size() != count) {
            mesh.logical_to_cell = std::move(active);
            mesh.cell_to_logical = std::move(logical);
        }
        const auto n = mesh.shape();
        const std::array<std::size_t, 3> stride {n[1] * n[2], n[2], 1};
        for (std::size_t c = 0; c < cell_count(mesh); ++c) {
            const auto l = mesh.logical_to_cell.empty() ? c : mesh.cell_to_logical[c];
            const auto cell = cell_geometry(mesh, static_cast<CellId>(c));
            for (int a = 0; a < 3; ++a)
                for (int sign : {-1, 1}) {
                    const auto index = l / stride[a] % n[a];
                    if ((sign < 0 && index > 0) || (sign > 0 && index + 1 < n[a])) {
                        const auto other = sign < 0 ? l - stride[a] : l + stride[a];
                        if (mesh.logical_to_cell.empty() || mesh.logical_to_cell[other] != invalid_id)
                            continue;
                    }
                    model.boundary_facets.push_back(
                        boundary_facet(static_cast<CellId>(c), cell, static_cast<Axis>(a), sign));
                }
        }
        model.mesh = std::move(mesh);
    }
} // namespace mhs::compiler
