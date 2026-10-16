#include "compiler/fluid_preprocessor.hpp"
#include "compiler/fluid_physics.hpp"
#include "compiler/geometry_compiler.hpp"
#include "core/constants.hpp"
#include "numerics/linear/linear_solver.hpp"
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace mhs::sim::fluid {
    void build_domain(mhs::core::Model& model, const std::vector<mhs::model::FluidBoundarySpec>& boundaries,
        double scale, const FluidMaterialData& materials)
    {
        using namespace mhs::core;
        auto& fluid = model.fluid;
        const auto& mesh = model.discrete;
        fluid = {};
        fluid.global_to_fluid.assign(mesh.cells.size(), invalidIndex);
        std::vector<double> viscosity;
        for (Index c = 0; c < mesh.cells.size(); ++c) {
            const auto mat = model.cell_material[c];
            if (mat >= materials.initial_viscosity.size() || !materials.initial_viscosity[mat])
                continue;
            fluid.global_to_fluid[c] = fluid.fluid_to_global.size();
            fluid.fluid_to_global.push_back(c);
            viscosity.push_back(*materials.initial_viscosity[mat]);
        }
        const auto n = fluid.fluid_to_global.size();
        if (n == 0) {
            fluid.global_to_fluid.clear();
            return;
        }
        fluid.face_volume_flux.assign(mesh.faces.size(), 0);
        fluid.interface_heat_transfer_factor.resize(n);
        std::vector<Vec3> extents(n);
        // Rectangular-channel extents from connected fluid runs along each axis.
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<Index> parent(n);
            std::iota(parent.begin(), parent.end(), 0);
            auto root = [&](Index i) {
                while (parent[i] != i) {
                    parent[i] = parent[parent[i]];
                    i = parent[i];
                }
                return i;
            };
            for (const auto& f : mesh.faces) {
                if (f.axis != axis || !f.neighbor.valid())
                    continue;
                auto i = fluid.global_to_fluid[f.owner.value], j = fluid.global_to_fluid[f.neighbor.value];
                if (i != invalidIndex && j != invalidIndex)
                    parent[root(i)] = root(j);
            }
            std::vector<double> lo(n, std::numeric_limits<double>::infinity()),
                hi(n, -std::numeric_limits<double>::infinity());
            for (Index i = 0; i < n; ++i) {
                auto r = root(i);
                const auto& c = mesh.cells[fluid.fluid_to_global[i]];
                lo[r] = std::min(lo[r], c.min[axis]);
                hi[r] = std::max(hi[r], c.max[axis]);
            }
            for (Index i = 0; i < n; ++i)
                extents[i][axis] = hi[root(i)] - lo[root(i)];
        }
        std::vector<double> mobility(n), rho(n);
        for (Index i = 0; i < n; ++i) {
            auto lengths = extents[i];
            std::sort(lengths.begin(), lengths.end());
            double w = lengths[0], h = lengths[1], diameter = 2 * w * h / (w + h);
            fluid.interface_heat_transfer_factor[i] = mhs::utils::nusselt_rectangular(w, h) / diameter;
            mobility[i] = diameter * diameter / (2 * mhs::utils::f_re_rectangular(w, h) * viscosity[i]);
            const auto c = fluid.fluid_to_global[i];
            const auto p = mesh.cells[c].center();
            rho[i] = model.material_table[model.cell_material[c]].rho.eval(
                {p[0], p[1], p[2], model.initial_temperature, 0});
        }
        // Use the same rectangle intersection algorithm as thermal projection,
        // against the original mesh and an independent set of regions.
        std::vector<CompiledBoundaryRegion> regions;
        for (TableIndex id = 0; id < boundaries.size(); ++id)
            for (const auto& r : boundaries[id].regions) {
                CompiledBoundaryRegion region;
                region.axis = r.axis;
                region.coordinate = r.coordinate * scale;
                region.parameter_index = id;
                for (const auto& rect : r.rectangles)
                    region.rectangles.push_back(
                        {rect.a_min * scale, rect.a_max * scale, rect.b_min * scale, rect.b_max * scale});
                regions.push_back(std::move(region));
            }
        auto projected = project_surface_regions(mesh, regions);
        std::erase_if(projected, [&](const auto& term) {
            return term.parameter == invalidTableIndex
                || fluid.global_to_fluid[mesh.faces[term.face.value].owner.value] == invalidIndex
                || boundaries[term.parameter].kind == mhs::model::FluidBoundaryKind::None;
        });
        std::vector<double> total_area(boundaries.size(), 0);
        for (const auto& term : projected)
            total_area[term.parameter] += projected_face(mesh, term).area;
        std::vector<double> conductance(mesh.faces.size(), 0), diagonal(n, 0);
        for (Index id = 0; id < mesh.faces.size(); ++id) {
            const auto& f = mesh.faces[id];
            if (!f.neighbor.valid())
                continue;
            auto i = fluid.global_to_fluid[f.owner.value], j = fluid.global_to_fluid[f.neighbor.value];
            if (i == invalidIndex || j == invalidIndex)
                continue;
            auto g = f.area / (f.owner_distance / mobility[i] + f.neighbor_distance / mobility[j]);
            conductance[id] = g;
            diagonal[i] += g;
            diagonal[j] += g;
        }
        bool reference = false;
        std::vector<double> boundary_conductance(projected.size(), 0);
        std::vector<Eigen::Triplet<double>> entries;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
        for (Index t = 0; t < projected.size(); ++t) {
            const auto& term = projected[t];
            const auto f = projected_face(mesh, term);
            const auto i = fluid.global_to_fluid[f.owner.value];
            const auto& bc = boundaries[term.parameter];
            if (bc.kind == mhs::model::FluidBoundaryKind::MassFlowInlet)
                rhs[i] += bc.value * f.area / total_area[term.parameter] / rho[i];
            else {
                reference = true;
                const double g = f.area * mobility[i] / f.owner_distance;
                boundary_conductance[t] = g;
                diagonal[i] += g;
                rhs[i] += g * bc.value;
            }
        }
        if (!reference)
            throw std::runtime_error(
                "fluid domain has no pressure reference; a pressure inlet or an outlet is required");
        for (Index i = 0; i < n; ++i)
            entries.emplace_back(i, i, diagonal[i]);
        for (Index id = 0; id < mesh.faces.size(); ++id)
            if (conductance[id] != 0) {
                const auto& f = mesh.faces[id];
                auto i = fluid.global_to_fluid[f.owner.value], j = fluid.global_to_fluid[f.neighbor.value];
                entries.emplace_back(i, j, -conductance[id]);
                entries.emplace_back(j, i, -conductance[id]);
            }
        Eigen::SparseMatrix<double> matrix(n, n);
        matrix.setFromTriplets(entries.begin(), entries.end());
        auto solver = mhs::sim::create_solver();
        solver->compute(matrix);
        auto pressure = solver->solve(rhs, Eigen::VectorXd::Zero(n));
        if (!solver->success())
            throw std::runtime_error("fluid pressure solve failed");
        for (Index t = 0; t < projected.size(); ++t) {
            const auto& term = projected[t];
            const auto f = projected_face(mesh, term);
            const auto i = fluid.global_to_fluid[f.owner.value];
            const auto& bc = boundaries[term.parameter];
            const double mass = bc.kind == mhs::model::FluidBoundaryKind::MassFlowInlet
                ? -bc.value * f.area / total_area[term.parameter]
                : rho[i] * boundary_conductance[t] * (pressure[i] - bc.value);
            fluid.boundary_terms.push_back(
                {term.face, mass, std::isnan(bc.inlet_temperature) ? model.initial_temperature : bc.inlet_temperature});
        }
        for (Index id = 0; id < mesh.faces.size(); ++id)
            if (conductance[id] != 0) {
                const auto& f = mesh.faces[id];
                auto i = fluid.global_to_fluid[f.owner.value], j = fluid.global_to_fluid[f.neighbor.value];
                fluid.face_volume_flux[id] = (pressure[i] - pressure[j]) * conductance[id];
            }
    }
}
