#include "compiler/fluid_preprocessor.hpp"
#include "compiler/fluid_physics.hpp"
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
            const auto mat = model.cells.material_id[c];
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
        fluid.boundary_outflux.assign(n, std::numeric_limits<double>::quiet_NaN());
        fluid.boundary_temperature = fluid.boundary_outflux;
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
            rho[i] = model.material_table[model.cells.material_id[c]].rho.eval(
                {p[0], p[1], p[2], model.initial_temperature, 0});
        }
        struct BC {
            mhs::model::FluidBoundaryKind kind = mhs::model::FluidBoundaryKind::None;
            double value = 0;
        };
        std::vector<BC> bcs(n);
        for (const auto& boundary : boundaries) {
            std::vector<double> area(n, 0);
            for (const auto& f : mesh.faces) {
                if (f.neighbor.valid())
                    continue;
                auto i = fluid.global_to_fluid[f.owner.value];
                if (i == invalidIndex)
                    continue;
                auto p = f.center();
                for (const auto& r : boundary.regions) {
                    if (static_cast<int>(r.axis) != f.axis
                        || std::abs(r.coordinate * scale - p[f.axis]) >= geometry_eps)
                        continue;
                    const int a = f.axis == 0 ? 1 : 0, b = f.axis == 2 ? 1 : 2;
                    bool match = std::any_of(r.rectangles.begin(), r.rectangles.end(), [&](const auto& rect) {
                        return p[a] >= rect.a_min * scale - geometry_eps && p[a] <= rect.a_max * scale + geometry_eps
                            && p[b] >= rect.b_min * scale - geometry_eps && p[b] <= rect.b_max * scale + geometry_eps;
                    });
                    if (match) {
                        area[i] += f.area;
                        break;
                    }
                }
            }
            const double total = std::accumulate(area.begin(), area.end(), 0.0);
            if (total == 0)
                continue;
            for (Index i = 0; i < n; ++i)
                if (area[i] > 0) {
                    double value = boundary.value;
                    if (boundary.kind == mhs::model::FluidBoundaryKind::MassFlowInlet) {
                        value *= area[i] / total;
                        fluid.boundary_outflux[i] = value;
                    }
                    bcs[i] = {boundary.kind, value};
                    if (!std::isnan(boundary.inlet_temperature))
                        fluid.boundary_temperature[i] = boundary.inlet_temperature;
                }
        }
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
        auto fixed = [&](Index i) {
            return bcs[i].kind == mhs::model::FluidBoundaryKind::PressureInlet
                || bcs[i].kind == mhs::model::FluidBoundaryKind::Outlet;
        };
        bool reference = false;
        std::vector<Eigen::Triplet<double>> entries;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
        for (Index i = 0; i < n; ++i) {
            entries.emplace_back(i, i, diagonal[i]);
            if (fixed(i)) {
                reference = true;
                rhs[i] = bcs[i].value * diagonal[i];
            }
            else if (bcs[i].kind == mhs::model::FluidBoundaryKind::MassFlowInlet)
                rhs[i] = bcs[i].value / rho[i];
        }
        if (!reference)
            throw std::runtime_error(
                "fluid domain has no pressure reference; a pressure inlet or an outlet is required");
        for (Index id = 0; id < mesh.faces.size(); ++id)
            if (conductance[id] != 0) {
                const auto& f = mesh.faces[id];
                auto i = fluid.global_to_fluid[f.owner.value], j = fluid.global_to_fluid[f.neighbor.value];
                if (!fixed(i))
                    entries.emplace_back(i, j, -conductance[id]);
                if (!fixed(j))
                    entries.emplace_back(j, i, -conductance[id]);
            }
        Eigen::SparseMatrix<double> matrix(n, n);
        matrix.setFromTriplets(entries.begin(), entries.end());
        auto solver = mhs::sim::create_solver();
        solver->compute(matrix);
        auto pressure = solver->solve(rhs, Eigen::VectorXd::Zero(n));
        if (!solver->success())
            throw std::runtime_error("fluid pressure solve failed");
        for (Index id = 0; id < mesh.faces.size(); ++id)
            if (conductance[id] != 0) {
                const auto& f = mesh.faces[id];
                auto i = fluid.global_to_fluid[f.owner.value], j = fluid.global_to_fluid[f.neighbor.value];
                fluid.face_volume_flux[id] = (pressure[i] - pressure[j]) * conductance[id];
            }
    }
}
