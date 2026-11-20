#pragma once

#include "model/project.hpp"
#include <utility>

namespace mhs::solver {

    inline const model::BoundaryCondition* boundary_override(const model::BoundaryOverrides& overrides,
        model::BoundaryGroupId boundary_group, const model::BoundaryCondition* fallback = nullptr)
    {
        return boundary_group < overrides.size() && overrides[boundary_group] ? &*overrides[boundary_group] : fallback;
    }

    // Unit-area Fourier conduction resistance; no fluid/interface model involved.
    inline double side_thermal_resistance(double d, double k) { return d / k; }

    inline double cell_conductivity(
        const model::Fom& fom, model::CellId cell, std::size_t axis, double temperature, double time)
    {
        const auto p = fom.discrete.cells[cell].center();
        const auto& m = fom.material_table[fom.cell_assignment[cell].material];
        return (axis == 0 ? m.kx : axis == 1 ? m.ky : m.kz).eval({p[0], p[1], p[2], temperature, time});
    }

    inline double boundary_resistance(const model::Fom& fom, model::FacetId id, double k)
    {
        const auto cell = fom.discrete.cell_facets[id].owner();
        const auto axis = static_cast<std::size_t>(fom.discrete.facets[id].axis);
        return side_thermal_resistance(fom.discrete.cells[cell].size()[axis] / 2, k);
    }

    inline double boundary_resistance(const model::Fom& fom, model::FacetId id, double temperature, double time)
    {
        const auto& facet = fom.discrete.facets[id];
        return boundary_resistance(fom, id,
            cell_conductivity(
                fom, fom.discrete.cell_facets[id].owner(), static_cast<std::size_t>(facet.axis), temperature, time));
    }

    inline double convection_conductance(double area, double h, double resistance)
    {
        return std::abs(h) <= 1 ? area * h / (1 + resistance * h) : area / (resistance + 1 / h);
    }

    inline double contact_resistance(double left, double right, double layer) { return left + right + layer; }

    // Fluid bulk-to-wall film resistance, separate from the conductive side law.
    inline double fluid_film_resistance(double nu_over_d, double k) { return 1.0 / (nu_over_d * k); }

    // Unit-area resistances from the minus/plus cell centers to an interior
    // facet. For solid-fluid interfaces the fluid-side resistance is the film.
    std::pair<double, double> interior_facet_resistances(
        const mhs::model::Fom& model, mhs::model::FacetId id, double k_minus, double k_plus);

    // Algebraic elimination of an external boundary's surface temperature.
    struct BoundaryEvaluation {
        double temperature = 0.0;
        double conductance = 0.0;
        double load = 0.0;
        bool dirichlet = false;
    };

    BoundaryEvaluation evaluate_boundary(const model::BoundaryCondition& condition, const model::Facet& patch,
        double temperature, double time, double resistance);

    // Boundary expressions and point probes use component coordinates.
    inline model::Facet local_patch(
        const model::Facet& facet, const model::Rect& world_bounds, const model::Vec3& offset)
    {
        auto result = facet;
        const auto [u, v] = model::tangential_axes(facet.axis);
        for (std::size_t i = 0; i < world_bounds.size(); ++i) {
            const double shift = offset[i < 2 ? u : v];
            result.bounds[i] = world_bounds[i] == facet.bounds[i] + shift ? facet.bounds[i] : world_bounds[i] - shift;
        }
        return result;
    }

    inline std::pair<double, double> boundary_terms(
        model::Facet facet, const model::FacePartition* partition, const auto& evaluate)
    {
        if (!partition) {
            const auto trace = evaluate(facet);
            return {trace.conductance, trace.load};
        }
        double conductance = 0, load = 0;
        for (const auto& bounds : partition->exposed) {
            facet.bounds = bounds;
            const auto trace = evaluate(facet);
            conductance += trace.conductance;
            load += trace.load;
        }
        return {conductance, load};
    }

    inline bool is_adiabatic(const model::BoundaryCondition& condition)
    {
        const auto* flux = std::get_if<model::Neumann>(&condition);
        return flux && flux->heat_flux.is_constant() && flux->heat_flux.constant_value() == 0;
    }

} // namespace mhs::solver
