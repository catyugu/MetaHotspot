#pragma once

#include "model/fom.hpp"
#include <utility>

namespace mhs::solver {

    // Unit-area Fourier conduction resistance; no fluid/interface model involved.
    inline double side_thermal_resistance(double d, double k) { return d / k; }

    // Fluid bulk-to-wall film resistance, separate from the conductive side law.
    inline double fluid_film_resistance(double nu_over_d, double k) { return 1.0 / (nu_over_d * k); }

    // Unit-area resistances from the minus/plus cell centers to an interior
    // facet. For solid-fluid interfaces the fluid-side resistance is the film.
    std::pair<double, double> interior_facet_resistances(
        const mhs::model::Fom& model, mhs::model::FacetId id, double k_minus, double k_plus);

    // Algebraic elimination of an external boundary's surface temperature.
    struct BoundaryTrace {
        double temperature = 0.0;
        double conductance = 0.0;
        double load = 0.0;
    };

    // Physical law on one boundary integration region. Context holds the cell
    // temperature and time; conductance is cell-to-surface, in W/K.
    BoundaryTrace evaluate_boundary(const mhs::model::BoundaryCondition& condition,
        mhs::expression::FieldContext context, double area, double conductive);

    inline BoundaryTrace evaluate_boundary(const model::BoundaryCondition& condition, const model::Facet& patch,
        double temperature, double time, double resistance)
    {
        const auto p = patch.center();
        const double area = patch.area();
        return evaluate_boundary(condition, {p[0], p[1], p[2], temperature, time}, area, area / resistance);
    }

    inline bool is_adiabatic(const model::BoundaryCondition& condition)
    {
        const auto* flux = std::get_if<model::Neumann>(&condition);
        return flux && flux->heat_flux.is_constant() && flux->heat_flux.constant_value() == 0;
    }

} // namespace mhs::solver
