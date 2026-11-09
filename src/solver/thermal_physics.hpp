#pragma once

#include "model/model.hpp"

namespace mhs::solver {

    // Unit-area Fourier conduction resistance; no fluid/interface model involved.
    inline double side_thermal_resistance(double d, double k) { return d / k; }

    // Fluid bulk-to-wall film resistance, separate from the conductive side law.
    inline double fluid_film_resistance(double nu_over_d, double k) { return 1.0 / (nu_over_d * k); }

    // Algebraic elimination of an external boundary's surface temperature.
    struct BoundaryTrace {
        double temperature = 0.0;
        double conductance = 0.0;
        double load = 0.0;
    };

    // Evaluate a thermal boundary law locally, with no mesh or Model dependency.
    BoundaryTrace evaluate_boundary(const mhs::model::BoundaryCondition& condition, const mhs::model::Vec3& center,
        double area, double normal_distance, double cell_temperature, double normal_conductivity, double time);

} // namespace mhs::solver
