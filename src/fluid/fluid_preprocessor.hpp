#pragma once

#include "data/internal_model.hpp"

namespace mhs::sim {

    /**
     * @brief Solves the fluid pressure field and computes flow properties.
     *
     * Four-phase pipeline:
     *   1. init_cell_hydro_properties  — Hele-Shaw hydraulic conductance per axis
     *   2. apply_pressure_boundary_conditions — marks already-set pressure BCs
     *   3. solve_pressure              — Poisson matrix + SparseLU solve
     *   4. precompute_flow_axes        — dominant flow axis per fluid cell
     *
     * If no fluid cells exist (is_fluid all false), the method returns immediately.
     */
    class FluidPreprocessor {
    public:
        FluidPreprocessor() = default;
        ~FluidPreprocessor() = default;

        void solveFlow(mhs::core::InternalModel& model);

    private:
        void initCellHydroProperties(mhs::core::InternalModel& model);
        void solvePressure(mhs::core::InternalModel& model);
        void precomputeFlowAxes(mhs::core::InternalModel& model);

        /// Harmonic mean of two conductances
        static double harmonicConductance(double a, double b)
        {
            if (a < 1e-30 || b < 1e-30)
                return 0.0;
            return (2.0 * a * b) / (a + b);
        }
    };

} // namespace mhs::sim
