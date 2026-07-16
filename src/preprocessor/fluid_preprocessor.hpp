#pragma once

#include "data/model_definition.hpp"
#include "data/model.hpp"

namespace mhs::sim {

    /**
     * @brief Prepare the fluid-domain data already present in a model definition.
     *
     * Marks fluid cells, builds the fluid indirection mapping (fluid_to_global /
     * global_to_fluid), sets up pressure boundary conditions, and computes per-cell
     * channel geometry (hydraulic diameter, width, height).
     *
     * If no cells reference fluid materials, the model is left unchanged.
     */
    void prepare_fluid_domain(mhs::core::Model& model,
        const std::vector<mhs::core::FluidBoundary>& boundaries, double si_scale);

    /**
     * @brief Solve the fluid flow field after the fluid domain is prepared.
     *
     * Three-phase pipeline (all internal):
     *   1. initCellHydroProperties  — Hele-Shaw hydraulic conductance per axis
     *   2. solvePressure            — Poisson matrix + EigenBiCGSTAB solve
     *   3. precomputeFlowAxes       — dominant flow axis per fluid cell
     *
     * If no fluid cells exist (is_fluid all false), returns immediately.
     */
    void solve_fluid_flow(mhs::core::Model& model);

} // namespace mhs::sim
