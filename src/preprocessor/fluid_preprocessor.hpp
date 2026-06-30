#pragma once

#include "data/internal_model.hpp"
#include "data/io_model.hpp"
#include "expr/expr.hpp"
#include <optional>

namespace mhs::sim {

    /**
     * @brief Apply a fluid overlay to a loaded model.
     *
     * Marks fluid cells, builds the fluid indirection mapping (fluid_to_global /
     * global_to_fluid), sets up pressure boundary conditions, and computes per-cell
     * channel geometry (hydraulic diameter, width, height).
     *
     * If overlay is empty or no fluid materials match, the model is left unchanged.
     */
    void applyFluidOverlay(mhs::core::InternalModel& model, const std::optional<mhs::core::FluidOverlay>& overlay,
        const mhs::core::IOStructure& ioStructure, const mhs::core::SymbolTable& symbols);

    /**
     * @brief Solve the fluid flow field after the overlay is applied.
     *
     * Three-phase pipeline (all internal):
     *   1. initCellHydroProperties  — Hele-Shaw hydraulic conductance per axis
     *   2. solvePressure            — Poisson matrix + BiCGSTAB solve
     *   3. precomputeFlowAxes       — dominant flow axis per fluid cell
     *
     * If no fluid cells exist (is_fluid all false), returns immediately.
     */
    void solveFluidFlow(mhs::core::InternalModel& model);

} // namespace mhs::sim
