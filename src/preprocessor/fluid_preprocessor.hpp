#pragma once

#include "data/io_structure.hpp"
#include "data/model.hpp"
#include "expr/expr.hpp"

#include <optional>
#include <string>
#include <vector>

namespace mhs::sim {

    // Scratch data shared by the fluid-domain builder and the one-time flow
    // solve. It is owned by Preprocessor::load and released when preprocessing
    // finishes; no process-global state is retained between simulations.
    struct FluidPreprocessWorkspace {
        std::vector<mhs::Index> active_to_grid;
    };

    std::optional<FluidPreprocessWorkspace> buildFluidDomain(mhs::core::Model& model,
        const mhs::core::FluidOverlay& overlay, const mhs::core::IOStructure& io_structure,
        const mhs::core::SymbolTable& symbols, const std::vector<std::string>& material_names);

    void solveFluidFlow(mhs::core::Model& model, const FluidPreprocessWorkspace& workspace);

} // namespace mhs::sim
