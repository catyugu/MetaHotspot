#pragma once

#include "data/io_model.hpp"
#include "data/model.hpp"
#include <memory>
#include <optional>


namespace mhs::sim {

    /**
     * @brief Reads mhs::core::IOStructure and converts to internal SoA representation
     *
     * If `fluidOverlay` is provided, the fluid overlay (viscosity expressions,
     * pressure BCs, hydraulic channel geometry) is applied before returning.
     * The overlay's viscosity expressions are compiled with the same SymbolTable
     * the rest of the model is built from, so they share variable / native names.
     */
    class Preprocessor {
    public:
        Preprocessor() = default;
        ~Preprocessor() = default;

        std::unique_ptr<mhs::core::Model> load(const mhs::core::IOStructure& ioStructure,
            const std::optional<mhs::core::FluidOverlay>& fluidOverlay = std::nullopt);
    };

} // namespace mhs::sim
