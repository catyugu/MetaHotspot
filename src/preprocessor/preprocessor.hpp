#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "data/internal_model.hpp"
#include "data/io_model.hpp"

namespace mhs::sim {

    /**
     * @brief Reads mhs::core::IOStructure and converts to internal SoA representation
     */
    class Preprocessor {
    public:
        Preprocessor() = default;
        ~Preprocessor() = default;

        std::unique_ptr<mhs::core::InternalModel> load(const mhs::core::IOStructure& ioStructure);

        /**
         * @brief Apply a fluid overlay to an already-loaded model.
         *
         * If overlay is empty or no fluid materials match, the model is left unchanged.
         */
        void applyFluidOverlay(mhs::core::InternalModel& model, const std::optional<mhs::core::FluidOverlay>& overlay,
            const mhs::core::IOStructure& ioStructure);
    };

} // namespace mhs::sim
