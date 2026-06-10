#pragma once

#include <memory>

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
    };

} // namespace mhs::sim
