#pragma once

#include <memory>

#include "common/internal_model.hpp"
#include "common/io_model.hpp"

namespace mhs {

    /**
     * @brief Reads IOStructure and converts to internal SoA representation
     */
    class Preprocessor {
    public:
        Preprocessor() = default;
        ~Preprocessor() = default;

        std::unique_ptr<model::InternalModel> load(const model::IOStructure& ioStructure);
    };

} // namespace mhs
