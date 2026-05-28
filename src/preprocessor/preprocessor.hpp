#pragma once

#include <memory>
#include <string>

#include "model/internal_model.hpp"
#include "model/io_model.hpp"

namespace mhs {

    /**
     * @brief Reads XML model file and converts to internal SoA representation
     */
    class Preprocessor {
    public:
        Preprocessor() = default;
        ~Preprocessor() = default;

        std::unique_ptr<model::InternalModel> load(const std::string& xmlPath);
    };

} // namespace mhs
