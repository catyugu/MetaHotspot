#pragma once

#include "model/internal_model.hpp"
#include <string>
#include <vector>

namespace mhs {

    class Postprocessor {
    public:
        Postprocessor() = default;
        ~Postprocessor() = default;

        void writeVTU(const std::string& path, const model::InternalModel& model, const std::vector<double>& solution);

        void writeXML(const std::string& path, const model::InternalModel& model, const std::vector<double>& solution);
    };

} // namespace mhs
