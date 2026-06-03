#pragma once

#include "common/internal_model.hpp"
#include "common/io_model.hpp"
#include <string>
#include <vector>

namespace mhs::io {

    model::IOStructure read_xml(const std::string& xml_path);

    void write_vtu(const std::string& path,
        const model::InternalModel& model,
        const std::vector<double>& node_temperature);

    void write_xml(const std::string& input_path,
        const std::string& output_path,
        const model::InternalModel& model,
        const std::vector<double>& node_temperature);

} // namespace mhs::io