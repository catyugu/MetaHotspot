#pragma once

#include <string>
#include <vector>

#include "common/internal_model.hpp"
#include "common/io_model.hpp"

namespace mhs::io {

    mhs::IOStructure read_xml(const std::string& xml_path);

    void write_vtu(
        const std::string& path, const mhs::InternalModel& model, const std::vector<double>& node_temperature);

    // observation_traces 默认空：稳态 case 走原路径，不写 Result0DTransient。
    void write_xml(const std::string& input_path, const std::string& output_path, const mhs::InternalModel& model,
        const std::vector<double>& node_temperature, const std::vector<mhs::ProbeTrace>& observation_traces = {});

} // namespace mhs::io