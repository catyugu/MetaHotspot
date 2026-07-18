#pragma once

#include "runtime/model.hpp"
#include "runtime/solution.hpp"

#include <string>
#include <vector>

namespace mhs::io {

    void write_vtu(const std::string& path, const mhs::core::Model& model, const std::vector<double>& node_temperature);

    // observation_traces 默认空：稳态 case 走原路径，不写 Result0DTransient。
    void write_xml(const std::string& input_path, const std::string& output_path, const mhs::core::Model& model,
        const std::vector<double>& node_temperature, const std::vector<mhs::core::ProbeTrace>& observation_traces = {});

} // namespace mhs::io
