#pragma once

#include "mhs/model.hpp"
#include "mhs/solution.hpp"

#include <span>
#include <string>
#include <vector>

namespace mhs::io {

    void write_vtu(const std::string& path, const mhs::core::Model& model, std::span<const double> cell_temperature);

    // observation_traces 默认空：稳态 case 走原路径，不写 Result0DTransient。
    void write_xml(const std::string& input_path, const std::string& output_path, const mhs::core::Model& model,
        std::span<const double> node_temperature, const std::vector<mhs::core::ProbeTrace>& observation_traces = {});

} // namespace mhs::io
