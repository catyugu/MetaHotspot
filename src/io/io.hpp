#pragma once

#include <optional>
#include <string>
#include <vector>

#include "data/io_model.hpp"
#include "data/model.hpp"


namespace mhs::io {

    mhs::core::IOStructure read_xml(const std::string& xml_path);

    // Read fluid overlay XML; returns std::nullopt if file doesn't exist or has no FluidOverlay element.
    std::optional<mhs::core::FluidOverlay> read_fluid_overlay_xml(const std::string& xml_path);

    void write_vtu(const std::string& path, const mhs::core::Model& model, const std::vector<double>& node_temperature);

    // observation_traces 默认空：稳态 case 走原路径，不写 Result0DTransient。
    void write_xml(const std::string& input_path, const std::string& output_path, const mhs::core::Model& model,
        const std::vector<double>& node_temperature, const std::vector<mhs::core::ProbeTrace>& observation_traces = {});

} // namespace mhs::io