#pragma once

#include "core/model.hpp"
#include "core/solution.hpp"

#include "core/vector.hpp"
#include <span>
#include <string>

namespace mhs::io {

    void write_vtu(const std::string& path, const mhs::core::Model& model, std::span<const double> cell_temperature);

    // Every trace follows observation_times. With no traces, omit Result0DTransient.
    void write_xml(const std::string& input_path, const std::string& output_path, const mhs::core::Model& model,
        std::span<const double> node_temperature,
        const mhs::core::Vector<mhs::core::ProbeTrace>& observation_traces = {},
        std::span<const double> observation_times = {});

} // namespace mhs::io
