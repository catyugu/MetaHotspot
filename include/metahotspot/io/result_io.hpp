#pragma once

#include "model/model.hpp"
#include "solver/solution.hpp"

#include "core/vector.hpp"
#include <span>
#include <string>

namespace mhs::io {

    void write_vtu(const std::string& path, const mhs::model::Model& model, std::span<const double> cell_temperature);

    // Every trace follows observation_times. With no traces, omit Result0DTransient.
    void write_xml(const std::string& input_path, const std::string& output_path, const mhs::model::Model& model,
        std::span<const double> node_temperature,
        const mhs::core::Vector<mhs::solver::ProbeTrace>& observation_traces = {},
        std::span<const double> observation_times = {});

} // namespace mhs::io
