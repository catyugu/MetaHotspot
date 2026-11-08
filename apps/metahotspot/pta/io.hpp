#pragma once

#include "core/vector.hpp"
#include "model/model_definition.hpp"
#include "solver/solution.hpp"
#include "solver/solver.hpp"

#include <array>
#include <span>
#include <string>

namespace mhs::app::pta {

    struct Input : mhs::model::ModelDefinition {
        mhs::solver::Study study;
    };

    Input read_xml(const std::string& xml_path);

    // Every trace follows observation_times. With no traces, omit Result0DTransient.
    void write_xml(const std::string& input_path, const std::string& output_path,
        const std::array<std::size_t, 3>& shape, std::span<const double> node_temperature,
        const mhs::core::Vector<mhs::solver::ProbeTrace>& observation_traces = {},
        std::span<const double> observation_times = {});

} // namespace mhs::app::pta
