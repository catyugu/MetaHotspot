#pragma once

#include <string>
#include <vector>

#include "compiler/runtime_model.hpp"
#include "solver/solution.hpp"
#include "model/model_definition.hpp"

namespace mhs::io {

    mhs::model::ModelDefinition read_xml(const std::string& xml_path);

    // Merge an optional fluid XML document into an existing model definition.
    // Returns false when the file cannot be loaded or has no FluidOverlay element.
    bool merge_fluid_xml(const std::string& xml_path, mhs::model::ModelDefinition& definition);

    void write_vtu(const std::string& path, const mhs::core::Model& model, const std::vector<double>& node_temperature);

    // observation_traces 默认空：稳态 case 走原路径，不写 Result0DTransient。
    void write_xml(const std::string& input_path, const std::string& output_path, const mhs::core::Model& model,
        const std::vector<double>& node_temperature, const std::vector<mhs::core::ProbeTrace>& observation_traces = {});

} // namespace mhs::io
