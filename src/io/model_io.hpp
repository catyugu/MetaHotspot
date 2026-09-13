#pragma once

#include "core/model_definition.hpp"

#include <string>

namespace mhs::io {

    mhs::model::ModelDefinition read_xml(const std::string& xml_path);

    // Merge an optional fluid XML document into an existing model definition.
    // Throws std::runtime_error if the file cannot be loaded.
    void merge_fluid_xml(const std::string& xml_path, mhs::model::ModelDefinition& definition);

} // namespace mhs::io
