#pragma once

#include "model/model_definition.hpp"

#include <string>

namespace mhs::io {

    mhs::model::ModelDefinition read_xml(const std::string& xml_path);

    // Merge an optional fluid XML document into an existing model definition.
    // Returns false when the file cannot be loaded or has no FluidOverlay element.
    bool merge_fluid_xml(const std::string& xml_path, mhs::model::ModelDefinition& definition);

} // namespace mhs::io
