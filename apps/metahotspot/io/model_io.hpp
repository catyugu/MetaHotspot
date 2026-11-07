#pragma once

#include "model/model_definition.hpp"

#include <string>

namespace mhs::io {

    mhs::model::ModelDefinition read_xml(const std::string& xml_path);

} // namespace mhs::io
