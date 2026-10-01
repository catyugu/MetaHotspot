#pragma once

#include "core/model_definition.hpp"

#include <string>

namespace mhs::io {

    mhs::model::ModelDefinition read_xml(const std::string& xml_path);

} // namespace mhs::io
