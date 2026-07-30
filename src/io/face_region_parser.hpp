#pragma once

#include "mhs/model_definition.hpp"

#include <string>

namespace mhs::io::detail {

    mhs::model::FaceRegion parse_face_region(const std::string& encoded);

} // namespace mhs::io::detail
