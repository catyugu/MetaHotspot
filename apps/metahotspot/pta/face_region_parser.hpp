#pragma once

#include "model/model_definition.hpp"

#include <string>

namespace mhs::app::pta::detail {

    mhs::core::Vector<mhs::model::SurfaceRegion> parse_face_region(const std::string& encoded);

} // namespace mhs::app::pta::detail
