#pragma once

#include "common/model_definition.hpp"

#include <initializer_list>

namespace mhs::test {

    inline mhs::model::FaceRegion face_region(
        mhs::model::Axis axis, double coordinate, std::initializer_list<mhs::model::RegionRect> rectangles)
    { return {axis, coordinate, rectangles}; }

} // namespace mhs::test
