#pragma once

#include <array>
#include <string>
#include <vector>

namespace mhs::utils {

    struct FaceKeyInfo {
        char axis = 'Z';
        char side = 'E';
        double coord_value = 0.0;
        std::vector<std::array<double, 4>> rects;
    };

    FaceKeyInfo parse_face_key(const std::string& key, double si_scale);

    bool point_in_face_rects(const FaceKeyInfo& fk, double a, double b);

} // namespace mhs::utils
