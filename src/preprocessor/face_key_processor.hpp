#pragma once

#include "model/internal_model.hpp"
#include "model/io_model.hpp"

namespace mhs::preprocessor {

    struct FaceKeyInfo {
        char axis = 'Z';
        char side = 'E';
        double coord_value = 0.0;
        std::vector<std::array<double, 4>> rects; // {a_min, a_max, b_min, b_max} in SI
    };

    // Parse a face key string like "Z|E|0|0,50,50,100;50,100,0,50"
    FaceKeyInfo parse_face_key(const std::string& key, double si_scale);

    // Check if a 2D point is inside any of the face key rectangles
    bool point_in_face_rects(const FaceKeyInfo& fk, double a, double b);

    // Resolve BCs: assign CellBC per cell per face from boundaries + other_bc + virtual neighbors
    void resolve_face_keys(const std::vector<model::Boundary>& boundaries,
        model::ThermalBCType other_bc_type,
        const model::FirstTypeThermalBC& other_bc_first,
        const model::SecondTypeThermalBC& other_bc_second,
        const model::ThirdTypeThermalBC& other_bc_third,
        const model::MeshGeometry& mesh,
        model::CellFields& cells,
        model::BCParamTable& bc_params,
        double si_scale);

} // namespace mhs::preprocessor