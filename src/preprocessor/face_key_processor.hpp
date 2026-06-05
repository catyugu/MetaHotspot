#pragma once

#include "common/internal_model.hpp"
#include "common/io_model.hpp"

#include <functional>
#include <string>

namespace mhs::preprocessor {

    struct FaceKeyInfo {
        char axis = 'Z';
        char side = 'E';
        double coord_value = 0.0;
        std::vector<std::array<double, 4>> rects; // {a_min, a_max, b_min, b_max} in SI
    };

    // Parse a face key string. Two formats are supported, selected by axis:
    //
    //   Z-face (4 parts, comma/semicolon rect list):
    //     "Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100"
    //     -> Face|Direction|CoordValue|xmin,xmax,ymin,ymax;...
    //
    //   X/Y-face (7 parts, pipe-delimited rect):
    //     "X|E|5|-7.5|7.5|26|29"
    //     -> Face|Direction|CoordValue|Min1|Max1|Min2|Max2
    //
    // For X-faces the 2D rect is (cy, cz); for Y-faces it is (cx, cz);
    // for Z-faces it is (cx, cy). The caller picks the right cell-center
    // pair when matching, see resolve_face_keys.
    FaceKeyInfo parse_face_key(const std::string& key, double si_scale);

    // Check if a 2D point is inside any of the face key rectangles
    bool point_in_face_rects(const FaceKeyInfo& fk, double a, double b);

    // Resolve BCs: assign CellBC per cell per face from boundaries + other_bc + virtual neighbors.
    // The `rewriter` is applied to every BC string (temperature / heat_flux / convection_coeff /
    // T_inf) before parsing — typically the 字面替换 that turns `name(x)` into `name(T)`.
    void resolve_face_keys(const std::vector<Boundary>& boundaries, ThermalBCType other_bc_type,
        const FirstTypeThermalBC& other_bc_first, const SecondTypeThermalBC& other_bc_second,
        const ThirdTypeThermalBC& other_bc_third, const MeshGeometry& mesh, CellFields& cells,
        BCParamTable& bc_params, double si_scale,
        const std::function<std::string(const std::string&)>& rewriter);

} // namespace mhs::preprocessor