#pragma once

#include "data/io_structure.hpp"
#include "data/model.hpp"
#include "expr/expr.hpp"

#include <functional>
#include <string>

namespace mhs::sim {

    struct FaceKeyInfo {
        char axis = 'Z';
        char side = 'E';
        double coord_value = 0.0;
        std::vector<std::array<double, 4>> rects; // {a_min, a_max, b_min, b_max} in SI
    };

    struct ParsedFaceKey {
        FaceKeyInfo fk;
        mhs::core::BcType bc_enum;
        uint16_t param_idx;
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

    // Flatten all (boundary, face_key) pairs and push their BC params into
    // bc_params.  Returns the flattened ParsedFaceKey vector used by the
    // single-pass grid traversal inside resolve_layers.
    //
    // The `rewriter` is applied to every BC string (temperature / heat_flux /
    // convection_coeff / T_inf) before parsing — typically the 字面替换 that
    // turns `name(x)` into `name(T)`.
    // `symbols` is forwarded into every parse() call so the resulting
    // CompiledExpression captures the correct natives/variables.
    std::vector<ParsedFaceKey> parse_all_face_keys(const std::vector<mhs::core::Boundary>& boundaries,
        mhs::core::BCParamTable& bc_params, double si_scale,
        const std::function<std::string(const std::string&)>& rewriter, const mhs::core::SymbolTable& symbols);

} // namespace mhs::sim