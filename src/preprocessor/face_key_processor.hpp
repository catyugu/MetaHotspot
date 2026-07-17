#pragma once

#include "data/model.hpp"
#include "data/model_definition.hpp"
#include "expr/expr.hpp"
#include "utils/face_key.hpp"

#include <functional>
#include <string>

namespace mhs::sim {

    /// `overloaded` helper for std::visit: `std::visit(overloaded{lambda1, lambda2, ...}, variant)`.
    template <typename... Ts> struct overloaded : Ts... {
        using Ts::operator()...;
    };
    template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

    using mhs::utils::FaceKeyInfo;
    using mhs::utils::parse_face_key;
    using mhs::utils::point_in_face_rects;

    struct ParsedFaceKey {
        FaceKeyInfo fk;
        mhs::core::BcType bc_enum;
        uint16_t param_idx;
    };

    /// Fallback boundary condition used for faces that don't match any explicit boundary key.
    struct OtherBC {
        mhs::core::BcType type = mhs::core::BcType::None;
        uint16_t param_idx = 0;
    };

    // Flatten all (boundary, face_key) pairs and push their BC params into
    // bc_params.  Returns the flattened ParsedFaceKey vector consumed by
    // resolve_boundary_patches to match boundary conditions against cell faces.
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
