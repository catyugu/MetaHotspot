#include "data/types.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"

namespace mhs::sim {

    std::vector<ParsedFaceKey> parse_all_face_keys(const std::vector<mhs::core::Boundary>& boundaries,
        mhs::core::BCParamTable& bc_params, double si_scale,
        const std::function<std::string(const std::string&)>& rewriter, const mhs::core::SymbolTable& symbols)
    {
        std::vector<ParsedFaceKey> parsed_keys;

        for (const auto& boundary : boundaries) {
            OtherBC other_bc;
            std::visit(overloaded {
                           [&](const mhs::core::FirstTypeThermalBC& b) {
                               other_bc.type = mhs::core::BcType::FirstType;
                               other_bc.param_idx = static_cast<uint16_t>(bc_params.dirichlet_T.size());
                               bc_params.dirichlet_T.push_back(mhs::core::parse(rewriter(b.temperature), symbols));
                           },
                           [&](const mhs::core::SecondTypeThermalBC& b) {
                               other_bc.type = mhs::core::BcType::SecondType;
                               other_bc.param_idx = static_cast<uint16_t>(bc_params.neumann_q.size());
                               bc_params.neumann_q.push_back(mhs::core::parse(rewriter(b.heat_flux), symbols));
                           },
                           [&](const mhs::core::ThirdTypeThermalBC& b) {
                               other_bc.type = mhs::core::BcType::ThirdType;
                               other_bc.param_idx = static_cast<uint16_t>(bc_params.cauchy_h.size());
                               bc_params.cauchy_h.push_back(mhs::core::parse(rewriter(b.convection_coeff), symbols));
                               bc_params.cauchy_T_inf.push_back(mhs::core::parse(rewriter(b.T_inf), symbols));
                           },
                       },
                boundary.bc);

            for (const auto& key_str : boundary.face_keys) {
                parsed_keys.push_back({parse_face_key(key_str, si_scale), other_bc.type, other_bc.param_idx});
            }
        }

        return parsed_keys;
    }

} // namespace mhs::sim
