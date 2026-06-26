#include "data/types.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"

namespace mhs::sim {

    namespace {
        // Split a string by a single-character delimiter.
        std::vector<std::string> split(const std::string& s, char delim)
        {
            std::vector<std::string> out;
            std::string token;
            for (char c : s) {
                if (c == delim) {
                    out.push_back(token);
                    token.clear();
                }
                else {
                    token += c;
                }
            }
            if (!token.empty())
                out.push_back(token);
            return out;
        }

        // Parse the comma/semicolon rect list used by Z-face keys:
        //   "0,50,50,100;50,100,0,50" -> two rects, each {xmin,xmax,ymin,ymax}
        std::vector<std::array<double, 4>> parse_z_rects(const std::string& rect_str, double si_scale)
        {
            std::vector<std::array<double, 4>> rects;
            for (const auto& rp : split(rect_str, ';')) {
                std::vector<double> vals;
                for (const auto& v : split(rp, ','))
                    vals.push_back(std::stod(v) * si_scale);
                if (vals.size() == 4)
                    rects.push_back({vals[0], vals[1], vals[2], vals[3]});
            }
            return rects;
        }

        // Parse the four pipe-delimited rect numbers used by X/Y-face keys:
        //   parts[3..6] = Min1, Max1, Min2, Max2
        std::vector<std::array<double, 4>> parse_xy_rect(const std::vector<std::string>& parts, double si_scale)
        {
            if (parts.size() < 7)
                return {};
            return {{std::stod(parts[3]) * si_scale, std::stod(parts[4]) * si_scale, std::stod(parts[5]) * si_scale,
                std::stod(parts[6]) * si_scale}};
        }
    } // anonymous namespace

    FaceKeyInfo parse_face_key(const std::string& key, double si_scale)
    {
        FaceKeyInfo info;
        auto parts = split(key, '|');
        if (parts.size() < 3)
            return info;

        info.axis = parts[0][0];
        info.side = parts[1][0];
        info.coord_value = std::stod(parts[2]) * si_scale;

        if (info.axis == 'Z') {
            if (parts.size() == 4)
                info.rects = parse_z_rects(parts[3], si_scale);
        }
        else if (info.axis == 'X' || info.axis == 'Y') {
            info.rects = parse_xy_rect(parts, si_scale);
        }
        return info;
    }

    bool point_in_face_rects(const FaceKeyInfo& fk, double a, double b)
    {
        constexpr double EPS = 1e-9;
        for (const auto& rect : fk.rects) {
            if (a >= rect[0] - EPS && a <= rect[1] + EPS && b >= rect[2] - EPS && b <= rect[3] + EPS) {
                return true;
            }
        }
        return false;
    }

    std::vector<ParsedFaceKey> parse_all_face_keys(const std::vector<mhs::core::Boundary>& boundaries,
        mhs::core::BCParamTable& bc_params, double si_scale,
        const std::function<std::string(const std::string&)>& rewriter, const mhs::core::SymbolTable& symbols)
    {
        std::vector<ParsedFaceKey> parsed_keys;

        for (const auto& boundary : boundaries) {
            uint16_t bc_param_idx = 0;
            mhs::core::BcType bc_enum = mhs::core::BcType::None;

            switch (boundary.bc_type) {
            case mhs::core::ThermalBCType::FirstType:
                bc_enum = mhs::core::BcType::FirstType;
                bc_param_idx = (uint16_t)bc_params.dirichlet_T.size();
                bc_params.dirichlet_T.push_back(mhs::core::parse(rewriter(boundary.first.temperature), symbols));
                break;
            case mhs::core::ThermalBCType::SecondType:
                bc_enum = mhs::core::BcType::SecondType;
                bc_param_idx = (uint16_t)bc_params.neumann_q.size();
                bc_params.neumann_q.push_back(mhs::core::parse(rewriter(boundary.second.heat_flux), symbols));
                break;
            case mhs::core::ThermalBCType::ThirdType:
                bc_enum = mhs::core::BcType::ThirdType;
                bc_param_idx = (uint16_t)bc_params.cauchy_h.size();
                bc_params.cauchy_h.push_back(mhs::core::parse(rewriter(boundary.third.convection_coeff), symbols));
                bc_params.cauchy_T_inf.push_back(mhs::core::parse(rewriter(boundary.third.T_inf), symbols));
                break;
            }

            for (const auto& key_str : boundary.face_keys) {
                parsed_keys.push_back({parse_face_key(key_str, si_scale), bc_enum, bc_param_idx});
            }
        }

        return parsed_keys;
    }

} // namespace mhs::sim
