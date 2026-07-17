#include "utils/face_key.hpp"

#include "data/tolerance_config.hpp"

namespace mhs::utils {

    namespace {
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

        std::vector<std::array<double, 4>> parse_xy_rect(const std::vector<std::string>& parts, double si_scale)
        {
            if (parts.size() < 7)
                return {};
            return {{std::stod(parts[3]) * si_scale, std::stod(parts[4]) * si_scale, std::stod(parts[5]) * si_scale,
                std::stod(parts[6]) * si_scale}};
        }
    } // namespace

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
        for (const auto& rect : fk.rects) {
            if (a >= rect[0] - mhs::core::geometry_eps && a <= rect[1] + mhs::core::geometry_eps
                && b >= rect[2] - mhs::core::geometry_eps && b <= rect[3] + mhs::core::geometry_eps) {
                return true;
            }
        }
        return false;
    }

} // namespace mhs::utils
