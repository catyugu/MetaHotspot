#include "common/face_dir_tables.hpp"
#include "common/types.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include <cmath>

namespace mhs::sim {

    namespace {
        // 核心抽象：判断一个面的外侧是否“暴露”（即邻居是域外或者虚拟单元）
        bool is_face_exposed(mhs::core::FaceDir dir, int ix, int iy, int iz, const mhs::core::MeshGeometry& mesh,
            const mhs::core::CellFields& cells)
        {
            int nix = mhs::core::neighbor_ix(dir, ix);
            int niy = mhs::core::neighbor_iy(dir, iy);
            int niz = mhs::core::neighbor_iz(dir, iz);

            // 1. 如果超出网格边界，说明是域外，绝对暴露
            if (nix < 0 || nix >= mesh.nx || niy < 0 || niy >= mesh.ny || niz < 0 || niz >= mesh.nz) {
                return true;
            }

            // 2. 如果内部对应邻居是空洞（虚拟单元），说明面暴露在了内部孔隙中
            int neighbor = nix * mesh.ny * mesh.nz + niy * mesh.nz + niz;
            return cells.valid_mask[neighbor] == 0;
        }

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

    void resolve_face_keys(const std::vector<mhs::core::Boundary>& boundaries, mhs::core::ThermalBCType other_bc_type,
        const mhs::core::FirstTypeThermalBC& other_bc_first, const mhs::core::SecondTypeThermalBC& other_bc_second,
        const mhs::core::ThirdTypeThermalBC& other_bc_third, const mhs::core::MeshGeometry& mesh,
        mhs::core::CellFields& cells, mhs::core::BCParamTable& bc_params, double si_scale,
        const std::function<std::string(const std::string&)>& rewriter)
    {
        // 1. 初始化所有 BC 为 None
        for (int c = 0; c < cells.cell_count; c++) {
            for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
                cells.cell_bcs[c].types[f] = mhs::core::BcType::None;
                cells.cell_bcs[c].param_idxs[f] = 0;
            }
        }

        // 2. 预存 other_bc 参数（索引始终为 0）
        uint16_t other_idx = 0;
        mhs::core::BcType other_bc_enum = mhs::core::BcType::None;
        switch (other_bc_type) {
        case mhs::core::ThermalBCType::FirstType:
            other_bc_enum = mhs::core::BcType::FirstType;
            bc_params.dirichlet_T.push_back(mhs::core::parse(rewriter(other_bc_first.temperature)));
            break;
        case mhs::core::ThermalBCType::SecondType:
            other_bc_enum = mhs::core::BcType::SecondType;
            bc_params.neumann_q.push_back(mhs::core::parse(rewriter(other_bc_second.heat_flux)));
            break;
        case mhs::core::ThermalBCType::ThirdType:
            other_bc_enum = mhs::core::BcType::ThirdType;
            bc_params.cauchy_h.push_back(mhs::core::parse(rewriter(other_bc_third.convection_coeff)));
            bc_params.cauchy_T_inf.push_back(mhs::core::parse(rewriter(other_bc_third.T_inf)));
            break;
        }

        // 3. 将所有 (boundary, face_key) 组合展平为 ParsedFaceKey 数组，
        //    使得后续只需对网格做单次遍历。
        struct ParsedFaceKey {
            FaceKeyInfo fk;
            mhs::core::BcType bc_enum;
            uint16_t param_idx;
        };
        std::vector<ParsedFaceKey> parsed_keys;

        for (const auto& boundary : boundaries) {
            uint16_t bc_param_idx = 0;
            mhs::core::BcType bc_enum = mhs::core::BcType::None;

            switch (boundary.bc_type) {
            case mhs::core::ThermalBCType::FirstType:
                bc_enum = mhs::core::BcType::FirstType;
                bc_param_idx = (uint16_t)bc_params.dirichlet_T.size();
                bc_params.dirichlet_T.push_back(mhs::core::parse(rewriter(boundary.first.temperature)));
                break;
            case mhs::core::ThermalBCType::SecondType:
                bc_enum = mhs::core::BcType::SecondType;
                bc_param_idx = (uint16_t)bc_params.neumann_q.size();
                bc_params.neumann_q.push_back(mhs::core::parse(rewriter(boundary.second.heat_flux)));
                break;
            case mhs::core::ThermalBCType::ThirdType:
                bc_enum = mhs::core::BcType::ThirdType;
                bc_param_idx = (uint16_t)bc_params.cauchy_h.size();
                bc_params.cauchy_h.push_back(mhs::core::parse(rewriter(boundary.third.convection_coeff)));
                bc_params.cauchy_T_inf.push_back(mhs::core::parse(rewriter(boundary.third.T_inf)));
                break;
            }

            for (const auto& key_str : boundary.face_keys) {
                parsed_keys.push_back({parse_face_key(key_str, si_scale), bc_enum, bc_param_idx});
            }
        }

        // 4. 单次遍历网格：对每个活跃单元的每个暴露面，
        //    依次检查 parsed_keys 是否命中；若全部未命中且面暴露，则赋予 other_bc。
        //    （原 Phase 2 + Phase 3 合并为一步。）
        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    if (cells.valid_mask[old_idx] == 0)
                        continue;
                    int c_idx = (int)cells.index_map[old_idx];

                    for (mhs::core::FaceDir dir : mhs::core::FACE_DIRS) {
                        // 计算面属性（轴、坐标、矩形投影点）
                        char face_axis;
                        double face_coord, a_val, b_val;

                        if (dir == mhs::core::FaceDir::XM || dir == mhs::core::FaceDir::XP) {
                            face_axis = 'X';
                            face_coord = (dir == mhs::core::FaceDir::XM) ? mesh.node_x_left(ix) : mesh.node_x_right(ix);
                            a_val = mesh.cy[iy];
                            b_val = mesh.cz[iz];
                        }
                        else if (dir == mhs::core::FaceDir::YM || dir == mhs::core::FaceDir::YP) {
                            face_axis = 'Y';
                            face_coord = (dir == mhs::core::FaceDir::YM) ? mesh.node_y_left(iy) : mesh.node_y_right(iy);
                            a_val = mesh.cx[ix];
                            b_val = mesh.cz[iz];
                        }
                        else { // ZM, ZP
                            face_axis = 'Z';
                            face_coord = (dir == mhs::core::FaceDir::ZM) ? mesh.node_z_left(iz) : mesh.node_z_right(iz);
                            a_val = mesh.cx[ix];
                            b_val = mesh.cy[iy];
                        }

                        if (!is_face_exposed(dir, ix, iy, iz, mesh, cells))
                            continue;

                        // 优先匹配 parsed_keys
                        bool matched = false;
                        for (const auto& pk : parsed_keys) {
                            if (pk.fk.axis == face_axis && std::abs(face_coord - pk.fk.coord_value) < 1e-10) {
                                if (point_in_face_rects(pk.fk, a_val, b_val)) {
                                    cells.cell_bcs[c_idx].types[(size_t)dir] = pk.bc_enum;
                                    cells.cell_bcs[c_idx].param_idxs[(size_t)dir] = pk.param_idx;
                                    matched = true;
                                    break;
                                }
                            }
                        }

                        // 未命中 → 兜底 other_bc
                        if (!matched && other_bc_enum != mhs::core::BcType::None) {
                            cells.cell_bcs[c_idx].types[(size_t)dir] = other_bc_enum;
                            cells.cell_bcs[c_idx].param_idxs[(size_t)dir] = other_idx;
                        }
                    }
                }
            }
        }
    }

} // namespace mhs::sim