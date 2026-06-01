#include "face_key_processor.hpp"
#include "expr/expr.hpp"
#include "model/types.hpp"
#include <cmath>

namespace mhs::preprocessor {

    namespace {
        // 核心抽象：判断一个面的外侧是否“暴露”（即邻居是域外或者虚拟单元）
        bool is_face_exposed(mhs::FaceDir dir, int ix, int iy, int iz,
            const model::MeshGeometry& mesh,
            const model::CellFields& cells)
        {
            int nix = ix, niy = iy, niz = iz;
            switch (dir) {
            case FaceDir::XM:
                nix--;
                break;
            case FaceDir::XP:
                nix++;
                break;
            case FaceDir::YM:
                niy--;
                break;
            case FaceDir::YP:
                niy++;
                break;
            case FaceDir::ZM:
                niz--;
                break;
            case FaceDir::ZP:
                niz++;
                break;
            }

            // 1. 如果超出网格边界，说明是域外，绝对暴露
            if (nix < 0 || nix >= mesh.nx || niy < 0 || niy >= mesh.ny || niz < 0 || niz >= mesh.nz) {
                return true;
            }

            // 2. 如果内部对应邻居是空洞（虚拟单元），说明面暴露在了内部孔隙中
            int neighbor = nix * mesh.ny * mesh.nz + niy * mesh.nz + niz;
            return cells.valid_mask[neighbor] == 0;
        }
    } // anonymous namespace

    FaceKeyInfo parse_face_key(const std::string& key, double si_scale)
    {
        FaceKeyInfo info;
        std::vector<std::string> parts;
        std::string token;
        for (char c : key) {
            if (c == '|') {
                parts.push_back(token);
                token.clear();
            }
            else {
                token += c;
            }
        }
        if (!token.empty())
            parts.push_back(token);
        if (parts.size() < 3)
            return info;

        info.axis = parts[0][0];
        info.side = parts[1][0];
        info.coord_value = std::stod(parts[2]) * si_scale;

        if (parts.size() == 4) {
            std::string rect_str = parts[3];
            std::vector<std::string> rect_parts;
            std::string rt;
            for (char c : rect_str) {
                if (c == ';') {
                    rect_parts.push_back(rt);
                    rt.clear();
                }
                else {
                    rt += c;
                }
            }
            if (!rt.empty())
                rect_parts.push_back(rt);

            for (const auto& rp : rect_parts) {
                std::vector<double> vals;
                std::string v;
                for (char c : rp) {
                    if (c == ',') {
                        vals.push_back(std::stod(v) * si_scale);
                        v.clear();
                    }
                    else {
                        v += c;
                    }
                }
                if (!v.empty())
                    vals.push_back(std::stod(v) * si_scale);
                if (vals.size() == 4)
                    info.rects.push_back({vals[0], vals[1], vals[2], vals[3]});
            }
        }
        else if (parts.size() >= 7) {
            std::vector<double> vals;
            for (size_t i = 3; i < parts.size(); i++)
                vals.push_back(std::stod(parts[i]) * si_scale);
            if (vals.size() >= 4)
                info.rects.push_back({vals[0], vals[1], vals[2], vals[3]});
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

    void resolve_face_keys(const std::vector<model::Boundary>& boundaries,
        model::ThermalBCType other_bc_type,
        const model::FirstTypeThermalBC& other_bc_first,
        const model::SecondTypeThermalBC& other_bc_second,
        const model::ThirdTypeThermalBC& other_bc_third,
        const model::MeshGeometry& mesh,
        model::CellFields& cells,
        model::BCParamTable& bc_params,
        double si_scale)
    {
        // 1. 初始化所有 BC 为 None
        for (int c = 0; c < cells.cell_count; c++) {
            for (size_t f = 0; f < FACE_COUNT; f++) {
                cells.cell_bcs[c].types[f] = BcType::None;
                cells.cell_bcs[c].param_idxs[f] = 0;
            }
        }

        // 2. 预存 other_bc 参数（索引始终为 0）
        uint16_t other_idx = 0;
        BcType other_bc_enum = BcType::None;
        switch (other_bc_type) {
        case model::ThermalBCType::FirstType:
            other_bc_enum = BcType::FirstType;
            bc_params.dirichlet_T.push_back(expr::parse(other_bc_first.temperature));
            break;
        case model::ThermalBCType::SecondType:
            other_bc_enum = BcType::SecondType;
            bc_params.neumann_q.push_back(expr::parse(other_bc_second.heat_flux));
            break;
        case model::ThermalBCType::ThirdType:
            other_bc_enum = BcType::ThirdType;
            bc_params.cauchy_h.push_back(expr::parse(other_bc_third.convection_coeff));
            bc_params.cauchy_T_inf.push_back(expr::parse(other_bc_third.T_inf));
            break;
        }

        // 3. 拾取并分配显式定义的 FaceKey 边界
        for (const auto& boundary : boundaries) {
            uint16_t bc_param_idx = 0;
            BcType bc_enum = BcType::None;

            switch (boundary.bc_type) {
            case model::ThermalBCType::FirstType:
                bc_enum = BcType::FirstType;
                bc_param_idx = (uint16_t)bc_params.dirichlet_T.size();
                bc_params.dirichlet_T.push_back(expr::parse(boundary.first.temperature));
                break;
            case model::ThermalBCType::SecondType:
                bc_enum = BcType::SecondType;
                bc_param_idx = (uint16_t)bc_params.neumann_q.size();
                bc_params.neumann_q.push_back(expr::parse(boundary.second.heat_flux));
                break;
            case model::ThermalBCType::ThirdType:
                bc_enum = BcType::ThirdType;
                bc_param_idx = (uint16_t)bc_params.cauchy_h.size();
                bc_params.cauchy_h.push_back(expr::parse(boundary.third.convection_coeff));
                bc_params.cauchy_T_inf.push_back(expr::parse(boundary.third.T_inf));
                break;
            }

            for (const auto& key_str : boundary.face_keys) {
                FaceKeyInfo fk = parse_face_key(key_str, si_scale);

                for (int ix = 0; ix < mesh.nx; ix++) {
                    for (int iy = 0; iy < mesh.ny; iy++) {
                        for (int iz = 0; iz < mesh.nz; iz++) {
                            int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                            if (cells.valid_mask[old_idx] == 0)
                                continue;
                            int c_idx = (int)cells.index_map[old_idx];

                            // 泛化迭代该活跃单元的 6 个面
                            for (FaceDir dir : FACE_DIRS) {
                                char face_axis;
                                double face_coord, a_val, b_val;

                                if (dir == FaceDir::XM || dir == FaceDir::XP) {
                                    face_axis = 'X';
                                    face_coord = (dir == FaceDir::XM) ? mesh.vertex_x[ix] : mesh.vertex_x[ix + 1];
                                    a_val = mesh.cy[iy];
                                    b_val = mesh.cz[iz];
                                }
                                else if (dir == FaceDir::YM || dir == FaceDir::YP) {
                                    face_axis = 'Y';
                                    face_coord = (dir == FaceDir::YM) ? mesh.vertex_y[iy] : mesh.vertex_y[iy + 1];
                                    a_val = mesh.cx[ix];
                                    b_val = mesh.cz[iz];
                                }
                                else { // ZM, ZP
                                    face_axis = 'Z';
                                    face_coord = (dir == FaceDir::ZM) ? mesh.vertex_z[iz] : mesh.vertex_z[iz + 1];
                                    a_val = mesh.cx[ix];
                                    b_val = mesh.cy[iy];
                                }

                                // 严格条件验证：同轴 -> 坐标重合 -> 面朝外暴露 -> 落入指定框内
                                if (fk.axis == face_axis && std::abs(face_coord - fk.coord_value) < 1e-10) {
                                    if (is_face_exposed(dir, ix, iy, iz, mesh, cells)) {
                                        if (point_in_face_rects(fk, a_val, b_val)) {
                                            cells.cell_bcs[c_idx].types[(size_t)dir] = bc_enum;
                                            cells.cell_bcs[c_idx].param_idxs[(size_t)dir] = bc_param_idx;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4. 为所有依然是 BcType::None 且【暴露在外/侧壁】的面，分配兜底的 other_bc
        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    if (cells.valid_mask[old_idx] == 0)
                        continue;
                    int c_idx = (int)cells.index_map[old_idx];

                    for (FaceDir dir : FACE_DIRS) {
                        if (cells.cell_bcs[c_idx].types[(size_t)dir] == BcType::None) {
                            if (is_face_exposed(dir, ix, iy, iz, mesh, cells)) {
                                cells.cell_bcs[c_idx].types[(size_t)dir] = other_bc_enum;
                                cells.cell_bcs[c_idx].param_idxs[(size_t)dir] = other_idx;
                            }
                        }
                    }
                }
            }
        }
    }

} // namespace mhs::preprocessor