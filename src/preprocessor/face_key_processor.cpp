#include "face_key_processor.hpp"
#include "expr/expr.hpp"
#include "model/types.hpp"

namespace mhs::preprocessor {

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
        if (!token.empty()) {
            parts.push_back(token);
        }

        if (parts.size() < 3) {
            return info;
        }

        info.axis = parts[0][0];
        info.side = parts[1][0];
        info.coord_value = std::stod(parts[2]) * si_scale;

        if (parts.size() == 4) {
            // Format: semicolon-separated rects in parts[3]
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
            if (!rt.empty()) {
                rect_parts.push_back(rt);
            }

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
                if (!v.empty()) {
                    vals.push_back(std::stod(v) * si_scale);
                }
                if (vals.size() == 4) {
                    info.rects.push_back({vals[0], vals[1], vals[2], vals[3]});
                }
            }
        }
        else if (parts.size() >= 7) {
            // Format: pipe-separated values after axis|side|coord
            std::vector<double> vals;
            for (size_t i = 3; i < parts.size(); i++) {
                vals.push_back(std::stod(parts[i]) * si_scale);
            }
            if (vals.size() >= 4) {
                info.rects.push_back({vals[0], vals[1], vals[2], vals[3]});
            }
        }

        return info;
    }

    bool point_in_face_rects(const FaceKeyInfo& fk, double a, double b)
    {
        for (const auto& rect : fk.rects) {
            if (a >= rect[0] && a <= rect[1] && b >= rect[2] && b <= rect[3]) {
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
        // Initialize all BCs to None
        for (int c = 0; c < cells.cell_count; c++) {
            for (size_t f = 0; f < FACE_COUNT; f++) {
                cells.cell_bcs[c].types[f] = BcType::None;
                cells.cell_bcs[c].param_idxs[f] = 0;
            }
        }

        // Build "other_bc" parameter entries (index 0)
        uint16_t other_idx = 0;
        switch (other_bc_type) {
        case model::ThermalBCType::FirstType:
            bc_params.dirichlet_T.push_back(expr::parse(other_bc_first.temperature));
            break;
        case model::ThermalBCType::SecondType:
            bc_params.neumann_q.push_back(expr::parse(other_bc_second.heat_flux));
            break;
        case model::ThermalBCType::ThirdType:
            bc_params.cauchy_h.push_back(expr::parse(other_bc_third.convection_coeff));
            bc_params.cauchy_T_inf.push_back(expr::parse(other_bc_third.T_inf));
            break;
        }

        // Process each explicit boundary
        for (const auto& boundary : boundaries) {
            uint16_t bc_param_idx = 0;

            switch (boundary.bc_type) {
            case model::ThermalBCType::FirstType:
                bc_param_idx = (uint16_t)bc_params.dirichlet_T.size();
                bc_params.dirichlet_T.push_back(expr::parse(boundary.first.temperature));
                break;
            case model::ThermalBCType::SecondType:
                bc_param_idx = (uint16_t)bc_params.neumann_q.size();
                bc_params.neumann_q.push_back(expr::parse(boundary.second.heat_flux));
                break;
            case model::ThermalBCType::ThirdType:
                bc_param_idx = (uint16_t)bc_params.cauchy_h.size();
                bc_params.cauchy_h.push_back(expr::parse(boundary.third.convection_coeff));
                bc_params.cauchy_T_inf.push_back(expr::parse(boundary.third.T_inf));
                break;
            }

            BcType bc_enum;
            switch (boundary.bc_type) {
            case model::ThermalBCType::FirstType:
                bc_enum = BcType::FirstType;
                break;
            case model::ThermalBCType::SecondType:
                bc_enum = BcType::SecondType;
                break;
            case model::ThermalBCType::ThirdType:
                bc_enum = BcType::ThirdType;
                break;
            default:
                bc_enum = BcType::None;
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

                            double cx_val = mesh.cx[ix];
                            double cy_val = mesh.cy[iy];
                            double cz_val = mesh.cz[iz];

                            FaceDir dir;
                            bool on_boundary = false;

                            if (fk.axis == 'Z') {
                                double z_min = mesh.vertex_z[0];
                                double z_max = mesh.vertex_z[mesh.nz];

                                if (fk.coord_value <= z_min + 1e-10) {
                                    if (iz == 0) {
                                        on_boundary = true;
                                        dir = FaceDir::ZM;
                                    }
                                }
                                else if (fk.coord_value >= z_max - 1e-10) {
                                    if (iz == mesh.nz - 1) {
                                        on_boundary = true;
                                        dir = FaceDir::ZP;
                                    }
                                }
                                else {
                                    for (int k = 0; k < mesh.nz; k++) {
                                        double vk = mesh.vertex_z[k];
                                        if (std::abs(vk - fk.coord_value) < 1e-10) {
                                            if (iz == k - 1) {
                                                on_boundary = true;
                                                dir = FaceDir::ZP;
                                            }
                                            else if (iz == k) {
                                                on_boundary = true;
                                                dir = FaceDir::ZM;
                                            }
                                            break;
                                        }
                                    }
                                }

                                if (on_boundary && point_in_face_rects(fk, cx_val, cy_val)) {
                                    cells.cell_bcs[c_idx].types[(size_t)dir] = bc_enum;
                                    cells.cell_bcs[c_idx].param_idxs[(size_t)dir] = bc_param_idx;
                                }
                            }
                            else if (fk.axis == 'Y') {
                                double y_min = mesh.vertex_y[0];
                                double y_max = mesh.vertex_y[mesh.ny];

                                if (fk.coord_value <= y_min + 1e-10) {
                                    if (iy == 0) {
                                        on_boundary = true;
                                        dir = FaceDir::YM;
                                    }
                                }
                                else if (fk.coord_value >= y_max - 1e-10) {
                                    if (iy == mesh.ny - 1) {
                                        on_boundary = true;
                                        dir = FaceDir::YP;
                                    }
                                }

                                if (on_boundary && point_in_face_rects(fk, cx_val, cz_val)) {
                                    cells.cell_bcs[c_idx].types[(size_t)dir] = bc_enum;
                                    cells.cell_bcs[c_idx].param_idxs[(size_t)dir] = bc_param_idx;
                                }
                            }
                            else if (fk.axis == 'X') {
                                double x_min = mesh.vertex_x[0];
                                double x_max = mesh.vertex_x[mesh.nx];

                                if (fk.coord_value <= x_min + 1e-10) {
                                    if (ix == 0) {
                                        on_boundary = true;
                                        dir = FaceDir::XM;
                                    }
                                }
                                else if (fk.coord_value >= x_max - 1e-10) {
                                    if (ix == mesh.nx - 1) {
                                        on_boundary = true;
                                        dir = FaceDir::XP;
                                    }
                                }

                                if (on_boundary && point_in_face_rects(fk, cy_val, cz_val)) {
                                    cells.cell_bcs[c_idx].types[(size_t)dir] = bc_enum;
                                    cells.cell_bcs[c_idx].param_idxs[(size_t)dir] = bc_param_idx;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fill other_bc for domain boundary faces not explicitly specified
        BcType other_bc_enum;
        switch (other_bc_type) {
        case model::ThermalBCType::FirstType:
            other_bc_enum = BcType::FirstType;
            break;
        case model::ThermalBCType::SecondType:
            other_bc_enum = BcType::SecondType;
            break;
        case model::ThermalBCType::ThirdType:
            other_bc_enum = BcType::ThirdType;
            break;
        default:
            other_bc_enum = BcType::SecondType;
        }

        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    if (cells.valid_mask[old_idx] == 0)
                        continue;
                    int c_idx = (int)cells.index_map[old_idx];

                    if (ix == 0 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::XM] == BcType::None) {
                        cells.cell_bcs[c_idx].types[(size_t)FaceDir::XM] = other_bc_enum;
                        cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::XM] = other_idx;
                    }
                    if (ix == mesh.nx - 1 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::XP] == BcType::None) {
                        cells.cell_bcs[c_idx].types[(size_t)FaceDir::XP] = other_bc_enum;
                        cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::XP] = other_idx;
                    }
                    if (iy == 0 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::YM] == BcType::None) {
                        cells.cell_bcs[c_idx].types[(size_t)FaceDir::YM] = other_bc_enum;
                        cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::YM] = other_idx;
                    }
                    if (iy == mesh.ny - 1 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::YP] == BcType::None) {
                        cells.cell_bcs[c_idx].types[(size_t)FaceDir::YP] = other_bc_enum;
                        cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::YP] = other_idx;
                    }
                    if (iz == 0 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::ZM] == BcType::None) {
                        cells.cell_bcs[c_idx].types[(size_t)FaceDir::ZM] = other_bc_enum;
                        cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::ZM] = other_idx;
                    }
                    if (iz == mesh.nz - 1 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::ZP] == BcType::None) {
                        cells.cell_bcs[c_idx].types[(size_t)FaceDir::ZP] = other_bc_enum;
                        cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::ZP] = other_idx;
                    }
                }
            }
        }

        // Handle virtual-cell neighbor faces
        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    if (cells.valid_mask[old_idx] == 0)
                        continue;
                    int c_idx = (int)cells.index_map[old_idx];

                    if (ix > 0) {
                        int neighbor = (ix - 1) * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                        if (cells.valid_mask[neighbor] == 0 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::XM] == BcType::None) {
                            cells.cell_bcs[c_idx].types[(size_t)FaceDir::XM] = other_bc_enum;
                            cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::XM] = other_idx;
                        }
                    }
                    if (ix < mesh.nx - 1) {
                        int neighbor = (ix + 1) * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                        if (cells.valid_mask[neighbor] == 0 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::XP] == BcType::None) {
                            cells.cell_bcs[c_idx].types[(size_t)FaceDir::XP] = other_bc_enum;
                            cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::XP] = other_idx;
                        }
                    }
                    if (iy > 0) {
                        int neighbor = ix * mesh.ny * mesh.nz + (iy - 1) * mesh.nz + iz;
                        if (cells.valid_mask[neighbor] == 0 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::YM] == BcType::None) {
                            cells.cell_bcs[c_idx].types[(size_t)FaceDir::YM] = other_bc_enum;
                            cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::YM] = other_idx;
                        }
                    }
                    if (iy < mesh.ny - 1) {
                        int neighbor = ix * mesh.ny * mesh.nz + (iy + 1) * mesh.nz + iz;
                        if (cells.valid_mask[neighbor] == 0 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::YP] == BcType::None) {
                            cells.cell_bcs[c_idx].types[(size_t)FaceDir::YP] = other_bc_enum;
                            cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::YP] = other_idx;
                        }
                    }
                    if (iz > 0) {
                        int neighbor = ix * mesh.ny * mesh.nz + iy * mesh.nz + (iz - 1);
                        if (cells.valid_mask[neighbor] == 0 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::ZM] == BcType::None) {
                            cells.cell_bcs[c_idx].types[(size_t)FaceDir::ZM] = other_bc_enum;
                            cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::ZM] = other_idx;
                        }
                    }
                    if (iz < mesh.nz - 1) {
                        int neighbor = ix * mesh.ny * mesh.nz + iy * mesh.nz + (iz + 1);
                        if (cells.valid_mask[neighbor] == 0 && cells.cell_bcs[c_idx].types[(size_t)FaceDir::ZP] == BcType::None) {
                            cells.cell_bcs[c_idx].types[(size_t)FaceDir::ZP] = other_bc_enum;
                            cells.cell_bcs[c_idx].param_idxs[(size_t)FaceDir::ZP] = other_idx;
                        }
                    }
                }
            }
        }
    }

} // namespace mhs::preprocessor