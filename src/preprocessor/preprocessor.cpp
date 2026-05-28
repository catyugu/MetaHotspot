#include "preprocessor.hpp"
#include "expr/expr.hpp"
#include "logger/logger.hpp"
#include <array>
#include <cmath>
#include <sstream>

namespace mhs {

namespace {

double to_millimeters(double value, model::LengthUnit unit)
{
    switch (unit) {
        case model::LengthUnit::M:
            return value * 1000.0;
        case model::LengthUnit::Mm:
            return value;
        case model::LengthUnit::Um:
            return value / 1000.0;
        case model::LengthUnit::Nm:
            return value / 1000000.0;
        case model::LengthUnit::Inch:
            return value * 25.4;
        case model::LengthUnit::Mil:
            return value * 0.0254;
    }
    return value;
}

double eval_expression(const std::string& expr, double x = 0, double y = 0)
{
    if (expr.empty() || expr == "0") {
        return 0.0;
    }
    mhs::expr::clear_registry();
    mhs::expr::set_variable("x", x);
    mhs::expr::set_variable("y", y);
    double val = mhs::expr::eval_geometry(expr);
    return std::isfinite(val) ? val : 0.0;
}

} // namespace

std::unique_ptr<model::InternalModel> Preprocessor::load(const model::IOStructure& ioStructure)
{
    auto model = std::make_unique<model::InternalModel>();
    model->initial_temperature = ioStructure.initial_temperature;
    model->ambient_temperature = ioStructure.ambient_temperature;
    model->study_type = ioStructure.study_type;
    model->transient_duration = ioStructure.transient_duration;
    model->transient_time_step = ioStructure.transient_time_step;

    if (ioStructure.dimension == model::Dimension::Dimension2D) {
        MHS_LOG_ERROR("Dimension2D is not supported. Only Dimension3D is implemented.");
    }

    // Set up variables from ioStructure
    for (const auto& var : ioStructure.variables) {
        try {
            double val = std::stod(var.value);
            mhs::expr::set_variable(var.name, val);
        } catch (...) {
            // Try to evaluate as expression
        }
    }

    // Process layers to determine mesh dimensions
    double domain_x_min = 0, domain_x_max = 0;
    double domain_y_min = 0, domain_y_max = 0;
    double domain_z_min = 0, domain_z_max = 0;

    for (const auto& layer : ioStructure.layers) {
        double thickness = eval_expression(layer.thickness_expr);
        domain_z_max += thickness;
    }

    // Find domain bounds
    for (const auto& layer : ioStructure.layers) {
        double x_offset = eval_expression(layer.x_offset_expr);
        double y_offset = eval_expression(layer.y_offset_expr);

        for (const auto& block : layer.blocks) {
            double x_off = eval_expression(block.x_offset_expr);
            double y_off = eval_expression(block.y_offset_expr);

            for (const auto& rect : block.all_rects) {
                if (!rect.add_sub) {
                    continue; // Skip subtraction rects for extent calculation
                }

                double x = eval_expression(rect.x_expr, x_off, 0) + x_offset;
                double y = eval_expression(rect.y_expr, y_off, 0) + y_offset;
                double w = eval_expression(rect.width_expr);
                double h = eval_expression(rect.height_expr);

                domain_x_min = std::min(domain_x_min, x);
                domain_y_min = std::min(domain_y_min, y);
                domain_x_max = std::max(domain_x_max, x + w);
                domain_y_max = std::max(domain_y_max, y + h);
            }
        }
    }

    // Convert to mm
    double scale = 1.0;
    if (ioStructure.length_unit != model::LengthUnit::Mm) {
        scale = to_millimeters(1.0, ioStructure.length_unit);
    }
    domain_x_min *= scale;
    domain_x_max *= scale;
    domain_y_min *= scale;
    domain_y_max *= scale;
    domain_z_max *= scale;

    // Determine mesh size
    int nx = 10, ny = 10, nz = 10;
    if (!ioStructure.layers.empty()) {
        const auto& top_layer = ioStructure.layers[0];
        double ms_x = eval_expression(top_layer.mesh_size_x_expr);
        double ms_y = eval_expression(top_layer.mesh_size_y_expr);
        double ms_z = eval_expression(top_layer.mesh_size_z_expr);

        if (ms_x > 0) {
            nx = std::max(2, static_cast<int>(std::round((domain_x_max - domain_x_min) / ms_x)));
        }
        if (ms_y > 0) {
            ny = std::max(2, static_cast<int>(std::round((domain_y_max - domain_y_min) / ms_y)));
        }
        if (ms_z > 0) {
            nz = std::max(2, static_cast<int>(std::round(domain_z_max / ms_z)));
        }
    }

    // Ensure minimum mesh
    nx = std::max(nx, 3);
    ny = std::max(ny, 3);
    nz = std::max(nz, 3);

    // Create mesh geometry
    auto& mesh = model->mesh;
    mesh.nx = nx;
    mesh.ny = ny;
    mesh.nz = nz;
    mesh.cell_count = nx * ny * nz;

    mesh.vertex_x.resize(nx + 1);
    mesh.vertex_y.resize(ny + 1);
    mesh.vertex_z.resize(nz + 1);

    for (int i = 0; i <= nx; ++i) {
        mesh.vertex_x[i] = domain_x_min + (domain_x_max - domain_x_min) * i / nx;
    }
    for (int j = 0; j <= ny; ++j) {
        mesh.vertex_y[j] = domain_y_min + (domain_y_max - domain_y_min) * j / ny;
    }
    for (int k = 0; k <= nz; ++k) {
        mesh.vertex_z[k] = domain_z_min + domain_z_max * k / nz;
    }

    // Cell center coordinates
    mesh.cx.resize(nx);
    mesh.cy.resize(ny);
    mesh.cz.resize(nz);
    for (int i = 0; i < nx; ++i) {
        mesh.cx[i] = (mesh.vertex_x[i] + mesh.vertex_x[i + 1]) * 0.5;
    }
    for (int j = 0; j < ny; ++j) {
        mesh.cy[j] = (mesh.vertex_y[j] + mesh.vertex_y[j + 1]) * 0.5;
    }
    for (int k = 0; k < nz; ++k) {
        mesh.cz[k] = (mesh.vertex_z[k] + mesh.vertex_z[k + 1]) * 0.5;
    }

    // Cell sizes
    mesh.dx.resize(nx);
    mesh.dy.resize(ny);
    mesh.dz.resize(nz);
    for (int i = 0; i < nx; ++i) {
        mesh.dx[i] = mesh.vertex_x[i + 1] - mesh.vertex_x[i];
    }
    for (int j = 0; j < ny; ++j) {
        mesh.dy[j] = mesh.vertex_y[j + 1] - mesh.vertex_y[j];
    }
    for (int k = 0; k < nz; ++k) {
        mesh.dz[k] = mesh.vertex_z[k + 1] - mesh.vertex_z[k];
    }

    // Initialize cell fields
    auto& cells = model->cells;
    cells.cell_count = mesh.cell_count;
    cells.material_id.resize(mesh.cell_count);
    cells.layer_id.resize(mesh.cell_count);
    cells.bc_flags.resize(mesh.cell_count, 0);

    // Initialize BC arrays
    auto& face_bcs = model->face_bcs;
    size_t xy_size = static_cast<size_t>(nx) * ny;
    size_t xz_size = static_cast<size_t>(nx) * nz;
    size_t yz_size = static_cast<size_t>(ny) * nz;

    face_bcs.bc_type_zm.resize(xy_size, BcType::None);
    face_bcs.bc_param_idx_zm.resize(xy_size, 0);
    face_bcs.bc_type_zp.resize(xy_size, BcType::None);
    face_bcs.bc_param_idx_zp.resize(xy_size, 0);
    face_bcs.bc_type_ym.resize(xz_size, BcType::None);
    face_bcs.bc_param_idx_ym.resize(xz_size, 0);
    face_bcs.bc_type_yp.resize(xz_size, BcType::None);
    face_bcs.bc_param_idx_yp.resize(xz_size, 0);
    face_bcs.bc_type_xm.resize(yz_size, BcType::None);
    face_bcs.bc_param_idx_xm.resize(yz_size, 0);
    face_bcs.bc_type_xp.resize(yz_size, BcType::None);
    face_bcs.bc_param_idx_xp.resize(yz_size, 0);

    // BC Param table
    auto& bc_params = model->bc_params;
    bc_params.dirichlet_T.push_back(mhs::expr::parse(ioStructure.other_bc_first.temperature));
    bc_params.neumann_q.push_back(mhs::expr::parse(ioStructure.other_bc_second.heat_flux));
    bc_params.cauchy_h.push_back(mhs::expr::parse(ioStructure.other_bc_third.convection_coeff));
    bc_params.cauchy_T_inf.push_back(mhs::expr::parse(ioStructure.other_bc_third.T_inf));

    // Assign materials and layers to cells
    for (const auto& mat_pair : ioStructure.materials) {
        const auto& mat = mat_pair.second;
        model::MaterialProps props;
        props.k = mhs::expr::parse(mat.daore_xishu);
        props.rho = mhs::expr::parse(mat.midu);
        props.c = mhs::expr::parse(mat.bi_rerong);
        model->material_table.push_back(props);
    }

    // Assign default material (copper)
    size_t default_mat_id = 0;
    if (ioStructure.materials.find("copper") != ioStructure.materials.end()) {
        default_mat_id = 0;
    }

    // For each cell, determine its layer based on z position
    double z_offset = 0.0;
    for (size_t layer_idx = 0; layer_idx < ioStructure.layers.size(); ++layer_idx) {
        const auto& layer = ioStructure.layers[layer_idx];
        double layer_thickness = eval_expression(layer.thickness_expr) * scale;
        double layer_x_offset = eval_expression(layer.x_offset_expr) * scale;
        double layer_y_offset = eval_expression(layer.y_offset_expr) * scale;

        for (int k = 0; k < nz; ++k) {
            double cell_z_min = mesh.vertex_z[k];
            double cell_z_max = mesh.vertex_z[k + 1];
            double cell_z_center = (cell_z_min + cell_z_max) * 0.5;

            if (cell_z_center >= z_offset && cell_z_center < z_offset + layer_thickness) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        size_t cell_idx = static_cast<size_t>(k) * nx * ny + j * nx + i;
                        cells.layer_id[cell_idx] = layer_idx;

                        // Check if cell is within any block in this layer
                        bool in_block = false;
                        size_t block_mat_id = default_mat_id;

                        for (size_t block_idx = 0; block_idx < layer.blocks.size(); ++block_idx) {
                            const auto& block = layer.blocks[block_idx];
                            double block_x_off = eval_expression(block.x_offset_expr) * scale;
                            double block_y_off = eval_expression(block.y_offset_expr) * scale;

                            for (const auto& rect : block.all_rects) {
                                double rect_x = eval_expression(rect.x_expr) * scale + layer_x_offset + block_x_off;
                                double rect_y = eval_expression(rect.y_expr) * scale + layer_y_offset + block_y_off;
                                double rect_w = eval_expression(rect.width_expr) * scale;
                                double rect_h = eval_expression(rect.height_expr) * scale;

                                double cell_x = mesh.cx[i];
                                double cell_y = mesh.cy[j];

                                bool in_rect = (cell_x >= rect_x && cell_x < rect_x + rect_w &&
                                               cell_y >= rect_y && cell_y < rect_y + rect_h);

                                if (rect.add_sub && in_rect) {
                                    in_block = true;
                                    if (ioStructure.materials.find(block.material_name) != ioStructure.materials.end()) {
                                        size_t mat_id = 0;
                                        for (const auto& mp : ioStructure.materials) {
                                            if (mp.first == block.material_name) break;
                                            mat_id++;
                                        }
                                        block_mat_id = mat_id;
                                    }
                                    break;
                                } else if (!rect.add_sub && in_rect) {
                                    in_block = false;
                                    break;
                                }
                            }
                            if (in_block) break;
                        }

                        cells.material_id[cell_idx] = in_block ? block_mat_id : default_mat_id;

                        // Parse heat source
                        for (const auto& block : layer.blocks) {
                            if (!block.ti_reyuan_expr.empty() && block.ti_reyuan_expr != "0") {
                                cells.heat_source.push_back(mhs::expr::parse(block.ti_reyuan_expr));
                            }
                        }
                        if (cells.heat_source.size() <= cell_idx) {
                            cells.heat_source.push_back(mhs::expr::parse("0"));
                        }
                    }
                }
            }
        }

        z_offset += layer_thickness;
    }

    // Parse boundary conditions
    // bc_param_count reserved for future use
    for (const auto& boundary : ioStructure.boundaries) {
        BcType bc_type = BcType::None;
        size_t param_idx = 0;

        switch (boundary.bc_type) {
            case model::ThermalBCType::FirstType:
                bc_type = BcType::FirstType;
                bc_params.dirichlet_T.push_back(mhs::expr::parse(boundary.first.temperature));
                param_idx = bc_params.dirichlet_T.size() - 1;
                break;
            case model::ThermalBCType::SecondType:
                bc_type = BcType::SecondType;
                bc_params.neumann_q.push_back(mhs::expr::parse(boundary.second.heat_flux));
                param_idx = bc_params.neumann_q.size() - 1;
                break;
            case model::ThermalBCType::ThirdType:
                bc_type = BcType::ThirdType;
                bc_params.cauchy_h.push_back(mhs::expr::parse(boundary.third.convection_coeff));
                bc_params.cauchy_T_inf.push_back(mhs::expr::parse(boundary.third.T_inf));
                param_idx = bc_params.cauchy_h.size() - 1;
                break;
        }

        // Parse face keys
        for (const auto& face_key : boundary.face_keys) {
            parse_face_key(face_key, mesh, face_bcs, bc_type, param_idx);
        }
    }

    return model;
}

void Preprocessor::parse_face_key(const std::string& face_key,
                                  const model::MeshGeometry& mesh,
                                  model::FaceBCFields& face_bcs,
                                  BcType bc_type,
                                  size_t param_idx)
{
    // Format: Face|Direction|LayerIndex|Coords...
    // Example: "Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100"

    std::stringstream ss(face_key);
    std::string part;
    std::vector<std::string> parts;

    while (std::getline(ss, part, '|')) {
        parts.push_back(part);
    }

    if (parts.size() < 3) {
        return;
    }

    char face = parts[0][0]; // Z, Y, or X
    // layer_idx would be used to match boundary to specific layer
    std::ignore = parts[2];

    // Parse coordinates
    std::string coords_str = parts.size() > 3 ? parts[3] : "";
    std::stringstream coord_ss(coords_str);
    std::string rect_str;
    std::vector<std::array<double, 4>> rects;

    while (std::getline(coord_ss, rect_str, ';')) {
        std::stringstream rect_ss(rect_str);
        std::string coord;
        std::array<double, 4> coords = {0, 0, 0, 0};
        int i = 0;
        while (std::getline(rect_ss, coord, ',') && i < 4) {
            coords[i++] = std::stod(coord);
        }
        rects.push_back(coords);
    }

    // Apply BC to matching faces
    int nx = mesh.nx;
    int ny = mesh.ny;

    if (face == 'Z') {
        // Z- face
        if (parts.size() > 4 && parts[4].find("E") != std::string::npos && parts[4].find("0") != std::string::npos) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double x = mesh.cx[i];
                    double y = mesh.cy[j];
                    for (const auto& r : rects) {
                        if (x >= r[0] && x < r[2] && y >= r[1] && y < r[3]) {
                            size_t idx = j * nx + i;
                            if (face_bcs.bc_type_zm[idx] == BcType::None) {
                                face_bcs.bc_type_zm[idx] = bc_type;
                                face_bcs.bc_param_idx_zm[idx] = static_cast<uint16_t>(param_idx);
                            }
                        }
                    }
                }
            }
        }
    }
    // Additional face key parsing can be extended
}

} // namespace mhs