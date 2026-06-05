#include <filesystem>
#include <tinyxml2.h>

#include <algorithm>
#include <stdexcept>

#include "io.hpp"

namespace mhs::io {

    using namespace tinyxml2;

    static std::string trim(const std::string& str)
    {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (std::string::npos == first)
            return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

    static std::string get_text(const XMLElement* elem)
    {
        if (!elem)
            return "";
        const char* text = elem->GetText();
        return text ? trim(text) : "";
    }

    static double parse_double(const std::string& s)
    {
        if (s.empty()) {
            return 0.0;
        }
        return std::stod(s);
    }

    // Helpers for the <Functions> block: pull one double-typed child or no-op.
    static void read_double_member(const XMLElement* parent, const char* tag, double& target)
    {
        if (const XMLElement* e = parent->FirstChildElement(tag)) {
            target = parse_double(get_text(e));
        }
    }
    static void read_string_member(const XMLElement* parent, const char* tag, std::string& target)
    {
        if (const XMLElement* e = parent->FirstChildElement(tag)) {
            target = get_text(e);
        }
    }
    static void read_draw_range(const XMLElement* parent, double& min_x, double& max_x)
    {
        read_double_member(parent, "b:DrawMinX", min_x);
        read_double_member(parent, "b:DrawMaxX", max_x);
    }

    IOStructure read_xml(const std::string& xml_path)
    {
        XMLDocument doc;
        XMLError err = doc.LoadFile(xml_path.c_str());
        if (err != XML_SUCCESS) {
            throw std::runtime_error("Failed to load XML file: " + xml_path);
        }

        IOStructure structure;

        const XMLElement* root = doc.FirstChildElement("Structure");
        if (!root) {
            throw std::runtime_error("No Structure element found");
        }

        // Basic attributes
        const char* study_type_str = root->Attribute("StudyType");
        if (study_type_str) {
            if (std::string(study_type_str) == "Steady") {
                structure.study_type = StudyType::Steady;
            }
            else {
                structure.study_type = StudyType::Transient;
            }
        }
        else {
            // Try parsing from child element (for namespace-prefixed XML)
            const XMLElement* study_elem = root->FirstChildElement("StudyType");
            if (study_elem) {
                std::string val = get_text(study_elem);
                if (val == "Steady") {
                    structure.study_type = StudyType::Steady;
                }
                else {
                    structure.study_type = StudyType::Transient;
                }
            }
        }

        const char* dim_str = root->Attribute("Dimension");
        if (dim_str) {
            if (std::string(dim_str) == "Dimension3D") {
                structure.dimension = Dimension::Dimension3D;
            }
            else {
                structure.dimension = Dimension::Dimension2D;
            }
        }
        else {
            // Try parsing from child element (for namespace-prefixed XML)
            const XMLElement* dim_elem = root->FirstChildElement("Dimension");
            if (dim_elem) {
                std::string val = get_text(dim_elem);
                if (val == "Dimension3D") {
                    structure.dimension = Dimension::Dimension3D;
                }
                else {
                    structure.dimension = Dimension::Dimension2D;
                }
            }
        }

        // Length unit
        const char* unit_str = root->Attribute("LengthUnit");
        if (unit_str) {
            std::string u = unit_str;
            if (u == "M") {
                structure.length_unit = LengthUnit::M;
            }
            else if (u == "Mm") {
                structure.length_unit = LengthUnit::Mm;
            }
            else if (u == "Um") {
                structure.length_unit = LengthUnit::Um;
            }
            else if (u == "Nm") {
                structure.length_unit = LengthUnit::Nm;
            }
            else if (u == "Inch") {
                structure.length_unit = LengthUnit::Inch;
            }
            else if (u == "Mil") {
                structure.length_unit = LengthUnit::Mil;
            }
        }
        else {
            // Try parsing from child element
            const XMLElement* unit_elem = root->FirstChildElement("LengthUnit");
            if (unit_elem) {
                std::string u = get_text(unit_elem);
                if (u == "M") {
                    structure.length_unit = LengthUnit::M;
                }
                else if (u == "Mm") {
                    structure.length_unit = LengthUnit::Mm;
                }
                else if (u == "Um") {
                    structure.length_unit = LengthUnit::Um;
                }
                else if (u == "Nm") {
                    structure.length_unit = LengthUnit::Nm;
                }
                else if (u == "Inch") {
                    structure.length_unit = LengthUnit::Inch;
                }
                else if (u == "Mil") {
                    structure.length_unit = LengthUnit::Mil;
                }
            }
        }

        // Temperature settings
        if (const XMLElement* amb = root->FirstChildElement("AmbientTemperature")) {
            structure.ambient_temperature = parse_double(get_text(amb));
        }
        if (const XMLElement* init = root->FirstChildElement("InitialTemperature")) {
            structure.initial_temperature = parse_double(get_text(init));
        }

        // Transient settings
        if (const XMLElement* trans = root->FirstChildElement("TransientStudyDuration")) {
            structure.transient_duration = parse_double(get_text(trans));
        }
        if (const XMLElement* step = root->FirstChildElement("TransientStudyTimeStep")) {
            structure.transient_time_step = parse_double(get_text(step));
        }
        if (const XMLElement* unit = root->FirstChildElement("TransientTimeUnit")) {
            structure.transient_time_unit = get_text(unit);
        }

        // OtherThermalBoundary (default BC)
        if (const XMLElement* other = root->FirstChildElement("OtherThermalBondary")) {
            const char* type = other->Attribute("i:type");
            std::string type_str = type ? type : "";
            if (type_str.find("FirstType") != std::string::npos) {
                structure.other_bc_type = ThermalBCType::FirstType;
                if (const XMLElement* temp = other->FirstChildElement("a:Temperature")) {
                    structure.other_bc_first.temperature = get_text(temp);
                }
            }
            else if (type_str.find("SecondType") != std::string::npos) {
                structure.other_bc_type = ThermalBCType::SecondType;
                if (const XMLElement* flux = other->FirstChildElement("a:HeatFlux")) {
                    structure.other_bc_second.heat_flux = get_text(flux);
                }
            }
            else if (type_str.find("ThirdType") != std::string::npos) {
                structure.other_bc_type = ThermalBCType::ThirdType;
                if (const XMLElement* h = other->FirstChildElement("a:ConvectionCoefficient")) {
                    structure.other_bc_third.convection_coeff = get_text(h);
                }
                if (const XMLElement* t = other->FirstChildElement("a:EnvironmentTemperature")) {
                    structure.other_bc_third.T_inf = get_text(t);
                }
            }
        }

        // Variables
        if (const XMLElement* vars = root->FirstChildElement("Variables")) {
            for (const XMLElement* kv = vars->FirstChildElement("a:KeyValueOfstringdouble"); kv;
                kv = kv->NextSiblingElement("a:KeyValueOfstringdouble")) {
                Variable var;
                if (const XMLElement* key = kv->FirstChildElement("a:Key")) {
                    var.name = get_text(key);
                }
                if (const XMLElement* val = kv->FirstChildElement("a:Value")) {
                    var.value = get_text(val);
                }
                if (!var.name.empty()) {
                    structure.variables.push_back(var);
                }
            }
        }

        // Materials
        if (const XMLElement* mats = root->FirstChildElement("Materials")) {
            for (const XMLElement* kv = mats->FirstChildElement("a:KeyValueOfstringMaterialGyu7GfTz"); kv;
                kv = kv->NextSiblingElement("a:KeyValueOfstringMaterialGyu7GfTz")) {
                Material mat;
                if (const XMLElement* key = kv->FirstChildElement("a:Key")) {
                    mat.name = get_text(key);
                }
                const XMLElement* val = kv->FirstChildElement("a:Value");
                if (val) {
                    if (const XMLElement* daore = val->FirstChildElement("DaoreXishu")) {
                        mat.daore_xishu = get_text(daore);
                    }
                    if (const XMLElement* midu = val->FirstChildElement("Midu")) {
                        mat.midu = get_text(midu);
                    }
                    if (const XMLElement* birerong = val->FirstChildElement("BiRerong")) {
                        mat.bi_rerong = get_text(birerong);
                    }
                }
                if (!mat.name.empty()) {
                    structure.materials[mat.name] = mat;
                }
            }
        }

        // Functions (5 类单变元函数)
        if (const XMLElement* funcs = root->FirstChildElement("Functions")) {
            for (const XMLElement* kv = funcs->FirstChildElement("a:KeyValueOfstringFunctionAdzryM2O"); kv;
                kv = kv->NextSiblingElement("a:KeyValueOfstringFunctionAdzryM2O")) {
                std::string name;
                if (const XMLElement* key = kv->FirstChildElement("a:Key")) {
                    name = get_text(key);
                }
                const XMLElement* val = kv->FirstChildElement("a:Value");
                Function fn;
                if (val) {
                    const char* type = val->Attribute("i:type");
                    std::string type_str = type ? type : "";
                    if (type_str.find("ExpressionFunction") != std::string::npos) {
                        fn.type = FunctionType::Expression;
                        read_string_member(val, "b:Expression", fn.expression.expression);
                        read_draw_range(val, fn.expression.draw_min_x, fn.expression.draw_max_x);
                    }
                    else if (type_str.find("DoubleExponentialFunction") != std::string::npos) {
                        fn.type = FunctionType::DoubleExponential;
                        read_double_member(val, "b:A", fn.double_exp.a);
                        read_double_member(val, "b:Alpha", fn.double_exp.alpha);
                        read_double_member(val, "b:Beta", fn.double_exp.beta);
                        read_draw_range(val, fn.double_exp.draw_min_x, fn.double_exp.draw_max_x);
                    }
                    else if (type_str.find("GaussFunction") != std::string::npos) {
                        fn.type = FunctionType::Gauss;
                        read_double_member(val, "b:A", fn.gauss.a);
                        read_double_member(val, "b:Tau", fn.gauss.tau);
                        read_double_member(val, "b:X0", fn.gauss.x0);
                        read_draw_range(val, fn.gauss.draw_min_x, fn.gauss.draw_max_x);
                    }
                    else if (type_str.find("SineFunction") != std::string::npos) {
                        fn.type = FunctionType::Sine;
                        read_double_member(val, "b:A", fn.sine.a);
                        read_double_member(val, "b:Omega", fn.sine.omega);
                        read_double_member(val, "b:Phi", fn.sine.phi);
                        read_draw_range(val, fn.sine.draw_min_x, fn.sine.draw_max_x);
                    }
                    else if (type_str.find("PieceWiseFunction") != std::string::npos) {
                        fn.type = FunctionType::PieceWise;
                        if (const XMLElement* points = val->FirstChildElement("b:Points")) {
                            for (const XMLElement* pt = points->FirstChildElement("b:PieceWiseFunction.Point"); pt;
                                pt = pt->NextSiblingElement("b:PieceWiseFunction.Point")) {
                                PieceWiseFunction::Point p;
                                read_double_member(pt, "b:X", p.x);
                                read_double_member(pt, "b:Y", p.y);
                                fn.piecewise.points.push_back(p);
                            }
                            // Pre-sort by X so the closure can binary-search without
                            // sorting again at registration time.
                            std::sort(fn.piecewise.points.begin(), fn.piecewise.points.end(),
                                [](const PieceWiseFunction::Point& a, const PieceWiseFunction::Point& b) {
                                    return a.x < b.x;
                                });
                        }
                        read_draw_range(val, fn.piecewise.draw_min_x, fn.piecewise.draw_max_x);
                    }
                    else if (!type_str.empty()) {
                        throw std::runtime_error("Unknown function i:type: " + type_str);
                    }
                }
                if (!name.empty()) {
                    structure.functions[name] = fn;
                }
            }
        }

        // Layers
        if (const XMLElement* layers_elem = root->FirstChildElement("Layers")) {
            for (const XMLElement* layer_elem = layers_elem->FirstChildElement("Layer"); layer_elem;
                layer_elem = layer_elem->NextSiblingElement("Layer")) {
                Layer layer;

                if (const XMLElement* name = layer_elem->FirstChildElement("Name")) {
                    layer.name = get_text(name);
                }
                if (const XMLElement* thickness = layer_elem->FirstChildElement("ThicknessExpression")) {
                    layer.thickness_expr = get_text(thickness);
                }
                if (const XMLElement* xoff = layer_elem->FirstChildElement("XOffsetExpression")) {
                    layer.x_offset_expr = get_text(xoff);
                }
                if (const XMLElement* yoff = layer_elem->FirstChildElement("YOffsetExpression")) {
                    layer.y_offset_expr = get_text(yoff);
                }
                if (const XMLElement* period = layer_elem->FirstChildElement("PeriodWidth")) {
                    layer.period_width = std::stoi(get_text(period));
                }
                if (const XMLElement* top = layer_elem->FirstChildElement("IsTopLayer")) {
                    layer.is_top_layer = std::string(get_text(top)) == "true";
                }

                // Blocks
                if (const XMLElement* blocks_elem = layer_elem->FirstChildElement("Blocks")) {
                    for (const XMLElement* block_elem = blocks_elem->FirstChildElement("Block"); block_elem;
                        block_elem = block_elem->NextSiblingElement("Block")) {
                        Block block;

                        if (const XMLElement* name = block_elem->FirstChildElement("Name")) {
                            block.name = get_text(name);
                        }
                        if (const XMLElement* mat = block_elem->FirstChildElement("MaterialName")) {
                            block.material_name = get_text(mat);
                        }
                        if (const XMLElement* ti = block_elem->FirstChildElement("TiReyuan")) {
                            block.ti_reyuan_expr = get_text(ti);
                        }
                        if (const XMLElement* xoff = block_elem->FirstChildElement("XOffsetExpression")) {
                            block.x_offset_expr = get_text(xoff);
                        }
                        if (const XMLElement* yoff = block_elem->FirstChildElement("YOffsetExpression")) {
                            block.y_offset_expr = get_text(yoff);
                        }
                        if (const XMLElement* normal = block_elem->FirstChildElement("IsNormalMaterial")) {
                            block.is_normal_material = std::string(get_text(normal)) == "true";
                        }

                        // Rects (AllRects)
                        if (const XMLElement* rects_elem = block_elem->FirstChildElement("AllRects")) {
                            for (const XMLElement* rect_elem = rects_elem->FirstChildElement("Rect"); rect_elem;
                                rect_elem = rect_elem->NextSiblingElement("Rect")) {
                                Rect rect;
                                if (const XMLElement* adds = rect_elem->FirstChildElement("Add_sub")) {
                                    rect.add_sub = std::string(get_text(adds)) == "true";
                                }
                                if (const XMLElement* name = rect_elem->FirstChildElement("Name")) {
                                    rect.name = get_text(name);
                                }
                                if (const XMLElement* w = rect_elem->FirstChildElement("WidthExpression")) {
                                    rect.width_expr = get_text(w);
                                }
                                if (const XMLElement* h = rect_elem->FirstChildElement("HeightExpression")) {
                                    rect.height_expr = get_text(h);
                                }
                                if (const XMLElement* x = rect_elem->FirstChildElement("XExpression")) {
                                    rect.x_expr = get_text(x);
                                }
                                if (const XMLElement* y = rect_elem->FirstChildElement("YExpression")) {
                                    rect.y_expr = get_text(y);
                                }
                                if (const XMLElement* xs = rect_elem->FirstChildElement("XSizeExpression")) {
                                    rect.x_size_expr = get_text(xs);
                                }
                                if (const XMLElement* ys = rect_elem->FirstChildElement("YSizeExpression")) {
                                    rect.y_size_expr = get_text(ys);
                                }
                                if (const XMLElement* xi = rect_elem->FirstChildElement("XIntervalExpression")) {
                                    rect.x_interval_expr = get_text(xi);
                                }
                                if (const XMLElement* yi = rect_elem->FirstChildElement("YIntervalExpression")) {
                                    rect.y_interval_expr = get_text(yi);
                                }
                                block.all_rects.push_back(rect);
                            }
                        }

                        layer.blocks.push_back(block);
                    }
                }

                structure.layers.push_back(layer);
            }
        }

        // Boundaries
        if (const XMLElement* bounds_elem = root->FirstChildElement("Boundaries")) {
            for (const XMLElement* bound_elem = bounds_elem->FirstChildElement("Boundary"); bound_elem;
                bound_elem = bound_elem->NextSiblingElement("Boundary")) {
                Boundary boundary;
                boundary.category = BoundaryCategory::Electrical;

                if (const XMLElement* name = bound_elem->FirstChildElement("Name")) {
                    boundary.name = get_text(name);
                }

                // FaceKeys
                if (const XMLElement* fkeys = bound_elem->FirstChildElement("FaceKeys")) {
                    for (const XMLElement* fk = fkeys->FirstChildElement("a:string"); fk;
                        fk = fk->NextSiblingElement("a:string")) {
                        std::string key = get_text(fk);
                        if (!key.empty()) {
                            boundary.face_keys.push_back(key);
                        }
                    }
                }

                // ThermalBoundary type
                const XMLElement* thermal = bound_elem->FirstChildElement("ThermalBoundary");
                if (thermal) {
                    const char* type = thermal->Attribute("i:type");
                    std::string type_str = type ? type : "";
                    if (type_str.find("FirstType") != std::string::npos) {
                        boundary.bc_type = ThermalBCType::FirstType;
                        if (const XMLElement* t = thermal->FirstChildElement("a:Temperature")) {
                            boundary.first.temperature = get_text(t);
                        }
                    }
                    else if (type_str.find("SecondType") != std::string::npos) {
                        boundary.bc_type = ThermalBCType::SecondType;
                        if (const XMLElement* q = thermal->FirstChildElement("a:HeatFlux")) {
                            boundary.second.heat_flux = get_text(q);
                        }
                    }
                    else if (type_str.find("ThirdType") != std::string::npos) {
                        boundary.bc_type = ThermalBCType::ThirdType;
                        if (const XMLElement* h = thermal->FirstChildElement("a:ConvectionCoefficient")) {
                            boundary.third.convection_coeff = get_text(h);
                        }
                        if (const XMLElement* t = thermal->FirstChildElement("a:EnvironmentTemperature")) {
                            boundary.third.T_inf = get_text(t);
                        }
                    }
                }

                structure.boundaries.push_back(boundary);
            }
        }

        // Mesh vertex coordinates from Results[0].Mesh (XArray/YArray/ZArray)
        if (const XMLElement* results_elem = root->FirstChildElement("Results")) {
            if (const XMLElement* any_type = results_elem->FirstChildElement("a:anyType")) {
                if (const XMLElement* mesh_elem = any_type->FirstChildElement("Mesh")) {
                    if (const XMLElement* x_array = mesh_elem->FirstChildElement("b:XArray")) {
                        for (const XMLElement* val = x_array->FirstChildElement("a:double"); val;
                            val = val->NextSiblingElement("a:double")) {
                            structure.mesh_vertex_x.push_back(parse_double(get_text(val)));
                        }
                    }
                    if (const XMLElement* y_array = mesh_elem->FirstChildElement("b:YArray")) {
                        for (const XMLElement* val = y_array->FirstChildElement("a:double"); val;
                            val = val->NextSiblingElement("a:double")) {
                            structure.mesh_vertex_y.push_back(parse_double(get_text(val)));
                        }
                    }
                    if (const XMLElement* z_array = mesh_elem->FirstChildElement("b:ZArray")) {
                        for (const XMLElement* val = z_array->FirstChildElement("a:double"); val;
                            val = val->NextSiblingElement("a:double")) {
                            structure.mesh_vertex_z.push_back(parse_double(get_text(val)));
                        }
                    }
                }
            }
        }

        return structure;
    }

    void write_vtu(const std::string& path, const InternalModel& model, const std::vector<double>& node_temperature)
    {
        using namespace tinyxml2;
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;
        int node_nx = mesh.nx + 1;
        int node_ny = mesh.ny + 1;
        int node_nz = mesh.nz + 1;

        // Build node remapping: only include nodes whose temperature is not NaN
        int total_nodes = node_nx * node_ny * node_nz;
        std::vector<int> node_remap(total_nodes, -1);
        std::vector<double> active_coords;
        std::vector<double> active_temps;

        auto node_idx = [](int vx, int vy, int vz, int nny, int nnz) { return vx * nny * nnz + vy * nnz + vz; };

        char buf[64];
        for (int vx = 0; vx < node_nx; vx++) {
            for (int vy = 0; vy < node_ny; vy++) {
                for (int vz = 0; vz < node_nz; vz++) {
                    int i = node_idx(vx, vy, vz, node_ny, node_nz);
                    double T = node_temperature[i];
                    if (std::isnan(T))
                        continue;
                    node_remap[i] = (int)active_temps.size();
                    active_temps.push_back(T);
                }
            }
        }

        int num_points = (int)active_temps.size();

        // Build string buffers
        std::string coords_str;
        for (int vx = 0; vx < node_nx; vx++) {
            for (int vy = 0; vy < node_ny; vy++) {
                for (int vz = 0; vz < node_nz; vz++) {
                    int i = node_idx(vx, vy, vz, node_ny, node_nz);
                    if (node_remap[i] < 0)
                        continue;
                    snprintf(
                        buf, sizeof(buf), "%.8g %.8g %.8g\n", mesh.vertex_x[vx], mesh.vertex_y[vy], mesh.vertex_z[vz]);
                    coords_str += buf;
                }
            }
        }

        std::string temp_str;
        for (double T : active_temps) {
            snprintf(buf, sizeof(buf), "%.8g\n", T);
            temp_str += buf;
        }

        // Build cell connectivity using remapped node indices
        std::string conn_str;
        std::string off_str;
        std::string type_str;
        int cell_num = 0;

        for (int ix = 0; ix < mesh.nx; ix++) {
            for (int iy = 0; iy < mesh.ny; iy++) {
                for (int iz = 0; iz < mesh.nz; iz++) {
                    int old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    if (cells.valid_mask[old_idx] == 0)
                        continue;

                    // VTK hex ordering: 0-3 bottom face, 4-7 top face
                    // Node indices in original grid
                    int n[8] = {node_idx(ix, iy, iz, node_ny, node_nz), node_idx(ix + 1, iy, iz, node_ny, node_nz),
                        node_idx(ix + 1, iy + 1, iz, node_ny, node_nz), node_idx(ix, iy + 1, iz, node_ny, node_nz),
                        node_idx(ix, iy, iz + 1, node_ny, node_nz), node_idx(ix + 1, iy, iz + 1, node_ny, node_nz),
                        node_idx(ix + 1, iy + 1, iz + 1, node_ny, node_nz),
                        node_idx(ix, iy + 1, iz + 1, node_ny, node_nz)};

                    // Remap to compact node indices
                    snprintf(buf, sizeof(buf), "%d %d %d %d %d %d %d %d\n", node_remap[n[0]], node_remap[n[1]],
                        node_remap[n[2]], node_remap[n[3]], node_remap[n[4]], node_remap[n[5]], node_remap[n[6]],
                        node_remap[n[7]]);
                    conn_str += buf;

                    cell_num++;
                    snprintf(buf, sizeof(buf), "%d\n", cell_num * 8);
                    off_str += buf;
                    type_str += "12\n";
                }
            }
        }

        // Assemble XML document
        XMLDocument doc;
        XMLElement* vtk_elem = doc.NewElement("VTKFile");
        vtk_elem->SetAttribute("type", "UnstructuredGrid");
        vtk_elem->SetAttribute("version", "0.1");
        vtk_elem->SetAttribute("byte_order", "LittleEndian");
        doc.InsertFirstChild(vtk_elem);

        XMLElement* grid_elem = doc.NewElement("UnstructuredGrid");
        vtk_elem->InsertEndChild(grid_elem);

        XMLElement* piece_elem = doc.NewElement("Piece");
        piece_elem->SetAttribute("NumberOfPoints", num_points);
        piece_elem->SetAttribute("NumberOfCells", cell_num);
        grid_elem->InsertEndChild(piece_elem);

        // Points
        XMLElement* points_elem = doc.NewElement("Points");
        piece_elem->InsertEndChild(points_elem);
        XMLElement* coords_arr = doc.NewElement("DataArray");
        coords_arr->SetAttribute("type", "Float64");
        coords_arr->SetAttribute("NumberOfComponents", "3");
        coords_arr->SetAttribute("format", "ascii");
        coords_arr->SetText(coords_str.c_str());
        points_elem->InsertEndChild(coords_arr);

        // PointData (temperature)
        XMLElement* point_data = doc.NewElement("PointData");
        piece_elem->InsertEndChild(point_data);
        XMLElement* temp_arr = doc.NewElement("DataArray");
        temp_arr->SetAttribute("type", "Float64");
        temp_arr->SetAttribute("Name", "Temperature");
        temp_arr->SetAttribute("NumberOfComponents", "1");
        temp_arr->SetAttribute("format", "ascii");
        temp_arr->SetText(temp_str.c_str());
        point_data->InsertEndChild(temp_arr);

        // Cells
        XMLElement* cells_elem = doc.NewElement("Cells");
        piece_elem->InsertEndChild(cells_elem);

        XMLElement* conn_arr_el = doc.NewElement("DataArray");
        conn_arr_el->SetAttribute("type", "Int32");
        conn_arr_el->SetAttribute("Name", "connectivity");
        conn_arr_el->SetAttribute("format", "ascii");
        conn_arr_el->SetText(conn_str.c_str());
        cells_elem->InsertEndChild(conn_arr_el);

        XMLElement* offsets_arr = doc.NewElement("DataArray");
        offsets_arr->SetAttribute("type", "Int32");
        offsets_arr->SetAttribute("Name", "offsets");
        offsets_arr->SetAttribute("format", "ascii");
        offsets_arr->SetText(off_str.c_str());
        cells_elem->InsertEndChild(offsets_arr);

        XMLElement* types_arr = doc.NewElement("DataArray");
        types_arr->SetAttribute("type", "UInt8");
        types_arr->SetAttribute("Name", "types");
        types_arr->SetAttribute("format", "ascii");
        types_arr->SetText(type_str.c_str());
        cells_elem->InsertEndChild(types_arr);

        std::filesystem::path dirPath(path);
        if (!std::filesystem::exists(dirPath.parent_path())) {
            std::filesystem::create_directories(dirPath.parent_path());
        }

        doc.SaveFile(path.c_str());
    }

    void write_xml(const std::string& input_path, const std::string& output_path, const InternalModel& model,
        const std::vector<double>& node_temperature)
    {
        using namespace tinyxml2;

        // Load the original XML
        XMLDocument doc;
        XMLError err = doc.LoadFile(input_path.c_str());
        if (err != XML_SUCCESS) {
            return;
        }

        XMLElement* results_elem = doc.FirstChildElement("Structure")->FirstChildElement("Results");
        if (!results_elem) {
            return;
        }

        XMLElement* any_type = results_elem->FirstChildElement("a:anyType");
        if (!any_type) {
            return;
        }

        XMLElement* values_elem = any_type->FirstChildElement("Values");
        if (!values_elem) {
            return;
        }

        XMLElement* data_elem = values_elem->FirstChildElement("Data");
        if (!data_elem) {
            return;
        }

        // Remove old data values
        while (XMLElement* child = data_elem->FirstChildElement("a:double")) {
            data_elem->DeleteChild(child);
        }

        // Node temperature layout: vx * node_ny * node_nz + vy * node_nz + vz
        // Reference data ordering: index = vz + SizeZ * vy + SizeZ * SizeY * vx
        // (X outermost, Y middle, Z innermost)
        // Note: in the reference formula, 'x' maps to our Y dimension (stride = SizeZ)
        // and 'y' maps to our X dimension (stride = SizeZ * SizeY).
        int node_nx = model.mesh.nx + 1;
        int node_ny = model.mesh.ny + 1;
        int node_nz = model.mesh.nz + 1;

        // Write new temperature values in reference ordering: (vx, vy, vz)
        // Reference index = vz + SizeZ * vy + SizeZ * SizeY * vx
        for (int vx = 0; vx < node_nx; vx++) {
            for (int vy = 0; vy < node_ny; vy++) {
                for (int vz = 0; vz < node_nz; vz++) {
                    double val = node_temperature[vx * node_ny * node_nz + vy * node_nz + vz];

                    XMLElement* double_elem = doc.NewElement("a:double");
                    if (std::isnan(val)) {
                        double_elem->SetText("NaN");
                    }
                    else {
                        char buf[64];
                        snprintf(buf, sizeof(buf), "%.6f", val);
                        double_elem->SetText(buf);
                    }
                    data_elem->InsertEndChild(double_elem);
                }
            }
        }

        // Update SizeX, SizeY, SizeZ
        XMLElement* sx = values_elem->FirstChildElement("SizeX");
        if (sx)
            sx->SetText(node_nx);
        XMLElement* sy = values_elem->FirstChildElement("SizeY");
        if (sy)
            sy->SetText(node_ny);
        XMLElement* sz = values_elem->FirstChildElement("SizeZ");
        if (sz)
            sz->SetText(node_nz);
        std::filesystem::path dirPath(output_path);
        if (!std::filesystem::exists(dirPath.parent_path())) {
            std::filesystem::create_directories(dirPath.parent_path());
        }
        doc.SaveFile(output_path.c_str());
    }

} // namespace mhs::io