#include <filesystem>
#include <fstream>
#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/logger.hpp"
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

    // 清空 parent 的全部 <a:double> 子节点（O(1) 调用 DeleteChildren），再依次追加 data。
    // allow_nan=false 时遇到 NaN 写 "NaN"（参考 XML 风格）。
    static void refill_double_list(
        tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* parent, const std::vector<double>& data, bool allow_nan)
    {
        if (!parent)
            return;
        parent->DeleteChildren();
        for (double v : data) {
            auto* d = doc.NewElement("a:double");
            if (allow_nan && std::isnan(v)) {
                d->SetText("NaN");
            }
            else {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.6f", v);
                d->SetText(buf);
            }
            parent->InsertEndChild(d);
        }
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

    mhs::core::IOStructure read_xml(const std::string& xml_path)
    {
        XMLDocument doc;
        XMLError err = doc.LoadFile(xml_path.c_str());
        if (err != XML_SUCCESS) {
            throw std::runtime_error("Failed to load XML file: " + xml_path);
        }

        mhs::core::IOStructure structure;

        const XMLElement* root = doc.FirstChildElement("Structure");
        if (!root) {
            throw std::runtime_error("No Structure element found");
        }

        // Basic attributes
        const char* study_type_str = root->Attribute("StudyType");
        if (study_type_str) {
            if (std::string(study_type_str) == "Steady") {
                structure.study_type = mhs::core::StudyType::Steady;
            }
            else {
                structure.study_type = mhs::core::StudyType::Transient;
            }
        }
        else {
            // Try parsing from child element (for namespace-prefixed XML)
            const XMLElement* study_elem = root->FirstChildElement("StudyType");
            if (study_elem) {
                std::string val = get_text(study_elem);
                if (val == "Steady") {
                    structure.study_type = mhs::core::StudyType::Steady;
                }
                else {
                    structure.study_type = mhs::core::StudyType::Transient;
                }
            }
        }

        const char* dim_str = root->Attribute("Dimension");
        if (dim_str) {
            if (std::string(dim_str) == "Dimension3D") {
                structure.dimension = mhs::core::Dimension::Dimension3D;
            }
            else {
                structure.dimension = mhs::core::Dimension::Dimension2D;
            }
        }
        else {
            // Try parsing from child element (for namespace-prefixed XML)
            const XMLElement* dim_elem = root->FirstChildElement("Dimension");
            if (dim_elem) {
                std::string val = get_text(dim_elem);
                if (val == "Dimension3D") {
                    structure.dimension = mhs::core::Dimension::Dimension3D;
                }
                else {
                    structure.dimension = mhs::core::Dimension::Dimension2D;
                }
            }
        }

        // Length unit
        const char* unit_str = root->Attribute("LengthUnit");
        if (unit_str) {
            std::string u = unit_str;
            if (u == "M") {
                structure.length_unit = mhs::core::LengthUnit::M;
            }
            else if (u == "Mm") {
                structure.length_unit = mhs::core::LengthUnit::Mm;
            }
            else if (u == "Um") {
                structure.length_unit = mhs::core::LengthUnit::Um;
            }
            else if (u == "Nm") {
                structure.length_unit = mhs::core::LengthUnit::Nm;
            }
            else if (u == "Inch") {
                structure.length_unit = mhs::core::LengthUnit::Inch;
            }
            else if (u == "Mil") {
                structure.length_unit = mhs::core::LengthUnit::Mil;
            }
        }
        else {
            // Try parsing from child element
            const XMLElement* unit_elem = root->FirstChildElement("LengthUnit");
            if (unit_elem) {
                std::string u = get_text(unit_elem);
                if (u == "M") {
                    structure.length_unit = mhs::core::LengthUnit::M;
                }
                else if (u == "Mm") {
                    structure.length_unit = mhs::core::LengthUnit::Mm;
                }
                else if (u == "Um") {
                    structure.length_unit = mhs::core::LengthUnit::Um;
                }
                else if (u == "Nm") {
                    structure.length_unit = mhs::core::LengthUnit::Nm;
                }
                else if (u == "Inch") {
                    structure.length_unit = mhs::core::LengthUnit::Inch;
                }
                else if (u == "Mil") {
                    structure.length_unit = mhs::core::LengthUnit::Mil;
                }
            }
        }

        // Temperature settings
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
                structure.other_bc_type = mhs::core::ThermalBCType::FirstType;
                if (const XMLElement* temp = other->FirstChildElement("a:Temperature")) {
                    structure.other_bc_first.temperature = get_text(temp);
                }
            }
            else if (type_str.find("SecondType") != std::string::npos) {
                structure.other_bc_type = mhs::core::ThermalBCType::SecondType;
                if (const XMLElement* flux = other->FirstChildElement("a:HeatFlux")) {
                    structure.other_bc_second.heat_flux = get_text(flux);
                }
            }
            else if (type_str.find("ThirdType") != std::string::npos) {
                structure.other_bc_type = mhs::core::ThermalBCType::ThirdType;
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
                mhs::core::Variable var;
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
                mhs::core::Material mat;
                if (const XMLElement* key = kv->FirstChildElement("a:Key")) {
                    mat.name = get_text(key);
                }
                const XMLElement* val = kv->FirstChildElement("a:Value");
                if (val) {
                    if (const XMLElement* daore = val->FirstChildElement("DaoreXishu")) {
                        std::string raw = get_text(daore);
                        std::vector<std::string> segs;
                        size_t start = 0;
                        while (true) {
                            size_t end = raw.find(',', start);
                            std::string token
                                = (end == std::string::npos) ? raw.substr(start) : raw.substr(start, end - start);
                            size_t f = token.find_first_not_of(" \t\r\n");
                            size_t l = (f == std::string::npos) ? std::string::npos : token.find_last_not_of(" \t\r\n");
                            token = (f == std::string::npos) ? std::string() : token.substr(f, l - f + 1);
                            segs.push_back(token);
                            if (end == std::string::npos)
                                break;
                            start = end + 1;
                        }
                        if (segs.size() == 1) {
                            mat.kx = mat.ky = mat.kz = segs[0];
                        }
                        else if (segs.size() == 3) {
                            for (const auto& s : segs) {
                                if (s.empty()) {
                                    std::string preview = raw.substr(0, 200);
                                    MHS_FATAL("DaoreXishu: empty segment in '{}'", preview);
                                }
                            }
                            mat.kx = segs[0];
                            mat.ky = segs[1];
                            mat.kz = segs[2];
                        }
                        else {
                            std::string preview = raw.substr(0, 200);
                            MHS_FATAL("DaoreXishu must have 1 or 3 comma-separated expressions, got {}: '{}'",
                                segs.size(), preview);
                        }
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
                mhs::core::Function fn;
                if (val) {
                    const char* type = val->Attribute("i:type");
                    std::string type_str = type ? type : "";
                    if (type_str.find("ExpressionFunction") != std::string::npos) {
                        fn.type = mhs::core::FunctionType::Expression;
                        read_string_member(val, "b:Expression", fn.expression.expression);
                        read_draw_range(val, fn.expression.draw_min_x, fn.expression.draw_max_x);
                    }
                    else if (type_str.find("DoubleExponentialFunction") != std::string::npos) {
                        fn.type = mhs::core::FunctionType::DoubleExponential;
                        read_double_member(val, "b:A", fn.double_exp.a);
                        read_double_member(val, "b:Alpha", fn.double_exp.alpha);
                        read_double_member(val, "b:Beta", fn.double_exp.beta);
                        read_draw_range(val, fn.double_exp.draw_min_x, fn.double_exp.draw_max_x);
                    }
                    else if (type_str.find("GaussFunction") != std::string::npos) {
                        fn.type = mhs::core::FunctionType::Gauss;
                        read_double_member(val, "b:A", fn.gauss.a);
                        read_double_member(val, "b:Tau", fn.gauss.tau);
                        read_double_member(val, "b:X0", fn.gauss.x0);
                        read_draw_range(val, fn.gauss.draw_min_x, fn.gauss.draw_max_x);
                    }
                    else if (type_str.find("SineFunction") != std::string::npos) {
                        fn.type = mhs::core::FunctionType::Sine;
                        read_double_member(val, "b:A", fn.sine.a);
                        read_double_member(val, "b:Omega", fn.sine.omega);
                        read_double_member(val, "b:Phi", fn.sine.phi);
                        read_draw_range(val, fn.sine.draw_min_x, fn.sine.draw_max_x);
                    }
                    else if (type_str.find("PieceWiseFunction") != std::string::npos) {
                        fn.type = mhs::core::FunctionType::PieceWise;
                        if (const XMLElement* points = val->FirstChildElement("b:Points")) {
                            for (const XMLElement* pt = points->FirstChildElement("b:PieceWiseFunction.Point"); pt;
                                pt = pt->NextSiblingElement("b:PieceWiseFunction.Point")) {
                                mhs::core::PieceWiseFunction::Point p;
                                read_double_member(pt, "b:X", p.x);
                                read_double_member(pt, "b:Y", p.y);
                                fn.piecewise.points.push_back(p);
                            }
                            // Pre-sort by X so the closure can binary-search without
                            // sorting again at registration time.
                            std::sort(fn.piecewise.points.begin(), fn.piecewise.points.end(),
                                [](const mhs::core::PieceWiseFunction::Point& a,
                                    const mhs::core::PieceWiseFunction::Point& b) { return a.x < b.x; });
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
                mhs::core::Layer layer;

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
                        mhs::core::Block block;

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
                        if (const XMLElement* thickness = block_elem->FirstChildElement("ThicknessExpression")) {
                            block.thickness_expr = get_text(thickness);
                        }
                        if (const XMLElement* btype = block_elem->FirstChildElement("BlockType")) {
                            std::string bt = get_text(btype);
                            if (bt == "SmartMacro" || bt == "smart_macro") {
                                block.block_type = mhs::core::BlockType::SmartMacro;
                            }
                        }
                        if (const XMLElement* mfile = block_elem->FirstChildElement("ModelFile")) {
                            block.model_file = get_text(mfile);
                        }

                        // Rects (AllRects)
                        if (const XMLElement* rects_elem = block_elem->FirstChildElement("AllRects")) {
                            for (const XMLElement* rect_elem = rects_elem->FirstChildElement("Rect"); rect_elem;
                                rect_elem = rect_elem->NextSiblingElement("Rect")) {
                                mhs::core::Rect rect;
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
                mhs::core::Boundary boundary;
                boundary.category = mhs::core::BoundaryCategory::Electrical;

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
                        boundary.bc_type = mhs::core::ThermalBCType::FirstType;
                        if (const XMLElement* t = thermal->FirstChildElement("a:Temperature")) {
                            boundary.first.temperature = get_text(t);
                        }
                    }
                    else if (type_str.find("SecondType") != std::string::npos) {
                        boundary.bc_type = mhs::core::ThermalBCType::SecondType;
                        if (const XMLElement* q = thermal->FirstChildElement("a:HeatFlux")) {
                            boundary.second.heat_flux = get_text(q);
                        }
                    }
                    else if (type_str.find("ThirdType") != std::string::npos) {
                        boundary.bc_type = mhs::core::ThermalBCType::ThirdType;
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

        // ObservePoints3D — 用户坐标系下的探针列表。3D 专用；2D 路径暂不支持。
        // x/y/z 保留为 muparser 表达式字符串，由 preprocessor 在加载时统一求值。
        if (const XMLElement* obs3d = root->FirstChildElement("ObservePoints3D")) {
            for (const XMLElement* pt = obs3d->FirstChildElement("ObservePoint3D"); pt;
                pt = pt->NextSiblingElement("ObservePoint3D")) {
                mhs::core::ObservationPoint3D op;
                if (const XMLElement* name = pt->FirstChildElement("Name")) {
                    op.name = get_text(name);
                }
                if (const XMLElement* x = pt->FirstChildElement("X")) {
                    op.x = get_text(x);
                }
                if (const XMLElement* y = pt->FirstChildElement("Y")) {
                    op.y = get_text(y);
                }
                if (const XMLElement* z = pt->FirstChildElement("Z")) {
                    op.z = get_text(z);
                }
                structure.observation_points.push_back(op);
            }
        }

        return structure;
    }

    // =========================================================================
    // Fluid overlay XML parser
    // =========================================================================

    std::optional<mhs::core::FluidOverlay> read_fluid_overlay_xml(const std::string& xml_path)
    {
        XMLDocument doc;
        XMLError err = doc.LoadFile(xml_path.c_str());
        if (err != XML_SUCCESS) {
            return std::nullopt;
        }

        const XMLElement* root = doc.FirstChildElement("FluidOverlay");
        if (!root) {
            return std::nullopt;
        }

        mhs::core::FluidOverlay overlay;

        // Parse FluidMaterial nodes
        for (const XMLElement* mat_elem = root->FirstChildElement("FluidMaterial"); mat_elem;
            mat_elem = mat_elem->NextSiblingElement("FluidMaterial")) {
            mhs::core::FluidMaterialOverlay fm;
            // name is an XML attribute, not a child element
            if (const char* attr = mat_elem->Attribute("name")) {
                fm.name = attr;
            }
            if (const XMLElement* visc = mat_elem->FirstChildElement("DynamicViscosity")) {
                fm.dynamic_viscosity = get_text(visc);
            }
            if (!fm.name.empty()) {
                overlay.fluid_materials.push_back(std::move(fm));
            }
        }

        // Parse Boundary nodes (fluidic)
        for (const XMLElement* bound_elem = root->FirstChildElement("Boundary"); bound_elem;
            bound_elem = bound_elem->NextSiblingElement("Boundary")) {
            mhs::core::FluidBoundaryOverlay fb;

            if (const XMLElement* name = bound_elem->FirstChildElement("Name")) {
                fb.name = get_text(name);
            }

            // FaceKeys
            if (const XMLElement* fkeys = bound_elem->FirstChildElement("FaceKeys")) {
                for (const XMLElement* fk = fkeys->FirstChildElement("string"); fk;
                    fk = fk->NextSiblingElement("string")) {
                    std::string key = get_text(fk);
                    if (!key.empty()) {
                        fb.face_keys.push_back(key);
                    }
                }
            }

            // Pressure / MassFlowRate / Velocity (mutually exclusive, drives fb.kind)
            if (const XMLElement* p = bound_elem->FirstChildElement("Pressure")) {
                fb.value = parse_double(get_text(p));
                fb.kind = mhs::core::FluidBCType::PressureType;
            }
            else if (const XMLElement* mfr = bound_elem->FirstChildElement("MassFlowRate")) {
                fb.value = parse_double(get_text(mfr));
                fb.kind = mhs::core::FluidBCType::MassFlowRateType;
            }
            else if (const XMLElement* vel = bound_elem->FirstChildElement("Velocity")) {
                fb.value = parse_double(get_text(vel));
                fb.kind = mhs::core::FluidBCType::VelocityType;
            }

            // InletTemperature (optional)
            if (const XMLElement* tin = bound_elem->FirstChildElement("InletTemperature")) {
                fb.inlet_temperature = parse_double(get_text(tin));
            }

            if (!fb.name.empty()) {
                overlay.boundaries.push_back(std::move(fb));
            }
        }

        return overlay;
    }

    void write_vtu(const std::string& path, const mhs::core::Model& model, const std::vector<double>& node_temperature)
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
                    double node_x = (vx == 0) ? mesh.cx[0] - mesh.dx[0] * 0.5 : mesh.cx[vx - 1] + mesh.dx[vx - 1] * 0.5;
                    double node_y = (vy == 0) ? mesh.cy[0] - mesh.dy[0] * 0.5 : mesh.cy[vy - 1] + mesh.dy[vy - 1] * 0.5;
                    double node_z = (vz == 0) ? mesh.cz[0] - mesh.dz[0] * 0.5 : mesh.cz[vz - 1] + mesh.dz[vz - 1] * 0.5;
                    snprintf(buf, sizeof(buf), "%.8g %.8g %.8g\n", node_x, node_y, node_z);
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
                    if (cells.index_map[old_idx] == mhs::core::invalidIndex)
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

    void write_xml(const std::string& input_path, const std::string& output_path, const mhs::core::Model& model,
        const std::vector<double>& node_temperature, const std::vector<mhs::core::ProbeTrace>& observation_traces)
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

        // 注入 Result0DTransient 节点（每个观察点一个）。
        // - 已有 PointName 节点：清空其 <Times>/<Values> 内的 <a:double>，重新填充。
        // - 没有则：在 Results 末尾新建。
        if (!observation_traces.empty()) {
            for (const auto& trace : observation_traces) {
                XMLElement* target = nullptr;
                for (XMLElement* cand = results_elem->FirstChildElement("a:anyType"); cand;
                    cand = cand->NextSiblingElement("a:anyType")) {
                    const char* t = cand->Attribute("i:type");
                    if (!t || std::string(t).find("Result0DTransient") == std::string::npos)
                        continue;
                    const XMLElement* pn = cand->FirstChildElement("PointName");
                    if (pn && get_text(pn) == trace.name) {
                        target = cand;
                        break;
                    }
                }

                if (!target) {
                    target = doc.NewElement("a:anyType");
                    target->SetAttribute("i:type", "Result0DTransient");
                    // 顺序：PhysicsName / PointName / TimeUnit / Times / UnitName / Values
                    // 与参考 XML 对齐，避免节点顺序变化引入差异。
                    XMLElement* phys = doc.NewElement("PhysicsName");
                    phys->SetText("温度");
                    target->InsertEndChild(phys);
                    XMLElement* pn = doc.NewElement("PointName");
                    pn->SetText(trace.name.c_str());
                    target->InsertEndChild(pn);
                    XMLElement* tu = doc.NewElement("TimeUnit");
                    tu->SetText("S");
                    target->InsertEndChild(tu);
                    XMLElement* times = doc.NewElement("Times");
                    target->InsertEndChild(times);
                    XMLElement* un = doc.NewElement("UnitName");
                    un->SetText("K");
                    target->InsertEndChild(un);
                    XMLElement* values = doc.NewElement("Values");
                    target->InsertEndChild(values);
                    results_elem->InsertEndChild(target);
                }

                refill_double_list(doc, target->FirstChildElement("Times"), trace.times, /*allow_nan=*/false);
                refill_double_list(doc, target->FirstChildElement("Values"), trace.values, /*allow_nan=*/true);
            }
        }

        std::filesystem::path dirPath(output_path);
        if (!std::filesystem::exists(dirPath.parent_path())) {
            std::filesystem::create_directories(dirPath.parent_path());
        }
        doc.SaveFile(output_path.c_str());
    }

    // =========================================================================
    // SmartMacro model loading
    // =========================================================================

    mhs::core::SmartMacroModelData read_smart_macro_model(const std::string& xml_path)
    {
        XMLDocument doc;
        if (doc.LoadFile(xml_path.c_str()) != XML_SUCCESS) {
            throw std::runtime_error("Failed to load SmartMacro model XML: " + xml_path);
        }

        const auto* root = doc.FirstChildElement("SmartMacroModel");
        if (!root) {
            throw std::runtime_error("No <SmartMacroModel> element found in " + xml_path);
        }

        mhs::core::SmartMacroModelData result;

        // Name
        if (const auto* name_elem = root->FirstChildElement("Name")) {
            if (const char* text = name_elem->GetText())
                result.name = std::string(text);
        }

        // NPorts
        int n_ports = 0;
        if (const auto* np_elem = root->FirstChildElement("NPorts")) {
            if (const char* text = np_elem->GetText())
                n_ports = std::stoi(text);
        }
        else {
            throw std::runtime_error("Missing <NPorts> in SmartMacro model");
        }

        if (n_ports <= 0) {
            throw std::runtime_error("Invalid NPorts=" + std::to_string(n_ports) + " in SmartMacro model");
        }

        // NModes (POD format, required)
        int n_modes = 0;
        if (const auto* nm_elem = root->FirstChildElement("NModes")) {
            if (const char* text = nm_elem->GetText())
                n_modes = std::stoi(text);
        }
        else {
            throw std::runtime_error("Missing <NModes> in SmartMacro model (POD format required)");
        }

        if (n_modes <= 0) {
            throw std::runtime_error("Invalid NModes=" + std::to_string(n_modes) + " in SmartMacro model");
        }

        result.n_modes = n_modes;

        // PortOrder (with Dir for face-level ports)
        result.port_ix.reserve(n_ports);
        result.port_iy.reserve(n_ports);
        result.port_iz.reserve(n_ports);
        result.port_dir.reserve(n_ports);

        const auto* po_elem = root->FirstChildElement("PortOrder");
        if (!po_elem) {
            throw std::runtime_error("Missing <PortOrder> in SmartMacro model");
        }

        for (const auto* port_elem = po_elem->FirstChildElement("Port"); port_elem;
            port_elem = port_elem->NextSiblingElement("Port")) {

            int ix = 0, iy = 0, iz = 0, dir = 0;
            if (const auto* e = port_elem->FirstChildElement("IX"))
                ix = std::stoi(e->GetText() ? e->GetText() : "0");
            if (const auto* e = port_elem->FirstChildElement("IY"))
                iy = std::stoi(e->GetText() ? e->GetText() : "0");
            if (const auto* e = port_elem->FirstChildElement("IZ"))
                iz = std::stoi(e->GetText() ? e->GetText() : "0");
            if (const auto* e = port_elem->FirstChildElement("Dir"))
                dir = std::stoi(e->GetText() ? e->GetText() : "0");

            result.port_ix.push_back(ix);
            result.port_iy.push_back(iy);
            result.port_iz.push_back(iz);
            result.port_dir.push_back(dir);
        }

        if (static_cast<int>(result.port_ix.size()) != n_ports) {
            throw std::runtime_error("PortOrder has " + std::to_string(result.port_ix.size())
                + " entries but NPorts=" + std::to_string(n_ports));
        }

        // DataFile: resolve relative to XML directory
        std::string data_file;
        if (const auto* df_elem = root->FirstChildElement("DataFile")) {
            if (const char* text = df_elem->GetText())
                data_file = std::string(text);
        }
        if (data_file.empty()) {
            throw std::runtime_error("Missing <DataFile> in SmartMacro model");
        }

        auto data_path = std::filesystem::path(xml_path).parent_path() / data_file;

        // Read binary data: [K_modal: M*M doubles, row-major][phi_basis: N*M doubles, row-major]
        // NOTE: f_modal not stored — always zero for BC-agnostic training.
        std::ifstream bin(data_path, std::ios::binary);
        if (!bin) {
            throw std::runtime_error("Failed to open binary data file: " + data_path.string());
        }

        // Read K_modal (row-major flat vector, M x M)
        result.K_modal.resize(static_cast<size_t>(n_modes) * static_cast<size_t>(n_modes));
        bin.read(reinterpret_cast<char*>(result.K_modal.data()),
            static_cast<std::streamsize>(result.K_modal.size() * sizeof(double)));
        if (!bin) {
            throw std::runtime_error("Failed to read K_modal from binary data");
        }

        // Read phi_basis (row-major: N_ports x n_modes)
        result.phi_basis.resize(static_cast<size_t>(n_ports) * static_cast<size_t>(n_modes));
        bin.read(reinterpret_cast<char*>(result.phi_basis.data()),
            static_cast<std::streamsize>(result.phi_basis.size() * sizeof(double)));
        if (!bin) {
            throw std::runtime_error("Failed to read phi_basis from binary data");
        }

        return result;
    }

    std::vector<mhs::core::SmartMacroModelData> load_smart_macro_models(
        const mhs::core::IOStructure& io, const std::string& case_dir)
    {
        if (case_dir.empty())
            return {};

        std::vector<mhs::core::SmartMacroModelData> result;
        for (const auto& layer : io.layers) {
            for (const auto& block : layer.blocks) {
                if (block.block_type != mhs::core::BlockType::SmartMacro)
                    continue;
                const std::string model_path = (std::filesystem::path(case_dir) / block.model_file).string();
                result.push_back(read_smart_macro_model(model_path));
            }
        }
        return result;
    }

} // namespace mhs::io