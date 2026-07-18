#include <tinyxml2.h>

#include <algorithm>
#include <stdexcept>

#include "io/model_io.hpp"
#include "io/face_region_parser.hpp"
#include "io/xml_helpers.hpp"
#include "logging/logger.hpp"

namespace mhs::io {

    using namespace tinyxml2;
    using detail::get_text;
    using detail::parse_double;

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

    mhs::model::ModelDefinition read_xml(const std::string& xml_path)
    {
        XMLDocument doc;
        XMLError err = doc.LoadFile(xml_path.c_str());
        if (err != XML_SUCCESS) {
            throw std::runtime_error("Failed to load XML file: " + xml_path);
        }

        mhs::model::ModelDefinition structure;

        const XMLElement* root = doc.FirstChildElement("Structure");
        if (!root) {
            throw std::runtime_error("No Structure element found");
        }

        // Basic attributes
        const char* study_type_str = root->Attribute("StudyType");
        if (study_type_str) {
            if (std::string(study_type_str) == "Steady") {
                structure.settings.study_type = mhs::model::StudyType::Steady;
            }
            else {
                structure.settings.study_type = mhs::model::StudyType::Transient;
            }
        }
        else {
            // Try parsing from child element (for namespace-prefixed XML)
            const XMLElement* study_elem = root->FirstChildElement("StudyType");
            if (study_elem) {
                std::string val = get_text(study_elem);
                if (val == "Steady") {
                    structure.settings.study_type = mhs::model::StudyType::Steady;
                }
                else {
                    structure.settings.study_type = mhs::model::StudyType::Transient;
                }
            }
        }

        // Length unit
        const char* unit_str = root->Attribute("LengthUnit");
        if (unit_str) {
            std::string u = unit_str;
            if (u == "M") {
                structure.settings.length_unit = mhs::model::LengthUnit::Meter;
            }
            else if (u == "Mm") {
                structure.settings.length_unit = mhs::model::LengthUnit::Millimeter;
            }
            else if (u == "Um") {
                structure.settings.length_unit = mhs::model::LengthUnit::Micrometer;
            }
            else if (u == "Nm") {
                structure.settings.length_unit = mhs::model::LengthUnit::Nanometer;
            }
            else if (u == "Inch") {
                structure.settings.length_unit = mhs::model::LengthUnit::Inch;
            }
            else if (u == "Mil") {
                structure.settings.length_unit = mhs::model::LengthUnit::Mil;
            }
        }
        else {
            // Try parsing from child element
            const XMLElement* unit_elem = root->FirstChildElement("LengthUnit");
            if (unit_elem) {
                std::string u = get_text(unit_elem);
                if (u == "M") {
                    structure.settings.length_unit = mhs::model::LengthUnit::Meter;
                }
                else if (u == "Mm") {
                    structure.settings.length_unit = mhs::model::LengthUnit::Millimeter;
                }
                else if (u == "Um") {
                    structure.settings.length_unit = mhs::model::LengthUnit::Micrometer;
                }
                else if (u == "Nm") {
                    structure.settings.length_unit = mhs::model::LengthUnit::Nanometer;
                }
                else if (u == "Inch") {
                    structure.settings.length_unit = mhs::model::LengthUnit::Inch;
                }
                else if (u == "Mil") {
                    structure.settings.length_unit = mhs::model::LengthUnit::Mil;
                }
            }
        }

        // Temperature settings
        if (const XMLElement* init = root->FirstChildElement("InitialTemperature")) {
            structure.settings.initial_temperature = parse_double(get_text(init));
        }

        // Transient settings
        if (const XMLElement* trans = root->FirstChildElement("TransientStudyDuration")) {
            structure.settings.transient_duration = parse_double(get_text(trans));
        }
        if (const XMLElement* step = root->FirstChildElement("TransientStudyTimeStep")) {
            structure.settings.transient_output_interval = parse_double(get_text(step));
        }
        // OtherThermalBoundary (default BC)
        if (const XMLElement* other = root->FirstChildElement("OtherThermalBondary")) {
            const char* type = other->Attribute("i:type");
            std::string type_str = type ? type : "";
            if (type_str.find("FirstType") != std::string::npos) {
                mhs::model::DirichletBoundary bc;
                if (const XMLElement* temp = other->FirstChildElement("a:Temperature")) {
                    bc.temperature = get_text(temp);
                }
                structure.default_boundary = std::move(bc);
            }
            else if (type_str.find("SecondType") != std::string::npos) {
                mhs::model::NeumannBoundary bc;
                if (const XMLElement* flux = other->FirstChildElement("a:HeatFlux")) {
                    bc.heat_flux = get_text(flux);
                }
                structure.default_boundary = std::move(bc);
            }
            else if (type_str.find("ThirdType") != std::string::npos) {
                mhs::model::ConvectionBoundary bc;
                if (const XMLElement* h = other->FirstChildElement("a:ConvectionCoefficient")) {
                    bc.coefficient = get_text(h);
                }
                if (const XMLElement* t = other->FirstChildElement("a:EnvironmentTemperature")) {
                    bc.ambient_temperature = get_text(t);
                }
                structure.default_boundary = std::move(bc);
            }
        }

        // Variables
        if (const XMLElement* vars = root->FirstChildElement("Variables")) {
            for (const XMLElement* kv = vars->FirstChildElement("a:KeyValueOfstringdouble"); kv;
                kv = kv->NextSiblingElement("a:KeyValueOfstringdouble")) {
                mhs::model::VariableSpec var;
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
                mhs::model::MaterialSpec mat;
                std::string name;
                if (const XMLElement* key = kv->FirstChildElement("a:Key")) {
                    name = get_text(key);
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
                            mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = segs[0];
                        }
                        else if (segs.size() == 3) {
                            for (const auto& s : segs) {
                                if (s.empty()) {
                                    std::string preview = raw.substr(0, 200);
                                    MHS_LOG_WARN("DaoreXishu: empty segment, skipping.");
                                    continue;
                                }
                            }
                            mat.conductivity_x = segs[0];
                            mat.conductivity_y = segs[1];
                            mat.conductivity_z = segs[2];
                        }
                        else {
                            std::string preview = raw.substr(0, 200);
                            MHS_LOG_WARN("Invalid input! DaoreXishu must have 1 or 3 comma-separated expressions.");
                        }
                    }
                    if (const XMLElement* density = val->FirstChildElement("Midu")) {
                        mat.density = get_text(density);
                    }
                    if (const XMLElement* birerong = val->FirstChildElement("BiRerong")) {
                        mat.specific_heat = get_text(birerong);
                    }
                }
                if (!name.empty()) {
                    structure.materials.push_back({std::move(name), std::move(mat)});
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
                mhs::model::FunctionSpec fn;
                if (val) {
                    const char* type = val->Attribute("i:type");
                    std::string type_str = type ? type : "";
                    if (type_str.find("ExpressionFunction") != std::string::npos) {
                        mhs::model::ExpressionFunctionSpec expr;
                        read_string_member(val, "b:Expression", expr.expression);
                        fn = std::move(expr);
                    }
                    else if (type_str.find("DoubleExponentialFunction") != std::string::npos) {
                        mhs::model::DoubleExponentialFunctionSpec de;
                        read_double_member(val, "b:A", de.amplitude);
                        read_double_member(val, "b:Alpha", de.alpha);
                        read_double_member(val, "b:Beta", de.beta);
                        fn = std::move(de);
                    }
                    else if (type_str.find("GaussFunction") != std::string::npos) {
                        mhs::model::GaussFunctionSpec g;
                        read_double_member(val, "b:A", g.amplitude);
                        read_double_member(val, "b:Tau", g.tau);
                        read_double_member(val, "b:X0", g.center);
                        fn = std::move(g);
                    }
                    else if (type_str.find("SineFunction") != std::string::npos) {
                        mhs::model::SineFunctionSpec s;
                        read_double_member(val, "b:A", s.amplitude);
                        read_double_member(val, "b:Omega", s.angular_frequency);
                        read_double_member(val, "b:Phi", s.phase);
                        fn = std::move(s);
                    }
                    else if (type_str.find("PieceWiseFunction") != std::string::npos) {
                        mhs::model::PiecewiseFunctionSpec pw;
                        if (const XMLElement* points = val->FirstChildElement("b:Points")) {
                            for (const XMLElement* pt = points->FirstChildElement("b:PieceWiseFunction.Point"); pt;
                                pt = pt->NextSiblingElement("b:PieceWiseFunction.Point")) {
                                mhs::model::PiecewiseFunctionSpec::Point p;
                                read_double_member(pt, "b:X", p.x);
                                read_double_member(pt, "b:Y", p.y);
                                pw.points.push_back(p);
                            }
                            // Pre-sort by X so the closure can binary-search without
                            // sorting again at registration time.
                            std::sort(pw.points.begin(), pw.points.end(),
                                [](const mhs::model::PiecewiseFunctionSpec::Point& a,
                                    const mhs::model::PiecewiseFunctionSpec::Point& b) { return a.x < b.x; });
                        }
                        fn = std::move(pw);
                    }
                    else if (!type_str.empty()) {
                        throw std::runtime_error("Unknown function i:type: " + type_str);
                    }
                }
                if (!name.empty()) {
                    structure.functions.push_back({std::move(name), std::move(fn)});
                }
            }
        }

        // Layers
        if (const XMLElement* layers_elem = root->FirstChildElement("Layers")) {
            for (const XMLElement* layer_elem = layers_elem->FirstChildElement("Layer"); layer_elem;
                layer_elem = layer_elem->NextSiblingElement("Layer")) {
                mhs::model::LayerSpec layer;

                if (const XMLElement* thickness = layer_elem->FirstChildElement("ThicknessExpression")) {
                    layer.thickness = get_text(thickness);
                }
                if (const XMLElement* xoff = layer_elem->FirstChildElement("XOffsetExpression")) {
                    layer.x_offset = get_text(xoff);
                }
                if (const XMLElement* yoff = layer_elem->FirstChildElement("YOffsetExpression")) {
                    layer.y_offset = get_text(yoff);
                }
                // Blocks
                if (const XMLElement* blocks_elem = layer_elem->FirstChildElement("Blocks")) {
                    for (const XMLElement* block_elem = blocks_elem->FirstChildElement("Block"); block_elem;
                        block_elem = block_elem->NextSiblingElement("Block")) {
                        mhs::model::BlockSpec block;

                        if (const XMLElement* mat = block_elem->FirstChildElement("MaterialName")) {
                            block.material = get_text(mat);
                        }
                        if (const XMLElement* ti = block_elem->FirstChildElement("TiReyuan")) {
                            block.volumetric_heat_source = get_text(ti);
                        }
                        if (const XMLElement* xoff = block_elem->FirstChildElement("XOffsetExpression")) {
                            block.x_offset = get_text(xoff);
                        }
                        if (const XMLElement* yoff = block_elem->FirstChildElement("YOffsetExpression")) {
                            block.y_offset = get_text(yoff);
                        }
                        if (const XMLElement* thickness = block_elem->FirstChildElement("ThicknessExpression")) {
                            block.thickness = get_text(thickness);
                        }

                        // Rects (AllRects)
                        if (const XMLElement* rects_elem = block_elem->FirstChildElement("AllRects")) {
                            for (const XMLElement* rect_elem = rects_elem->FirstChildElement("Rect"); rect_elem;
                                rect_elem = rect_elem->NextSiblingElement("Rect")) {
                                mhs::model::RectOperation rect;
                                if (const XMLElement* adds = rect_elem->FirstChildElement("Add_sub")) {
                                    rect.operation = get_text(adds) == "true" ? mhs::model::GeometryOperation::Add
                                                                              : mhs::model::GeometryOperation::Subtract;
                                }
                                if (const XMLElement* w = rect_elem->FirstChildElement("WidthExpression")) {
                                    rect.rect.width = get_text(w);
                                }
                                if (const XMLElement* h = rect_elem->FirstChildElement("HeightExpression")) {
                                    rect.rect.height = get_text(h);
                                }
                                if (const XMLElement* x = rect_elem->FirstChildElement("XExpression")) {
                                    rect.rect.x = get_text(x);
                                }
                                if (const XMLElement* y = rect_elem->FirstChildElement("YExpression")) {
                                    rect.rect.y = get_text(y);
                                }
                                block.geometry.push_back(std::move(rect));
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
                mhs::model::BoundaryPatch boundary;

                // FaceKeys
                if (const XMLElement* fkeys = bound_elem->FirstChildElement("FaceKeys")) {
                    for (const XMLElement* fk = fkeys->FirstChildElement("a:string"); fk;
                        fk = fk->NextSiblingElement("a:string")) {
                        std::string key = get_text(fk);
                        if (!key.empty()) {
                            boundary.regions.push_back(detail::parse_face_region(key));
                        }
                    }
                }

                // ThermalBoundary type
                const XMLElement* thermal = bound_elem->FirstChildElement("ThermalBoundary");
                if (thermal) {
                    const char* type = thermal->Attribute("i:type");
                    std::string type_str = type ? type : "";
                    if (type_str.find("FirstType") != std::string::npos) {
                        mhs::model::DirichletBoundary bc;
                        if (const XMLElement* t = thermal->FirstChildElement("a:Temperature")) {
                            bc.temperature = get_text(t);
                        }
                        boundary.condition = std::move(bc);
                    }
                    else if (type_str.find("SecondType") != std::string::npos) {
                        mhs::model::NeumannBoundary bc;
                        if (const XMLElement* q = thermal->FirstChildElement("a:HeatFlux")) {
                            bc.heat_flux = get_text(q);
                        }
                        boundary.condition = std::move(bc);
                    }
                    else if (type_str.find("ThirdType") != std::string::npos) {
                        mhs::model::ConvectionBoundary bc;
                        if (const XMLElement* h = thermal->FirstChildElement("a:ConvectionCoefficient")) {
                            bc.coefficient = get_text(h);
                        }
                        if (const XMLElement* t = thermal->FirstChildElement("a:EnvironmentTemperature")) {
                            bc.ambient_temperature = get_text(t);
                        }
                        boundary.condition = std::move(bc);
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
                            structure.mesh.x_vertices.push_back(parse_double(get_text(val)));
                        }
                    }
                    if (const XMLElement* y_array = mesh_elem->FirstChildElement("b:YArray")) {
                        for (const XMLElement* val = y_array->FirstChildElement("a:double"); val;
                            val = val->NextSiblingElement("a:double")) {
                            structure.mesh.y_vertices.push_back(parse_double(get_text(val)));
                        }
                    }
                    if (const XMLElement* z_array = mesh_elem->FirstChildElement("b:ZArray")) {
                        for (const XMLElement* val = z_array->FirstChildElement("a:double"); val;
                            val = val->NextSiblingElement("a:double")) {
                            structure.mesh.z_vertices.push_back(parse_double(get_text(val)));
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
                mhs::model::ObservationPointSpec op;
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

} // namespace mhs::io
