#include "io.hpp"
#include <stdexcept>
#include <tinyxml2.h>

using mhs::model::Dimension;
using mhs::model::LengthUnit;
using mhs::model::ThermalBCType;

namespace mhs::io {

    using namespace tinyxml2;

    static std::string get_text(const XMLElement* elem)
    {
        if (!elem)
            return "";
        const char* text = elem->GetText();
        return text ? text : "";
    }

    static double parse_double(const std::string& s)
    {
        if (s.empty()) {
            return 0.0;
        }
        return std::stod(s);
    }

    model::IOStructure read_xml(const std::string& xml_path)
    {
        XMLDocument doc;
        XMLError err = doc.LoadFile(xml_path.c_str());
        if (err != XML_SUCCESS) {
            throw std::runtime_error("Failed to load XML file: " + xml_path);
        }

        model::IOStructure structure;

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
                model::Variable var;
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
            for (const XMLElement* kv = mats->FirstChildElement("a:KeyValueOfstringMaterialGyu7GfTz");
                kv; kv = kv->NextSiblingElement("a:KeyValueOfstringMaterialGyu7GfTz")) {
                model::Material mat;
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

        // Layers
        if (const XMLElement* layers_elem = root->FirstChildElement("Layers")) {
            for (const XMLElement* layer_elem = layers_elem->FirstChildElement("Layer"); layer_elem;
                layer_elem = layer_elem->NextSiblingElement("Layer")) {
                model::Layer layer;

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
                        model::Block block;

                        if (const XMLElement* name = block_elem->FirstChildElement("Name")) {
                            block.name = get_text(name);
                        }
                        if (const XMLElement* mat = block_elem->FirstChildElement("MaterialName")) {
                            block.material_name = get_text(mat);
                        }
                        if (const XMLElement* thick = block_elem->FirstChildElement("ThicknessExpression")) {
                            block.thickness_expr = get_text(thick);
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
                        if (const XMLElement* zoff = block_elem->FirstChildElement("ZOffsetExpression")) {
                            block.z_offset_expr = get_text(zoff);
                        }
                        if (const XMLElement* normal = block_elem->FirstChildElement("IsNormalMaterial")) {
                            block.is_normal_material = std::string(get_text(normal)) == "true";
                        }

                        // Rects (AllRects)
                        if (const XMLElement* rects_elem = block_elem->FirstChildElement("AllRects")) {
                            for (const XMLElement* rect_elem = rects_elem->FirstChildElement("Rect");
                                rect_elem; rect_elem = rects_elem->NextSiblingElement("Rect")) {
                                model::Rect rect;
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
                model::Boundary boundary;
                boundary.category = mhs::model::BoundaryCategory::Electrical;

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

    void write_vtu(const std::string& path,
        const model::InternalModel& model,
        const std::vector<double>& node_temperature)
    {
        (void)model;
        (void)path;
        (void)node_temperature;
    }

    void write_xml(const std::string& input_path,
        const std::string& output_path,
        const model::InternalModel& model,
        const std::vector<double>& node_temperature)
    {
        // model parameter kept for API consistency; not used in current implementation
        (void)model;
        (void)output_path;
        (void)input_path;
        (void)node_temperature;
    }

} // namespace mhs::io