#include <tinyxml2.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/model_definition.hpp"
#include "io/face_region_parser.hpp"
#include "io/model_io.hpp"
#include "io/xml_helpers.hpp"

namespace mhs::io {

    using namespace tinyxml2;
    using detail::child_by_local_name;
    using detail::get_text;
    using detail::parse_double;

    static void read_fluid_boundary_fields(const XMLElement* source, mhs::model::FluidBoundarySpec& boundary)
    {
        const XMLElement* inlet = child_by_local_name(source, "InletTemperature");
        const XMLElement* pressure = child_by_local_name(source, "Pressure");
        const XMLElement* mass_flow = child_by_local_name(source, "MassFlowRate");
        if (inlet && pressure && !mass_flow) {
            boundary.kind = mhs::model::FluidBoundaryKind::PressureInlet;
            boundary.inlet_temperature = parse_double(get_text(inlet));
            boundary.value = parse_double(get_text(pressure));
        }
        else if (inlet && mass_flow && !pressure) {
            boundary.kind = mhs::model::FluidBoundaryKind::MassFlowInlet;
            boundary.inlet_temperature = parse_double(get_text(inlet));
            boundary.value = parse_double(get_text(mass_flow));
        }
        else if (!inlet && pressure && !mass_flow) {
            boundary.kind = mhs::model::FluidBoundaryKind::Outlet;
            boundary.value = parse_double(get_text(pressure));
        }
        else {
            throw std::runtime_error("fluid boundary fields must be InletTemperature+Pressure, "
                                     "InletTemperature+MassFlowRate, or Pressure only");
        }
    }

    static std::vector<mhs::model::FaceRegion> read_face_key_regions(const XMLElement* source)
    {
        std::vector<mhs::model::FaceRegion> regions;
        const XMLElement* face_keys = child_by_local_name(source, "FaceKeys");
        if (!face_keys)
            return regions;
        for (const XMLElement* face_key = face_keys->FirstChildElement(); face_key;
            face_key = face_key->NextSiblingElement()) {
            std::string_view child_name(face_key->Name());
            const size_t separator = child_name.rfind(':');
            if (separator != std::string_view::npos)
                child_name.remove_prefix(separator + 1);
            if (child_name != "string")
                continue;
            const std::string key = get_text(face_key);
            if (!key.empty())
                regions.push_back(detail::parse_face_region(key));
        }
        return regions;
    }

    // Helpers for the <Functions> block.
    static void read_double_member(const XMLElement* parent, const char* tag, double& target)
    {
        if (const XMLElement* e = child_by_local_name(parent, tag)) {
            target = parse_double(get_text(e));
        }
    }
    static void read_string_member(const XMLElement* parent, const char* tag, std::string& target)
    {
        if (const XMLElement* e = child_by_local_name(parent, tag)) {
            target = get_text(e);
        }
    }

    /// Split a comma-separated list of expressions, trimming whitespace around each token.
    static std::vector<std::string> split_csv(const std::string& raw)
    {
        std::vector<std::string> segs;
        size_t start = 0;
        while (true) {
            size_t end = raw.find(',', start);
            std::string token = (end == std::string::npos) ? raw.substr(start) : raw.substr(start, end - start);
            size_t f = token.find_first_not_of(" \t\r\n");
            size_t l = (f == std::string::npos) ? std::string::npos : token.find_last_not_of(" \t\r\n");
            token = (f == std::string::npos) ? std::string() : token.substr(f, l - f + 1);
            segs.push_back(token);
            if (end == std::string::npos)
                break;
            start = end + 1;
        }
        return segs;
    }

    static mhs::model::StudyType parse_study_type(const std::string& val)
    {
        if (val == "Steady")
            return mhs::model::StudyType::Steady;
        if (val == "Transient")
            return mhs::model::StudyType::Transient;
        throw std::runtime_error("unknown StudyType: '" + val + "' (expected Steady or Transient)");
    }

    static mhs::model::LengthUnit parse_length_unit(const std::string& val)
    {
        if (val == "M")
            return mhs::model::LengthUnit::Meter;
        if (val == "Mm")
            return mhs::model::LengthUnit::Millimeter;
        if (val == "Um")
            return mhs::model::LengthUnit::Micrometer;
        if (val == "Nm")
            return mhs::model::LengthUnit::Nanometer;
        if (val == "Inch")
            return mhs::model::LengthUnit::Inch;
        if (val == "Mil")
            return mhs::model::LengthUnit::Mil;
        throw std::runtime_error("unknown LengthUnit: '" + val + "'");
    }

    mhs::model::ModelDefinition read_xml(const std::string& xml_path)
    {
        XMLDocument doc;
        XMLError err = doc.LoadFile(xml_path.c_str());
        if (err != XML_SUCCESS) {
            throw std::runtime_error("failed to load XML file: " + xml_path);
        }

        mhs::model::ModelDefinition def;
        mhs::model::ThermalBoundary default_boundary = mhs::model::NeumannBoundary {};

        const XMLElement* root = doc.FirstChildElement("Structure");
        if (!root) {
            throw std::runtime_error("no <Structure> element found in " + xml_path);
        }

        // StudyType (child element only)
        if (const XMLElement* study_elem = root->FirstChildElement("StudyType")) {
            def.settings.study_type = parse_study_type(get_text(study_elem));
        }

        // LengthUnit (child element only)
        if (const XMLElement* unit_elem = root->FirstChildElement("LengthUnit")) {
            def.settings.length_unit = parse_length_unit(get_text(unit_elem));
        }

        // Temperature settings
        if (const XMLElement* init = root->FirstChildElement("InitialTemperature")) {
            def.settings.initial_temperature = parse_double(get_text(init));
        }
        else {
            throw std::runtime_error("missing required element <InitialTemperature>");
        }

        // Transient settings
        if (const XMLElement* trans = root->FirstChildElement("TransientStudyDuration")) {
            def.settings.transient_duration = parse_double(get_text(trans));
        }
        else if (def.settings.study_type == mhs::model::StudyType::Transient) {
            throw std::runtime_error("transient study requires <TransientStudyDuration>");
        }
        if (const XMLElement* step = root->FirstChildElement("TransientStudyTimeStep")) {
            def.settings.transient_output_interval = parse_double(get_text(step));
        }
        else if (def.settings.study_type == mhs::model::StudyType::Transient) {
            throw std::runtime_error("transient study requires <TransientStudyTimeStep>");
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
                default_boundary = std::move(bc);
            }
            else if (type_str.find("SecondType") != std::string::npos) {
                mhs::model::NeumannBoundary bc;
                if (const XMLElement* flux = other->FirstChildElement("a:HeatFlux")) {
                    bc.heat_flux = get_text(flux);
                }
                default_boundary = std::move(bc);
            }
            else if (type_str.find("ThirdType") != std::string::npos) {
                mhs::model::ConvectionBoundary bc;
                if (const XMLElement* h = other->FirstChildElement("a:ConvectionCoefficient")) {
                    bc.coefficient = get_text(h);
                }
                if (const XMLElement* t = other->FirstChildElement("a:EnvironmentTemperature")) {
                    bc.ambient_temperature = get_text(t);
                }
                default_boundary = std::move(bc);
            }
        }

        // Variables
        if (const XMLElement* vars = root->FirstChildElement("Variables")) {
            for (const XMLElement* kv = vars->FirstChildElement(); kv; kv = kv->NextSiblingElement()) {
                mhs::model::VariableSpec var;
                if (const XMLElement* key = child_by_local_name(kv, "Key")) {
                    var.name = get_text(key);
                }
                if (const XMLElement* val = child_by_local_name(kv, "Value")) {
                    var.value = get_text(val);
                }
                if (!var.name.empty()) {
                    def.variables.push_back(std::move(var));
                }
            }
        }

        // Materials
        if (const XMLElement* mats = root->FirstChildElement("Materials")) {
            for (const XMLElement* kv = mats->FirstChildElement(); kv; kv = kv->NextSiblingElement()) {
                mhs::model::MaterialSpec mat;
                std::string name;
                if (const XMLElement* key = child_by_local_name(kv, "Key")) {
                    name = get_text(key);
                }
                const XMLElement* val = child_by_local_name(kv, "Value");
                if (val) {
                    enum class MaterialField { Conductivity, Density, SpecificHeat };
                    static constexpr std::array<std::pair<const char*, MaterialField>, 3> material_fields = {{
                        {"ThermalConductivity", MaterialField::Conductivity},
                        {"Density", MaterialField::Density},
                        {"SpecificHeatCapacity", MaterialField::SpecificHeat},
                    }};
                    for (const auto& [tag, field] : material_fields) {
                        const XMLElement* property = val->FirstChildElement(tag);
                        if (!property)
                            continue;
                        const std::string expression = get_text(property);
                        if (field == MaterialField::Conductivity) {
                            auto segs = split_csv(expression);
                            if (segs.size() == 1) {
                                mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = segs[0];
                            }
                            else if (segs.size() == 3) {
                                mat.conductivity_x = segs[0];
                                mat.conductivity_y = segs[1];
                                mat.conductivity_z = segs[2];
                            }
                            else {
                                throw std::runtime_error(std::string(tag)
                                    + " must have 1 or 3 comma-separated expressions, got "
                                    + std::to_string(segs.size()));
                            }
                        }
                        else if (field == MaterialField::Density) {
                            mat.density = expression;
                        }
                        else {
                            mat.specific_heat = expression;
                        }
                    }
                    if (const XMLElement* viscosity = val->FirstChildElement("DynamicViscosity")) {
                        const char* is_nil = viscosity->Attribute("i:nil");
                        if (!(is_nil && std::string_view(is_nil) == "true")) {
                            const std::string expression = get_text(viscosity);
                            if (!expression.empty())
                                mat.dynamic_viscosity = expression;
                        }
                    }
                    if (const XMLElement* fluid = child_by_local_name(val, "FluidMaterial")) {
                        const std::string value = get_text(fluid);
                        if (value != "true" && value != "false")
                            throw std::runtime_error("FluidMaterial must be true or false");
                        mat.is_fluid = value == "true";
                    }
                    else {
                        throw std::runtime_error("material '" + name + "' is missing required <FluidMaterial>");
                    }
                }
                if (!name.empty()) {
                    def.materials.push_back({std::move(name), std::move(mat)});
                }
            }
        }

        // Functions
        if (const XMLElement* funcs = root->FirstChildElement("Functions")) {
            for (const XMLElement* kv = funcs->FirstChildElement(); kv; kv = kv->NextSiblingElement()) {
                std::string name;
                if (const XMLElement* key = child_by_local_name(kv, "Key")) {
                    name = get_text(key);
                }
                const XMLElement* val = child_by_local_name(kv, "Value");
                mhs::model::FunctionSpec fn;
                if (val) {
                    const char* type = val->Attribute("i:type");
                    std::string type_str = type ? type : "";
                    if (type_str.find("ExpressionFunction") != std::string::npos) {
                        mhs::model::ExpressionFunctionSpec expr;
                        read_string_member(val, "Expression", expr.expression);
                        fn = std::move(expr);
                    }
                    else if (type_str.find("DoubleExponentialFunction") != std::string::npos) {
                        mhs::model::DoubleExponentialFunctionSpec de;
                        read_double_member(val, "A", de.amplitude);
                        read_double_member(val, "Alpha", de.alpha);
                        read_double_member(val, "Beta", de.beta);
                        fn = std::move(de);
                    }
                    else if (type_str.find("GaussFunction") != std::string::npos) {
                        mhs::model::GaussFunctionSpec g;
                        read_double_member(val, "A", g.amplitude);
                        read_double_member(val, "Tau", g.tau);
                        read_double_member(val, "X0", g.center);
                        fn = std::move(g);
                    }
                    else if (type_str.find("SineFunction") != std::string::npos) {
                        mhs::model::SineFunctionSpec s;
                        read_double_member(val, "A", s.amplitude);
                        read_double_member(val, "Omega", s.angular_frequency);
                        read_double_member(val, "Phi", s.phase);
                        fn = std::move(s);
                    }
                    else if (type_str.find("PieceWiseFunction") != std::string::npos) {
                        mhs::model::PiecewiseFunctionSpec pw;
                        if (const XMLElement* points = child_by_local_name(val, "Points")) {
                            for (const XMLElement* pt = points->FirstChildElement(); pt;
                                pt = pt->NextSiblingElement()) {
                                mhs::model::PiecewiseFunctionSpec::Point p;
                                read_double_member(pt, "X", p.x);
                                read_double_member(pt, "Y", p.y);
                                pw.points.push_back(p);
                            }
                            std::sort(pw.points.begin(), pw.points.end(),
                                [](const mhs::model::PiecewiseFunctionSpec::Point& a,
                                    const mhs::model::PiecewiseFunctionSpec::Point& b) { return a.x < b.x; });
                        }
                        fn = std::move(pw);
                    }
                    else if (type_str.find("PeriodicPiecewiseConstantFunction") != std::string::npos) {
                        mhs::model::PeriodicPiecewiseConstantFunctionSpec periodic;
                        if (const XMLElement* interval = child_by_local_name(val, "Interval"))
                            periodic.period = parse_double(get_text(interval));
                        if (const XMLElement* values = child_by_local_name(val, "Values")) {
                            for (const XMLElement* item = values->FirstChildElement(); item;
                                item = item->NextSiblingElement()) {
                                if (const XMLElement* value = child_by_local_name(item, "Value"))
                                    periodic.values.push_back(parse_double(get_text(value)));
                            }
                        }
                        fn = std::move(periodic);
                    }
                    else if (!type_str.empty()) {
                        throw std::runtime_error("unknown function i:type: " + type_str);
                    }
                }
                if (!name.empty()) {
                    def.functions.push_back({std::move(name), std::move(fn)});
                }
            }
        }

        // Layers -> Blocks -> Rects
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

                // Blocks within this layer
                if (const XMLElement* blocks_elem = layer_elem->FirstChildElement("Blocks")) {
                    for (const XMLElement* block_elem = blocks_elem->FirstChildElement("Block"); block_elem;
                        block_elem = block_elem->NextSiblingElement("Block")) {
                        mhs::model::BlockSpec block;

                        if (const XMLElement* mat = block_elem->FirstChildElement("MaterialName")) {
                            block.material = get_text(mat);
                        }
                        const XMLElement* ti = block_elem->FirstChildElement("VolumetricHeatSource");
                        if (ti) {
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

                        // Rects within this block
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

                        layer.blocks.push_back(std::move(block));
                    }
                }

                def.layers.push_back(std::move(layer));
            }
        }

        // Boundaries
        if (const XMLElement* bounds_elem = root->FirstChildElement("Boundaries")) {
            for (const XMLElement* bound_elem = bounds_elem->FirstChildElement("Boundary"); bound_elem;
                bound_elem = bound_elem->NextSiblingElement("Boundary")) {
                mhs::model::BoundaryPatch boundary;

                auto regions = read_face_key_regions(bound_elem);

                // ThermalBoundary type
                const XMLElement* thermal = child_by_local_name(bound_elem, "ThermalBoundary");
                if (!thermal)
                    throw std::runtime_error("Boundary is missing <ThermalBoundary i:type>");
                const char* type = thermal->Attribute("i:type");
                const std::string type_str = type ? type : "";
                if (type_str.find("PressureInletBoundary") != std::string::npos
                    || type_str.find("MassFlowInletBoundary") != std::string::npos
                    || type_str.find("OutletBoundary") != std::string::npos) {
                    mhs::model::FluidBoundarySpec fluid_boundary;
                    fluid_boundary.regions = std::move(regions);
                    read_fluid_boundary_fields(thermal, fluid_boundary);
                    const bool pressure_inlet_type = type_str.find("PressureInletBoundary") != std::string::npos;
                    const bool mass_flow_inlet_type = type_str.find("MassFlowInletBoundary") != std::string::npos;
                    const bool outlet_type = type_str.find("OutletBoundary") != std::string::npos;
                    if ((pressure_inlet_type && fluid_boundary.kind != mhs::model::FluidBoundaryKind::PressureInlet)
                        || (mass_flow_inlet_type && fluid_boundary.kind != mhs::model::FluidBoundaryKind::MassFlowInlet)
                        || (outlet_type && fluid_boundary.kind != mhs::model::FluidBoundaryKind::Outlet))
                        throw std::runtime_error(
                            "fluid boundary fields do not match ThermalBoundary i:type: " + type_str);
                    if (fluid_boundary.regions.empty())
                        throw std::runtime_error("fluid boundary has no FaceKeys regions");
                    def.fluid_boundaries.push_back(std::move(fluid_boundary));
                    continue;
                }
                boundary.regions = std::move(regions);
                if (type_str.empty())
                    throw std::runtime_error("Boundary ThermalBoundary is missing i:type");
                {
                    if (type_str.find("FirstType") != std::string::npos) {
                        mhs::model::DirichletBoundary bc;
                        if (const XMLElement* t = child_by_local_name(thermal, "Temperature")) {
                            bc.temperature = get_text(t);
                        }
                        boundary.condition = std::move(bc);
                    }
                    else if (type_str.find("SecondType") != std::string::npos) {
                        mhs::model::NeumannBoundary bc;
                        if (const XMLElement* q = child_by_local_name(thermal, "HeatFlux")) {
                            bc.heat_flux = get_text(q);
                        }
                        boundary.condition = std::move(bc);
                    }
                    else if (type_str.find("ThirdType") != std::string::npos) {
                        mhs::model::ConvectionBoundary bc;
                        if (const XMLElement* h = child_by_local_name(thermal, "ConvectionCoefficient")) {
                            bc.coefficient = get_text(h);
                        }
                        if (const XMLElement* t = child_by_local_name(thermal, "EnvironmentTemperature")) {
                            bc.ambient_temperature = get_text(t);
                        }
                        boundary.condition = std::move(bc);
                    }
                    else {
                        throw std::runtime_error("unknown Boundary ThermalBoundary i:type: " + type_str);
                    }
                }

                def.boundaries.push_back(std::move(boundary));
            }
        }

        // Mesh vertex coordinates from Results -> a:anyType -> Mesh -> b:XArray/YArray/ZArray
        if (const XMLElement* results_elem = root->FirstChildElement("Results")) {
            if (const XMLElement* any_type = results_elem->FirstChildElement("a:anyType")) {
                if (const XMLElement* mesh_elem = any_type->FirstChildElement("Mesh")) {
                    if (const XMLElement* x_array = mesh_elem->FirstChildElement("b:XArray")) {
                        for (const XMLElement* val = x_array->FirstChildElement("a:double"); val;
                            val = val->NextSiblingElement("a:double")) {
                            def.mesh.x_vertices.push_back(parse_double(get_text(val)));
                        }
                    }
                    if (const XMLElement* y_array = mesh_elem->FirstChildElement("b:YArray")) {
                        for (const XMLElement* val = y_array->FirstChildElement("a:double"); val;
                            val = val->NextSiblingElement("a:double")) {
                            def.mesh.y_vertices.push_back(parse_double(get_text(val)));
                        }
                    }
                    if (const XMLElement* z_array = mesh_elem->FirstChildElement("b:ZArray")) {
                        for (const XMLElement* val = z_array->FirstChildElement("a:double"); val;
                            val = val->NextSiblingElement("a:double")) {
                            def.mesh.z_vertices.push_back(parse_double(get_text(val)));
                        }
                    }
                }
            }
        }
        if (def.mesh.x_vertices.empty() || def.mesh.y_vertices.empty() || def.mesh.z_vertices.empty()) {
            if (const XMLElement* generated_mesh = root->FirstChildElement("GeneratedMesh")) {
                const auto read_vertices = [&](const char* axis, std::vector<double>& vertices) {
                    const XMLElement* array = child_by_local_name(generated_mesh, axis);
                    if (!array)
                        return;
                    for (const XMLElement* value = array->FirstChildElement(); value;
                        value = value->NextSiblingElement()) {
                        std::string_view name(value->Name());
                        const size_t separator = name.rfind(':');
                        if (separator != std::string_view::npos)
                            name.remove_prefix(separator + 1);
                        if (name == "double")
                            vertices.push_back(parse_double(get_text(value)));
                    }
                };
                read_vertices("XArray", def.mesh.x_vertices);
                read_vertices("YArray", def.mesh.y_vertices);
                read_vertices("ZArray", def.mesh.z_vertices);
            }
        }

        // ObservePoints3D
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
                def.observation_points.push_back(std::move(op));
            }
        }

        def.default_boundary = std::move(default_boundary);
        return def;
    }

} // namespace mhs::io
