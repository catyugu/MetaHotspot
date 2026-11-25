#include "io.hpp"
#include "definition.hpp"
#include "xml.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace mhs::app::pta {
    using namespace tinyxml2;
    using detail::child_by_local_name;
    using detail::get_text;
    using detail::parse_double;

    namespace {
        mhs::core::Vector<std::string> split(const std::string& value, char delimiter)
        {
            mhs::core::Vector<std::string> parts;
            std::string current;
            for (char character : value) {
                if (character == delimiter) {
                    parts.push_back(current);
                    current.clear();
                }
                else {
                    current += character;
                }
            }
            if (!current.empty())
                parts.push_back(current);
            return parts;
        }

        mhs::model::Axis parse_axis(const std::string& value)
        {
            if (value == "X")
                return mhs::model::Axis::X;
            if (value == "Y")
                return mhs::model::Axis::Y;
            if (value == "Z")
                return mhs::model::Axis::Z;
            throw std::invalid_argument("invalid face region axis: " + value);
        }
    } // namespace

    static mhs::core::Vector<mhs::model::SurfaceRegion> parse_face_region(const std::string& encoded)
    {
        const auto parts = split(encoded, '|');
        mhs::model::SurfaceRegion region;
        mhs::core::Vector<mhs::model::SurfaceRegion> regions;
        if (parts.size() < 3)
            throw std::invalid_argument("malformed face region: " + encoded);

        region.axis = parse_axis(parts[0]);
        region.coordinate = std::stod(parts[2]);

        if (region.axis == mhs::model::Axis::Z && parts.size() == 4) {
            for (const auto& encoded_rectangle : split(parts[3], ';')) {
                const auto values = split(encoded_rectangle, ',');
                if (values.size() != 4)
                    throw std::invalid_argument("malformed face region rectangle: " + encoded);
                regions.push_back({region.axis, region.coordinate,
                    {std::stod(values[0]), std::stod(values[1]), std::stod(values[2]), std::stod(values[3])}});
            }
        }
        else if ((region.axis == mhs::model::Axis::X || region.axis == mhs::model::Axis::Y) && parts.size() == 7) {
            regions.push_back({region.axis, region.coordinate,
                {std::stod(parts[3]), std::stod(parts[4]), std::stod(parts[5]), std::stod(parts[6])}});
        }
        else {
            throw std::invalid_argument("malformed face region: " + encoded);
        }

        return regions;
    }

    static void read_fluid_boundary_fields(const XMLElement* source, mhs::model::FluidBoundarySpec& boundary)
    {
        const XMLElement* inlet = child_by_local_name(source, "InletTemperature");
        const XMLElement* pressure = child_by_local_name(source, "Pressure");
        const XMLElement* mass_flow = child_by_local_name(source, "MassFlowRate");
        const XMLElement* pump_resistance = child_by_local_name(source, "PumpResistance");
        if (pump_resistance && (!inlet || !pressure || mass_flow))
            throw std::runtime_error("PumpResistance is only valid for a pressure inlet");
        if (inlet && pressure && !mass_flow) {
            boundary.condition = mhs::model::PressureInlet {parse_double(get_text(pressure)),
                parse_double(get_text(inlet)), pump_resistance ? parse_double(get_text(pump_resistance)) : 0.0};
        }
        else if (inlet && mass_flow && !pressure) {
            boundary.condition
                = mhs::model::MassFlowInlet {parse_double(get_text(mass_flow)), parse_double(get_text(inlet))};
        }
        else if (!inlet && pressure && !mass_flow) {
            boundary.condition = mhs::model::PressureOutlet {parse_double(get_text(pressure))};
        }
        else {
            throw std::runtime_error("fluid boundary fields must be InletTemperature+Pressure, "
                                     "InletTemperature+MassFlowRate, or Pressure only");
        }
    }

    static mhs::core::Vector<mhs::model::SurfaceRegion> read_face_key_regions(const XMLElement* source)
    {
        mhs::core::Vector<mhs::model::SurfaceRegion> regions;
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
                for (const auto& region : parse_face_region(key))
                    regions.push_back(region);
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
    static mhs::core::Vector<std::string> split_csv(const std::string& raw)
    {
        mhs::core::Vector<std::string> segs;
        size_t start = 0;
        while (true) {
            size_t end = raw.find(',', start);
            segs.push_back(detail::trim(raw.substr(start, end == std::string::npos ? end : end - start)));
            if (end == std::string::npos)
                break;
            start = end + 1;
        }
        return segs;
    }

    static LengthUnit parse_length_unit(const std::string& val)
    {
        if (val == "M")
            return LengthUnit::Meter;
        if (val == "Mm")
            return LengthUnit::Millimeter;
        if (val == "Um")
            return LengthUnit::Micrometer;
        if (val == "Nm")
            return LengthUnit::Nanometer;
        if (val == "Inch")
            return LengthUnit::Inch;
        if (val == "Mil")
            return LengthUnit::Mil;
        throw std::runtime_error("unknown LengthUnit: '" + val + "'");
    }

    ModelInput read_model(const tinyxml2::XMLElement& structure, bool extraction)
    {
        Definition def;
        core::Vector<model::BoundaryGroupId> boundary_ids;
        mhs::model::ThermalBoundary default_thermal_boundary = mhs::model::NeumannBoundary {};
        const auto* root = &structure;

        // LengthUnit (child element only)
        if (const XMLElement* unit_elem = root->FirstChildElement("LengthUnit")) {
            def.length_unit = parse_length_unit(get_text(unit_elem));
        }

        // Temperature settings
        if (const XMLElement* init = root->FirstChildElement("InitialTemperature")) {
            def.model.reference_temperature = parse_double(get_text(init));
        }
        else {
            throw std::runtime_error("missing required element <InitialTemperature>");
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
                default_thermal_boundary = std::move(bc);
            }
            else if (type_str.find("SecondType") != std::string::npos) {
                mhs::model::NeumannBoundary bc;
                if (const XMLElement* flux = other->FirstChildElement("a:HeatFlux")) {
                    bc.heat_flux = get_text(flux);
                }
                default_thermal_boundary = std::move(bc);
            }
            else if (type_str.find("ThirdType") != std::string::npos) {
                mhs::model::ConvectionBoundary bc;
                if (const XMLElement* h = other->FirstChildElement("a:ConvectionCoefficient")) {
                    bc.coefficient = get_text(h);
                }
                if (const XMLElement* t = other->FirstChildElement("a:EnvironmentTemperature")) {
                    bc.ambient_temperature = get_text(t);
                }
                default_thermal_boundary = std::move(bc);
            }
        }

        // Variables
        if (const XMLElement* vars = root->FirstChildElement("Variables")) {
            for (const XMLElement* kv = vars->FirstChildElement(); kv; kv = kv->NextSiblingElement()) {
                VariableSpec var;
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
                        mat.kind = value == "true" ? mhs::model::MaterialKind::Fluid : mhs::model::MaterialKind::Solid;
                        if (!extraction && mat.kind == mhs::model::MaterialKind::Fluid && !mat.dynamic_viscosity)
                            throw std::runtime_error("fluid material '" + name + "' is missing DynamicViscosity");
                        if (mat.kind == mhs::model::MaterialKind::Solid)
                            mat.dynamic_viscosity.reset();
                    }
                    else {
                        throw std::runtime_error("material '" + name + "' is missing required <FluidMaterial>");
                    }
                }
                if (!name.empty()) {
                    def.model.materials.push_back({std::move(name), std::move(mat)});
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
                                p.x = parse_double(get_text(child_by_local_name(pt, "X")));
                                p.y = parse_double(get_text(child_by_local_name(pt, "Y")));
                                pw.points.push_back(p);
                            }
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
                    def.model.functions.push_back({std::move(name), std::move(fn)});
                }
            }
        }

        // The extraction contract uses explicit identities in PTA declaration
        // order. PowerDensity is deliberately irrelevant to a unit-power port.
        if (extraction) {
            if (const auto* sources = root->FirstChildElement("HeatSources"))
                for (const auto* element = sources->FirstChildElement("HeatSource"); element;
                    element = element->NextSiblingElement("HeatSource")) {
                    HeatSourceSpec source {
                        get_text(child_by_local_name(element, "Id")), get_text(child_by_local_name(element, "Name"))};
                    if (source.id.empty() || std::ranges::any_of(def.heat_sources, [&](const auto& other) {
                            return other.id == source.id;
                        }))
                        throw std::invalid_argument("missing or duplicate HeatSource Id");
                    def.heat_sources.push_back(std::move(source));
                }
        }

        // Layers -> Blocks -> Rects
        if (const XMLElement* layers_elem = root->FirstChildElement("Layers")) {
            for (const XMLElement* layer_elem = layers_elem->FirstChildElement("Layer"); layer_elem;
                layer_elem = layer_elem->NextSiblingElement("Layer")) {
                LayerSpec layer;

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
                        BlockSpec block;
                        if (const XMLElement* name = block_elem->FirstChildElement("Name"))
                            block.name = get_text(name);

                        if (const XMLElement* mat = block_elem->FirstChildElement("MaterialName")) {
                            block.material = get_text(mat);
                        }
                        if (extraction)
                            block.heat_source_id = get_text(child_by_local_name(block_elem, "HeatSourceId"));
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
                                // PTA array placeholders are not physical rectangles.
                                const XMLElement* disguise = rect_elem->FirstChildElement("ArrayDisguise");
                                if (disguise && get_text(disguise) == "true")
                                    continue;
                                RectOperation rect;
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
                mhs::model::ThermalBoundarySpec boundary;
                if (const XMLElement* name = child_by_local_name(bound_elem, "Name"))
                    boundary.name = get_text(name);

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
                    if ((pressure_inlet_type
                            && !std::holds_alternative<mhs::model::PressureInlet>(fluid_boundary.condition))
                        || (mass_flow_inlet_type
                            && !std::holds_alternative<mhs::model::MassFlowInlet>(fluid_boundary.condition))
                        || (outlet_type
                            && !std::holds_alternative<mhs::model::PressureOutlet>(fluid_boundary.condition)))
                        throw std::runtime_error(
                            "fluid boundary fields do not match ThermalBoundary i:type: " + type_str);
                    if (fluid_boundary.regions.empty())
                        throw std::runtime_error("fluid boundary has no FaceKeys regions");
                    boundary_ids.push_back(core::invalid_id);
                    def.model.fluid_boundaries.push_back(std::move(fluid_boundary));
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

                boundary_ids.push_back(static_cast<model::BoundaryGroupId>(def.model.thermal_boundaries.size()));
                def.model.thermal_boundaries.push_back(std::move(boundary));
            }
        }

        // A mesh is read as a single, complete source. Extraction prefers the
        // current GeneratedMesh; legacy studies retain their Results preference.
        auto read_mesh = [&](const XMLElement* element) {
            model::MeshSpec mesh;
            const auto read_vertices = [&](const char* axis, core::Vector<double>& vertices) {
                if (const auto* array = child_by_local_name(element, axis))
                    for (const auto* value = array->FirstChildElement(); value; value = value->NextSiblingElement()) {
                        std::string_view name(value->Name());
                        const auto separator = name.rfind(':');
                        if (separator != std::string_view::npos)
                            name.remove_prefix(separator + 1);
                        if (name == "double")
                            vertices.push_back(parse_double(get_text(value)));
                    }
            };
            read_vertices("XArray", mesh.x_vertices);
            read_vertices("YArray", mesh.y_vertices);
            read_vertices("ZArray", mesh.z_vertices);
            return mesh;
        };
        const auto* generated = root->FirstChildElement("GeneratedMesh");
        if (generated && generated->Attribute("i:nil", "true"))
            generated = nullptr;
        if (extraction && generated)
            def.mesh = read_mesh(generated);
        else {
            const auto* result = child_by_local_name(root->FirstChildElement("Results"), "anyType");
            def.mesh = read_mesh(child_by_local_name(result, "Mesh"));
            if (generated
                && (def.mesh.x_vertices.empty() || def.mesh.y_vertices.empty() || def.mesh.z_vertices.empty()))
                def.mesh = read_mesh(generated);
        }

        // ObservePoints3D
        if (const XMLElement* obs3d = root->FirstChildElement("ObservePoints3D")) {
            for (const XMLElement* pt = obs3d->FirstChildElement("ObservePoint3D"); pt;
                pt = pt->NextSiblingElement("ObservePoint3D")) {
                PointProbeSpec op;
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
                def.point_probes.push_back(std::move(op));
            }
        }

        def.model.default_thermal_boundary = std::move(default_thermal_boundary);
        normalize(def, extraction);
        return {std::move(def.model), std::move(def.mesh), std::move(boundary_ids)};
    }

    static mhs::solver::StudyType parse_study_type(const std::string& val)
    {
        if (val == "Steady")
            return mhs::solver::StudyType::Steady;
        if (val == "Transient")
            return mhs::solver::StudyType::Transient;
        throw std::runtime_error("unknown StudyType: '" + val + "' (expected Steady or Transient)");
    }

    Input read_xml(const std::string& xml_path)
    {
        XMLDocument doc;
        if (doc.LoadFile(xml_path.c_str()) != XML_SUCCESS)
            throw std::runtime_error("failed to load XML file: " + xml_path);
        const XMLElement* root = doc.FirstChildElement("Structure");
        if (!root)
            throw std::runtime_error("no <Structure> element found in " + xml_path);

        mhs::solver::Study study;
        // StudyType (child element only)
        if (const XMLElement* study_elem = root->FirstChildElement("StudyType")) {
            study.type = parse_study_type(get_text(study_elem));
        }

        // Transient settings
        if (const XMLElement* trans = root->FirstChildElement("TransientStudyDuration")) {
            study.duration = parse_double(get_text(trans));
        }
        else if (study.type == mhs::solver::StudyType::Transient) {
            throw std::runtime_error("transient study requires <TransientStudyDuration>");
        }
        if (const XMLElement* step = root->FirstChildElement("TransientStudyTimeStep")) {
            study.output_interval = parse_double(get_text(step));
        }
        else if (study.type == mhs::solver::StudyType::Transient) {
            throw std::runtime_error("transient study requires <TransientStudyTimeStep>");
        }

        auto input = read_model(*root);
        study.initial_temperature = input.model.reference_temperature;
        return {study, std::move(input.model), std::move(input.mesh)};
    }

    namespace {

        void refill_double_list(
            tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* parent, std::span<const double> data, bool allow_nan)
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
                    std::snprintf(buf, sizeof(buf), "%.6f", v);
                    d->SetText(buf);
                }
                parent->InsertEndChild(d);
            }
        }

    } // namespace

    void write_xml(const std::string& input_path, const std::string& output_path,
        const std::array<std::size_t, 3>& shape, std::span<const double> node_temperature,
        const mhs::core::Vector<mhs::solver::OutputTrace>& observation_traces,
        std::span<const double> observation_times)
    {
        if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0)
            throw std::invalid_argument("XML output requires a Cartesian sampling grid; use VTU for imported meshes");
        if (node_temperature.size() != shape[0] * shape[1] * shape[2])
            throw std::invalid_argument("XML node values do not match sampling-grid shape");

        XMLDocument doc;
        if (doc.LoadFile(input_path.c_str()) != XML_SUCCESS)
            throw std::runtime_error("failed to load XML output template: " + input_path);
        auto* root = doc.FirstChildElement("Structure");
        if (!root)
            throw std::runtime_error("XML output template has no Structure");
        auto ensure = [&](XMLElement* parent, const char* name) {
            auto* child = parent->FirstChildElement(name);
            if (!child) {
                child = doc.NewElement(name);
                parent->InsertEndChild(child);
            }
            return child;
        };
        auto* results_elem = ensure(root, "Results");
        results_elem->SetAttribute("xmlns:a", "http://schemas.microsoft.com/2003/10/Serialization/Arrays");
        auto* any_type = ensure(results_elem, "a:anyType");
        any_type->SetAttribute("i:type", "Result3D");
        any_type->SetAttribute("xmlns:i", "http://www.w3.org/2001/XMLSchema-instance");
        if (!any_type->FirstChildElement("Mesh")) {
            const auto* generated = root->FirstChildElement("GeneratedMesh");
            if (!generated)
                throw std::runtime_error("XML output template has no sampling mesh");
            auto* mesh = ensure(any_type, "Mesh");
            mesh->SetAttribute("xmlns:b", "http://schemas.datacontract.org/2004/07/ThermalSim.Models.Mesh");
            for (const auto* axis : {"XArray", "YArray", "ZArray"}) {
                auto* values = detail::child_by_local_name(generated, axis);
                if (!values)
                    throw std::runtime_error("GeneratedMesh is missing an axis");
                auto* array = ensure(mesh, (std::string("b:") + axis).c_str());
                for (auto* v = values->FirstChildElement(); v; v = v->NextSiblingElement()) {
                    auto* value = doc.NewElement("a:double");
                    value->SetText(v->GetText());
                    array->InsertEndChild(value);
                }
            }
            ensure(any_type, "PhysicsName")->SetText("温度");
            ensure(any_type, "UnitName")->SetText("K");
        }
        auto* values_elem = ensure(any_type, "Values");
        auto* data_elem = ensure(values_elem, "Data");
        for (const auto* name : {"SizeX", "SizeY", "SizeZ"})
            ensure(values_elem, name);

        refill_double_list(doc, data_elem, node_temperature, true);
        const auto [node_nx, node_ny, node_nz] = shape;

        XMLElement* sx = values_elem->FirstChildElement("SizeX");
        if (sx)
            sx->SetText(static_cast<std::uint64_t>(node_nx));
        XMLElement* sy = values_elem->FirstChildElement("SizeY");
        if (sy)
            sy->SetText(static_cast<std::uint64_t>(node_ny));
        XMLElement* sz = values_elem->FirstChildElement("SizeZ");
        if (sz)
            sz->SetText(static_cast<std::uint64_t>(node_nz));

        for (const auto& trace : observation_traces) {
            XMLElement* target = nullptr;
            for (XMLElement* candidate = results_elem->FirstChildElement("a:anyType"); candidate;
                candidate = candidate->NextSiblingElement("a:anyType")) {
                const char* type = candidate->Attribute("i:type");
                if (!type || std::string(type).find("Result0DTransient") == std::string::npos)
                    continue;
                const XMLElement* point_name = candidate->FirstChildElement("PointName");
                if (point_name && get_text(point_name) == trace.name) {
                    target = candidate;
                    break;
                }
            }

            if (!target) {
                target = doc.NewElement("a:anyType");
                target->SetAttribute("i:type", "Result0DTransient");

                // The downstream front-end reader matches this label verbatim,
                // so it is a fixed value of the output contract, not a
                // localizable string. Do not translate it.
                XMLElement* physics = doc.NewElement("PhysicsName");
                physics->SetText("温度");
                target->InsertEndChild(physics);
                XMLElement* point_name = doc.NewElement("PointName");
                point_name->SetText(trace.name.c_str());
                target->InsertEndChild(point_name);
                XMLElement* time_unit = doc.NewElement("TimeUnit");
                time_unit->SetText("S");
                target->InsertEndChild(time_unit);
                target->InsertEndChild(doc.NewElement("Times"));
                XMLElement* unit_name = doc.NewElement("UnitName");
                unit_name->SetText("K");
                target->InsertEndChild(unit_name);
                target->InsertEndChild(doc.NewElement("Values"));
                results_elem->InsertEndChild(target);
            }

            refill_double_list(doc, target->FirstChildElement("Times"), observation_times, false);
            refill_double_list(doc, target->FirstChildElement("Values"), trace.values, true);
        }

        const std::filesystem::path dir_path(output_path);
        if (!dir_path.parent_path().empty()) {
            std::filesystem::create_directories(dir_path.parent_path());
        }
        if (doc.SaveFile(output_path.c_str()) != XML_SUCCESS)
            throw std::runtime_error("failed to write XML: " + output_path);
    }
} // namespace mhs::app::pta
