#include "io/model_io.hpp"

#include "io/face_region_parser.hpp"
#include "io/xml_helpers.hpp"

#include <tinyxml2.h>

#include <algorithm>
#include <string>

namespace mhs::io {

    bool merge_fluid_xml(const std::string& xml_path, mhs::model::ModelDefinition& definition)
    {
        tinyxml2::XMLDocument doc;
        const tinyxml2::XMLError error = doc.LoadFile(xml_path.c_str());
        if (error != tinyxml2::XML_SUCCESS) {
            return false;
        }

        const tinyxml2::XMLElement* root = doc.FirstChildElement("FluidOverlay");
        if (!root) {
            return false;
        }

        for (const tinyxml2::XMLElement* material_element = root->FirstChildElement("FluidMaterial");
            material_element; material_element = material_element->NextSiblingElement("FluidMaterial")) {
            std::string name;
            if (const char* attribute = material_element->Attribute("name")) {
                name = attribute;
            }
            std::string dynamic_viscosity;
            if (const tinyxml2::XMLElement* viscosity = material_element->FirstChildElement("DynamicViscosity")) {
                dynamic_viscosity = detail::get_text(viscosity);
            }
            const auto material = std::find_if(definition.materials.begin(), definition.materials.end(),
                [&](const mhs::model::NamedMaterial& item) { return item.name == name; });
            if (material != definition.materials.end()) {
                material->value.dynamic_viscosity = std::move(dynamic_viscosity);
            }
        }

        for (const tinyxml2::XMLElement* boundary_element = root->FirstChildElement("Boundary"); boundary_element;
            boundary_element = boundary_element->NextSiblingElement("Boundary")) {
            mhs::model::FluidBoundarySpec boundary;

            if (const tinyxml2::XMLElement* face_keys = boundary_element->FirstChildElement("FaceKeys")) {
                for (const tinyxml2::XMLElement* face_key = face_keys->FirstChildElement("string"); face_key;
                    face_key = face_key->NextSiblingElement("string")) {
                    const std::string key = detail::get_text(face_key);
                    if (!key.empty()) {
                        boundary.regions.push_back(detail::parse_face_region(key));
                    }
                }
            }

            if (const tinyxml2::XMLElement* pressure = boundary_element->FirstChildElement("Pressure")) {
                boundary.value = detail::parse_double(detail::get_text(pressure));
                boundary.kind = mhs::model::FluidBoundaryKind::Pressure;
            }
            else if (const tinyxml2::XMLElement* mass_flow = boundary_element->FirstChildElement("MassFlowRate")) {
                boundary.value = detail::parse_double(detail::get_text(mass_flow));
                boundary.kind = mhs::model::FluidBoundaryKind::MassFlowRate;
            }
            else if (const tinyxml2::XMLElement* velocity = boundary_element->FirstChildElement("Velocity")) {
                boundary.value = detail::parse_double(detail::get_text(velocity));
                boundary.kind = mhs::model::FluidBoundaryKind::Velocity;
            }

            if (const tinyxml2::XMLElement* inlet_temperature
                = boundary_element->FirstChildElement("InletTemperature")) {
                boundary.inlet_temperature = detail::parse_double(detail::get_text(inlet_temperature));
            }

            if (boundary.kind != mhs::model::FluidBoundaryKind::None && !boundary.regions.empty()) {
                definition.fluid_boundaries.push_back(std::move(boundary));
            }
        }

        return true;
    }

} // namespace mhs::io
