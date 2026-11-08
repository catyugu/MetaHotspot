#include "core/vector.hpp"
#include "io.hpp"

#include "xml_helpers.hpp"
#include <tinyxml2.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace mhs::app::pta {

    namespace {

        std::string trim(const std::string& str)
        {
            const size_t first = str.find_first_not_of(" \t\r\n");
            if (first == std::string::npos)
                return "";
            const size_t last = str.find_last_not_of(" \t\r\n");
            return str.substr(first, last - first + 1);
        }

        std::string get_text(const tinyxml2::XMLElement* elem)
        {
            if (!elem)
                return "";
            const char* text = elem->GetText();
            return text ? trim(text) : "";
        }

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
        const mhs::core::Vector<mhs::solver::ProbeTrace>& observation_traces, std::span<const double> observation_times)
    {
        using namespace tinyxml2;

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
