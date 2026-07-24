#include "io/result_io.hpp"
#include <tinyxml2.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

namespace mhs::io {

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

    void write_xml(const std::string& input_path, const std::string& output_path, const mhs::core::Model& model,
        std::span<const double> node_temperature, const std::vector<mhs::core::ProbeTrace>& observation_traces)
    {
        using namespace tinyxml2;

        XMLDocument doc;
        const XMLError err = doc.LoadFile(input_path.c_str());
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

        while (XMLElement* child = data_elem->FirstChildElement("a:double")) {
            data_elem->DeleteChild(child);
        }

        const mhs::core::Index node_nx = model.mesh.nx + 1;
        const mhs::core::Index node_ny = model.mesh.ny + 1;
        const mhs::core::Index node_nz = model.mesh.nz + 1;

        for (mhs::core::Index vx = 0; vx < node_nx; vx++) {
            for (mhs::core::Index vy = 0; vy < node_ny; vy++) {
                for (mhs::core::Index vz = 0; vz < node_nz; vz++) {
                    const double val = node_temperature[vx * node_ny * node_nz + vy * node_nz + vz];

                    XMLElement* double_elem = doc.NewElement("a:double");
                    if (std::isnan(val)) {
                        double_elem->SetText("NaN");
                    }
                    else {
                        char buf[64];
                        std::snprintf(buf, sizeof(buf), "%.6f", val);
                        double_elem->SetText(buf);
                    }
                    data_elem->InsertEndChild(double_elem);
                }
            }
        }

        XMLElement* sx = values_elem->FirstChildElement("SizeX");
        if (sx)
            sx->SetText(node_nx);
        XMLElement* sy = values_elem->FirstChildElement("SizeY");
        if (sy)
            sy->SetText(node_ny);
        XMLElement* sz = values_elem->FirstChildElement("SizeZ");
        if (sz)
            sz->SetText(node_nz);

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

            refill_double_list(doc, target->FirstChildElement("Times"), trace.times, false);
            refill_double_list(doc, target->FirstChildElement("Values"), trace.values, true);
        }

        const std::filesystem::path dir_path(output_path);
        if (!dir_path.parent_path().empty() && !std::filesystem::exists(dir_path.parent_path())) {
            std::filesystem::create_directories(dir_path.parent_path());
        }
        doc.SaveFile(output_path.c_str());
    }

} // namespace mhs::io
