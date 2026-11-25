#include "erom/extraction.hpp"

#include <charconv>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tinyxml2.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace mhs::app::erom {
    namespace {
        using tinyxml2::XMLElement;

        std::string decimal(double value)
        {
            char buffer[64];
            const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general,
                std::numeric_limits<double>::max_digits10);
            if (result.ec != std::errc {})
                throw std::runtime_error("cannot encode EROM numeric value");
            return std::string(buffer, result.ptr);
        }

        XMLElement& append(tinyxml2::XMLNode& parent, const char* name)
        {
            auto* element = parent.GetDocument()->NewElement(name);
            parent.InsertEndChild(element);
            return *element;
        }

        void scalar(XMLElement& element, const char* name, double value)
        {
            element.SetAttribute(name, decimal(value).c_str());
        }

        void count(XMLElement& element, const char* name, std::size_t value)
        {
            element.SetAttribute(name, std::to_string(value).c_str());
        }

        void values(XMLElement& element, std::span<const double> data)
        {
            std::string text;
            for (const double value : data) {
                if (!text.empty())
                    text += ' ';
                text += decimal(value);
            }
            if (!text.empty())
                element.SetText(text.c_str());
        }

        void matrix(XMLElement& parent, const char* name, const core::Matrix<double>& data)
        {
            auto& element = append(parent, name);
            count(element, "rows", data.rows());
            count(element, "cols", data.cols());
            values(element, {data.data(), data.rows() * data.cols()});
        }

        void vector(XMLElement& parent, const char* name, std::span<const double> data)
        {
            auto& element = append(parent, name);
            count(element, "size", data.size());
            values(element, data);
        }

        void facet(XMLElement& element, const model::Facet& geometry)
        {
            const char* axes[] = {"x", "y", "z"};
            element.SetAttribute("axis", axes[static_cast<std::size_t>(geometry.axis)]);
            element.SetAttribute("normal", static_cast<int>(geometry.normal));
            scalar(element, "coordinate", geometry.coordinate);
            scalar(element, "uMin", geometry.bounds[0]);
            scalar(element, "uMax", geometry.bounds[1]);
            scalar(element, "vMin", geometry.bounds[2]);
            scalar(element, "vMax", geometry.bounds[3]);
        }

        struct Temporary {
            std::filesystem::path path;
            ~Temporary()
            {
                std::error_code error;
                std::filesystem::remove(path, error);
            }
        };
    } // namespace

    void write_erom(const std::filesystem::path& path, const model::Erom& e)
    {
        model::validate_erom(e);
        tinyxml2::XMLDocument doc;
        doc.InsertEndChild(doc.NewDeclaration("xml version=\"1.0\" encoding=\"UTF-8\""));
        auto& root = append(doc, "Erom");
        root.SetAttribute("version", "1");
        append(root, "ReferenceTemperature").SetText(decimal(e.reference_temperature).c_str());
        auto& operators = append(root, "Operators");
        matrix(operators, "K", e.K);
        matrix(operators, "C", e.C);
        matrix(operators, "B", e.B);
        matrix(operators, "L", e.L);
        matrix(operators, "Vb", e.Vb);
        vector(operators, "FixedLoad", e.f);
        vector(operators, "OutputOffset", e.output_offset);
        vector(root, "UniformTemperatureMode", e.uniform_temperature_mode);
        auto& groups = append(root, "BoundaryGroups");
        for (std::size_t i = 0; i < e.boundary_groups.size(); ++i) {
            const auto& surface = e.boundary_groups[i];
            auto& element = append(groups, "BoundaryGroup");
            count(element, "id", i);
            element.SetAttribute("name", surface.name.c_str());
            scalar(element, "htcMin", surface.minimum);
            scalar(element, "htcMax", surface.maximum);
        }
        auto& interfaces = append(root, "Interfaces");
        for (const auto& face : e.interfaces) {
            auto& element = append(interfaces, "Interface");
            count(element, "boundary", face.boundary_group);
            scalar(element, "resistance", face.resistance);
            scalar(element, "defaultHtc", face.default_coefficient);
            scalar(element, "ambientTemperature", face.default_ambient);
            facet(element, face.facet);
        }
        auto& outline = append(root, "Outline");
        for (const auto& group : e.outline) {
            auto& element = append(outline, "Group");
            element.SetAttribute("name", group.name.c_str());
            for (const auto& geometry : group.facets)
                facet(append(element, "Facet"), geometry);
        }
        auto& sources = append(root, "Sources");
        for (std::size_t i = 0; i < e.sources.size(); ++i) {
            auto& element = append(sources, "Source");
            count(element, "id", i);
            element.SetAttribute("name", e.sources[i].name.c_str());
        }
        auto& outputs = append(root, "Outputs");
        for (std::size_t i = 0; i < e.outputs.size(); ++i) {
            const auto& output = e.outputs[i];
            auto& element = append(outputs, "Output");
            count(element, "id", i);
            element.SetAttribute("name", output.name.c_str());
            if (output.point) {
                scalar(element, "x", (*output.point)[0]);
                scalar(element, "y", (*output.point)[1]);
                scalar(element, "z", (*output.point)[2]);
            }
        }
        std::random_device random;
        auto temporary = path;
        do {
            temporary = path;
            temporary += ".tmp-" + std::to_string(random()) + "-" + std::to_string(random());
        } while (std::filesystem::exists(temporary));
        const Temporary cleanup {temporary};
        tinyxml2::XMLPrinter printer;
        doc.Print(&printer);
        std::ofstream output(temporary, std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write(printer.CStr(), static_cast<std::streamsize>(printer.CStrSize() - 1));
        output.close();
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::system_error(
                static_cast<int>(GetLastError()), std::system_category(), "replace EROM XML output");
#else
        std::filesystem::rename(temporary, path);
#endif
    }
} // namespace mhs::app::erom
