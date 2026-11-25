#include "io/erom.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tinyxml2.h>
#include <type_traits>

namespace mhs::io {
    using namespace model;

    namespace {
        using tinyxml2::XMLElement;

        void require(bool valid, const char* message)
        {
            if (!valid)
                throw std::invalid_argument(std::string("EROM XML: ") + message);
        }

        const XMLElement& child(const XMLElement& parent, const char* name)
        {
            const auto* element = parent.FirstChildElement(name);
            require(element && !element->NextSiblingElement(name), "missing or duplicate required element");
            return *element;
        }

        std::string_view attribute(const XMLElement& element, const char* name)
        {
            const char* value = element.Attribute(name);
            require(value != nullptr, "missing required attribute");
            return value;
        }

        std::string_view trim(std::string_view text)
        {
            const auto begin = text.find_first_not_of(" \t\r\n");
            if (begin == std::string_view::npos)
                return {};
            return text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
        }

        std::string_view contents(const XMLElement& element)
        {
            require(element.FirstChildElement() == nullptr, "unexpected nested element in numeric field");
            const char* text = element.GetText();
            return text ? std::string_view(text) : std::string_view {};
        }

        template <typename T> T number(std::string_view text)
        {
            text = trim(text);
            if (!text.empty() && text.front() == '+')
                text.remove_prefix(1);
            require(!text.empty(), "missing numeric value");
            T value {};
            if constexpr (std::is_floating_point_v<T>) {
                // Floating from_chars is unavailable on older macOS deployment targets.
                std::istringstream input {std::string(text)};
                input.imbue(std::locale::classic());
                require(static_cast<bool>(input >> value) && input.eof(), "invalid numeric value");
                require(std::isfinite(value), "nonfinite numeric value");
            }
            else {
                const auto end = text.data() + text.size();
                const auto parsed = std::from_chars(text.data(), end, value);
                require(parsed.ec == std::errc {} && parsed.ptr == end, "invalid numeric value");
            }
            return value;
        }

        std::size_t count(const XMLElement& element, const char* name)
        {
            return number<std::size_t>(attribute(element, name));
        }

        double scalar(const XMLElement& element, const char* name) { return number<double>(attribute(element, name)); }

        void values(std::string_view text, std::span<double> destination)
        {
            std::istringstream input {std::string(text)};
            input.imbue(std::locale::classic());
            for (auto& value : destination) {
                require(static_cast<bool>(input >> value) && std::isfinite(value), "invalid numeric entry");
                const auto next = input.peek();
                require(next == std::char_traits<char>::eof() || next == ' ' || next == '\t' || next == '\r'
                        || next == '\n',
                    "numeric entries must be whitespace separated");
            }
            input >> std::ws;
            require(input.eof(), "unexpected trailing numeric entries");
        }

        core::Matrix<double> matrix(const XMLElement& element)
        {
            const auto rows = count(element, "rows"), cols = count(element, "cols");
            const auto text = contents(element);
            require(cols == 0 || rows <= text.size() / cols, "matrix dimensions exceed numeric data");
            core::Matrix<double> result(rows, cols);
            values(text, {result.data(), rows * cols});
            return result;
        }

        core::Vector<double> vector(const XMLElement& element)
        {
            const auto size = count(element, "size");
            const auto text = contents(element);
            require(size <= text.size(), "vector dimension exceeds numeric data");
            core::Vector<double> result(size);
            values(text, result);
            return result;
        }

        Facet facet(const XMLElement& element)
        {
            const auto axis = attribute(element, "axis");
            require(axis == "x" || axis == "y" || axis == "z", "invalid facet axis");
            const int normal = number<int>(attribute(element, "normal"));
            require(normal == -1 || normal == 1, "invalid facet normal");
            return {axis == "x"   ? Axis::X
                    : axis == "y" ? Axis::Y
                                  : Axis::Z,
                static_cast<std::int8_t>(normal), scalar(element, "coordinate"),
                {scalar(element, "uMin"), scalar(element, "uMax"), scalar(element, "vMin"), scalar(element, "vMax")}};
        }

        template <typename F> void records(const XMLElement& parent, const char* name, F read)
        {
            for (const auto* element = parent.FirstChildElement(); element; element = element->NextSiblingElement()) {
                require(std::string_view(element->Name()) == name, "unexpected record element");
                read(*element);
            }
        }
    } // namespace

    std::shared_ptr<const model::Erom> read_erom(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("cannot open EROM XML file: " + path.string());
        const std::string xml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        tinyxml2::XMLDocument doc;
        require(doc.Parse(xml.data(), xml.size()) == tinyxml2::XML_SUCCESS, "invalid XML document");
        const auto* root = doc.RootElement();
        require(
            root && std::string_view(root->Name()) == "Erom" && !root->NextSiblingElement(), "invalid root element");
        require(attribute(*root, "version") == "1", "unsupported version");
        auto e = std::make_shared<Erom>();
        e->reference_temperature = number<double>(contents(child(*root, "ReferenceTemperature")));
        const auto& operators = child(*root, "Operators");
        e->K = matrix(child(operators, "K"));
        e->C = matrix(child(operators, "C"));
        e->B = matrix(child(operators, "B"));
        e->L = matrix(child(operators, "L"));
        e->Vb = matrix(child(operators, "Vb"));
        e->f = vector(child(operators, "FixedLoad"));
        e->output_offset = vector(child(operators, "OutputOffset"));
        e->uniform_temperature_mode = vector(child(*root, "UniformTemperatureMode"));

        records(child(*root, "BoundaryGroups"), "BoundaryGroup", [&](const auto& element) {
            require(count(element, "id") == e->boundary_groups.size(), "invalid boundary group id");
            e->boundary_groups.push_back(
                {std::string(attribute(element, "name")), scalar(element, "htcMin"), scalar(element, "htcMax")});
        });
        records(child(*root, "Interfaces"), "Interface", [&](const auto& element) {
            const auto group = count(element, "boundary");
            require(group < e->boundary_groups.size() && group < core::invalid_id, "invalid interface surface");
            e->interfaces.push_back({facet(element), scalar(element, "resistance"), static_cast<BoundaryGroupId>(group),
                scalar(element, "defaultHtc"), scalar(element, "ambientTemperature")});
        });
        records(child(*root, "Outline"), "Group", [&](const auto& element) {
            EromOutline group {std::string(attribute(element, "name")), {}};
            records(element, "Facet", [&](const auto& geometry) { group.facets.push_back(facet(geometry)); });
            e->outline.push_back(std::move(group));
        });
        records(child(*root, "Sources"), "Source", [&](const auto& element) {
            require(count(element, "id") == e->sources.size(), "invalid source id");
            e->sources.push_back({std::string(attribute(element, "name"))});
        });
        records(child(*root, "Outputs"), "Output", [&](const auto& element) {
            require(count(element, "id") == e->outputs.size(), "invalid output id");
            EromOutput output {std::string(attribute(element, "name")), std::nullopt};
            if (element.Attribute("x") || element.Attribute("y") || element.Attribute("z"))
                output.point = Vec3 {scalar(element, "x"), scalar(element, "y"), scalar(element, "z")};
            e->outputs.push_back(std::move(output));
        });
        validate_erom(*e);
        return e;
    }
} // namespace mhs::io
