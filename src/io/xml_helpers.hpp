#pragma once

#include <tinyxml2.h>

#include <string>
#include <string_view>

namespace mhs::io::detail {

    inline const tinyxml2::XMLElement* child_by_local_name(const tinyxml2::XMLElement* parent, std::string_view name)
    {
        if (!parent)
            return nullptr;
        for (const tinyxml2::XMLElement* child = parent->FirstChildElement(); child;
            child = child->NextSiblingElement()) {
            std::string_view child_name(child->Name());
            const size_t separator = child_name.rfind(':');
            if (separator != std::string_view::npos)
                child_name.remove_prefix(separator + 1);
            if (child_name == name)
                return child;
        }
        return nullptr;
    }

    inline std::string trim(const std::string& value)
    {
        const size_t first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }
        const size_t last = value.find_last_not_of(" \t\r\n");
        return value.substr(first, last - first + 1);
    }

    inline std::string get_text(const tinyxml2::XMLElement* element)
    {
        if (!element) {
            return "";
        }
        const char* text = element->GetText();
        return text ? trim(text) : "";
    }

    inline double parse_double(const std::string& value) { return value.empty() ? 0.0 : std::stod(value); }

} // namespace mhs::io::detail
