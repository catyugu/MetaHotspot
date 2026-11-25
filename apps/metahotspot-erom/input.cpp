#include "input.hpp"
#include "pta/model_input.hpp"
#include "pta/xml_helpers.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mhs::app::erom {
    namespace {
        using tinyxml2::XMLElement;
        using pta::detail::get_text;

        const XMLElement* unique_child(const XMLElement& parent, const char* name, bool required = true)
        {
            const auto* child = parent.FirstChildElement(name);
            if ((!child && required) || (child && child->NextSiblingElement(name)))
                throw std::invalid_argument(std::string("missing or duplicate ") + name);
            return child;
        }

        template <typename T> T number(std::string_view text, const char* name)
        {
            if (!text.empty() && text.front() == '+')
                text.remove_prefix(1);
            T value {};
            if (text.empty())
                throw std::invalid_argument(std::string("missing ") + name);
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc {} || parsed.ptr != text.data() + text.size())
                throw std::invalid_argument(std::string("invalid ") + name);
            return value;
        }
    } // namespace

    Input read_input(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("cannot open input XML: " + path.string());
        const std::string xml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        tinyxml2::XMLDocument doc;
        if (doc.Parse(xml.data(), xml.size()) != tinyxml2::XML_SUCCESS)
            throw std::invalid_argument("invalid PTA XML: " + std::string(doc.ErrorStr()));
        const auto* root = doc.RootElement();
        if (!root || std::string_view(root->Name()) != "Structure" || root->NextSiblingElement())
            throw std::invalid_argument("expected PTA Structure root");
        const auto& config = *unique_child(*root, "EromExtraction");
        if (!config.Attribute("version", "1"))
            throw std::invalid_argument("unsupported EromExtraction version");

        Input result;
        if (const auto* tolerance = unique_child(config, "Tolerance", false))
            result.tolerance = number<double>(get_text(tolerance), "Tolerance");
        if (!std::isfinite(result.tolerance) || result.tolerance <= 0 || result.tolerance >= 1)
            throw std::invalid_argument("Tolerance must be in (0, 1)");

        auto physical = pta::read_model(*root, true);
        const auto& groups = *unique_child(config, "BoundaryGroups");
        for (const auto* group = groups.FirstChildElement(); group; group = group->NextSiblingElement()) {
            if (std::string_view(group->Name()) != "BoundaryGroup")
                throw std::invalid_argument("unexpected element in EromExtraction/BoundaryGroups");
            const auto* raw_index = group->Attribute("index");
            if (!raw_index)
                throw std::invalid_argument("BoundaryGroup requires index");
            const auto index = number<std::size_t>(raw_index, "BoundaryGroup index");
            if (index >= physical.boundary_ids.size())
                throw std::invalid_argument("unknown PTA BoundaryGroup index " + std::to_string(index));
            const auto id = physical.boundary_ids[index];
            if (id == core::invalid_id
                || !std::holds_alternative<model::ConvectionBoundary>(physical.model.thermal_boundaries[id].condition))
                throw std::invalid_argument("BoundaryGroup " + std::to_string(index) + " is not a convection boundary");
            const HtcRange range {number<double>(get_text(unique_child(*group, "HtcMin")), "HtcMin"),
                number<double>(get_text(unique_child(*group, "HtcMax")), "HtcMax")};
            if (!std::isfinite(range.minimum) || !std::isfinite(range.maximum) || range.minimum <= 0
                || range.maximum < range.minimum)
                throw std::invalid_argument("invalid HTC range for BoundaryGroup " + std::to_string(index));
            if (!result.ranges.emplace(id, range).second)
                throw std::invalid_argument("duplicate BoundaryGroup index " + std::to_string(index));
        }
        if (result.ranges.empty())
            throw std::invalid_argument("EromExtraction requires at least one BoundaryGroup");

        result.model = std::move(physical.model);
        result.mesh = std::move(physical.mesh);
        return result;
    }
} // namespace mhs::app::erom
