#include "model/erom.hpp"

#include <Eigen/Cholesky>
#include <tinyxml2.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace mhs::model {
    namespace {
        using tinyxml2::XMLElement;

        void require(bool valid, const char* message)
        {
            if (!valid)
                throw std::invalid_argument(std::string("EROM XML: ") + message);
        }

        bool finite(std::span<const double> values)
        {
            return std::ranges::all_of(values, [](double x) { return std::isfinite(x); });
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
            const auto end = text.data() + text.size();
            const auto parsed = std::from_chars(text.data(), end, value);
            require(parsed.ec == std::errc {} && parsed.ptr == end, "invalid numeric value");
            if constexpr (std::is_floating_point_v<T>)
                require(std::isfinite(value), "nonfinite numeric value");
            return value;
        }

        std::size_t count(const XMLElement& element, const char* name)
        {
            return number<std::size_t>(attribute(element, name));
        }

        double scalar(const XMLElement& element, const char* name)
        {
            return number<double>(attribute(element, name));
        }

        void values(std::string_view text, std::span<double> destination)
        {
            for (auto& value : destination) {
                text = trim(text);
                const auto stop = text.find_first_of(" \t\r\n");
                value = number<double>(text.substr(0, stop));
                text = stop == std::string_view::npos ? std::string_view {} : text.substr(stop);
            }
            require(trim(text).empty(), "unexpected trailing numeric entries");
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
            return {axis == "x" ? Axis::X : axis == "y" ? Axis::Y : Axis::Z,
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

    void validate_erom(const Erom& e)
    {
        const auto n = e.K.rows(), nf = e.Vb.rows();
        require(n > 0 && e.K.cols() == n && e.C.rows() == n && e.C.cols() == n, "invalid K/C dimensions");
        require(e.B.rows() == n && e.B.cols() == e.sources.size() && e.L.rows() == e.outputs.size() && e.L.cols() == n
                && e.Vb.cols() == n && nf == e.interfaces.size(),
            "invalid B/L/Vb dimensions");
        for (const auto* matrix : {&e.K, &e.C, &e.B, &e.L, &e.Vb})
            for (std::size_t i = 0; i < matrix->rows(); ++i)
                require(finite(matrix->row(i)), "nonfinite matrix entry");
        for (const auto* matrix : {&e.K, &e.C})
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < i; ++j)
                    require(std::abs((*matrix)(i, j) - (*matrix)(j, i))
                            <= 1e-12 * std::max({1.0, std::abs((*matrix)(i, j)), std::abs((*matrix)(j, i))}),
                        "K and C must be symmetric");
        for (std::size_t i = 0; i < n; ++i)
            require(e.C(i, i) > 0 && e.K(i, i) >= 0, "invalid capacity or stiffness diagonal");
        using Dense = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        const auto order = static_cast<Eigen::Index>(n);
        const Eigen::Map<const Dense> C(e.C.data(), order, order), K(e.K.data(), order, order);
        require(C.llt().info() == Eigen::Success, "C must be positive definite");
        Dense shifted = K;
        const double scale = K.cwiseAbs().maxCoeff();
        if (scale > 0) {
            shifted.diagonal().array() += 1e-10 * scale;
            require(shifted.llt().info() == Eigen::Success, "K must be positive semidefinite");
        }
        require(std::isfinite(e.reference_temperature) && e.uniform_temperature_mode.size() == n
                && finite(e.uniform_temperature_mode),
            "invalid temperature initialization");
        require(e.f.size() == n && finite(e.f), "invalid fixed load");
        require(e.output_offset.size() == e.outputs.size() && finite(e.output_offset), "invalid output offset");
        auto validate_facet = [](const Facet& facet) {
            require(static_cast<unsigned>(facet.axis) < 3 && (facet.normal == -1 || facet.normal == 1)
                    && std::isfinite(facet.coordinate) && finite(facet.bounds) && positive_rect(facet.bounds)
                    && std::isfinite(facet.area()) && facet.area() > 0,
                "invalid exterior facet geometry");
        };
        core::Vector<bool> used(e.boundary_groups.size(), false);
        for (const auto& face : e.interfaces) {
            validate_facet(face.facet);
            require(std::isfinite(face.resistance) && face.resistance > 0,
                "half-cell resistance must be positive and finite");
            require(face.boundary_group < e.boundary_groups.size(), "invalid interface surface");
            require(std::isfinite(face.default_coefficient) && face.default_coefficient >= 0
                    && std::isfinite(face.default_ambient), "invalid default convection parameters");
            used[face.boundary_group] = true;
        }
        require(std::ranges::all_of(used, [](bool x) { return x; }), "empty surface");
        for (const auto& h : e.boundary_groups)
            require(!h.name.empty() && std::isfinite(h.minimum) && std::isfinite(h.maximum)
                    && h.minimum > 0 && h.maximum >= h.minimum,
                "invalid HTC surface");
        for (const auto& group : e.outline) {
            require(!group.name.empty() && !group.facets.empty(), "invalid outline group");
            for (const auto& facet : group.facets)
                validate_facet(facet);
        }
        for (const auto& s : e.sources)
            require(!s.name.empty(), "invalid source");
        for (const auto& o : e.outputs)
            require(!o.name.empty() && (!o.point || finite(*o.point)), "invalid temperature output");
    }

    std::shared_ptr<const Erom> read_erom(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("cannot open EROM XML file: " + path.string());
        const std::string xml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        tinyxml2::XMLDocument doc;
        require(doc.Parse(xml.data(), xml.size()) == tinyxml2::XML_SUCCESS, "invalid XML document");
        const auto* root = doc.RootElement();
        require(root && std::string_view(root->Name()) == "Erom" && !root->NextSiblingElement(), "invalid root element");
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
            e->boundary_groups.push_back({std::string(attribute(element, "name")), scalar(element, "htcMin"),
                scalar(element, "htcMax")});
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
} // namespace mhs::model
