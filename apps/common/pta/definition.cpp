#include "definition.hpp"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace mhs::app::pta {
    namespace {
        bool identifier_start(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }

        bool identifier_part(char c) { return identifier_start(c) || (c >= '0' && c <= '9'); }

        class Parser {
        public:
            Parser(std::string_view input, const std::unordered_map<std::string, double>& values)
                : input_(input), values_(values)
            {
            }

            double evaluate()
            {
                skip_space();
                if (pos_ == input_.size())
                    return 0.0; // Optional PTA geometry fields default to zero.
                const double result = expression();
                skip_space();
                if (pos_ != input_.size())
                    fail("unexpected token");
                if (!std::isfinite(result))
                    fail("non-finite result");
                return result;
            }

        private:
            void skip_space()
            {
                while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_])))
                    ++pos_;
            }

            bool consume(char c)
            {
                skip_space();
                if (pos_ < input_.size() && input_[pos_] == c) {
                    ++pos_;
                    return true;
                }
                return false;
            }

            [[noreturn]] void fail(const std::string& message) const
            {
                throw std::invalid_argument("invalid PTA geometry expression '" + std::string(input_) + "' at "
                    + std::to_string(pos_) + ": " + message);
            }

            double expression()
            {
                double value = term();
                while (true) {
                    if (consume('+'))
                        value += term();
                    else if (consume('-'))
                        value -= term();
                    else
                        return value;
                }
            }

            double term()
            {
                double value = unary();
                while (true) {
                    if (consume('*'))
                        value *= unary();
                    else if (consume('/'))
                        value /= unary();
                    else
                        return value;
                }
            }

            double unary()
            {
                if (consume('+'))
                    return unary();
                if (consume('-'))
                    return -unary();
                return power();
            }

            double power()
            {
                double value = primary();
                if (consume('^'))
                    value = std::pow(value, unary()); // Right associative, tighter than unary sign.
                return value;
            }

            double primary()
            {
                if (consume('(')) {
                    const double value = expression();
                    if (!consume(')'))
                        fail("expected ')'");
                    return value;
                }
                skip_space();
                if (pos_ == input_.size())
                    fail("expected number or variable");
                if (identifier_start(input_[pos_])) {
                    const auto start = pos_++;
                    while (pos_ < input_.size() && identifier_part(input_[pos_]))
                        ++pos_;
                    const std::string name(input_.substr(start, pos_ - start));
                    auto it = values_.find(name);
                    if (it == values_.end())
                        fail("unknown variable '" + name + "'");
                    return it->second;
                }
                if ((input_[pos_] >= '0' && input_[pos_] <= '9') || input_[pos_] == '.') {
                    const char* begin = input_.data() + pos_;
                    const char* end = input_.data() + input_.size();
                    double value = 0.0;
                    const auto parsed = std::from_chars(begin, end, value, std::chars_format::general);
                    if (parsed.ec != std::errc() || parsed.ptr == begin)
                        fail("invalid number");
                    pos_ = static_cast<std::size_t>(parsed.ptr - input_.data());
                    return value;
                }
                fail("expected number or variable");
            }

            std::string_view input_;
            const std::unordered_map<std::string, double>& values_;
            std::size_t pos_ = 0;
        };

        // Private to the PTA adapter. Results use the input's length unit.
        class GeometryEvaluator {
        public:
            explicit GeometryEvaluator(const mhs::core::Vector<VariableSpec>& variables);
            double evaluate(std::string_view expr) const;

        private:
            std::unordered_map<std::string, double> values_;
        };

        GeometryEvaluator::GeometryEvaluator(const mhs::core::Vector<VariableSpec>& variables)
        {
            values_.reserve(variables.size());
            for (const auto& variable : variables) {
                const auto& name = variable.name;
                if (name.empty() || !identifier_start(name.front()))
                    throw std::invalid_argument("invalid PTA geometry variable name: '" + name + "'");
                for (char c : name)
                    if (!identifier_part(c))
                        throw std::invalid_argument("invalid PTA geometry variable name: '" + name + "'");
                if (values_.contains(name))
                    throw std::invalid_argument("duplicate PTA geometry variable: '" + name + "'");
                if (variable.value.empty())
                    throw std::invalid_argument("empty PTA geometry variable: '" + name + "'");
                try {
                    values_.emplace(name, evaluate(variable.value));
                }
                catch (const std::exception& e) {
                    throw std::invalid_argument("PTA geometry variable '" + name + "': " + e.what());
                }
            }
        }

        double GeometryEvaluator::evaluate(std::string_view expr) const { return Parser(expr, values_).evaluate(); }
    } // namespace

    inline double length_unit_to_si(LengthUnit unit)
    {
        switch (unit) {
        case LengthUnit::Meter:
            return 1.0;
        case LengthUnit::Millimeter:
            return 1e-3;
        case LengthUnit::Micrometer:
            return 1e-6;
        case LengthUnit::Nanometer:
            return 1e-9;
        case LengthUnit::Inch:
            return 0.0254;
        case LengthUnit::Mil:
            return 2.54e-5;
        }
        throw std::invalid_argument("invalid length unit");
    }

    // PTA uses a generic formal variable named x for several unrelated
    // physical laws. Normalize that front-end convention before the model
    // reaches the compiler; FomDefinition expressions are canonical.
    std::string rewrite_formal_argument(const std::string& expression, std::string_view replacement)
    {
        auto is_id_start = [](char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; };
        auto is_id_char = [&](char c) { return is_id_start(c) || (c >= '0' && c <= '9'); };

        std::string out;
        out.reserve(expression.size());
        for (std::size_t i = 0; i < expression.size();) {
            if (!is_id_start(expression[i])) {
                out += expression[i++];
                continue;
            }
            const std::size_t begin = i++;
            while (i < expression.size() && is_id_char(expression[i]))
                ++i;
            const std::string_view name(expression.data() + begin, i - begin);
            if (name == "x")
                out += replacement;
            else
                out.append(expression, begin, i - begin);
        }
        return out;
    }

    static void normalize_pta_expressions(Definition& def)
    {
        for (auto& named : def.model.materials) {
            auto& material = named.value;
            material.conductivity_x = rewrite_formal_argument(material.conductivity_x, "T");
            material.conductivity_y = rewrite_formal_argument(material.conductivity_y, "T");
            material.conductivity_z = rewrite_formal_argument(material.conductivity_z, "T");
            material.density = rewrite_formal_argument(material.density, "T");
            material.specific_heat = rewrite_formal_argument(material.specific_heat, "T");
            if (material.dynamic_viscosity)
                *material.dynamic_viscosity = rewrite_formal_argument(*material.dynamic_viscosity, "T");
        }

        auto normalize_boundary = [](mhs::model::ThermalBoundary& boundary) {
            std::visit(
                [](auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, mhs::model::DirichletBoundary>) {
                        value.temperature = rewrite_formal_argument(value.temperature, "T");
                    }
                    else if constexpr (std::is_same_v<T, mhs::model::NeumannBoundary>) {
                        value.heat_flux = rewrite_formal_argument(value.heat_flux, "T");
                    }
                    else {
                        value.coefficient = rewrite_formal_argument(value.coefficient, "T");
                        value.ambient_temperature = rewrite_formal_argument(value.ambient_temperature, "T");
                    }
                },
                boundary);
        };
        normalize_boundary(def.model.default_thermal_boundary);
        for (auto& boundary : def.model.thermal_boundaries)
            normalize_boundary(boundary.condition);

        for (auto& layer : def.layers)
            for (auto& block : layer.blocks) {
                const bool names_function = std::any_of(def.model.functions.begin(), def.model.functions.end(),
                    [&](const mhs::model::NamedFunction& function) {
                        return function.name == block.volumetric_heat_source;
                    });
                if (names_function)
                    block.volumetric_heat_source += "(t)";
                else
                    block.volumetric_heat_source = rewrite_formal_argument(block.volumetric_heat_source, "t");
            }
    }

    // In extraction mode a legacy source is harmless only if it is absent or
    // explicitly numeric zero. Never discard a possibly nonzero field law.
    static bool literal_zero(std::string_view text)
    {
        if (text.empty())
            return true;
        if (text.front() == '+')
            text.remove_prefix(1);
        double value = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        return parsed.ec == std::errc {} && parsed.ptr == text.data() + text.size() && value == 0;
    }

    void normalize(Definition& pta, bool extraction)
    {
        normalize_pta_expressions(pta);
        auto& out = pta.model;
        if (extraction) {
            for (const auto& source : pta.heat_sources)
                out.sources.push_back({source.name.empty() ? source.id : source.name, "0"});
        }
        const double scale = length_unit_to_si(pta.length_unit);
        const GeometryEvaluator geometry(pta.variables);
        auto eval = [&](const auto& e) { return geometry.evaluate(e) * scale; };
        mhs::core::Vector<double> thickness;
        for (std::size_t l = 0; l < pta.layers.size(); ++l) {
            double t = pta.layers[l].thickness.empty() ? 0 : eval(pta.layers[l].thickness);
            if (l == 0)
                for (const auto& b : pta.layers[l].blocks)
                    if (b.thickness)
                        t = std::max(t, eval(*b.thickness));
            thickness.push_back(t);
        }
        // Build from bottom to top so the first PTA layer wins at shared planes.
        double z = 0;
        for (std::size_t l = pta.layers.size(); l-- > 0;) {
            const auto& layer = pta.layers[l];
            for (const auto& b : layer.blocks) {
                auto mat = std::find_if(
                    out.materials.begin(), out.materials.end(), [&](const auto& m) { return m.name == b.material; });
                if (mat == out.materials.end())
                    throw std::invalid_argument("unknown material: " + b.material);
                mhs::model::VolumeSpec v;
                v.material = static_cast<mhs::model::MaterialId>(mat - out.materials.begin());
                if (!extraction) {
                    v.source = static_cast<mhs::model::SourceId>(out.sources.size());
                    out.sources.push_back({b.name, b.volumetric_heat_source});
                }
                else if (!b.heat_source_id.empty()) {
                    const auto source = std::find_if(pta.heat_sources.begin(), pta.heat_sources.end(),
                        [&](const auto& entry) { return entry.id == b.heat_source_id; });
                    if (source == pta.heat_sources.end())
                        throw std::invalid_argument(
                            "unknown HeatSourceId in block " + b.name + ": " + b.heat_source_id);
                    v.source = static_cast<mhs::model::SourceId>(source - pta.heat_sources.begin());
                }
                else if (!literal_zero(b.volumetric_heat_source)) {
                    throw std::invalid_argument("block " + b.name + " has a legacy heat source without HeatSourceId");
                }
                const double x = eval(layer.x_offset) + eval(b.x_offset), y = eval(layer.y_offset) + eval(b.y_offset);
                const double top = z + ((l == 0 && b.thickness) ? eval(*b.thickness) : thickness[l]);
                for (const auto& op : b.geometry) {
                    const auto& r = op.rect;
                    double x0 = x + eval(r.x), y0 = y + eval(r.y), x1 = x0 + eval(r.width), y1 = y0 + eval(r.height);
                    v.geometry.push_back({op.operation, {std::min(x0, x1), std::min(y0, y1), z},
                        {std::max(x0, x1), std::max(y0, y1), top}});
                }
                out.volumes.push_back(std::move(v));
            }
            z += thickness[l];
        }
        for (auto* axis : {&pta.mesh.x_vertices, &pta.mesh.y_vertices, &pta.mesh.z_vertices})
            for (auto& v : *axis)
                v *= scale;
        auto scale_regions = [&](auto& regions) {
            for (auto& r : regions) {
                r.coordinate *= scale;
                for (auto& x : r.bounds)
                    x *= scale;
            }
        };
        for (auto& b : out.thermal_boundaries)
            scale_regions(b.regions);
        for (auto& b : out.fluid_boundaries)
            scale_regions(b.regions);
        for (const auto& p : pta.point_probes)
            out.point_probes.push_back({p.name, eval(p.x), eval(p.y), eval(p.z)});
    }
} // namespace mhs::app::pta
