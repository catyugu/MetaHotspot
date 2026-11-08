#include "definition.hpp"
#include "compiler/model_compiler.hpp"
#include "expression/expression.hpp"
#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <type_traits>
namespace mhs::app::pta {
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
    // reaches the compiler; ModelDefinition expressions are canonical.
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
        auto& library = *def.model.components.front().library;
        for (auto& named : library.materials) {
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
        for (auto& boundary : library.thermal_boundaries)
            normalize_boundary(boundary);

        for (auto& layer : def.layers)
            for (auto& block : layer.blocks) {
                const bool names_function = std::any_of(library.functions.begin(),
                    library.functions.end(), [&](const mhs::model::NamedFunction& function) {
                        return function.name == block.volumetric_heat_source;
                    });
                if (names_function)
                    block.volumetric_heat_source += "(t)";
                else
                    block.volumetric_heat_source = rewrite_formal_argument(block.volumetric_heat_source, "t");
            }
    }

    mhs::model::ModelDefinition normalize(Definition pta)
    {
        normalize_pta_expressions(pta);
        auto out = std::move(pta.model);
        auto& library = *out.components.front().library;
        const double scale = length_unit_to_si(pta.length_unit);
        auto symbols = mhs::compiler::build_symbol_table(out);
        auto eval = [&](const auto& e) { return mhs::expression::eval_geometry(e, symbols) * scale; };
        mhs::core::Vector<double> thickness;
        double z = 0;
        for (std::size_t l = 0; l < pta.layers.size(); ++l) {
            double t = pta.layers[l].thickness.empty() ? 0 : eval(pta.layers[l].thickness);
            if (l == 0)
                for (const auto& b : pta.layers[l].blocks)
                    if (b.thickness)
                        t = std::max(t, eval(*b.thickness));
            thickness.push_back(t);
            z += t;
        }
        // Reverse the layer order so the first PTA layer wins at a shared plane.
        mhs::core::Vector<mhs::core::Vector<mhs::model::VolumeSpec>> layers;
        for (std::size_t l = 0; l < pta.layers.size(); ++l) {
            z -= thickness[l];
            const auto& layer = pta.layers[l];
            mhs::core::Vector<mhs::model::VolumeSpec> volumes;
            for (const auto& b : layer.blocks) {
                auto mat = std::find_if(library.materials.begin(), library.materials.end(),
                    [&](const auto& m) { return m.name == b.material; });
                if (mat == library.materials.end())
                    throw std::invalid_argument("unknown material: " + b.material);
                mhs::model::VolumeSpec v;
                v.material = static_cast<mhs::model::MaterialId>(mat - library.materials.begin());
                v.source = static_cast<mhs::model::SourceId>(library.sources.size());
                library.sources.push_back(b.volumetric_heat_source);
                const double x = eval(layer.x_offset) + eval(b.x_offset), y = eval(layer.y_offset) + eval(b.y_offset);
                const double top = z + ((l == 0 && b.thickness) ? eval(*b.thickness) : thickness[l]);
                for (const auto& op : b.geometry) {
                    const auto& r = op.rect;
                    double x0 = x + eval(r.x), y0 = y + eval(r.y), x1 = x0 + eval(r.width), y1 = y0 + eval(r.height);
                    v.geometry.push_back({op.operation, {std::min(x0, x1), std::min(y0, y1), z},
                        {std::max(x0, x1), std::max(y0, y1), top}});
                }
                volumes.push_back(std::move(v));
            }
            layers.push_back(std::move(volumes));
        }
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            for (auto& v : *it)
                out.fom(0).volumes.push_back(std::move(v));
        auto& mesh = out.fom(0).mesh;
        for (auto* axis : {&mesh.x_vertices, &mesh.y_vertices, &mesh.z_vertices})
            for (auto& v : *axis)
                v *= scale;
        auto scale_regions = [&](auto& regions) {
            for (auto& r : regions) {
                r.coordinate *= scale;
                r.rectangle.a_min *= scale;
                r.rectangle.a_max *= scale;
                r.rectangle.b_min *= scale;
                r.rectangle.b_max *= scale;
            }
        };
        for (auto& b : out.component(0).thermal_boundaries)
            scale_regions(b.regions);
        for (auto& b : out.fom(0).fluid_boundaries)
            scale_regions(b.regions);
        for (const auto& p : pta.point_probes)
            out.component(0).observations.push_back({{p.name, eval(p.x), eval(p.y), eval(p.z)}, {}, 0.0});
        return out;
    }
} // namespace mhs::app::pta
