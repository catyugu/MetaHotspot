#include "definition.hpp"
#include "geometry_expression.hpp"
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

    namespace {
        using Box = mhs::model::BoxOperation;
        using Boxes = mhs::core::Vector<Box>;

        // The difference of two aligned boxes contains at most six pieces.
        Boxes subtract_box(const Box& region, const Box& cut)
        {
            Box overlap = region;
            for (int a = 0; a < 3; ++a) {
                overlap.min[a] = std::max(region.min[a], cut.min[a]);
                overlap.max[a] = std::min(region.max[a], cut.max[a]);
                if (overlap.min[a] >= overlap.max[a])
                    return {region};
            }
            Boxes result;
            Box remaining = region;
            for (int a = 0; a < 3; ++a) {
                if (remaining.min[a] < overlap.min[a]) {
                    Box piece = remaining;
                    piece.max[a] = overlap.min[a];
                    result.push_back(piece);
                    remaining.min[a] = overlap.min[a];
                }
                if (overlap.max[a] < remaining.max[a]) {
                    Box piece = remaining;
                    piece.min[a] = overlap.max[a];
                    result.push_back(piece);
                    remaining.max[a] = overlap.max[a];
                }
            }
            return result;
        }

        Boxes subtract(Boxes regions, const Box& cut)
        {
            Boxes result;
            for (const auto& region : regions) {
                auto pieces = subtract_box(region, cut);
                result.insert(result.end(), pieces.begin(), pieces.end());
            }
            return result;
        }

        // Convert ordered Add/Subtract operations to disjoint positive boxes.
        Boxes disjoint_boxes(const mhs::model::VolumeGeometry& geometry)
        {
            Boxes result;
            for (const auto& op : geometry.operations) {
                if (op.operation == mhs::model::GeometryOperation::Subtract) {
                    result = subtract(std::move(result), op);
                    continue;
                }
                Boxes added {op};
                for (const auto& existing : result)
                    added = subtract(std::move(added), existing);
                result.insert(result.end(), added.begin(), added.end());
            }
            return result;
        }
    } // namespace

    mhs::model::ModelDefinition normalize(Definition pta)
    {
        normalize_pta_expressions(pta);
        auto out = std::move(pta.model);
        const double scale = length_unit_to_si(pta.length_unit);
        const GeometryEvaluator geometry(pta.variables);
        auto eval = [&](const auto& e) { return geometry.evaluate(e) * scale; };
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
        // PTA block ordering is retained for material precedence.
        struct BlockBody {
            mhs::model::BodySpec body;
            mhs::model::Expression power_density;
        };
        mhs::core::Vector<mhs::core::Vector<BlockBody>> layers;
        for (std::size_t l = 0; l < pta.layers.size(); ++l) {
            z -= thickness[l];
            const auto& layer = pta.layers[l];
            mhs::core::Vector<BlockBody> blocks;
            for (const auto& b : layer.blocks) {
                auto mat = std::find_if(
                    out.materials.begin(), out.materials.end(), [&](const auto& m) { return m.name == b.material; });
                if (mat == out.materials.end())
                    throw std::invalid_argument("unknown material: " + b.material);
                BlockBody block;
                block.body.material = static_cast<mhs::model::MaterialId>(mat - out.materials.begin());
                block.power_density = b.volumetric_heat_source;
                const double x = eval(layer.x_offset) + eval(b.x_offset), y = eval(layer.y_offset) + eval(b.y_offset);
                const double top = z + ((l == 0 && b.thickness) ? eval(*b.thickness) : thickness[l]);
                for (const auto& op : b.geometry) {
                    const auto& r = op.rect;
                    double x0 = x + eval(r.x), y0 = y + eval(r.y), x1 = x0 + eval(r.width), y1 = y0 + eval(r.height);
                    block.body.geometry.operations.push_back({op.operation, {std::min(x0, x1), std::min(y0, y1), z},
                        {std::max(x0, x1), std::max(y0, y1), top}});
                }
                blocks.push_back(std::move(block));
            }
            layers.push_back(std::move(blocks));
        }
        mhs::core::Vector<mhs::model::Expression> power;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            for (auto& block : *it) {
                power.push_back(std::move(block.power_density));
                out.bodies.push_back(std::move(block.body));
            }

        // PTA block power follows the visible portion of its Block. Resolve
        // precedence in this adapter; independent core sources always add.
        mhs::core::Vector<mhs::model::VolumeSourceSpec> sources(out.bodies.size());
        Boxes covered;
        for (std::size_t i = out.bodies.size(); i-- > 0;) {
            const auto region = disjoint_boxes(out.bodies[i].geometry);
            if (!power[i].empty() && power[i] != "0" && power[i] != "0.0") {
                Boxes visible = region;
                for (const auto& occluder : covered)
                    visible = subtract(std::move(visible), occluder);
                if (!visible.empty())
                    sources[i] = {{std::move(visible)}, std::move(power[i])};
            }
            covered.insert(covered.end(), region.begin(), region.end());
        }
        for (auto& source : sources)
            if (!source.geometry.operations.empty())
                out.volume_sources.push_back(std::move(source));
        for (auto* axis : {&out.mesh.x_vertices, &out.mesh.y_vertices, &out.mesh.z_vertices})
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
        for (auto& b : out.thermal_boundaries)
            scale_regions(b.regions);
        for (auto& b : out.fluid_boundaries)
            scale_regions(b.regions);
        for (const auto& p : pta.point_probes)
            out.point_probes.push_back({p.name, eval(p.x), eval(p.y), eval(p.z)});
        return out;
    }
} // namespace mhs::app::pta
