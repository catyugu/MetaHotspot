#include "project_output.hpp"
#include "contact.hpp"
#include "thermal_reconstruction.hpp"

namespace mhs::solver {
    using namespace model;

    namespace {
        BoundarySampler project_boundary(const Project& project, std::size_t i, const Fom& m,
            std::span<const double> state, std::span<const double> local, double time)
        {
            const auto& component = project.components()[i];
            if (project.contacts().empty() && component.boundaries.empty())
                return {};
            return [&project, &component, &m, i, state, local, time](
                       FacetId id, const Vec3* point) -> BoundaryEvaluation {
                const auto& face = m.mesh.exterior_facets[id];
                const auto& facet = face.geometry;
                const double temperature = local[face.owner];
                const double r = boundary_resistance(m, id, temperature, time);
                const auto surface = face.boundary_group;
                const auto& law
                    = *boundary_override(component.boundaries, surface, &m.boundary_groups[surface].condition);
                auto ambient = [&](const Facet& patch) { return evaluate_boundary(law, patch, temperature, time, r); };
                const auto* partition = project.face_partition(i, id);
                if (!partition)
                    return ambient(facet);
                double integral = 0;
                for (const auto index : partition->contacts) {
                    const auto& c = project.contacts()[index];
                    const auto geometry = local_patch(facet, c.bounds, component.offset);
                    const auto other = c.left.component_index == i ? c.right : c.left;
                    const double other_temperature = contact_side(project, other).temperature(state);
                    const auto& asset = project.components()[other.component_index].asset;
                    const auto* fom = std::get_if<std::shared_ptr<const Fom>>(&asset);
                    const double other_r = fom
                        ? boundary_resistance(**fom, other.face_index, other_temperature, time)
                        : std::get<std::shared_ptr<const Erom>>(asset)->interfaces[other.face_index].resistance;
                    const auto t = temperature
                        + (other_temperature - temperature) * r / contact_resistance(r, other_r, c.resistance);
                    if (point && geometry.contains(*point))
                        return {.temperature = t};
                    integral += c.area() * t;
                }
                for (const auto& bounds : partition->exposed) {
                    auto geometry = facet;
                    geometry.bounds = bounds;
                    const auto trace = ambient(geometry);
                    if (point && geometry.contains(*point))
                        return trace;
                    integral += geometry.area() * trace.temperature;
                }
                return {.temperature = integral / facet.area()};
            };
        }

    } // namespace

    core::Vector<OutputTrace> project_outputs(const Project& project)
    {
        core::Vector<OutputTrace> traces;
        for (std::size_t i = 0; i < project.components().size(); ++i)
            std::visit(
                [&](const auto& asset) {
                    using Asset = typename std::decay_t<decltype(asset)>::element_type;
                    const auto& outputs = [&]() -> const auto& {
                        if constexpr (std::is_same_v<Asset, const Fom>)
                            return asset->point_probes;
                        else
                            return asset->outputs;
                    }();
                    for (OutputId id = 0; id < outputs.size(); ++id)
                        traces.push_back({outputs[id].name, {}, i, id});
                },
                project.components()[i].asset);
        return traces;
    }

    void sample_outputs(
        const Project& project, core::Vector<OutputTrace>& traces, std::span<const double> state, double time)
    {
        std::size_t trace = 0;
        const auto components = project.components();
        for (std::size_t i = 0; i < components.size(); ++i) {
            const auto values = state.subspan(project.dof_offset(i), project.dof_size(i));
            if (const auto* fom = std::get_if<std::shared_ptr<const Fom>>(&components[i].asset)) {
                const auto& m = **fom;
                if (m.point_probes.empty())
                    continue;
                const auto boundary = project_boundary(project, i, m, state, values, time);
                for (const auto& p : m.point_probes)
                    traces[trace++].values.push_back(sample_point(m, values, time, p.cell, p.point, boundary));
            }
            else {
                const auto& erom = *std::get<std::shared_ptr<const Erom>>(components[i].asset);
                for (std::size_t o = 0; o < erom.outputs.size(); ++o) {
                    const auto row = erom.L.row(o);
                    traces[trace++].values.push_back(
                        std::inner_product(row.begin(), row.end(), values.begin(), erom.reference_temperature));
                }
            }
        }
    }
} // namespace mhs::solver
