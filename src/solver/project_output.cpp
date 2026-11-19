#include "project_output.hpp"
#include "contact.hpp"
#include "thermal_reconstruction.hpp"

namespace mhs::solver {
    using namespace model;

    namespace {
        double sample_project_fom(
            const Project& project, std::size_t i, const PointProbe& probe, std::span<const double> state, double time)
        {
            const auto& m = *std::get<std::shared_ptr<const Fom>>(project.components()[i].asset);
            const auto local = state.subspan(project.dof_offset(i), project.dof_size(i));
            const auto& component = project.components()[i];
            if (project.patches().empty() && component.boundaries.empty())
                return sample_point(m, local, time, probe.cell, probe.point);
            BoundarySampler sample = [&](FacetId id, const Vec3* point) -> BoundaryEvaluation {
                const auto& facet = m.discrete.facets[id];
                const double temperature = local[m.discrete.cell_facets[id].owner()];
                const double r = boundary_resistance(m, id, temperature, time);
                const auto surface = m.facet_surface[id];
                const auto& law = boundary_override(component.boundaries, surface, m.surfaces[surface].condition);
                auto ambient = [&](const Facet& patch) { return evaluate_boundary(law, patch, temperature, time, r); };
                const auto indices = project.facet_patches(i, id);
                if (indices.empty())
                    return ambient(facet);
                double integral = 0;
                for (const auto index : indices) {
                    const auto& patch = project.patches()[index];
                    auto geometry = facet;
                    geometry.bounds = patch.bounds;
                    if (const auto& contact = patch.contact) {
                        const auto& c = *contact;
                        const auto other = c.left.component == i ? c.right : c.left;
                        const double other_temperature = contact_side(project, other.component, other.facet).temperature(state);
                        const auto& asset = project.components()[other.component].asset;
                        const auto* fom = std::get_if<std::shared_ptr<const Fom>>(&asset);
                        const double other_r = fom ? boundary_resistance(**fom, other.facet, other_temperature, time)
                            : std::get<std::shared_ptr<const Erom>>(asset)->interfaces[other.facet].resistance;
                        const auto t = temperature
                            + (other_temperature - temperature) * r / contact_resistance(r, other_r, c.resistance);
                        if (point) {
                            auto world = *point;
                            for (std::size_t axis = 0; axis < world.size(); ++axis)
                                world[axis] += component.offset[axis];
                            geometry.coordinate += component.offset[static_cast<std::size_t>(facet.axis)];
                            if (geometry.contains(world))
                                return {.temperature = t};
                        }
                        integral += patch.area() * t;
                    }
                    else {
                        const auto trace = ambient(geometry);
                        if (point && geometry.contains(*point))
                            return trace;
                        integral += patch.area() * trace.temperature;
                    }
                }
                return {.temperature = integral / facet.area()};
            };
            return sample_point(m, local, time, probe.cell, probe.point, sample);
        }

    } // namespace

    void append_fom_traces(core::Vector<OutputTrace>& traces, const Fom& fom, ComponentId component)
    {
        for (std::size_t p = 0; p < fom.point_probes.size(); ++p)
            traces.push_back({fom.point_probes[p].name, {}, component, static_cast<OutputId>(p)});
    }

    core::Vector<OutputTrace> project_outputs(const Project& project)
    {
        core::Vector<OutputTrace> traces;
        for (const auto& c : project.components())
            std::visit(
                [&](const auto& asset) {
                    using Asset = typename std::decay_t<decltype(asset)>::element_type;
                    if constexpr (std::is_same_v<Asset, const Fom>)
                        append_fom_traces(traces, *asset, c.id);
                    else
                        for (OutputId id = 0; id < asset->outputs.size(); ++id)
                            traces.push_back({asset->outputs[id].name, {}, c.id, id});
                },
                c.asset);
        return traces;
    }

    void sample_outputs(
        const Project& project, core::Vector<OutputTrace>& traces, std::span<const double> state, double time)
    {
        std::size_t trace = 0;
        const auto components = project.components();
        for (std::size_t i = 0; i < components.size(); ++i) {
            if (const auto* fom = std::get_if<std::shared_ptr<const Fom>>(&components[i].asset)) {
                for (const auto& p : (*fom)->point_probes)
                    traces[trace++].values.push_back(sample_project_fom(project, i, p, state, time));
            }
            else {
                const auto values = state.subspan(project.dof_offset(i), project.dof_size(i));
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
