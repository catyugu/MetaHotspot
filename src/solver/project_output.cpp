#include "project_output.hpp"
#include "contact.hpp"
#include "thermal_reconstruction.hpp"

namespace mhs::solver {
    using namespace model;

    namespace {
        BoundarySampler project_boundary(const Project& project, std::size_t i, const FomInstance& fom,
            std::span<const double> state, std::span<const double> local, double time)
        {
            const auto& component = project.components()[i];
            const auto& m = *fom.asset;
            if (project.contacts().empty() && fom.boundaries.empty())
                return {};
            return [&project, &fom, &m, offset = component.offset, i, state, local, time](
                       FacetId id, const Vec3* point) -> BoundaryEvaluation {
                const auto& facet = m.discrete.facets[id];
                const double temperature = local[m.discrete.cell_facets[id].owner()];
                const double r = boundary_resistance(m, id, temperature, time);
                const auto surface = m.facet_boundary_group[id];
                const auto& law = boundary_law(fom, surface);
                auto ambient = [&](const Facet& patch) { return evaluate_boundary(law, patch, temperature, time, r); };
                const auto* partition = project.face_partition(i, id);
                if (!partition)
                    return ambient(facet);
                double integral = 0;
                for (const auto index : partition->contacts) {
                    const auto& c = project.contacts()[index];
                    const auto geometry = local_patch(facet, c.bounds, offset);
                    const auto other = c.left.component_index == i ? c.right : c.left;
                    const double other_temperature = contact_side(project, other).temperature(state);
                    const auto& model = project.components()[other.component_index].model;
                    const auto* other_facet = std::get_if<FacetId>(&other.face);
                    const double other_r = other_facet ? boundary_resistance(*std::get<FomInstance>(model).asset,
                                                             *other_facet, other_temperature, time)
                                                       : std::get<EromInstance>(model)
                                                             .asset->interface_at(std::get<EromInterfaceId>(other.face))
                                                             .resistance;
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
                [&](const auto& component) {
                    const auto& outputs = [&]() -> const auto& {
                        if constexpr (std::is_same_v<std::decay_t<decltype(component)>, FomInstance>)
                            return component.asset->point_probes;
                        else
                            return component.asset->outputs;
                    }();
                    for (OutputId id = 0; id < outputs.size(); ++id)
                        traces.push_back({outputs[id].name, {}, i, id});
                },
                project.components()[i].model);
        return traces;
    }

    void sample_outputs(
        const Project& project, core::Vector<OutputTrace>& traces, std::span<const double> state, double time)
    {
        std::size_t trace = 0;
        const auto components = project.components();
        for (std::size_t i = 0; i < components.size(); ++i) {
            const auto values = state.subspan(project.dof_offset(i), project.dof_size(i));
            if (const auto* fom = std::get_if<FomInstance>(&components[i].model)) {
                const auto& m = *fom->asset;
                if (m.point_probes.empty())
                    continue;
                const auto boundary = project_boundary(project, i, *fom, state, values, time);
                for (const auto& p : m.point_probes)
                    traces[trace++].values.push_back(sample_point(m, values, time, p.cell, p.point, boundary));
            }
            else {
                const auto& erom = *std::get<EromInstance>(components[i].model).asset;
                for (std::size_t o = 0; o < erom.outputs.size(); ++o) {
                    const auto row = erom.L.row(o);
                    traces[trace++].values.push_back(
                        std::inner_product(row.begin(), row.end(), values.begin(), erom.reference_temperature));
                }
            }
        }
    }
} // namespace mhs::solver
