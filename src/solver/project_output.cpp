#include "project_output.hpp"
#include "contact.hpp"
#include "thermal_physics.hpp"
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
            const auto& contacts = project.facet_contacts(i);
            if (contacts.empty() && component.boundaries.empty())
                return sample_point(m, local, time, probe.cell, probe.point);
            auto sample_r = [&](std::size_t index, FacetId id) {
                const auto& asset = project.components()[index].asset;
                if (const auto* erom = std::get_if<std::shared_ptr<const Erom>>(&asset))
                    return (*erom)->R[id];
                const auto& fom = *std::get<std::shared_ptr<const Fom>>(asset);
                const auto axis = static_cast<std::size_t>(fom.discrete.facets[id].axis);
                const auto cell = fom.discrete.cell_facets[id].owner();
                const auto temperature = state[project.dof_offset(index) + cell];
                return boundary_resistance(fom, id, cell_conductivity(fom, cell, axis, temperature, time));
            };
            BoundarySampler sample = [&](FacetId id, const Vec3* point) -> BoundarySample {
                const auto& facet = m.discrete.facets[id];
                const auto r = sample_r(i, id);
                const double temperature = contact_side(project, i, id).temperature(state);
                const auto surface = m.facet_surface[id];
                const auto& law = boundary_override(component.boundaries, surface, m.boundary_conditions[surface]);
                auto ambient = [&](const Facet& patch) {
                    const auto t = evaluate_boundary(law, patch, temperature, time, r);
                    return BoundarySample {t.temperature, std::holds_alternative<Dirichlet>(law)};
                };
                const auto record = contacts.find(id);
                if (record == contacts.end())
                    return ambient(facet);
                double integral = 0;
                for (const auto index : record->second.contacts) {
                    const auto& c = project.contacts()[index];
                    const auto other = c.left == i ? c.right : c.left;
                    const auto other_id = c.left == i ? c.right_facet : c.left_facet;
                    const auto difference = contact_side(project, other, other_id).temperature(state) - temperature;
                    const auto t
                        = temperature + difference * r / contact_resistance(r, sample_r(other, other_id), c.resistance);
                    if (point) {
                        const auto u = facet.axis == Axis::X ? 1 : 0, v = facet.axis == Axis::Z ? 1 : 2;
                        const auto a = (*point)[u] + component.offset[u], b = (*point)[v] + component.offset[v];
                        if (a >= c.overlap[0] - geometry_eps && a <= c.overlap[1] + geometry_eps
                            && b >= c.overlap[2] - geometry_eps && b <= c.overlap[3] + geometry_eps)
                            return {t, false};
                    }
                    integral += c.area() * t;
                }
                for (const auto& patch : record->second.exposed) {
                    const auto trace = ambient(patch);
                    if (point && patch.contains(*point))
                        return trace;
                    integral += patch.area() * trace.temperature;
                }
                return {integral / facet.area(), false};
            };
            return sample_point(m, local, time, probe.cell, probe.point, sample);
        }

    } // namespace

    core::Vector<OutputTrace> project_outputs(const Project& project)
    {
        core::Vector<OutputTrace> traces;
        for (const auto& c : project.components())
            std::visit(
                [&](const auto& asset) {
                    using Asset = typename std::decay_t<decltype(asset)>::element_type;
                    if constexpr (std::is_same_v<Asset, const Fom>) {
                        for (std::size_t p = 0; p < asset->point_probes.size(); ++p)
                            traces.push_back({asset->point_probes[p].name, {}, c.id, static_cast<OutputId>(p)});
                    }
                    else
                        for (const auto& output : asset->outputs)
                            traces.push_back({output.name, {}, c.id, output.id});
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
