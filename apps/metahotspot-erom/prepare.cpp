#include "extraction.hpp"
#include "solver/assembler.hpp"
#include "solver/thermal_reconstruction.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace mhs::app::erom {
    using namespace mhs::model;
    using Weights = std::map<CellId, double>;

    double constant(const expression::CompiledExpression& e, std::string_view name)
    {
        if (!e.is_constant() || !std::isfinite(e.constant_value()))
            throw std::invalid_argument(std::string(name) + " must be constant and finite for EROM extraction");
        return e.constant_value();
    }

    struct Source {
        EromSource channel;
        Weights volume;
        double total_volume = 0;
    };

    struct Output {
        EromOutput channel;
        Weights weights;
    };

    Output project_probe(const Fom& model, const Selection& selection, OutputId id, core::Vector<double>& state)
    {
        const auto& probe = model.point_probes[id];
        if (probe.cell == core::invalid_id)
            throw std::invalid_argument("cannot retain point probe outside the FOM: " + probe.name);
        const auto facets = model.discrete.incident_facets(probe.cell);
        const auto& mesh = model.discrete;
        const auto center = mesh.cells[probe.cell].center();
        const bool on_boundary = std::ranges::any_of(
            facets, [&](FacetId f) { return mesh.cell_facets[f].exterior() && mesh.facets[f].contains(probe.point); });
        std::set<CellId> support {probe.cell};
        for (const auto f : facets) {
            const auto& cells = mesh.cell_facets[f];
            const auto& facet = mesh.facets[f];
            if (!cells.exterior()) {
                support.insert(cells.minus == probe.cell ? cells.plus : cells.minus);
                continue;
            }
            const auto axis = static_cast<std::size_t>(facet.axis);
            if (selection.htc.contains(model.facet_surface[f])
                && (on_boundary ? facet.contains(probe.point) : probe.point[axis] != center[axis]))
                throw std::invalid_argument("probe " + probe.name
                    + " depends directly on HTC boundary temperature and cannot be represented by L");
        }
        auto sample = [&] { return solver::sample_point(model, state, 0, probe.cell, probe.point); };
        const double baseline = sample();
        Output output {{id, probe.name, EromOutputKind::PointTemperature, baseline, probe.point}, {}};
        for (const auto cell : support)
            state[cell] = 0;
        const double zero = sample();
        for (const auto cell : support) {
            state[cell] = 1;
            output.weights[cell] = sample() - zero;
            state[cell] = 0;
        }
        for (const auto cell : support)
            state[cell] = model.reference_temperature;
        return output;
    }

    Prepared prepare(const Fom& model, double initial_temperature, Selection selection)
    {
        const auto& mesh = model.discrete;
        if (mesh.cells.empty() || !model.fluid.fluid_to_global.empty())
            throw std::invalid_argument("EROM extraction requires a nonempty solid FOM");

        const std::set<MaterialId> used_materials(model.cell_material.begin(), model.cell_material.end());
        for (const auto id : used_materials) {
            const auto& material = model.material_table[id];
            for (const auto* property : {&material.kx, &material.ky, &material.kz, &material.rho, &material.c})
                if (constant(*property, "Material property") <= 0)
                    throw std::invalid_argument(
                        "EROM extraction requires positive material properties on active cells");
        }

        for (const auto& group : selection.htc)
            if (const auto id = group.first; id >= model.surfaces.size() || model.surfaces[id].facets.empty())
                throw std::invalid_argument("unknown or empty HTC surface " + std::to_string(id));

        core::Vector<const Surface*> surfaces;
        for (const auto& surface : model.surfaces) {
            if (surface.facets.empty())
                continue;
            surfaces.push_back(&surface);
            const auto training = selection.htc.find(surface.id);
            const auto condition_id = model.facet_thermal_condition[surface.facets.front()];
            for (const auto id : surface.facets)
                if (model.facet_thermal_condition[id] != condition_id)
                    throw std::invalid_argument("EROM surface requires a uniform thermal condition");
            const auto& condition = model.boundary_conditions[condition_id];
            if (const auto* htc = std::get_if<Convection>(&condition)) {
                const double h = constant(htc->coefficient, "HTC");
                if (h < 0 || (h > 0 && training == selection.htc.end()))
                    throw std::invalid_argument(
                        "surface " + std::to_string(surface.id) + " has an untrained or negative HTC");
                if (training != selection.htc.end()) {
                    training->second.coefficient = h;
                    training->second.ambient = constant(htc->ambient_temperature, "Ambient temperature");
                }
            }
            else if (const auto* flux = std::get_if<Neumann>(&condition);
                !flux || constant(flux->heat_flux, "Neumann heat flux") != 0)
                throw std::invalid_argument("surface " + std::to_string(surface.id)
                    + " has a Dirichlet or nonzero Neumann condition unsupported by EROM extraction");
            else if (training != selection.htc.end())
                training->second.ambient = model.reference_temperature;
        }

        std::map<SourceId, Source> sources;
        std::map<SourceId, double> source_density;
        for (const auto& term : model.source_terms) {
            const auto id = term.source;
            auto [density, inserted] = source_density.try_emplace(id);
            if (inserted)
                density->second = constant(model.heat_source_table[id], "Volumetric source");
            // PTA emits a zero-valued source entry even for an unpowered block.
            if (density->second == 0)
                continue;
            if (!sources.contains(id)) {
                auto name = model.source_names.empty() ? std::string {} : model.source_names[id];
                if (name.empty())
                    name = "source " + std::to_string(id);
                sources.emplace(id, Source {{id, std::move(name), 0}, {}, 0});
            }
            auto& source = sources.at(id);
            source.volume[term.cell] += term.weight;
            source.total_volume += term.weight;
        }
        for (auto& [id, source] : sources)
            source.channel.default_power = source_density.at(id) * source.total_volume;

        core::Vector<Output> outputs;
        core::Vector<double> state(mesh.cells.size(), model.reference_temperature);
        for (std::size_t i = 0; i < model.point_probes.size(); ++i) {
            const auto& probe = model.point_probes[i];
            if (selection.all_probes || selection.probes.contains(probe.name))
                outputs.push_back(project_probe(model, selection, static_cast<OutputId>(i), state));
        }
        for (const auto& requested : selection.probes)
            if (!std::ranges::any_of(model.point_probes, [&](const auto& probe) { return probe.name == requested; }))
                throw std::invalid_argument("unknown point probe: " + requested);

        solver::ThermalAssembler assembler(model);
        const auto& operators = assembler.assemble_internal(state, 0);
        Prepared result;
        const auto n = static_cast<Eigen::Index>(mesh.cells.size());
        result.K = Eigen::Map<const Sparse>(n, n, static_cast<Eigen::Index>(operators.K.nnz()),
            operators.K.row_ptr().data(), operators.K.col_idx().data(), operators.K.values().data());
        result.capacity = Eigen::Map<const Vector>(operators.C.values().data(), n);
        result.reference = model.reference_temperature;
        result.initial = initial_temperature;
        for (const auto* surface : surfaces) {
            SurfaceData data {surface->id, surface->name, std::nullopt, {}, {}, {}};
            const auto range = selection.htc.find(surface->id);
            if (range != selection.htc.end())
                data.training = range->second;
            for (const auto id : surface->facets) {
                data.facets.push_back(mesh.facets[id]);
                if (data.training) {
                    data.cells.push_back(mesh.cell_facets[id].owner());
                    data.resistance.push_back(assembler.resistance(id));
                }
            }
            result.surfaces.push_back(std::move(data));
        }
        result.B = Matrix::Zero(n, static_cast<Eigen::Index>(sources.size()));
        Eigen::Index column = 0;
        for (const auto& [id, source] : sources) {
            result.sources.push_back(source.channel);
            for (const auto& [cell, volume] : source.volume)
                result.B(cell, column) = volume / source.total_volume;
            ++column;
        }
        core::Vector<Eigen::Triplet<double>> weights;
        auto add_output = [&](EromOutput output, const Weights& values, double scale) {
            const auto row = static_cast<int>(result.outputs.size());
            result.outputs.push_back(std::move(output));
            for (const auto& [cell, value] : values)
                weights.emplace_back(row, static_cast<int>(cell), value * scale);
        };
        for (const auto& output : outputs)
            add_output(output.channel, output.weights, 1);
        if (selection.source_temperatures) {
            std::size_t id = model.point_probes.size();
            for (const auto& [source_id, source] : sources)
                add_output({static_cast<OutputId>(id++), source.channel.name + " temperature",
                               EromOutputKind::SourceTemperature, model.reference_temperature, std::nullopt, source_id},
                    source.volume, 1 / source.total_volume);
        }
        result.L.resize(static_cast<Eigen::Index>(result.outputs.size()), n);
        result.L.setFromTriplets(weights.begin(), weights.end());
        return result;
    }
} // namespace mhs::app::erom
