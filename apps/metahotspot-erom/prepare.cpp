#include "extraction.hpp"
#include "solver/assembler.hpp"
#include "solver/thermal_reconstruction.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace mhs::app::erom {
    using namespace mhs::model;
    namespace {
        using Weights = core::Vector<Eigen::Triplet<double>>;

        double constant(const expression::CompiledExpression& e, std::string_view name)
        {
            if (!e.is_constant() || !std::isfinite(e.constant_value()))
                throw std::invalid_argument(std::string(name) + " must be constant and finite for EROM extraction");
            return e.constant_value();
        }

        void project_probe(
            const Fom& model, const HtcRanges& ranges, OutputId id, core::Vector<double>& state, Weights& weights)
        {
            const auto& probe = model.point_probes[id];
            if (probe.cell == core::invalid_id)
                throw std::invalid_argument("cannot retain point probe outside the FOM: " + probe.name);
            const auto facets = model.discrete.incident_facets(probe.cell);
            const auto& mesh = model.discrete;
            const auto center = mesh.cells[probe.cell].center();
            const bool on_boundary = std::ranges::any_of(facets,
                [&](FacetId f) { return mesh.cell_facets[f].exterior() && mesh.facets[f].contains(probe.point); });
            std::set<CellId> support {probe.cell};
            for (const auto f : facets) {
                const auto& cells = mesh.cell_facets[f];
                const auto& facet = mesh.facets[f];
                if (!cells.exterior()) {
                    support.insert(cells.minus == probe.cell ? cells.plus : cells.minus);
                    continue;
                }
                const auto axis = static_cast<std::size_t>(facet.axis);
                if (ranges.contains(model.facet_surface[f])
                    && (on_boundary ? facet.contains(probe.point) : probe.point[axis] != center[axis]))
                    throw std::invalid_argument("probe " + probe.name
                        + " depends directly on HTC boundary temperature and cannot be represented by L");
            }
            const solver::BoundarySampler adiabatic
                = [&](FacetId, const Vec3*) { return solver::BoundaryEvaluation {.temperature = state[probe.cell]}; };
            for (const auto cell : support) {
                state[cell] = 1;
                weights.emplace_back(
                    id, cell, solver::sample_point(model, state, 0, probe.cell, probe.point, adiabatic));
                state[cell] = 0;
            }
        }

        bool exterior_region(const DiscreteMesh& mesh, const SurfaceRegion& region)
        {
            const auto& r = region.rectangle;
            double covered = 0;
            for (std::size_t i = 0; i < mesh.facets.size(); ++i) {
                const auto& f = mesh.facets[i];
                if (f.axis != region.axis || std::abs(f.coordinate - region.coordinate) >= geometry_eps)
                    continue;
                const auto a = std::min(f.bounds[1], r.a_max) - std::max(f.bounds[0], r.a_min);
                const auto b = std::min(f.bounds[3], r.b_max) - std::max(f.bounds[2], r.b_min);
                if (a > 0 && b > 0) {
                    if (!mesh.cell_facets[i].exterior())
                        return false;
                    covered += a * b;
                }
            }
            return covered >= (r.a_max - r.a_min) * (r.b_max - r.b_min) * (1 - 1e-8);
        }

        core::Vector<model::Facet> outline(core::Vector<model::Facet> facets)
        {
            for (const int lo : {0, 2}) {
                const int other = 2 - lo;
                auto key = [other](const Facet& f) {
                    return std::tuple {f.axis, f.normal, f.coordinate, f.bounds[other], f.bounds[other + 1]};
                };
                std::ranges::stable_sort(facets, {}, [&](const Facet& f) { return std::pair {key(f), f.bounds[lo]}; });
                core::Vector<Facet> merged;
                for (const auto& f : facets) {
                    if (!merged.empty() && key(merged.back()) == key(f)
                        && merged.back().bounds[lo + 1] == f.bounds[lo]) {
                        merged.back().bounds[lo + 1] = f.bounds[lo + 1];
                        continue;
                    }
                    merged.push_back(f);
                }
                facets = std::move(merged);
            }
            return facets;
        }

    } // namespace

    Prepared prepare(const Fom& model, const FomDefinition& definition, const HtcRanges& ranges)
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

        core::Vector<core::Vector<FacetId>> facets_by_surface(model.surfaces.size());
        for (std::size_t id = 0; id < model.facet_surface.size(); ++id)
            if (const auto surface = model.facet_surface[id]; surface != core::invalid_id)
                facets_by_surface[surface].push_back(static_cast<FacetId>(id));
        for (const auto& [id, range] : ranges) {
            if (id >= model.surfaces.size())
                throw std::invalid_argument("unknown or empty HTC group " + std::to_string(id));
            if (id < definition.thermal_boundaries.size())
                for (const auto& region : definition.thermal_boundaries[id].regions)
                    if (!exterior_region(mesh, region))
                        throw std::invalid_argument("HTC group " + std::to_string(id) + " must lie on the exterior");
            if (facets_by_surface[id].empty())
                throw std::invalid_argument("unknown or empty HTC group " + std::to_string(id));
        }

        solver::ThermalAssembler assembler(model);
        core::Vector<double> state(mesh.cells.size(), model.reference_temperature);
        const auto& operators = assembler.assemble_internal(state, 0);
        Prepared result;
        const auto n = static_cast<Eigen::Index>(mesh.cells.size());
        result.K = Eigen::Map<const Sparse>(n, n, static_cast<Eigen::Index>(operators.K.nnz()),
            operators.K.row_ptr().data(), operators.K.col_idx().data(), operators.K.values().data());
        result.capacity = Eigen::Map<const Vector>(operators.C.values().data(), n);
        auto& asset = result.asset;
        asset.reference_temperature = model.reference_temperature;
        for (const bool trained : {true, false})
            for (SurfaceId surface = 0; surface < model.surfaces.size(); ++surface) {
                const auto& facets = facets_by_surface[surface];
                const auto range = ranges.find(surface);
                if (facets.empty() || trained != (range != ranges.end()))
                    continue;
                EromSurface target {surface, model.surfaces[surface].name, std::nullopt};
                const auto& condition = model.surfaces[surface].condition;
                const auto* htc = std::get_if<Convection>(&condition);
                const double h = htc ? constant(htc->coefficient, "HTC") : 0;
                if (h < 0 || (h > 0 && !trained))
                    throw std::invalid_argument(
                        "surface " + std::to_string(surface) + " has an untrained or negative HTC");
                if (const auto* flux = std::get_if<Neumann>(&condition);
                    !htc && (!flux || constant(flux->heat_flux, "Neumann heat flux") != 0))
                    throw std::invalid_argument("surface " + std::to_string(surface)
                        + " has a Dirichlet or nonzero Neumann condition unsupported by EROM extraction");

                if (trained) {
                    target.training = {range->second.minimum, range->second.maximum, h,
                        htc ? constant(htc->ambient_temperature, "Ambient temperature") : asset.reference_temperature};
                    for (const auto id : facets) {
                        asset.facets.push_back(mesh.facets[id]);
                        result.boundary_cells.push_back(mesh.cell_facets[id].owner());
                        asset.R.push_back(assembler.resistance(id));
                    }
                }
                else {
                    core::Vector<Facet> geometry;
                    for (const auto id : facets)
                        geometry.push_back(mesh.facets[id]);
                    const auto merged = outline(std::move(geometry));
                    asset.facets.insert(asset.facets.end(), merged.begin(), merged.end());
                }
                asset.facet_surface.resize(asset.facets.size(), static_cast<SurfaceId>(asset.surfaces.size()));
                asset.surfaces.push_back(std::move(target));
            }

        Weights weights;
        std::ranges::fill(state, 0);
        for (std::size_t i = 0; i < model.point_probes.size(); ++i) {
            const auto id = static_cast<OutputId>(i);
            project_probe(model, ranges, id, state, weights);
            asset.outputs.push_back({id, model.point_probes[i].name, model.point_probes[i].point});
        }

        core::Vector<double> volumes(model.heat_source_table.size());
        for (const auto& term : model.source_terms)
            volumes[term.source] += mesh.cells[term.cell].volume();
        core::Vector<SourceId> columns(volumes.size(), core::invalid_id);
        for (SourceId id = 0; id < volumes.size(); ++id) {
            if (volumes[id] == 0)
                continue;
            const double density = constant(model.heat_source_table[id], "Volumetric source");
            if (density == 0)
                continue;
            auto name = definition.source_names.empty() ? std::string {} : definition.source_names[id];
            if (name.empty())
                name = "source " + std::to_string(id);
            columns[id] = static_cast<SourceId>(asset.sources.size());
            asset.outputs.push_back({static_cast<OutputId>(asset.outputs.size()), name + " temperature", id});
            asset.sources.push_back({id, std::move(name), density * volumes[id]});
        }

        result.B = Matrix::Zero(n, static_cast<Eigen::Index>(asset.sources.size()));
        for (const auto& term : model.source_terms)
            if (const auto col = columns[term.source]; col != core::invalid_id) {
                const double weight = mesh.cells[term.cell].volume() / volumes[term.source];
                result.B(term.cell, col) += weight;
                weights.emplace_back(model.point_probes.size() + col, term.cell, weight);
            }
        result.L.resize(static_cast<Eigen::Index>(asset.outputs.size()), n);
        result.L.setFromTriplets(weights.begin(), weights.end());
        return result;
    }
} // namespace mhs::app::erom
