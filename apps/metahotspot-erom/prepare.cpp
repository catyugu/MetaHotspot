#include "compiler/fom_compiler.hpp"
#include "compiler/model_functions.hpp"
#include "extraction.hpp"
#include "solver/assembler.hpp"
#include "solver/thermal_physics.hpp"
#include "solver/thermal_reconstruction.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mhs::app::erom {
    using namespace mhs::model;
    namespace {
        using Weights = core::Vector<Eigen::Triplet<double>>;

        double field_value(const expression::CompiledExpression& field, const expression::FieldContext& ctx)
        {
            const double value = field.eval(ctx);
            if (!std::isfinite(value))
                throw std::invalid_argument("EROM extraction requires finite frozen field values");
            return value;
        }

        Fom compile_input(const FomDefinition& definition, const MeshSpec& mesh)
        {
            // Only this temporary definition is adapted for extraction. The FOM
            // compiler's contracts (including explicit Dirichlet data) stay unchanged.
            auto input = definition;
            for (const auto& volume : input.volumes) {
                if (volume.material >= input.materials.size())
                    throw std::out_of_range("volume material ID out of range");
                if (input.materials[volume.material].value.kind == MaterialKind::Fluid)
                    throw std::invalid_argument("EROM extraction does not support fluid blocks");
            }
            for (auto& material : input.materials)
                if (material.value.kind == MaterialKind::Fluid) {
                    // Unreferenced fluid records need no hydraulic properties.
                    material.value.kind = MaterialKind::Solid;
                    material.value.dynamic_viscosity.reset();
                }
            for (auto& source : input.sources)
                source.density = "0";
            auto clear_dirichlet = [](ThermalBoundary& law) {
                if (auto* value = std::get_if<DirichletBoundary>(&law))
                    value->temperature = "0";
            };
            for (auto& group : input.thermal_boundaries)
                clear_dirichlet(group.condition);
            clear_dirichlet(input.default_thermal_boundary);
            return compiler::compile_fom(input, mesh);
        }

        void freeze_probe_conductivity(Fom& model)
        {
            // sample_point is evaluated on unit temperature-rise states. Give
            // its existing Green-Gauss stencil constant conductivities sampled
            // at the reference state, without touching the recovery kernel.
            auto freeze = [&](CellId cell) {
                auto material = model.material_table[model.cell_material[cell]];
                if (material.kx.is_constant() && material.ky.is_constant() && material.kz.is_constant())
                    return;
                const auto p = model.mesh.cell(cell).center();
                const expression::FieldContext ctx {p[0], p[1], p[2], model.reference_temperature, 0};
                for (auto* k : {&material.kx, &material.ky, &material.kz})
                    *k = expression::CompiledExpression::make_constant(field_value(*k, ctx));
                if (model.material_table.size() >= core::invalid_id)
                    throw std::length_error("too many frozen probe materials");
                model.cell_material[cell] = static_cast<MaterialId>(model.material_table.size());
                model.material_table.push_back(std::move(material));
            };
            for (const auto& probe : model.point_probes) {
                if (probe.cell == core::invalid_id)
                    throw std::invalid_argument("cannot retain point probe outside the FOM: " + probe.name);
                freeze(probe.cell);
                for (std::size_t axis = 0; axis < 3; ++axis)
                    for (int sign : {-1, 1})
                        if (const auto neighbor = model.mesh.neighbor(probe.cell, axis, sign);
                            neighbor != core::invalid_id)
                            freeze(neighbor);
            }
        }

        struct FrozenTrace {
            solver::BoundaryEvaluation evaluation; // at zero temperature rise
            double slope; // derivative with respect to the cell temperature rise
        };

        double project_probe(const Fom& model, const HtcRanges& ranges, OutputId id, core::Vector<double>& state,
            const core::Vector<FrozenTrace>& frozen, Weights& weights)
        {
            const auto& probe = model.point_probes[id];
            const auto& mesh = model.mesh;
            const auto facets = mesh.boundary_ids(probe.cell);
            const auto center = mesh.cell(probe.cell).center();
            const bool on_boundary = std::ranges::any_of(
                facets, [&](FacetId f) { return mesh.exterior_facets[f].geometry.contains(probe.point); });
            const bool prescribed = std::ranges::any_of(facets, [&](FacetId f) {
                return frozen[f].evaluation.dirichlet && mesh.exterior_facets[f].geometry.contains(probe.point);
            });
            for (const auto f : facets) {
                const auto& face = mesh.exterior_facets[f];
                const auto& facet = face.geometry;
                const auto axis = static_cast<std::size_t>(facet.axis);
                if (ranges.contains(face.boundary_group)
                    && (on_boundary ? !prescribed && facet.contains(probe.point) : probe.point[axis] != center[axis]))
                    throw std::invalid_argument("probe " + probe.name
                        + " depends directly on HTC boundary temperature and cannot be represented by L");
            }
            std::set<CellId> support {probe.cell};
            if (!on_boundary)
                for (std::size_t axis = 0; axis < 3; ++axis)
                    for (int sign : {-1, 1})
                        if (const auto neighbor = mesh.neighbor(probe.cell, axis, sign); neighbor != core::invalid_id)
                            support.insert(neighbor);
            const solver::BoundarySampler boundary = [&](FacetId f, const Vec3*) {
                auto value = frozen[f].evaluation;
                value.temperature += frozen[f].slope * state[mesh.exterior_facets[f].owner];
                return value;
            };
            const auto sample
                = [&] { return solver::sample_point(model, state, 0, probe.cell, probe.point, boundary); };
            const double offset = sample();
            for (const auto cell : support) {
                state[cell] = 1;
                const double weight = sample() - offset;
                state[cell] = 0;
                if (!std::isfinite(weight))
                    throw std::invalid_argument("invalid frozen reconstruction for probe " + probe.name);
                weights.emplace_back(id, cell, weight);
            }
            if (!std::isfinite(offset))
                throw std::invalid_argument("invalid frozen reconstruction for probe " + probe.name);
            return offset;
        }

        bool on_box_face(const Facet& face, const CellGeometry& box)
        {
            const auto axis = static_cast<std::size_t>(face.axis);
            // Compiled facet planes and active cells use the same mesh coordinates.
            return face.coordinate == (face.normal < 0 ? box.min[axis] : box.max[axis]);
        }

        bool covered_by_box_faces(const ConformalMesh& mesh, const SurfaceRegion& region, const CellGeometry& box)
        {
            double covered = 0;
            for (const auto& face : mesh.exterior_facets) {
                const auto& f = face.geometry;
                if (f.axis != region.axis || std::abs(f.coordinate - region.coordinate) >= geometry_eps)
                    continue;
                if (const auto hit = rect_intersection(f.bounds, region.bounds)) {
                    if (!on_box_face(f, box))
                        return false;
                    covered += rect_area(*hit);
                }
            }
            return covered >= rect_area(region.bounds) * (1 - 1e-8);
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

    Prepared prepare(const FomDefinition& definition, const MeshSpec& input_mesh, const HtcRanges& ranges)
    {
        auto model = compile_input(definition, input_mesh);
        const auto& mesh = model.mesh;
        if (mesh.cell_count() == 0 || !model.fluid.fluid_to_global.empty())
            throw std::invalid_argument("EROM extraction requires a nonempty solid FOM");

        for (CellId id = 0; id < mesh.cell_count(); ++id) {
            const auto& material = model.material_table[model.cell_material[id]];
            const auto center = mesh.cell(id).center();
            const expression::FieldContext ctx {center[0], center[1], center[2], model.reference_temperature, 0};
            for (const auto* property : {&material.kx, &material.ky, &material.kz, &material.rho, &material.c})
                if (field_value(*property, ctx) <= 0)
                    throw std::invalid_argument("EROM extraction requires positive frozen material properties");
        }

        core::Vector<core::Vector<FacetId>> facets_by_surface(model.boundary_groups.size());
        for (FacetId id = 0; id < mesh.exterior_facets.size(); ++id)
            facets_by_surface[mesh.exterior_facets[id].boundary_group].push_back(id);
        auto bounds = mesh.cell(0);
        for (std::size_t c = 1; c < mesh.cell_count(); ++c) {
            const auto cell = mesh.cell(c);
            for (std::size_t axis = 0; axis < bounds.min.size(); ++axis) {
                bounds.min[axis] = std::min(bounds.min[axis], cell.min[axis]);
                bounds.max[axis] = std::max(bounds.max[axis], cell.max[axis]);
            }
        }
        for (const auto& [id, range] : ranges) {
            if (id >= definition.thermal_boundaries.size() || facets_by_surface[id].empty())
                throw std::invalid_argument("unknown or empty HTC group " + std::to_string(id));
            if (!std::holds_alternative<Convection>(model.boundary_groups[id].condition))
                throw std::invalid_argument("HTC training requires an explicit convection boundary group");
            const bool invalid_regions = std::ranges::any_of(definition.thermal_boundaries[id].regions,
                [&](const auto& region) { return !covered_by_box_faces(mesh, region, bounds); });
            if (invalid_regions || std::ranges::any_of(facets_by_surface[id], [&](FacetId f) {
                    return !on_box_face(mesh.exterior_facets[f].geometry, bounds);
                }))
                throw std::invalid_argument(
                    "HTC group " + std::to_string(id) + " must lie entirely on the faces of the FOM bounding box");
        }

        freeze_probe_conductivity(model);
        solver::ThermalAssembler assembler(model);
        core::Vector<double> state(mesh.cell_count(), model.reference_temperature);
        const auto& operators = assembler.assemble_internal(state, 0);
        Prepared result {};
        const auto n = static_cast<Eigen::Index>(mesh.cell_count());
        result.K = Eigen::Map<const Sparse>(n, n, static_cast<Eigen::Index>(operators.K.nnz()),
            operators.K.row_ptr().data(), operators.K.col_idx().data(), operators.K.values().data());
        result.capacity = Eigen::Map<const Vector>(operators.C.values().data(), n);
        result.reference_temperature = model.reference_temperature;
        result.origin = bounds.min;
        result.fixed_diagonal = Vector::Zero(n);
        result.fixed_load = Vector::Zero(n);
        core::Vector<FrozenTrace> frozen(mesh.exterior_facets.size());
        std::optional<expression::SymbolTable> dirichlet_symbols;
        for (BoundaryGroupId surface = 0; surface < model.boundary_groups.size(); ++surface) {
            const auto& facets = facets_by_surface[surface];
            const auto range = ranges.find(surface);
            auto condition = model.boundary_groups[surface].condition;
            const auto& original = surface < definition.thermal_boundaries.size()
                ? definition.thermal_boundaries[surface].condition
                : definition.default_thermal_boundary;
            // A temperature-dependent prescribed value is meaningful only after
            // extraction freezes it. Compile the original field here, not in Fom.
            if (const auto* prescribed = std::get_if<DirichletBoundary>(&original)) {
                if (!dirichlet_symbols) {
                    dirichlet_symbols.emplace();
                    compiler::register_all_functions(*dirichlet_symbols, definition.functions);
                }
                condition = Dirichlet {expression::parse(prescribed->temperature, *dirichlet_symbols)};
            }
            if (facets.empty())
                continue;
            const bool trained = range != ranges.end();
            const auto& name = model.boundary_groups[surface].name;
            const auto group = static_cast<BoundaryGroupId>(result.training.size());
            const auto begin = static_cast<Eigen::Index>(result.interfaces.size());
            core::Vector<Facet> geometry;
            for (const auto id : facets) {
                const auto& face = mesh.exterior_facets[id];
                const auto& facet = face.geometry;
                const auto center = facet.center();
                const expression::FieldContext ctx {center[0], center[1], center[2], result.reference_temperature, 0};
                const double resistance = assembler.resistance(id);
                if (!std::isfinite(resistance) || resistance <= 0)
                    throw std::invalid_argument("invalid frozen boundary resistance");
                if (trained) {
                    const auto& htc = std::get<Convection>(condition);
                    const double h = field_value(htc.coefficient, ctx);
                    const double ambient = field_value(htc.ambient_temperature, ctx);
                    if (h < 0)
                        throw std::invalid_argument("negative frozen HTC");
                    result.interfaces.push_back({facet, resistance, group, h, ambient});
                    result.boundary_cells.push_back(face.owner);
                    // A retained probe cannot depend on this trace. Use the
                    // cell value for irrelevant trained facets during reconstruction.
                    frozen[id] = {{}, 1.0};
                    continue;
                }
                using expression::CompiledExpression;
                auto constant = [](double value) { return CompiledExpression::make_constant(value); };
                // Freeze each field at the original facet center. Work directly
                // in temperature-rise coordinates to avoid subtracting absolute loads.
                const auto law = std::visit(
                    [&](const auto& value) -> BoundaryCondition {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, Dirichlet>)
                            return Dirichlet {
                                constant(field_value(value.temperature, ctx) - result.reference_temperature)};
                        else if constexpr (std::is_same_v<T, Neumann>)
                            return Neumann {constant(field_value(value.heat_flux, ctx))};
                        else {
                            const double h = field_value(value.coefficient, ctx);
                            if (h < 0)
                                throw std::invalid_argument("negative frozen HTC");
                            return Convection {constant(h),
                                constant(field_value(value.ambient_temperature, ctx) - result.reference_temperature)};
                        }
                    },
                    condition);
                const auto evaluation = solver::evaluate_boundary(law, facet, 0, 0, resistance);
                if (!std::isfinite(evaluation.conductance) || evaluation.conductance < 0
                    || !std::isfinite(evaluation.load) || !std::isfinite(evaluation.temperature))
                    throw std::invalid_argument("invalid frozen fixed boundary");
                // The condensed trace is affine with slope 1 - g R / A.
                const double slope
                    = evaluation.dirichlet ? 0.0 : 1 - evaluation.conductance / (facet.area() / resistance);
                frozen[id] = {evaluation, slope};
                result.fixed_diagonal[face.owner] += evaluation.conductance;
                result.fixed_load[face.owner] += evaluation.load;
                geometry.push_back(facet);
            }
            if (trained)
                result.training.push_back({{name, range->second.minimum, range->second.maximum}, begin,
                    static_cast<Eigen::Index>(result.interfaces.size())});
            else
                result.outline.push_back({name, outline(std::move(geometry))});
        }
        if (!result.fixed_diagonal.allFinite() || !result.fixed_load.allFinite())
            throw std::invalid_argument("nonfinite fixed boundary contribution");
        result.K = add_diagonal(result.K, result.fixed_diagonal);
        for (Eigen::Index i = 0; i < result.K.nonZeros(); ++i)
            if (!std::isfinite(result.K.valuePtr()[i]))
                throw std::invalid_argument("nonfinite frozen stiffness");

        Weights weights;
        std::ranges::fill(state, 0);
        for (std::size_t i = 0; i < model.point_probes.size(); ++i) {
            const auto id = static_cast<OutputId>(i);
            result.output_offset.push_back(project_probe(model, ranges, id, state, frozen, weights));
            result.outputs.push_back({model.point_probes[i].name, model.point_probes[i].point});
        }

        core::Vector<double> volumes(model.heat_source_table.size());
        for (const auto& term : model.source_terms)
            volumes[term.source] += mesh.cell(term.cell).volume();
        core::Vector<SourceId> columns(volumes.size(), core::invalid_id);
        for (SourceId id = 0; id < volumes.size(); ++id) {
            if (volumes[id] == 0)
                continue;
            auto name = definition.sources[id].name;
            if (name.empty())
                name = "source " + std::to_string(id);
            columns[id] = static_cast<SourceId>(result.sources.size());
            result.outputs.push_back({name + " temperature", std::nullopt});
            result.sources.push_back({std::move(name)});
        }

        result.B = Matrix::Zero(n, static_cast<Eigen::Index>(result.sources.size()));
        for (const auto& term : model.source_terms)
            if (const auto col = columns[term.source]; col != core::invalid_id) {
                const double weight = mesh.cell(term.cell).volume() / volumes[term.source];
                result.B(term.cell, col) += weight;
                weights.emplace_back(model.point_probes.size() + col, term.cell, weight);
            }
        result.output_offset.resize(result.outputs.size(), 0.0);
        result.L.resize(static_cast<Eigen::Index>(result.outputs.size()), n);
        result.L.setFromTriplets(weights.begin(), weights.end());
        return result;
    }
} // namespace mhs::app::erom
