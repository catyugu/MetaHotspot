#include "solver/assembler.hpp"

#include "thermal_physics.hpp"
#include "thermal_reconstruction.hpp"
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

namespace mhs::solver {
    namespace {
        using namespace model;
        using core::Vector;

        struct Dofs {
            std::size_t first, count;
        };

        struct Block {
            const Model* fom = nullptr;
            const Erom* erom = nullptr;
            std::unique_ptr<ThermalAssembler> thermal;
            Vector<BoundaryCondition> erom_defaults;
        };

        std::size_t slot(const core::CSRMatrix<double>& matrix, std::size_t row, std::size_t col)
        {
            const auto columns = matrix.col_idx();
            return std::lower_bound(
                       columns.begin() + matrix.row_ptr()[row], columns.begin() + matrix.row_ptr()[row + 1], col)
                - columns.begin();
        }

        struct Side {
            Dofs dofs;
            std::span<const double> modes;
            double reference;

            double value(std::size_t i) const { return modes.empty() ? 1.0 : modes[i]; }

            double temperature(std::span<const double> state) const
            {
                const auto values = state.subspan(dofs.first, dofs.count);
                return reference
                    + (modes.empty() ? values.front()
                                     : std::inner_product(modes.begin(), modes.end(), values.begin(), 0.0));
            }
        };

        void add_block(core::CSRMatrix<double>& matrix, const Side& rows, const Side& cols, double scale)
        {
            for (std::size_t i = 0; i < rows.dofs.count; ++i) {
                const auto idx = slot(matrix, rows.dofs.first + i, cols.dofs.first);
                const auto factor = scale * rows.value(i);
                for (std::size_t j = 0; j < cols.dofs.count; ++j)
                    matrix.values()[idx + j] += factor * cols.value(j);
            }
        }

        void add_load(Vector<double>& rhs, const Side& rows, double scale)
        {
            for (std::size_t i = 0; i < rows.dofs.count; ++i)
                rhs[rows.dofs.first + i] += scale * rows.value(i);
        }

        void copy_sparse(const core::CSRMatrix<double>& local, core::CSRMatrix<double>& global, std::size_t offset)
        {
            for (std::size_t r = 0; r < local.rows(); ++r) {
                const auto row = r + offset;
                auto pos = global.row_ptr()[row];
                for (auto i = local.row_ptr()[r]; i < local.row_ptr()[r + 1]; ++i) {
                    const auto col = local.col_idx()[i] + static_cast<core::SparseIndex>(offset);
                    while (global.col_idx()[pos] != col)
                        ++pos;
                    global.values()[pos++] = local.values()[i];
                }
            }
        }

        void copy_dense(const core::Matrix<double>& local, core::CSRMatrix<double>& global, std::size_t offset)
        {
            for (std::size_t r = 0; r < local.rows(); ++r)
                std::ranges::copy(local.row(r), global.values().begin() + slot(global, r + offset, offset));
        }

        const BoundaryCondition& boundary_override(
            const Component& c, SurfaceId surface, const BoundaryCondition& fallback)
        {
            const auto it = c.boundaries.find(surface);
            return it == c.boundaries.end() ? fallback : it->second;
        }
    } // namespace

    struct ProjectAssembler::Workspace {
        explicit Workspace(const Project& source) : project(source)
        {
            for (const auto& component : project.components()) {
                auto& block = blocks.emplace_back();
                if (const auto* model = std::get_if<std::shared_ptr<const Model>>(&component.asset)) {
                    block.fom = model->get();
                    block.thermal = std::make_unique<ThermalAssembler>(*block.fom);
                }
                else {
                    block.erom = std::get<std::shared_ptr<const Erom>>(component.asset).get();
                    for (const auto& htc : block.erom->htc) {
                        using expression::CompiledExpression;
                        if (htc)
                            block.erom_defaults.push_back(
                                Convection {CompiledExpression::make_constant(htc->coefficient),
                                    CompiledExpression::make_constant(htc->ambient)});
                        else
                            block.erom_defaults.emplace_back(Neumann {});
                    }
                }
            }

            for (const auto& contact : project.connections()) {
                auto record = [&](std::size_t component, FacetId facet) {
                    if (blocks[component].fom && !blocks[component].fom->point_probes.empty())
                        sampled_contacts[{component, facet}].push_back(&contact);
                };
                record(contact.left, contact.left_facet);
                record(contact.right, contact.right_facet);
            }
            single_fom = blocks.size() == 1 && blocks.front().fom;
            if (single_fom)
                return;
            operators = {core::CSRMatrix<double>(pattern(false)), core::CSRMatrix<double>(pattern(true)),
                Vector<double>(project.dof_count())};
            for (std::size_t i = 0; i < blocks.size(); ++i)
                if (const auto* erom = blocks[i].erom)
                    copy_dense(erom->C, operators.C, project.dof_offset(i));
        }

        Side side(std::size_t i, FacetId facet) const
        {
            const auto& b = blocks[i];
            const auto offset = project.dof_offset(i);
            if (b.fom)
                return {{offset + b.fom->discrete.cell_facets[facet].owner(), 1}, {}, 0};
            return {{offset, b.erom->K.rows()}, b.erom->Vb.row(facet), b.erom->reference_temperature};
        }

        std::shared_ptr<const core::CSRPattern> pattern(bool capacity) const
        {
            using core::SparseIndex;
            const auto size = project.dof_count();
            Vector<Vector<SparseIndex>> columns(size);
            auto dense = [&](Dofs rows, Dofs cols) {
                for (auto i = rows.first; i < rows.first + rows.count; ++i)
                    for (auto j = cols.first; j < cols.first + cols.count; ++j)
                        columns[i].push_back(static_cast<SparseIndex>(j));
            };
            for (std::size_t i = 0; i < blocks.size(); ++i) {
                const auto offset = project.dof_offset(i);
                if (const auto& thermal = blocks[i].thermal) {
                    const auto& matrix = capacity ? thermal->operators().C : thermal->operators().K;
                    for (std::size_t r = 0; r < matrix.rows(); ++r)
                        for (auto j = matrix.row_ptr()[r]; j < matrix.row_ptr()[r + 1]; ++j)
                            columns[offset + r].push_back(static_cast<SparseIndex>(offset) + matrix.col_idx()[j]);
                }
                else {
                    const Dofs modes {offset, project.dof_size(i)};
                    dense(modes, modes);
                }
            }
            if (!capacity) {
                std::set<std::array<std::size_t, 4>> seen;
                for (const auto& c : project.connections()) {
                    const auto l = side(c.left, c.left_facet).dofs;
                    const auto r = side(c.right, c.right_facet).dofs;
                    if (seen.insert({l.first, l.count, r.first, r.count}).second) {
                        dense(l, r);
                        dense(r, l);
                    }
                }
            }
            Vector<SparseIndex> ptr(size + 1), indices;
            for (std::size_t i = 0; i < size; ++i) {
                auto& row = columns[i];
                std::ranges::sort(row);
                row.erase(std::unique(row.begin(), row.end()), row.end());
                core::detail::checked_sparse_size(indices.size() + row.size());
                indices.insert(indices.end(), row.begin(), row.end());
                ptr[i + 1] = static_cast<SparseIndex>(indices.size());
            }
            return std::make_shared<const core::CSRPattern>(size, size, std::move(ptr), std::move(indices));
        }

        double resistance(std::size_t i, FacetId facet) const
        {
            const auto& b = blocks[i];
            return b.fom ? b.thermal->resistance(facet) : b.erom->R[facet];
        }

        void environment(Operators& ops, std::size_t i, FacetId id, const Facet& facet, const BoundaryCondition& law,
            std::span<const double> state, double time) const
        {
            if (const auto* flux = std::get_if<Neumann>(&law);
                flux && flux->heat_flux.is_constant() && flux->heat_flux.constant_value() == 0)
                return;
            const auto dofs = side(i, id);
            const auto r = resistance(i, id);
            const auto temperature = dofs.temperature(state);
            auto apply = [&](const Facet& patch) {
                const auto p = patch.center();
                const double area = patch.area();
                const auto trace = evaluate_boundary(law, {p[0], p[1], p[2], temperature, time}, area, area / r);
                add_block(ops.K, dofs, dofs, trace.conductance);
                add_load(ops.f, dofs, trace.load - trace.conductance * dofs.reference);
            };
            const auto& exposed = project.exposed(i);
            if (const auto it = exposed.find(id); it != exposed.end()) {
                for (const auto& patch : it->second)
                    apply(patch);
            }
            else
                apply(facet);
        }

        void fom_environment(Operators& ops, std::size_t i, std::span<const double> state, double time) const
        {
            const auto& model = *blocks[i].fom;
            const auto& component = project.components()[i];
            for (std::size_t j = 0; j < model.facet_thermal_condition.size(); ++j) {
                const auto condition = model.facet_thermal_condition[j];
                if (condition != core::invalid_id)
                    environment(ops, i, static_cast<FacetId>(j), model.discrete.facets[j],
                        boundary_override(component, condition, model.boundary_conditions[condition]), state, time);
            }
        }

        const Operators& assemble(std::span<const double> state, double time)
        {
            if (state.size() != project.dof_count())
                throw std::invalid_argument("project assembly: state size differs from DOF count");
            if (single_fom) {
                auto& local = blocks.front().thermal->assemble_internal(state, time);
                fom_environment(local, 0, state, time);
                return local;
            }

            std::ranges::fill(operators.K.values(), 0.0);
            std::ranges::fill(operators.f, 0.0);
            for (std::size_t i = 0; i < blocks.size(); ++i) {
                const auto& b = blocks[i];
                const auto& component = project.components()[i];
                const auto offset = project.dof_offset(i);
                if (b.fom) {
                    const auto& local = b.thermal->assemble_internal(state.subspan(offset, project.dof_size(i)), time);
                    copy_sparse(local.K, operators.K, offset);
                    copy_sparse(local.C, operators.C, offset);
                    std::ranges::copy(local.f, operators.f.begin() + offset);
                    fom_environment(operators, i, state, time);
                    continue;
                }

                const auto& e = *b.erom;
                copy_dense(e.K, operators.K, offset);
                for (std::size_t s = 0; s < e.sources.size(); ++s) {
                    const auto& source = e.sources[s];
                    const auto it = component.powers.find(source.id);
                    const double power = it == component.powers.end() ? source.default_power : it->second(time);
                    if (!std::isfinite(power))
                        throw std::invalid_argument("EROM power must be finite");
                    for (std::size_t j = 0; j < e.K.rows(); ++j)
                        operators.f[offset + j] += e.B(j, s) * power;
                }
                for (std::size_t s = 0; s < e.surfaces.size(); ++s)
                    if (e.htc[s]) {
                        const auto& surface = e.surfaces[s];
                        const auto& law = boundary_override(component, surface.id, b.erom_defaults[s]);
                        for (const auto id : surface.facets)
                            environment(operators, i, id, e.facets[id], law, state, time);
                    }
            }

            for (const auto& connection : project.connections()) {
                const auto& c = connection;
                const auto left = side(c.left, c.left_facet), right = side(c.right, c.right_facet);
                const auto g
                    = c.area / (resistance(c.left, c.left_facet) + resistance(c.right, c.right_facet) + c.resistance);
                add_block(operators.K, left, left, g);
                add_block(operators.K, right, right, g);
                add_block(operators.K, left, right, -g);
                add_block(operators.K, right, left, -g);
                const auto load = g * (right.reference - left.reference);
                add_load(operators.f, left, load);
                add_load(operators.f, right, -load);
            }
            return operators;
        }

        double sample_fom(std::size_t i, const PointProbe& probe, std::span<const double> state, double time) const
        {
            const auto& m = *blocks[i].fom;
            const auto local = state.subspan(project.dof_offset(i), project.dof_size(i));
            const auto& component = project.components()[i];
            if (project.exposed(i).empty() && component.boundaries.empty())
                return sample_point(m, local, time, probe.cell, probe.point);
            auto sample_r = [&](std::size_t index, FacetId id) {
                const auto& block = blocks[index];
                if (block.erom)
                    return block.erom->R[id];
                const auto& fom = *block.fom;
                const auto axis = fom.discrete.facets[id].axis;
                const auto cell = fom.discrete.cell_facets[id].owner();
                const auto& material = fom.material_table[fom.cell_material[cell]];
                const auto& k = axis == Axis::X ? material.kx : axis == Axis::Y ? material.ky : material.kz;
                const auto& geometry = fom.discrete.cells[cell];
                const auto p = geometry.center();
                return side_thermal_resistance(geometry.size()[static_cast<std::size_t>(axis)] / 2,
                    k.eval({p[0], p[1], p[2], state[project.dof_offset(index) + cell], time}));
            };
            BoundarySampler sample = [&](FacetId id, const Vec3* point) -> BoundarySample {
                const auto& facet = m.discrete.facets[id];
                const auto r = sample_r(i, id);
                const double temperature = side(i, id).temperature(state);
                const auto condition = m.facet_thermal_condition[id];
                const auto& law = boundary_override(component, condition, m.boundary_conditions[condition]);
                auto ambient = [&](const Facet& patch) {
                    const auto p = patch.center();
                    const auto area = patch.area();
                    const auto t = evaluate_boundary(law, {p[0], p[1], p[2], temperature, time}, area, area / r);
                    return BoundarySample {t.temperature, std::holds_alternative<Dirichlet>(law)};
                };
                const auto exposed = project.exposed(i).find(id);
                if (exposed == project.exposed(i).end())
                    return ambient(facet);
                double integral = 0;
                for (const auto* c : sampled_contacts.at({i, id})) {
                    const auto other = c->left == i ? c->right : c->left;
                    const auto other_id = c->left == i ? c->right_facet : c->left_facet;
                    const auto difference = side(other, other_id).temperature(state) - temperature;
                    const auto t = temperature + difference * r / (r + sample_r(other, other_id) + c->resistance);
                    if (point) {
                        const auto u = facet.axis == Axis::X ? 1 : 0, v = facet.axis == Axis::Z ? 1 : 2;
                        const auto a = (*point)[u] + component.offset[u], b = (*point)[v] + component.offset[v];
                        if (a >= c->overlap[0] - geometry_eps && a <= c->overlap[1] + geometry_eps
                            && b >= c->overlap[2] - geometry_eps && b <= c->overlap[3] + geometry_eps)
                            return {t, false};
                    }
                    integral += c->area * t;
                }
                for (const auto& patch : exposed->second) {
                    const auto trace = ambient(patch);
                    if (point && patch.contains(*point))
                        return trace;
                    integral += patch.area() * trace.temperature;
                }
                return {integral / facet.area(), false};
            };
            return sample_point(m, local, time, probe.cell, probe.point, sample);
        }

        const Project& project;
        std::map<std::pair<std::size_t, FacetId>, Vector<const Connection*>> sampled_contacts;
        Vector<Block> blocks;
        Operators operators;
        bool single_fom = false;
    };

    ProjectAssembler::ProjectAssembler(const model::Project& project) : workspace_(std::make_unique<Workspace>(project))
    {
    }
    ProjectAssembler::~ProjectAssembler() = default;

    const Operators& ProjectAssembler::assemble(std::span<const double> state, double time)
    {
        return workspace_->assemble(state, time);
    }

    double ProjectAssembler::sample_fom(
        std::size_t component, const model::PointProbe& probe, std::span<const double> state, double time) const
    {
        return workspace_->sample_fom(component, probe, state, time);
    }

    Operators ProjectAssembler::take_operators() &&
    {
        if (workspace_->single_fom)
            return std::move(*workspace_->blocks.front().thermal).take_operators();
        return std::move(workspace_->operators);
    }

    model::Project single_model_project(const model::Model& model)
    {
        // The legacy synchronous call borrows the caller's model for its entire lifetime.
        auto borrowed = std::shared_ptr<const model::Model>(std::shared_ptr<const model::Model> {}, &model);
        return model::compile_project({{{0, {}, std::move(borrowed), {}, {}}}, {}});
    }

    Operators assemble_thermal(const model::Model& model, std::span<const double> temperature, double time)
    {
        const auto project = single_model_project(model);
        ProjectAssembler assembler(project);
        assembler.assemble(temperature, time);
        return std::move(assembler).take_operators();
    }
} // namespace mhs::solver
