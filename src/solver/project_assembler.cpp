#include "solver/assembler.hpp"

#include "contact.hpp"
#include "thermal_physics.hpp"
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

namespace mhs::solver {
    namespace {
        using namespace model;
        using core::Vector;

        std::size_t slot(const core::CSRMatrix<double>& matrix, std::size_t row, std::size_t col)
        {
            const auto columns = matrix.col_idx();
            return std::lower_bound(
                       columns.begin() + matrix.row_ptr()[row], columns.begin() + matrix.row_ptr()[row + 1], col)
                - columns.begin();
        }

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
    } // namespace

    struct ProjectAssembler::Workspace {
        explicit Workspace(const Project& source) : project(source)
        {
            for (const auto& component : project.components()) {
                const auto* fom = std::get_if<std::shared_ptr<const Fom>>(&component.asset);
                thermal.push_back(fom ? std::make_unique<ThermalAssembler>(**fom) : nullptr);
            }
            if (thermal.size() == 1 && thermal.front())
                return;
            operators = {core::CSRMatrix<double>(pattern(false)), core::CSRMatrix<double>(pattern(true)),
                Vector<double>(project.dof_count())};
            for (std::size_t i = 0; i < thermal.size(); ++i)
                if (!thermal[i])
                    copy_dense(erom(i).C, operators.C, project.dof_offset(i));
            // Only reduced-only projects cache a global contact operator: a mixed
            // project's FOM rows would otherwise allocate a mostly empty copy.
            if (!project.contacts().empty()
                && std::ranges::none_of(thermal, [](const auto& assembler) { return bool(assembler); })) {
                assemble_contacts();
                cached_contact_K.assign(operators.K.values().begin(), operators.K.values().end());
                cached_contact_f = operators.f;
            }
        }

        void assemble_contacts()
        {
            for (const auto& c : project.contacts()) {
                const auto left = contact_side(project, c.left, c.left_facet);
                const auto right = contact_side(project, c.right, c.right_facet);
                const auto g = c.area()
                    / contact_resistance(
                        resistance(c.left, c.left_facet), resistance(c.right, c.right_facet), c.resistance);
                add_block(operators.K, left, left, g);
                add_block(operators.K, right, right, g);
                add_block(operators.K, left, right, -g);
                add_block(operators.K, right, left, -g);
                const auto load = g * (right.reference - left.reference);
                add_load(operators.f, left, load);
                add_load(operators.f, right, -load);
            }
        }

        std::shared_ptr<const core::CSRPattern> pattern(bool capacity) const
        {
            using core::SparseIndex;
            const auto size = project.dof_count();
            // Count and fill one flat buffer; avoid a synchronized pool allocation per row.
            auto visit_columns = [&](const auto& append) {
                auto dense = [&](Dofs rows, Dofs cols) {
                    for (auto i = rows.first; i < rows.first + rows.count; ++i)
                        for (auto j = cols.first; j < cols.first + cols.count; ++j)
                            append(i, static_cast<SparseIndex>(j));
                };
                for (std::size_t i = 0; i < thermal.size(); ++i) {
                    const auto offset = project.dof_offset(i);
                    if (const auto& assembler = thermal[i]) {
                        const auto& matrix = capacity ? assembler->operators().C : assembler->operators().K;
                        for (std::size_t r = 0; r < matrix.rows(); ++r)
                            for (auto j = matrix.row_ptr()[r]; j < matrix.row_ptr()[r + 1]; ++j)
                                append(offset + r, static_cast<SparseIndex>(offset) + matrix.col_idx()[j]);
                    }
                    else {
                        const Dofs modes {offset, project.dof_size(i)};
                        dense(modes, modes);
                    }
                }
                if (!capacity) {
                    std::set<std::array<std::size_t, 4>> seen;
                    for (const auto& c : project.contacts()) {
                        const auto l = contact_side(project, c.left, c.left_facet).dofs;
                        const auto r = contact_side(project, c.right, c.right_facet).dofs;
                        if (seen.insert({l.first, l.count, r.first, r.count}).second) {
                            dense(l, r);
                            dense(r, l);
                        }
                    }
                }
            };
            Vector<SparseIndex> ptr(size + 1);
            visit_columns([&](std::size_t row, SparseIndex) {
                ptr[row + 1] = static_cast<SparseIndex>(
                    core::detail::checked_sparse_size(static_cast<std::size_t>(ptr[row + 1]) + 1));
            });
            for (std::size_t i = 0; i < size; ++i)
                ptr[i + 1] = static_cast<SparseIndex>(
                    core::detail::checked_sparse_size(static_cast<std::size_t>(ptr[i]) + ptr[i + 1]));
            Vector<SparseIndex> indices(ptr.back());
            auto cursor = ptr;
            visit_columns([&](std::size_t row, SparseIndex col) { indices[cursor[row]++] = col; });
            SparseIndex first = 0, count = 0;
            for (std::size_t i = 0; i < size; ++i) {
                const auto last = ptr[i + 1];
                auto begin = indices.begin() + first, end = indices.begin() + last;
                std::sort(begin, end);
                end = std::unique(begin, end);
                ptr[i] = count;
                for (auto it = begin; it != end; ++it)
                    indices[count++] = *it;
                first = last;
            }
            ptr.back() = count;
            indices.resize(count);
            return std::make_shared<const core::CSRPattern>(size, size, std::move(ptr), std::move(indices));
        }

        const Erom& erom(std::size_t i) const
        {
            return *std::get<std::shared_ptr<const Erom>>(project.components()[i].asset);
        }

        double resistance(std::size_t i, FacetId facet) const
        {
            return thermal[i] ? thermal[i]->resistance(facet) : erom(i).R[facet];
        }

        void erom_environment(std::size_t i, FacetId id, const Erom& e, const EromSurfaceTraining& training,
            const BoundaryCondition* law, std::span<const double> state, double time)
        {
            const auto side = contact_side(project, i, id);
            const double r = e.R[id];
            const double temperature = law ? side.temperature(state) : 0;
            const auto [conductance, load]
                = boundary_terms(project.facet_contacts(i), id, e.facets[id], [&](const Facet& patch) {
                      if (law)
                          return evaluate_boundary(*law, patch, temperature, time, r);
                      const double g = convection_conductance(patch.area(), training.coefficient, r);
                      return BoundaryTrace {0, g, g * training.ambient};
                  });
            if (conductance != 0)
                add_block(operators.K, side, side, conductance);
            if (const double rhs = load - conductance * side.reference; rhs != 0)
                add_load(operators.f, side, rhs);
        }

        const Operators& assemble(std::span<const double> state, double time)
        {
            if (state.size() != project.dof_count())
                throw std::invalid_argument("project assembly: state size differs from DOF count");
            if (thermal.size() == 1 && thermal.front())
                return thermal.front()->assemble(state, time, {}, project.components().front().boundaries);

            std::ranges::fill(operators.K.values(), 0.0);
            std::ranges::fill(operators.f, 0.0);
            for (std::size_t i = 0; i < thermal.size(); ++i) {
                const auto& component = project.components()[i];
                const auto offset = project.dof_offset(i);
                if (const auto& assembler = thermal[i]) {
                    const auto& local = assembler->assemble(state.subspan(offset, project.dof_size(i)), time,
                        project.facet_contacts(i), component.boundaries);
                    copy_sparse(local.K, operators.K, offset);
                    copy_sparse(local.C, operators.C, offset);
                    std::ranges::copy(local.f, operators.f.begin() + offset);
                    continue;
                }

                const auto& e = erom(i);
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
                    if (const auto& training = e.surface_training[s]) {
                        const auto& surface = e.surfaces[s];
                        const auto it = component.boundaries.find(surface.id);
                        const auto* law = it == component.boundaries.end() ? nullptr : &it->second;
                        if (law ? is_adiabatic(*law) : training->coefficient == 0)
                            continue;
                        for (const auto id : surface.facets)
                            erom_environment(i, id, e, *training, law, state, time);
                    }
            }

            if (cached_contact_K.empty())
                assemble_contacts();
            else {
                for (std::size_t j = 0; j < cached_contact_K.size(); ++j)
                    operators.K.values()[j] += cached_contact_K[j];
                for (std::size_t j = 0; j < cached_contact_f.size(); ++j)
                    operators.f[j] += cached_contact_f[j];
            }
            return operators;
        }

        const Project& project;
        Vector<std::unique_ptr<ThermalAssembler>> thermal;
        Operators operators;
        Vector<double> cached_contact_K, cached_contact_f;
    };

    ProjectAssembler::ProjectAssembler(const model::Project& project) : workspace_(std::make_unique<Workspace>(project))
    {
    }
    ProjectAssembler::~ProjectAssembler() = default;

    const Operators& ProjectAssembler::assemble(std::span<const double> state, double time)
    {
        return workspace_->assemble(state, time);
    }

} // namespace mhs::solver
