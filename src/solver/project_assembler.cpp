#include "solver/assembler.hpp"

#include "contact.hpp"
#include "thermal_physics.hpp"
#include <cmath>
#include <set>
#include <stdexcept>

namespace mhs::solver {
    using namespace model;
    using core::Vector;

    namespace {

        std::size_t slot(const core::CSRMatrix<double>& matrix, std::size_t row, std::size_t col)
        {
            const auto columns = matrix.col_idx();
            return std::lower_bound(
                       columns.begin() + matrix.row_ptr()[row], columns.begin() + matrix.row_ptr()[row + 1], col)
                - columns.begin();
        }

        void add_block(core::CSRMatrix<double>& matrix, const Side& rows, const Side& cols, double scale)
        {
            for (std::size_t i = 0; i < rows.count(); ++i) {
                const auto idx = slot(matrix, rows.first + i, cols.first);
                const auto factor = scale * rows.value(i);
                for (std::size_t j = 0; j < cols.count(); ++j)
                    matrix.values()[idx + j] += factor * cols.value(j);
            }
        }

        void add_load(Vector<double>& rhs, const Side& rows, double scale)
        {
            for (std::size_t i = 0; i < rows.count(); ++i)
                rhs[rows.first + i] += scale * rows.value(i);
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

    ProjectAssembler::ProjectAssembler(const Project& project) : project_(project)
    {
        for (const auto& component : project_.components()) {
            const auto* fom = std::get_if<std::shared_ptr<const Fom>>(&component.asset);
            thermal_.push_back(fom ? std::make_unique<ThermalAssembler>(**fom) : nullptr);
        }
        if (thermal_.size() == 1 && thermal_.front())
            return;
        operators_ = make_operators();
        const bool reduced_only = std::ranges::none_of(thermal_, [](const auto& assembler) { return bool(assembler); });
        for (std::size_t i = 0; i < thermal_.size(); ++i)
            if (!thermal_[i]) {
                const auto offset = project_.dof_offset(i);
                const auto& e = *std::get<std::shared_ptr<const Erom>>(project_.components()[i].asset);
                std::copy_n(e.C.data(), e.C.rows() * e.C.cols(),
                    operators_.C.values().begin() + operators_.C.row_ptr()[offset]);
                if (reduced_only)
                    copy_dense(e.K, operators_.K, offset);
            }
        if (reduced_only) {
            assemble_contacts();
            fixed_K_.assign(operators_.K.values().begin(), operators_.K.values().end());
            fixed_f_ = operators_.f;
        }
    }

    void ProjectAssembler::assemble_contacts()
    {
        for (const auto& c : project_.contacts()) {
            const auto left = contact_side(project_, c.left, c.left_facet);
            const auto right = contact_side(project_, c.right, c.right_facet);
            auto resistance = [&](std::size_t i, FacetId facet) {
                return thermal_[i] ? thermal_[i]->resistance(facet)
                                   : std::get<std::shared_ptr<const Erom>>(project_.components()[i].asset)->R[facet];
            };
            const auto g = c.area()
                / contact_resistance(
                    resistance(c.left, c.left_facet), resistance(c.right, c.right_facet), c.resistance);
            add_block(operators_.K, left, left, g);
            add_block(operators_.K, right, right, g);
            add_block(operators_.K, left, right, -g);
            add_block(operators_.K, right, left, -g);
            const auto load = g * (right.reference - left.reference);
            add_load(operators_.f, left, load);
            add_load(operators_.f, right, -load);
        }
    }

    Operators ProjectAssembler::make_operators() const
    {
        using core::SparseIndex;
        const auto size = project_.dof_count();
        // Each contact joins disjoint component ranges. Deduplicate the directed blocks before expanding them.
        std::set<std::array<std::size_t, 4>> contacts;
        for (const auto& c : project_.contacts()) {
            const auto l = contact_side(project_, c.left, c.left_facet);
            const auto r = contact_side(project_, c.right, c.right_facet);
            contacts.insert({l.first, r.first, l.count(), r.count()});
            contacts.insert({r.first, l.first, r.count(), l.count()});
        }

        std::size_t conductance_nnz = 0, capacity_nnz = 0;
        for (const auto& [row, col, rows, cols] : contacts)
            conductance_nnz = core::detail::checked_sparse_size(conductance_nnz + rows * cols);
        for (std::size_t i = 0; i < thermal_.size(); ++i) {
            const auto count = project_.dof_size(i);
            const auto& local = thermal_[i];
            conductance_nnz = core::detail::checked_sparse_size(
                conductance_nnz + (local ? local->operators().K.nnz() : count * count));
            capacity_nnz = core::detail::checked_sparse_size(capacity_nnz + (local ? count : count * count));
        }

        Vector<SparseIndex> ptr(size + 1), indices, capacity_ptr(size + 1), capacity_indices;
        indices.reserve(conductance_nnz);
        capacity_indices.reserve(capacity_nnz);
        auto first_contact = contacts.begin();
        auto dense = [](Vector<SparseIndex>& columns, std::size_t first, std::size_t count) {
            for (auto col = first; col < first + count; ++col)
                columns.push_back(static_cast<SparseIndex>(col));
        };
        for (std::size_t i = 0; i < thermal_.size(); ++i) {
            const auto offset = project_.dof_offset(i), rows = project_.dof_size(i);
            const auto& assembler = thermal_[i];
            const auto* local = assembler ? &assembler->operators().K : nullptr;
            for (std::size_t r = 0; r < rows; ++r) {
                const auto row = offset + r, contact_row = local ? row : offset;
                while (first_contact != contacts.end() && (*first_contact)[0] < contact_row)
                    ++first_contact;
                auto contact = first_contact;
                for (; contact != contacts.end() && (*contact)[0] == contact_row && (*contact)[1] < offset; ++contact)
                    dense(indices, (*contact)[1], (*contact)[3]);
                if (local)
                    for (auto j = local->row_ptr()[r]; j < local->row_ptr()[r + 1]; ++j)
                        indices.push_back(static_cast<SparseIndex>(offset) + local->col_idx()[j]);
                else
                    dense(indices, offset, rows);
                for (; contact != contacts.end() && (*contact)[0] == contact_row; ++contact)
                    dense(indices, (*contact)[1], (*contact)[3]);
                ptr[row + 1] = static_cast<SparseIndex>(indices.size());
                dense(capacity_indices, local ? row : offset, local ? 1 : rows);
                capacity_ptr[row + 1] = static_cast<SparseIndex>(capacity_indices.size());
            }
        }
        return {{size, size, std::move(ptr), std::move(indices), Vector<double>(conductance_nnz)},
            {size, size, std::move(capacity_ptr), std::move(capacity_indices), Vector<double>(capacity_nnz)},
            Vector<double>(size)};
    }

    const Operators& ProjectAssembler::assemble(std::span<const double> state, double time)
    {
        if (state.size() != project_.dof_count())
            throw std::invalid_argument("project assembly: state size differs from DOF count");
        if (thermal_.size() == 1 && thermal_.front())
            return thermal_.front()->assemble(state, time, {}, project_.components().front().boundaries);

        if (fixed_K_.empty()) {
            std::ranges::fill(operators_.K.values(), 0.0);
            std::ranges::fill(operators_.f, 0.0);
        }
        else {
            std::ranges::copy(fixed_K_, operators_.K.values().begin());
            std::ranges::copy(fixed_f_, operators_.f.begin());
        }
        for (std::size_t i = 0; i < thermal_.size(); ++i) {
            const auto& component = project_.components()[i];
            const auto offset = project_.dof_offset(i);
            if (const auto& assembler = thermal_[i]) {
                const auto& local = assembler->assemble(state.subspan(offset, project_.dof_size(i)), time,
                    project_.facet_contacts(i), component.boundaries);
                copy_sparse(local.K, operators_.K, offset);
                std::ranges::copy(local.C.values(), operators_.C.values().begin() + operators_.C.row_ptr()[offset]);
                std::ranges::copy(local.f, operators_.f.begin() + offset);
                continue;
            }

            const auto& e = *std::get<std::shared_ptr<const Erom>>(component.asset);
            if (fixed_K_.empty())
                copy_dense(e.K, operators_.K, offset);
            for (std::size_t s = 0; s < e.sources.size(); ++s) {
                const auto& source = e.sources[s];
                const auto it = component.powers.find(source.id);
                const double power = it == component.powers.end() ? source.default_power : it->second(time);
                if (!std::isfinite(power))
                    throw std::invalid_argument("EROM power must be finite");
                for (std::size_t j = 0; j < e.K.rows(); ++j)
                    operators_.f[offset + j] += e.B(j, s) * power;
            }
            for (const auto& surface : e.surfaces)
                if (const auto& training = surface.training) {
                    const auto it = component.boundaries.find(surface.id);
                    const auto* law = it == component.boundaries.end() ? nullptr : &it->second;
                    if (law ? is_adiabatic(*law) : training->coefficient == 0)
                        continue;
                    const auto& contacts = project_.facet_contacts(i);
                    for (const auto id : surface.facets) {
                        const Side side {offset, e.Vb.row(id), e.reference_temperature};
                        const double temperature = law ? side.temperature(state) : 0;
                        const auto [g, load] = boundary_terms(contacts, id, e.facets[id], [&](const Facet& patch) {
                            if (law)
                                return evaluate_boundary(*law, patch, temperature, time, e.R[id]);
                            const double g = convection_conductance(patch.area(), training->coefficient, e.R[id]);
                            return BoundaryTrace {0, g, g * training->ambient};
                        });
                        if (g != 0)
                            add_block(operators_.K, side, side, g);
                        if (const double rhs = load - g * side.reference; rhs != 0)
                            add_load(operators_.f, side, rhs);
                    }
                }
        }

        if (fixed_K_.empty())
            assemble_contacts();
        return operators_;
    }

} // namespace mhs::solver
