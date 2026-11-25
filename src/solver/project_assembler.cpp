#include "solver/assembler.hpp"

#include "contact.hpp"
#include "core/csr_ops.hpp"
#include "thermal_physics.hpp"
#include <cmath>
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
            // Every row of a modal block has the same column pattern.
            const auto offsets = matrix.row_ptr();
            const auto column = slot(matrix, rows.first, cols.first) - offsets[rows.first];
            for (std::size_t i = 0; i < rows.weights.size(); ++i) {
                auto* values = matrix.values().data() + offsets[rows.first + i] + column;
                const auto factor = scale * rows.weights[i];
                for (std::size_t j = 0; j < cols.weights.size(); ++j)
                    values[j] += factor * cols.weights[j];
            }
        }

        void add_load(Vector<double>& rhs, const Side& rows, double scale)
        {
            for (std::size_t i = 0; i < rows.weights.size(); ++i)
                rhs[rows.first + i] += scale * rows.weights[i];
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

    ProjectAssembler::ProjectAssembler(const Project& project)
        : project_(project), thermal_(project.components().size())
    {
        for (std::size_t i = 0; i < thermal_.size(); ++i)
            if (const auto* fom = std::get_if<std::shared_ptr<const Fom>>(&project_.components()[i].asset))
                thermal_[i].emplace(**fom);
        if (thermal_.size() == 1 && thermal_.front())
            return;
        operators_ = make_operators();
        for (std::size_t i = 0; i < thermal_.size(); ++i)
            if (!thermal_[i]) {
                const auto offset = project_.dof_offset(i);
                const auto& e = *std::get<std::shared_ptr<const Erom>>(project_.components()[i].asset);
                std::copy_n(e.C.data(), e.C.rows() * e.C.cols(),
                    operators_.C.values().begin() + operators_.C.row_ptr()[offset]);
            }
    }

    void ProjectAssembler::assemble_contacts()
    {
        auto resistance = [&](ContactSide side) {
            const auto i = side.component_index;
            if (thermal_[i])
                return thermal_[i]->resistance(side.face_index);
            const auto& erom = *std::get<std::shared_ptr<const Erom>>(project_.components()[i].asset);
            return erom.interfaces[side.face_index].resistance;
        };
        for (const auto& c : project_.contacts()) {
            const auto left = contact_side(project_, c.left);
            const auto right = contact_side(project_, c.right);
            const auto g = c.area / contact_resistance(resistance(c.left), resistance(c.right), c.resistance);
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
        // Build adjacency over contact blocks before expanding dense modal columns.
        Vector<core::Triplet<double>> edges;
        edges.reserve(2 * project_.contacts().size());
        for (const auto& c : project_.contacts()) {
            const auto left = static_cast<SparseIndex>(contact_side(project_, c.left).first);
            const auto right = static_cast<SparseIndex>(contact_side(project_, c.right).first);
            edges.push_back({left, right, 0});
            edges.push_back({right, left, 0});
        }
        const auto adjacency = core::build_csr(size, size, std::move(edges));
        const auto contact_offsets = adjacency.row_ptr(), contact_blocks = adjacency.col_idx();
        Vector<SparseIndex> block_width(size, 1);
        std::size_t stiffness_nnz = 0, capacity_nnz = 0;
        for (std::size_t i = 0; i < thermal_.size(); ++i) {
            const auto order = project_.dof_size(i);
            if (!thermal_[i])
                block_width[project_.dof_offset(i)] = static_cast<SparseIndex>(order);
            stiffness_nnz += thermal_[i] ? thermal_[i]->operators().K.nnz() : order * order;
            capacity_nnz += thermal_[i] ? order : order * order;
        }
        for (std::size_t i = 0; i < size; ++i)
            for (auto j = contact_offsets[i]; j < contact_offsets[i + 1]; ++j)
                stiffness_nnz += static_cast<std::size_t>(block_width[i]) * block_width[contact_blocks[j]];

        Vector<SparseIndex> k_offsets(size + 1), c_offsets(size + 1), k_columns, c_columns;
        k_columns.reserve(core::detail::checked_sparse_size(stiffness_nnz));
        c_columns.reserve(core::detail::checked_sparse_size(capacity_nnz));
        auto dense = [](Vector<SparseIndex>& columns, SparseIndex start, SparseIndex count) {
            for (auto c = start; c < start + count; ++c)
                columns.push_back(c);
        };
        for (std::size_t i = 0; i < thermal_.size(); ++i) {
            const auto offset = static_cast<SparseIndex>(project_.dof_offset(i));
            const auto* local = thermal_[i] ? &thermal_[i]->operators().K : nullptr;
            for (std::size_t r = 0; r < project_.dof_size(i); ++r) {
                const auto row = offset + static_cast<SparseIndex>(r);
                const auto block = local ? row : offset;
                auto edge = contact_offsets[block];
                const auto last = contact_offsets[block + 1];
                for (; edge < last && contact_blocks[edge] < offset; ++edge)
                    dense(k_columns, contact_blocks[edge], block_width[contact_blocks[edge]]);
                if (local)
                    for (auto j = local->row_ptr()[r]; j < local->row_ptr()[r + 1]; ++j)
                        k_columns.push_back(offset + local->col_idx()[j]);
                else
                    dense(k_columns, block, block_width[block]);
                for (; edge < last; ++edge)
                    dense(k_columns, contact_blocks[edge], block_width[contact_blocks[edge]]);
                dense(c_columns, block, block_width[block]);
                k_offsets[row + 1] = static_cast<SparseIndex>(k_columns.size());
                c_offsets[row + 1] = static_cast<SparseIndex>(c_columns.size());
            }
        }
        return {{size, size, std::move(k_offsets), std::move(k_columns), Vector<double>(stiffness_nnz)},
            {size, size, std::move(c_offsets), std::move(c_columns), Vector<double>(capacity_nnz)},
            Vector<double>(size)};
    }

    const Operators& ProjectAssembler::assemble(std::span<const double> state, double time)
    {
        if (state.size() != project_.dof_count())
            throw std::invalid_argument("project assembly: state size differs from DOF count");
        if (thermal_.size() == 1 && thermal_.front())
            return thermal_.front()->assemble(
                state, time, project_.components().front().boundaries.empty() ? nullptr : &project_);

        std::ranges::fill(operators_.K.values(), 0.0);
        std::ranges::fill(operators_.f, 0.0);
        for (std::size_t i = 0; i < thermal_.size(); ++i) {
            const auto& component = project_.components()[i];
            const auto offset = project_.dof_offset(i);
            if (auto& assembler = thermal_[i]) {
                const auto& local
                    = assembler->assemble(state.subspan(offset, project_.dof_size(i)), time, &project_, i);
                copy_sparse(local.K, operators_.K, offset);
                std::ranges::copy(local.C.values(), operators_.C.values().begin() + operators_.C.row_ptr()[offset]);
                std::ranges::copy(local.f, operators_.f.begin() + offset);
                continue;
            }

            const auto& e = *std::get<std::shared_ptr<const Erom>>(component.asset);
            copy_dense(e.K, operators_.K, offset);
            std::ranges::copy(e.f, operators_.f.begin() + offset);
            for (SourceId s = 0; s < e.sources.size(); ++s) {
                const double power = s < component.powers.size() && component.powers[s] ? component.powers[s](time)
                                                                                        : 0.0;
                if (!std::isfinite(power))
                    throw std::invalid_argument("EROM power must be finite");
                for (std::size_t j = 0; j < e.K.rows(); ++j)
                    operators_.f[offset + j] += e.B(j, s) * power;
            }
            for (core::EntityIndex interface_index = 0; interface_index < e.interfaces.size(); ++interface_index) {
                const auto& face = e.interfaces[interface_index];
                const auto* law = boundary_override(component.boundaries, face.boundary_group);
                if (law ? is_adiabatic(*law) : face.default_coefficient == 0)
                    continue;
                const Side side {offset, e.Vb.row(interface_index), e.reference_temperature};
                const double temperature = law ? side.temperature(state) : 0;
                const auto [g, load]
                    = boundary_terms(face.facet, project_.face_patches(i, interface_index), [&](const Facet& patch) {
                          if (law)
                              return evaluate_boundary(*law, patch, temperature, time, face.resistance);
                          const double g = convection_conductance(patch.area(), face.default_coefficient, face.resistance);
                          return BoundaryEvaluation {.conductance = g, .load = g * face.default_ambient};
                      });
                if (g != 0)
                    add_block(operators_.K, side, side, g);
                if (const double rhs = load - g * side.reference; rhs != 0)
                    add_load(operators_.f, side, rhs);
            }
        }

        assemble_contacts();
        return operators_;
    }

} // namespace mhs::solver
