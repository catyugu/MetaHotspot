#pragma once

#include "solver/linear_system.hpp"

#include "model/project.hpp"
#include <span>
#include <utility>

namespace mhs::solver {

    // Binds an immutable model for the workspace lifetime. Each assembly
    // overwrites numerical buffers, including explicit zero coefficients.
    class ThermalAssembler {
    public:
        explicit ThermalAssembler(const mhs::model::Fom& model);
        const Operators& assemble(std::span<const double> temperature, double time,
            const model::FacetContacts& contacts = {}, const model::BoundaryOverrides& overrides = {});
        const Operators& assemble_internal(std::span<const double> temperature, double time);
        const Operators& operators() const { return operators_; }
        double resistance(model::FacetId id) const;

        Operators take_operators() && { return std::move(operators_); }

    private:
        const mhs::model::Fom& model_;
        Operators operators_;
        mhs::core::Vector<mhs::core::SparseIndex> diagonal_positions_;
        mhs::core::Vector<mhs::model::Vec3> conductivity_;
        mhs::core::Vector<double> density_, specific_heat_;
    };

    class ProjectAssembler {
    public:
        explicit ProjectAssembler(const model::Project& project);
        const Operators& assemble(std::span<const double> state, double time);

    private:
        Operators make_operators() const;
        double resistance(std::size_t component, model::FacetId facet) const;
        void assemble_contacts();

        const model::Project& project_;
        core::Vector<std::unique_ptr<ThermalAssembler>> thermal_;
        Operators operators_;
        core::Vector<double> fixed_K_, fixed_f_;
    };

    // Owning export for the C API and standalone operator inspection.
    Operators assemble_thermal(const mhs::model::Fom& model, std::span<const double> temperature, double time);

} // namespace mhs::solver
