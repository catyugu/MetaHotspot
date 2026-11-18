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
        explicit ThermalAssembler(const mhs::model::Model& model);
        Operators& assemble_internal(std::span<const double> temperature, double time);
        const Operators& operators() const { return operators_; }
        double resistance(model::FacetId id) const;

        Operators take_operators() && { return std::move(operators_); }

    private:
        const mhs::model::Model& model_;
        Operators operators_;
        mhs::core::Vector<mhs::core::SparseIndex> diagonal_positions_;
        mhs::core::Vector<mhs::model::Vec3> conductivity_;
        mhs::core::Vector<double> density_, specific_heat_;
    };

    class ProjectAssembler {
    public:
        explicit ProjectAssembler(const model::Project& project);
        ~ProjectAssembler();
        const Operators& assemble(std::span<const double> state, double time);
        Operators take_operators() &&;
        double sample_fom(
            std::size_t component, const model::PointProbe& probe, std::span<const double> state, double time) const;

    private:
        struct Workspace;
        std::unique_ptr<Workspace> workspace_;
    };

    model::Project single_model_project(const model::Model& model);

    // Owning export for the C API and standalone operator inspection.
    Operators assemble_thermal(const mhs::model::Model& model, std::span<const double> temperature, double time);

} // namespace mhs::solver
