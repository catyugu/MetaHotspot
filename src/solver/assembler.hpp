#pragma once

#include "solver/frozen_flow.hpp"
#include "solver/linear_system.hpp"

#include "model/model.hpp"
#include <span>
#include <utility>

namespace mhs::solver {

    // Binds an immutable model for the workspace lifetime. Each assembly
    // overwrites numerical buffers, including explicit zero coefficients.
    class ThermalAssembler {
    public:
        ThermalAssembler(const mhs::model::FomDomain& model, mhs::solver::FrozenFlow flow);
        const Operators& assemble(std::span<const double> temperature, double time);

        Operators take_operators() && { return std::move(operators_); }

    private:
        const mhs::model::FomDomain& model_;
        mhs::solver::FrozenFlow flow_;
        Operators operators_;
        mhs::core::Vector<mhs::core::SparseIndex> diagonal_positions_;
        mhs::core::Vector<mhs::model::Vec3> conductivity_;
        mhs::core::Vector<double> specific_heat_;
    };

    class ModelAssembler {
    public:
        ModelAssembler(const model::CompiledModel& model, std::span<const double> initial, double time = 0.0);
        const Operators& assemble(std::span<const double> state, double time);
        const core::Vector<core::Vector<double>>& temperatures() const { return temperatures_; }
        void gather(std::span<const double> state);

    private:
        const model::CompiledModel& model_;
        core::Vector<core::Vector<double>> temperatures_;
        core::Vector<core::Vector<model::DofId>> dofs_;
        core::Vector<std::unique_ptr<ThermalAssembler>> domains_;
        Operators operators_;
        bool identity_ = false;
    };

    Operators assemble_thermal(const mhs::model::CompiledModel& compiled, std::span<const double> state, double time);

    // Owning export for the C API and standalone operator inspection.
    Operators assemble_thermal(const mhs::model::FomDomain& model, std::span<const double> temperature, double time);

} // namespace mhs::solver
