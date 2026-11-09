#pragma once

#include "solver/frozen_flow.hpp"
#include "solver/linear_system.hpp"
#include "solver/stokes_flow.hpp"

#include "model/model.hpp"
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace mhs::solver {

    // Binds an immutable model for the workspace lifetime. Each assembly
    // overwrites numerical buffers, including explicit zero coefficients.
    class ThermalAssembler {
    public:
        ThermalAssembler(const mhs::model::FomDomain& model, const mhs::model::CompiledLibrary& library,
            mhs::solver::FrozenFlow flow);
        const Operators& assemble(std::span<const double> temperature, double time);

        Operators take_operators() && { return std::move(operators_); }
        void set_flow(FrozenFlow flow) { flow_ = std::move(flow); }

    private:
        const mhs::model::FomDomain& model_;
        const mhs::model::CompiledLibrary& library_;
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

    private:
        const model::CompiledModel& model_;
        struct ComponentAssembly {
            std::unique_ptr<ThermalAssembler> thermal;
            std::unique_ptr<StokesAssembler> stokes;
            std::optional<Operators> reduced;
        };
        core::Vector<ComponentAssembly> components_;
        Operators operators_;
    };

    Operators assemble_thermal(const mhs::model::CompiledModel& compiled, std::span<const double> state, double time);

} // namespace mhs::solver
