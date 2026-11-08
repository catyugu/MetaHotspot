#pragma once
#include "core/csr_matrix.hpp"
#include "model/model.hpp"
#include "solver/frozen_flow.hpp"
#include <span>

namespace mhs::solver {
    // A MAC / staggered finite-volume Stokes operator bound to immutable mesh
    // topology. Viscosity, density, pressure and velocity are evaluated anew
    // for each nonlinear iterate. Pressure rows are algebraic (zero capacity).
    class StokesAssembler {
    public:
        StokesAssembler(const model::CompiledModel& compiled, model::DomainId id);
        FrozenFlow assemble(std::span<const double> state, std::span<const double> temperature, double time,
            core::Vector<core::Triplet<double>>& stiffness, core::Vector<core::Triplet<double>>& capacity,
            core::Vector<double>& load) const;

    private:
        struct Node {
            model::FaceId face = core::invalid_id;
            model::DofId dof = core::invalid_id; // Invalid for prescribed normal wall/inlet velocity.
            model::CellId minus = core::invalid_id, plus = core::invalid_id;
            core::EntityIndex condition = core::invalid_id;
            std::size_t component = 0;
            model::Vec3 center {}, low {}, high {};
            std::array<bool, 6> joined {};
        };
        struct Link {
            std::size_t a = 0, b = 0;
            double area = 0, da = 0, db = 0;
        };
        struct Wall {
            std::size_t node = 0;
            double area = 0, distance = 0;
        };

        const model::FomDomain& domain_;
        const core::Vector<model::MaterialProps>& materials_;
        core::Vector<model::DofId> pressure_dof_, face_dof_;
        core::Vector<Node> nodes_;
        core::Vector<Link> links_;
        core::Vector<Wall> walls_;
        core::Vector<double> boundary_area_;
    };
} // namespace mhs::solver
