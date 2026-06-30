#pragma once

#include "data/linear_system.hpp"
#include "data/solution_history.hpp"

namespace mhs::sim::time_scheme {

    /// Time-integration method — pure linear algebra.
    ///
    /// These are the discrete operators; they do NOT own state or manage step size.
    /// Each kind corresponds to a specific stencil and order.
    enum class IntegratorKind { Bdf1, Bdf2 };

    /// Build the linear system A·x = b for the chosen integrator.
    ///
    /// \param kind   Which method to use (Bdf1 → backward Euler, Bdf2 → variable-step BDF2).
    /// \param ops    Raw spatial discretization from Assembler.
    /// \param hist   Accepted-solution history.  Bdf2 falls back to Bdf1 when the
    ///               history buffer holds fewer than 2 snapshots (startup phase).
    /// \param dt     Step size for this solve.
    LinearSystem build_system(
        IntegratorKind kind, const AssemblyResult& ops, const mhs::core::SolutionHistory& hist, double dt);

} // namespace mhs::sim::time_scheme
