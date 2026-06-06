#include "linear_solver/bicgstab_solver.hpp"
#include "linear_solver/linear_solver.hpp"
#include "linear_solver/sparse_lu_solver.hpp"
#ifdef MHS_ENABLE_PARDISO
#include "linear_solver/pardiso_solver.hpp"
#endif

namespace mhs::sim {

    // Factory: explicit branches for each SolverType. The default and the
    // Pardiso branch are guarded by MHS_ENABLE_PARDISO; without it, both
    // fall back to SparseLUSolver.
    std::unique_ptr<LinearSolver> LinearSolver::create(SolverType type)
    {
        switch (type) {
#ifdef MHS_ENABLE_PARDISO
        case SolverType::Pardiso:
            return std::make_unique<PardisoSolver>();
#endif
        case SolverType::SparseLU:
            return std::make_unique<SparseLUSolver>();
        case SolverType::BiCGSTAB:
            return std::make_unique<BiCGSTABSolver>();
        default:
            return std::make_unique<SparseLUSolver>();
        }
    }

} // namespace mhs::sim
