#include "linear_solver/eigen_bicgstab_solver.hpp"
#include "linear_solver/eigen_sparse_lu_solver.hpp"
#include "linear_solver/linear_solver.hpp"

#ifdef MHS_ENABLE_PARDISO
#include "linear_solver/pardiso_lu_solver.hpp"
#endif

namespace mhs::sim {

    // Factory: explicit branches for each SolverType. The default and the
    // Pardiso branch are guarded by MHS_ENABLE_PARDISO; without it, both
    // fall back to EigenSparseLUSolver.
    std::unique_ptr<LinearSolver> LinearSolver::create(SolverType type)
    {
        switch (type) {
#ifdef MHS_ENABLE_PARDISO
        case SolverType::Pardiso:
            return std::make_unique<PardisoLUSolver>();
#endif
        case SolverType::EigenSparseLU:
            return std::make_unique<EigenSparseLUSolver>();
        case SolverType::EigenBiCGSTAB:
            return std::make_unique<EigenBiCGSTABSolver>();
        default:
            return std::make_unique<EigenSparseLUSolver>();
        }
    }

} // namespace mhs::sim
