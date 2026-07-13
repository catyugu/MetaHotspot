#include "linear_solver/eigen_bicgstab_solver.hpp"
#include "linear_solver/eigen_sparse_lu_solver.hpp"
#include "linear_solver/linear_solver.hpp"

#ifdef MHS_ENABLE_PARDISO
#include "linear_solver/pardiso_lu_solver.hpp"
#endif

namespace mhs::sim {

    // Factory: builds the chosen solver and seeds it with the spec's config.
    // The spec default-constructs to SolverType::EigenSparseLU + default config,
    // The Pardiso branch is guarded by MHS_ENABLE_PARDISO; without it, requests for
    // Pardiso fall back to EigenSparseLUSolver.
    std::unique_ptr<LinearSolver> LinearSolver::create(const SolverSpec& spec)
    {
        std::unique_ptr<LinearSolver> solver = nullptr;
        switch (spec.type) {
#ifdef MHS_ENABLE_PARDISO
        case SolverType::Pardiso:
            solver = std::make_unique<PardisoLUSolver>();
            break;
#endif
        case SolverType::EigenSparseLU:
            solver = std::make_unique<EigenSparseLUSolver>();
            break;
        case SolverType::EigenBiCGSTAB:
            solver = std::make_unique<EigenBiCGSTABSolver>();
            break;
        default:
            solver = std::make_unique<EigenSparseLUSolver>();
            break;
        }
        solver->set_config(spec.config);
        return solver;
    }

} // namespace mhs::sim
