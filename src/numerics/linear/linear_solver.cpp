#include "numerics/linear/eigen_bicgstab_solver.hpp"
#include "numerics/linear/eigen_sparse_lu_solver.hpp"
#include "numerics/linear/linear_solver.hpp"

#ifdef MHS_ENABLE_PARDISO
#include "numerics/linear/pardiso_lu_solver.hpp"
#endif

namespace mhs::sim {

    // --- Factory ---

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
