#include "numerics/linear/linear_solver.hpp"
#include "numerics/linear/eigen_bicgstab_solver.hpp"
#include "numerics/linear/eigen_sparse_lu_solver.hpp"

#ifdef MHS_ENABLE_PARDISO
#include "numerics/linear/pardiso_lu_solver.hpp"
#endif

namespace mhs::sim {

    // --- Factory ---

    SolverHandle create_solver(const SolverSpec& spec)
    {
        switch (spec.type) {
#ifdef MHS_ENABLE_PARDISO
        case SolverType::Pardiso: {
            auto solver = std::make_unique<PardisoLUSolver>();
            solver->set_config(spec.config);
            return solver;
        }
#endif
        case SolverType::EigenSparseLU:
        default: { // Pardiso without MKL falls back to SparseLU
            auto solver = std::make_unique<EigenSparseLUSolver>();
            solver->set_config(spec.config);
            return solver;
        }
        case SolverType::EigenBiCGSTAB: {
            auto solver = std::make_unique<EigenBiCGSTABSolver>();
            solver->set_config(spec.config);
            return solver;
        }
        }
    }

} // namespace mhs::sim
