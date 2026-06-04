#include "solver/bicgstab_solver.hpp"
#include "solver/solver.hpp"
#include "solver/sparse_lu_solver.hpp"

namespace mhs {

    // Factory: explicit branches for each SolverType. Unknown types fall back
    // to SuperLUMTSolver, matching SolverConfig::type default (SuperLU_MT).
    std::unique_ptr<Solver> Solver::create(SolverType type)
    {
        switch (type) {
        case SolverType::SparseLU:
            return std::make_unique<SparseLUSolver>();
        case SolverType::BiCGSTAB:
            return std::make_unique<BiCGSTABSolver>();
        default:
            return std::make_unique<SparseLUSolver>();
        }
    }

} // namespace mhs