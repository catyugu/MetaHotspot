#include "numerics/linear/linear_solver.hpp"
#include "numerics/linear/amg_solver.hpp"

#ifdef MHS_ENABLE_PARDISO
#include "numerics/linear/pardiso_lu_solver.hpp"
#endif

#include <stdexcept>

namespace mhs::sim {

    // --- Factory ---

    SolverHandle create_solver(const SolverSpec& spec)
    {
        switch (spec.type) {
        case SolverType::Pardiso:
#ifdef MHS_ENABLE_PARDISO
        {
            auto solver = std::make_unique<PardisoLUSolver>();
            solver->set_config(spec.config);
            return solver;
        }
#else
            throw std::runtime_error("Pardiso solver requested but MKL/Pardiso is not enabled "
                                     "(build with USE_MKL=ON)");
#endif
        case SolverType::AmgCg: {
            auto solver = std::make_unique<AmgCgSolver>();
            solver->set_config(spec.config);
            return solver;
        }
        default:
            throw std::invalid_argument("create_solver: unknown solver type");
        }
    }

} // namespace mhs::sim
