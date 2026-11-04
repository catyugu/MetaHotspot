#include "linear/solver.hpp"
#include <stdexcept>

namespace mhs::linear {
    SolverPtr make_amg_solver(SolverConfig config);
#ifdef MHS_ENABLE_PARDISO
    SolverPtr make_pardiso_solver(SolverConfig config);
#endif

    SolverPtr create_solver(const SolverSpec& spec)
    {
        switch (spec.type) {
        case SolverType::Pardiso:
#ifdef MHS_ENABLE_PARDISO
            return make_pardiso_solver(spec.config);
#else
            throw std::runtime_error("Pardiso solver requested but MKL/Pardiso is not enabled "
                                     "(build with MHS_USE_MKL=ON)");
#endif
        case SolverType::Amg:
            return make_amg_solver(spec.config);
        default:
            throw std::invalid_argument("create_solver: unknown solver type");
        }
    }
} // namespace mhs::linear
