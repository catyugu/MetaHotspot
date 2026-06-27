#include "time_scheme/detail/build_ops.hpp"
#include "time_scheme/integrator.hpp"

namespace mhs::sim::time_scheme {

    LinearSystem build_system(
        IntegratorKind kind, const AssemblyResult& ops, const mhs::core::SolutionHistory& hist, double dt)
    {
        switch (kind) {
        case IntegratorKind::Bdf1:
            return detail::build_bdf1_ls(ops, hist, dt);
        case IntegratorKind::Bdf2:
            // BDF2 requires at least 2 preceding snapshots for the three-level
            // stencil; fall back to BDF1 during startup.
            if (hist.size() >= 2)
                return detail::build_bdf2_ls(ops, hist, dt);
            return detail::build_bdf1_ls(ops, hist, dt);
        }
        // unreachable — all cases covered.
        return detail::build_bdf1_ls(ops, hist, dt);
    }

} // namespace mhs::sim::time_scheme
