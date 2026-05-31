#include "nonlinear_solver.hpp"
#include "assembler/assembler.hpp"
#include "logger/logger.hpp"
#include <algorithm>
#include <cmath>

namespace mhs::nonlinear {

    SolveResult solve(const model::InternalModel& model,
                      model::GlobalState& state,
                      Solver& solver,
                      double underrelaxation,
                      int max_iterations,
                      double tolerance)
    {
        assembler::Assembler assembler(model);

        double omega = underrelaxation > 0.0 ? underrelaxation : 1.0;

        for (int iter = 0; iter < max_iterations; iter++) {
            auto linear_system = assembler.assemble(state);
            auto solve_result = solver.solve(linear_system.A, linear_system.b);

            if (!solve_result.success) {
                MHS_LOG_WARN("Linear solver failed at Newton iteration {}", iter);
            }

            double max_update = 0.0;
            for (int i = 0; i < state.cell_count; i++) {
                double update = solve_result.solution(i) - state.T[i];
                max_update = std::max(max_update, std::abs(update));
                state.T[i] += omega * update;
            }

            double max_residual = 0.0;
            for (int i = 0; i < state.cell_count; i++) {
                state.residual[i] = linear_system.residual(i);
                max_residual = std::max(max_residual, std::abs(state.residual[i]));
            }

            MHS_LOG_INFO("Newton iteration {}: max_update={:.6e}, max_residual={:.6e}",
                iter, max_update, max_residual);

            if (max_update < tolerance && max_residual < tolerance) {
                state.status = ConvergenceStatus::Converged;
                return {ConvergenceStatus::Converged, iter + 1};
            }
        }

        state.status = ConvergenceStatus::Diverged;
        return {ConvergenceStatus::Diverged, max_iterations};
    }

} // namespace mhs::nonlinear