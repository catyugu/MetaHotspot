#include "assembler/assembler.hpp"
#include "common/logger.hpp"
#include "nonlinear_solver.hpp"
#include <algorithm>
#include <cmath>

namespace mhs::nonlinear {

    NonLinearResult solve(const InternalModel& model, GlobalState& state, Solver& solver, const NonLinearConfig& cfg)
    {
        assembler::Assembler assembler(model);

        double omega = cfg.underrelaxation > 0.0 ? cfg.underrelaxation : 1.0;

        const double rel_tol = cfg.relative_tolerance;
        const double abs_tol = cfg.absolute_tolerance;

        for (int iter = 0; iter < cfg.max_iterations; iter++) {
            auto linear_system = assembler.assemble(state);

            double max_residual = 0.0;
            double max_b = 0.0;
            for (int i = 0; i < state.cell_count; i++) {
                state.residual[i] = linear_system.residual(i);
                max_residual = std::max(max_residual, std::abs(state.residual[i]));
                max_b = std::max(max_b, std::abs(linear_system.b(i)));
            }

            // 结合右端项规模的相对容差与绝对容差
            double residual_threshold = rel_tol * max_b + abs_tol;

            if (iter > 0 && max_residual <= residual_threshold) {
                return {true, iter};
            }

            auto solve_result = solver.solve(linear_system.A, linear_system.b);

            if (!solve_result.success) {
                MHS_LOG_WARN("Linear solver failed at Non-Linear iteration {}", iter);
            }

            double max_update = 0.0;
            double max_T = 0.0;
            for (int i = 0; i < state.cell_count; i++) {
                double update = solve_result.solution(i) - state.T[i];
                max_update = std::max(max_update, std::abs(update));
                max_T = std::max(max_T, std::abs(state.T[i]));

                state.T[i] += omega * update;
            }

            double update_threshold = rel_tol * max_T + abs_tol;

            MHS_LOG_INFO(
                "\t->Non-Linear iteration {}: max_update={:.6e}, max_residual={:.6e}", iter, max_update, max_residual);

            if (max_update <= update_threshold && max_residual <= residual_threshold) {
                return {true, iter + 1};
            }
        }

        return {false, cfg.max_iterations};
    }

} // namespace mhs::nonlinear