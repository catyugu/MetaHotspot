#include "assembler/assembler.hpp"
#include "common/logger.hpp"
#include "nonlinear_solver.hpp"
#include <algorithm>
#include <cmath>

namespace mhs::sim {

    static LinearSystem build_bdf1_linear_system(const StaticOpsResult& sops, const MassOpsResult& mops, double dt,
        const std::vector<double>& T_prev, const std::vector<double>& T_current)
    {
        const int N = static_cast<int>(sops.f_static.size());

        Eigen::SparseMatrix<double> A = sops.K;
        Eigen::VectorXd b = sops.f_static;

        for (int i = 0; i < N; ++i) {
            A.coeffRef(i, i) += mops.M_diag(i) / dt;
            b(i) += mops.M_diag(i) * T_prev[i] / dt;
        }

        Eigen::VectorXd T_vec(N);
        for (int i = 0; i < N; ++i)
            T_vec(i) = T_current[i];
        Eigen::VectorXd residual_vec = b - A * T_vec;

        return {A, b, residual_vec};
    }

    NonLinearResult nonlinear_solve(
        const mhs::core::InternalModel& model, mhs::core::GlobalState& state, LinearSolver& solver)
    {
        Assembler assembler(model);
        NonLinearConfig cfg;

        double omega = cfg.underrelaxation > 0.0 ? cfg.underrelaxation : 1.0;

        const double rel_tol = cfg.relative_tolerance;
        const double abs_tol = cfg.absolute_tolerance;

        // For Steady: dt=0, so no mass term — assemble_static alone is the full system.
        // For Transient: dt>0, we build A = K + M/dt*I, b = f_static + M*T_prev/dt.
        bool is_transient = (model.study_type == mhs::core::StudyType::Transient && state.dt > 0.0);

        // T_prev for the BDF1 fallback = history.latest() (== state.T after
        // the Scheduler has pushed the most recent accepted step).  The
        // global GlobalState::T_prev field is being phased out; the local
        // alias here keeps the slice-1 glue self-contained.
        const std::vector<double>& T_prev = state.history.size() > 0 ? state.history.latest() : state.T;

        for (int iter = 0; iter < cfg.max_iterations; iter++) {
            auto sops = assembler.assemble_static(state);

            LinearSystem linear_system;
            if (is_transient) {
                auto mops = assembler.assemble_mass(state);
                linear_system = build_bdf1_linear_system(sops, mops, state.dt, T_prev, state.T);
            }
            else {
                Eigen::VectorXd T_vec(static_cast<int>(state.T.size()));
                for (int i = 0; i < static_cast<int>(state.T.size()); ++i)
                    T_vec(i) = state.T[i];
                Eigen::VectorXd residual_vec = sops.f_static - sops.K * T_vec;
                linear_system = {sops.K, sops.f_static, residual_vec};
            }

            double max_residual = 0.0;
            double max_b = 0.0;
            const int N = static_cast<int>(state.T.size());
            for (int i = 0; i < N; i++) {
                state.residual[i] = linear_system.residual(i);
                max_residual = std::max(max_residual, std::abs(state.residual[i]));
                max_b = std::max(max_b, std::abs(linear_system.b(i)));
            }

            // Combined relative + absolute tolerance
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
            for (int i = 0; i < N; i++) {
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

} // namespace mhs::sim
