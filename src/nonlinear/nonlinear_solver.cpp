#include "nonlinear_solver.hpp"

#include "common/logger.hpp"

#include <Eigen/Dense>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace mhs::sim {

    namespace {
        struct AndersonMixer {
            int depth = 5; // m
            int warmup_iters = 2; // AA disabled for the first N iterations
            double dampening = 0.8; // 1.0 = full AA, 0.0 = plain Picard
            double max_growth = 1.5; // divergence guard (infinity-norm ratio)
            int reset_on_growth = 1; // reset history when guard trips

            // Ring buffers of size `depth`. Index 0 is most recent. When the
            // history is not yet full, hist_len < depth and we use fewer
            // columns in F.
            std::vector<Eigen::VectorXd> G_hist;
            std::vector<Eigen::VectorXd> x_hist;
            int hist_len = 0;
            int iter_count = 0;

            // Returns the next iterate proposal, or std::nullopt to fall back
            // to a plain damped Picard step. Caller is responsible for the
            // finite-value check on the returned vector.
            std::optional<Eigen::VectorXd> step(const Eigen::VectorXd& x_k, const Eigen::VectorXd& G_k)
            {
                // Disabled, in warm-up, or no history -> plain Picard path.
                if (depth == 0 || iter_count < warmup_iters || hist_len == 0) {
                    return std::nullopt;
                }

                const int N = static_cast<int>(x_k.size());
                const Eigen::VectorXd f_k = G_k - x_k;
                const int m_k = std::min(hist_len, depth);

                // Build F (N x m_k) and target b = f_k.
                Eigen::MatrixXd F(N, m_k);
                for (int j = 0; j < m_k; ++j) {
                    F.col(j) = f_k - (G_hist[j] - x_hist[j]);
                }

                // Solve F * alpha = f_k in the least-squares sense.
                Eigen::VectorXd alpha = F.colPivHouseholderQr().solve(f_k);

                // Mix G_k and the historical G's. Equivalently:
                //   x_prop = G_k - sum_i alpha_i * (G_k - G_{k-i})
                Eigen::VectorXd x_prop = (1.0 - alpha.sum()) * G_k;
                for (int j = 0; j < m_k; ++j) {
                    x_prop.noalias() += alpha(j) * G_hist[j];
                }

                // Optional outer blending with naive Picard (0 = Picard, 1 = AA).
                if (dampening < 1.0) {
                    const double omega = 1.0; // local omega for the blend; the caller's omega is applied after
                    const Eigen::VectorXd x_picard = x_k + omega * (G_k - x_k);
                    x_prop = (1.0 - dampening) * x_picard + dampening * x_prop;
                }

                // Divergence guard: compare infinity norms.
                if (max_growth > 0.0) {
                    const double prop_norm = (x_prop - x_k).cwiseAbs().maxCoeff();
                    const double naive_norm = (G_k - x_k).cwiseAbs().maxCoeff();
                    if (naive_norm > 0.0 && prop_norm > max_growth * naive_norm) {
                        if (reset_on_growth) {
                            G_hist.clear();
                            x_hist.clear();
                            hist_len = 0;
                        }
                        return std::nullopt;
                    }
                }
                return x_prop;
            }

            // Append (x_k, G_k) to the front of the history, trimming to depth.
            void push(const Eigen::VectorXd& x_k, const Eigen::VectorXd& G_k)
            {
                if (depth == 0) {
                    return;
                }
                G_hist.insert(G_hist.begin(), G_k);
                x_hist.insert(x_hist.begin(), x_k);
                if (static_cast<int>(G_hist.size()) > depth) {
                    G_hist.pop_back();
                    x_hist.pop_back();
                }
                hist_len = static_cast<int>(G_hist.size());
                ++iter_count;
            }

            void reset_history()
            {
                G_hist.clear();
                x_hist.clear();
                hist_len = 0;
            }
        };

    } // namespace

    NonLinearResult nonlinear_solve(LinearSystemProvider ls_provider, mhs::core::GlobalState& state,
        LinearSolver& solver, const NonLinearConfig& cfg)
    {
        const double omega = cfg.underrelaxation > 0.0 ? cfg.underrelaxation : 1.0;
        const double rel_tol = cfg.relative_tolerance;
        const double abs_tol = cfg.absolute_tolerance;
        const int N = static_cast<int>(state.T.size());

        AndersonMixer mixer;
        for (int iter = 0; iter < cfg.max_iterations; ++iter) {

            LinearSystem linear_system = ls_provider(state);
            Eigen::Map<const Eigen::VectorXd> T_map(state.T.data(), N);
            const Eigen::VectorXd residual_vec = linear_system.b - linear_system.A * T_map;

            // 一次性拷出残差（同时把逐元素 std::abs + max 的标量循环换成 Eigen SIMD）
            Eigen::Map<Eigen::VectorXd>(state.residual.data(), N) = residual_vec;
            const double max_residual = residual_vec.cwiseAbs().maxCoeff();
            const double max_b = linear_system.b.cwiseAbs().maxCoeff();

            // Combined relative + absolute tolerance
            const double residual_threshold = rel_tol * max_b + abs_tol;

            if (iter > 0 && max_residual <= residual_threshold) {
                return {true, iter};
            }

            auto solve_result = solver.solve(linear_system.A, linear_system.b);
            if (!solve_result.success) {
                MHS_LOG_WARN("Linear solver failed at Non-Linear iteration {}", iter);
            }

            const Eigen::VectorXd G_k = solve_result.solution;
            const Eigen::VectorXd x_k = T_map; // capture pre-update state

            std::optional<Eigen::VectorXd> x_prop = mixer.step(x_k, G_k);

            Eigen::VectorXd next(N);
            const bool use_aa = x_prop.has_value() && x_prop->allFinite();
            if (use_aa) {
                next = std::move(*x_prop);
            }
            else {
                for (int i = 0; i < N; ++i) {
                    next(i) = x_k(i) + omega * (G_k(i) - x_k(i));
                }
            }

            const double max_update = (next - x_k).cwiseAbs().maxCoeff();
            const double max_T = next.cwiseAbs().maxCoeff();
            Eigen::Map<Eigen::VectorXd>(state.T.data(), N) = next;

            const double update_threshold = rel_tol * max_T + abs_tol;

            MHS_LOG_DEBUG("\t->Non-Linear iteration {}: max_update={:.6e}, max_residual={:.6e} ({}AA)", iter,
                max_update, max_residual, use_aa ? "" : "no ");

            if (max_update <= update_threshold && max_residual <= residual_threshold) {
                mixer.push(x_k, G_k);
                return {true, iter + 1};
            }

            mixer.push(x_k, G_k);
        }

        return {false, cfg.max_iterations};
    }

} // namespace mhs::sim
