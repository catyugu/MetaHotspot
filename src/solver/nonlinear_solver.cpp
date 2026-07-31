#include "solver/nonlinear_solver.hpp"

#include "logging/logger.hpp"
#include "common/types.hpp"
#include <Eigen/QR>

#include <algorithm>
#include <cassert>
#include <deque>
#include <optional>

namespace mhs::sim {

    namespace {
        struct AndersonMixer {
            int depth = 5; // m
            int warmup_iters = 2; // AA disabled for the first N iterations
            double dampening = 0.8; // 1.0 = full AA, 0.0 = plain Picard
            double max_growth = 1.5; // divergence guard (infinity-norm ratio)
            int reset_on_growth = 1; // reset history when guard trips

            // History of size `depth`. Index 0 is most recent, stored as a
            // deque so that push_front is O(1) — insert(begin()) on vector
            // would be O(N) and we call push() every nonlinear iteration.
            std::deque<Eigen::VectorXd> G_hist;
            std::deque<Eigen::VectorXd> x_hist;
            int iter_count = 0;

            // Returns the next iterate proposal, or std::nullopt to fall back
            // to a plain damped Picard step. Caller is responsible for the
            // finite-value check on the returned vector.
            std::optional<Eigen::VectorXd> step(const Eigen::VectorXd& x_k, const Eigen::VectorXd& G_k)
            {
                // Disabled, in warm-up, or no history -> plain Picard path.
                if (depth == 0 || iter_count < warmup_iters || G_hist.empty()) {
                    return std::nullopt;
                }

                const Eigen::VectorXd f_k = G_k - x_k;
                const int m_k = std::min(static_cast<int>(G_hist.size()), depth);

                // AA solve: F * alpha = f_k  (LS, m_k columns).
                // Build the m_k x m_k normal-equation system FᵀF · α = Fᵀ·f_k
                // to avoid a full N x m_k QR (m_k is O(5), N is the cell count).
                Eigen::MatrixXd FtF(m_k, m_k);
                Eigen::VectorXd Ftf(m_k);
                for (int i = 0; i < m_k; ++i) {
                    const Eigen::VectorXd& f_i = f_k - (G_hist[i] - x_hist[i]);
                    Ftf(i) = f_i.dot(f_k);
                    for (int j = i; j < m_k; ++j) {
                        const Eigen::VectorXd& f_j = f_k - (G_hist[j] - x_hist[j]);
                        FtF(i, j) = f_i.dot(f_j);
                        FtF(j, i) = FtF(i, j);
                    }
                }
                Eigen::VectorXd alpha = FtF.colPivHouseholderQr().solve(Ftf);

                if (dampening < 1.0) {
                    alpha *= dampening;
                }

                Eigen::VectorXd x_prop = (1.0 - alpha.sum()) * G_k;
                for (int j = 0; j < m_k; ++j) {
                    x_prop.noalias() += alpha(j) * G_hist[j];
                }

                // Divergence guard: compare infinity norms.
                if (max_growth > 0.0) {
                    const double prop_norm = (x_prop - x_k).cwiseAbs().maxCoeff();
                    const double naive_norm = (G_k - x_k).cwiseAbs().maxCoeff();
                    if (naive_norm > 0.0 && prop_norm > max_growth * naive_norm) {
                        if (reset_on_growth) {
                            G_hist.clear();
                            x_hist.clear();
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
                G_hist.push_front(G_k);
                x_hist.push_front(x_k);
                if (static_cast<int>(G_hist.size()) > depth) {
                    G_hist.pop_back();
                    x_hist.pop_back();
                }
                ++iter_count;
            }
        };

    } // namespace

    NonLinearResult nonlinear_solve(
        LinearSystemProvider ls_provider, std::vector<double>& state, LinearSolver& solver, const NonLinearConfig& cfg)
    {
        const double omega = cfg.underrelaxation > 0.0 ? cfg.underrelaxation : 1.0;
        const double rel_tol = cfg.relative_tolerance;
        const double abs_tol = cfg.absolute_tolerance;
        const mhs::core::Index N = static_cast<mhs::core::Index>(state.size());
        assert(N <= static_cast<mhs::core::Index>(std::numeric_limits<Eigen::Index>::max()));
        const auto eigen_N = static_cast<Eigen::Index>(N);
        Eigen::Map<Eigen::VectorXd> state_map(state.data(), eigen_N);

        AndersonMixer mixer;
        for (int iter = 0; iter < cfg.max_iterations; ++iter) {

            LinearSystem linear_system = ls_provider(state);
            const Eigen::VectorXd residual_vec = linear_system.b - linear_system.A * state_map;

            const double max_residual = residual_vec.cwiseAbs().maxCoeff();
            const double max_b = linear_system.b.cwiseAbs().maxCoeff();

            // Combined relative + absolute tolerance
            const double residual_threshold = rel_tol * max_b + abs_tol;

            if (iter > 0 && max_residual <= residual_threshold) {
                return {true, iter};
            }

            solver.compute(linear_system.A);
            const Eigen::VectorXd G_k = solver.solve(linear_system.b);
            if (!solver.success()) {
                throw std::runtime_error("linear solver failed at iteration " + std::to_string(iter));
            }
            const Eigen::VectorXd x_k = state_map; // capture pre-update state

            std::optional<Eigen::VectorXd> x_prop = mixer.step(x_k, G_k);

            Eigen::VectorXd next(eigen_N);
            const bool use_aa = x_prop.has_value() && x_prop->allFinite();
            if (use_aa) {
                next = std::move(*x_prop);
            }
            else {
                next = x_k + omega * (G_k - x_k);
            }

            const double max_update = (next - x_k).cwiseAbs().maxCoeff();
            const double max_state = next.cwiseAbs().maxCoeff();
            state_map = next;

            const double update_threshold = rel_tol * max_state + abs_tol;

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
