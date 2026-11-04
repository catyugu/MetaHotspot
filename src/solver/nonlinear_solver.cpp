#include "solver/nonlinear_solver.hpp"

#include "core/csr_ops.hpp"
#include "core/matrix.hpp"
#include "core/vector_ops.hpp"
#include "logging/logger.hpp"
#include <Eigen/QR>
#include <deque>
#include <numeric>
#include <optional>

namespace mhs::solver {
    namespace {
        using core::Vector;
        struct AndersonMixer {
            int depth = 5;
            int warmup_iters = 2;
            double dampening = 0.8;
            double max_growth = 1.5;
            int reset_on_growth = 1;
            std::deque<Vector<double>> G_hist, f_hist;
            int iter_count = 0;

            std::optional<Vector<double>> step(std::span<const double> x_k, std::span<const double> G_k)
            {
                if (depth == 0 || iter_count < warmup_iters || G_hist.empty())
                    return std::nullopt;
                const auto m = std::min(G_hist.size(), static_cast<std::size_t>(depth));
                // Assemble the same small Gram system without full-length
                // temporary difference vectors for each pair of columns.
                core::Matrix<double> gram(m, m);
                Vector<double> rhs(m, 0.0), difference(m);
                for (std::size_t k = 0; k < x_k.size(); ++k) {
                    const double f = G_k[k] - x_k[k];
                    for (std::size_t i = 0; i < m; ++i) {
                        difference[i] = f - f_hist[i][k];
                        rhs[i] += difference[i] * f;
                        for (std::size_t j = 0; j <= i; ++j)
                            gram(i, j) += difference[i] * difference[j];
                    }
                }
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < i; ++j)
                        gram(j, i) = gram(i, j);
                using Dense = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                const auto count = static_cast<Eigen::Index>(m);
                Eigen::Map<const Dense> matrix(gram.data(), count, count);
                Eigen::Map<const Eigen::VectorXd> load(rhs.data(), count);
                Vector<double> alpha(m);
                Eigen::Map<Eigen::VectorXd> coefficients(alpha.data(), count);
                coefficients = matrix.colPivHouseholderQr().solve(load);
                if (dampening < 1.0)
                    for (auto& a : alpha)
                        a *= dampening;
                const double weight = 1.0 - std::accumulate(alpha.begin(), alpha.end(), 0.0);
                Vector<double> proposal(x_k.size());
                for (std::size_t k = 0; k < proposal.size(); ++k) {
                    proposal[k] = weight * G_k[k];
                    for (std::size_t j = 0; j < m; ++j)
                        proposal[k] += alpha[j] * G_hist[j][k];
                }
                if (max_growth > 0.0) {
                    const double naive_norm = core::distance_inf(G_k, x_k);
                    if (naive_norm > 0.0 && core::distance_inf(proposal, x_k) > max_growth * naive_norm) {
                        if (reset_on_growth) {
                            G_hist.clear();
                            f_hist.clear();
                        }
                        return std::nullopt;
                    }
                }
                return proposal;
            }
            void push(const Vector<double>& x_k, const Vector<double>& G_k)
            {
                if (depth == 0)
                    return;
                G_hist.push_front(G_k);
                auto& f = f_hist.emplace_front(x_k.size());
                for (std::size_t k = 0; k < f.size(); ++k)
                    f[k] = G_k[k] - x_k[k];
                if (G_hist.size() > static_cast<std::size_t>(depth)) {
                    G_hist.pop_back();
                    f_hist.pop_back();
                }
                ++iter_count;
            }
        };
    }

    bool nonlinear_solve(LinearSystemProvider ls_provider, core::Vector<double>& state, linear::SolverPtr& solver,
        const NonLinearConfig& cfg)
    {
        const double omega = cfg.underrelaxation > 0.0 ? cfg.underrelaxation : 1.0;
        const double rel_tol = cfg.relative_tolerance, abs_tol = cfg.absolute_tolerance;
        AndersonMixer mixer;
        auto G_k = state;
        core::Vector<double> residual(state.size()), next(state.size());
        for (int iter = 0; iter < cfg.max_iterations; ++iter) {
            auto system = ls_provider(state);
            core::spmv(system.A.view(), state, residual);
            const double max_residual = core::distance_inf(system.b, residual);
            const double residual_threshold = rel_tol * core::norm_inf(system.b) + abs_tol;
            if (iter > 0 && max_residual <= residual_threshold)
                return true;
            solver->compute(system.A.view());
            const auto result = solver->solve(system.b, G_k);
            if (!result.success)
                throw std::runtime_error("linear solver failed at iteration " + std::to_string(iter));
            auto proposal = mixer.step(state, G_k);
            const bool use_aa = proposal.has_value() && core::all_finite(*proposal);
            if (use_aa)
                next = std::move(*proposal);
            else
                for (std::size_t i = 0; i < next.size(); ++i)
                    next[i] = state[i] + omega * (G_k[i] - state[i]);
            const double max_update = core::distance_inf(next, state);
            const double update_threshold = rel_tol * core::norm_inf(next) + abs_tol;
            mixer.push(state, G_k);
            state.swap(next);
            MHS_LOG_DEBUG("\t->Non-Linear iteration {}: max_update={:.6e}, max_residual={:.6e} ({}AA)", iter,
                max_update, max_residual, use_aa ? "" : "no ");
            if (max_update <= update_threshold && max_residual <= residual_threshold)
                return true;
        }
        return false;
    }
} // namespace mhs::solver
