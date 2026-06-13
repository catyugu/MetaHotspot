#include "common/logger.hpp"
#include "time_scheme.hpp"
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>

namespace mhs::sim::time_scheme {

    namespace {

        LinearSystem build_bdf1_ls(
            const StaticOpsResult& sops, const MassOpsResult& mops, const mhs::core::TimeStepBuffer& history, double dt)
        {
            const int N = static_cast<int>(sops.f_static.size());
            const auto& T_prev = history.latest();
            Eigen::SparseMatrix<double> A = sops.K;
            Eigen::VectorXd b = sops.f_static;

            for (int i = 0; i < N; ++i) {
                A.coeffRef(i, i) += mops.M_diag(i) / dt;
                b(i) += mops.M_diag(i) * T_prev[i] / dt;
            }
            return {A, b, b - A * Eigen::Map<const Eigen::VectorXd>(T_prev.data(), N)};
        }

        LinearSystem build_bdf2_ls(
            const StaticOpsResult& sops, const MassOpsResult& mops, const mhs::core::TimeStepBuffer& history, double dt)
        {
            const int N = static_cast<int>(sops.f_static.size());
            const double h_n = dt, h_np1 = history.dt_to(1);
            const double delta = (h_np1 > 0.0) ? h_n / h_np1 : 1.0;
            const double alpha0 = (1.0 + 2.0 * delta) / (h_n * (1.0 + delta));
            const double alpha1 = -(1.0 + delta) / (h_n * delta);
            const double alpha2 = delta / (h_n * (1.0 + delta));

            const auto& T_n = history.latest();
            const auto& T_nm1 = history.at(1);
            const auto& T_nm2 = history.at(2);
            Eigen::SparseMatrix<double> A = sops.K;
            Eigen::VectorXd b = sops.f_static;

            for (int i = 0; i < N; ++i) {
                A.coeffRef(i, i) += alpha0 * mops.M_diag(i);
                b(i) += alpha0 * mops.M_diag(i) * T_n[i] + alpha1 * mops.M_diag(i) * T_nm1[i]
                    + alpha2 * mops.M_diag(i) * T_nm2[i];
            }
            return {A, b, b - A * Eigen::Map<const Eigen::VectorXd>(T_n.data(), N)};
        }

        class Bdf1Scheme : public TimeScheme {
            TimeSchemeConfig cfg_;

        public:
            explicit Bdf1Scheme(TimeSchemeConfig cfg) : cfg_(std::move(cfg)) { }
            StepDecision select_step(const mhs::core::TimeStepBuffer&, double, double) const override
            {
                return {cfg_.initial_dt, 1};
            }
            LinearSystem build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
                const mhs::core::TimeStepBuffer& h, std::size_t, double dt) const override
            {
                return build_bdf1_ls(sops, mops, h, dt);
            }
            const TimeSchemeConfig& config() const override { return cfg_; }
        };

        class Bdf2Scheme : public TimeScheme {
            TimeSchemeConfig cfg_;

        public:
            explicit Bdf2Scheme(TimeSchemeConfig cfg) : cfg_(std::move(cfg)) { }
            StepDecision select_step(const mhs::core::TimeStepBuffer& h, double, double) const override
            {
                return {cfg_.initial_dt, h.size() >= 3 ? 2 : 1ull};
            }
            LinearSystem build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
                const mhs::core::TimeStepBuffer& h, std::size_t order, double dt) const override
            {
                return (order == 1 || h.size() <= order) ? build_bdf1_ls(sops, mops, h, dt)
                                                         : build_bdf2_ls(sops, mops, h, dt);
            }
            const TimeSchemeConfig& config() const override { return cfg_; }
        };

        class AdaptiveBdfScheme : public TimeScheme {
            TimeSchemeConfig cfg_;
            mutable double last_dt_, next_dt_, optimal_dt_; // 新增 optimal_dt_ 用于记忆被截断前的理想步长
            mutable int output_step_ = 0;

        public:
            explicit AdaptiveBdfScheme(TimeSchemeConfig cfg)
                : cfg_(std::move(cfg))
                , last_dt_(cfg_.initial_dt)
                , next_dt_(cfg_.initial_dt)
                , optimal_dt_(cfg_.initial_dt)
            {
            }

            void initialize(mhs::core::TimeStepBuffer& history, mhs::core::GlobalState& state) const override
            {
                TimeScheme::initialize(history, state);
                optimal_dt_ = next_dt_ = last_dt_ = cfg_.initial_dt;
                output_step_ = 0;
            }

            StepDecision select_step(
                const mhs::core::TimeStepBuffer& h, double current_t, double duration) const override
            {
                // 先计算不受输出约束影响的理想步长
                optimal_dt_ = std::clamp(next_dt_ > 0.0 ? next_dt_ : cfg_.initial_dt, cfg_.min_dt, cfg_.max_dt);
                double dt = optimal_dt_;

                if (cfg_.output_dt > 0.0) {
                    double t_next = (output_step_ + 1) * cfg_.output_dt;
                    if (t_next > current_t + 1e-12 && t_next < current_t + dt) {
                        dt = t_next - current_t; // 步长被强制截断变小
                    }
                }
                last_dt_ = dt = std::min(dt, duration - current_t);
                return {dt, std::min(h.size() > cfg_.max_order ? cfg_.max_order : 1, 1ull)};
            }

            LinearSystem build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
                const mhs::core::TimeStepBuffer& h, std::size_t, double dt) const override
            {
                return build_bdf1_ls(sops, mops, h, dt);
            }

            void accept_or_reject(const std::vector<double>& err_est) const override
            {
                double err = 0.0;
                for (double v : err_est)
                    err = std::max(err, std::abs(v));
                double fac = cfg_.safety * std::pow(cfg_.abs_tol / std::max(err, 1e-30), 0.5); // order 1 estimation

                double calculated_dt = last_dt_ * std::clamp(fac, 0.5, 2.0);

                // 如果刚才这一步是被强制截断变小的（说明此时系统的物理变化并不剧烈）
                // 且误差估计认为当前步长是安全的，则应该恢复原本应该使用的最佳步长，防止无意义地失去爬坡势能
                if (last_dt_ < optimal_dt_ && fac >= 1.0) {
                    calculated_dt = std::max(calculated_dt, optimal_dt_);
                }

                next_dt_ = std::clamp(calculated_dt, cfg_.min_dt, cfg_.max_dt);
            }

            bool is_output_boundary(double t) const override
            {
                if (cfg_.output_dt <= 0.0)
                    return true;
                if (std::abs(t - (output_step_ + 1) * cfg_.output_dt) <= 1e-9 * std::max(1.0, t)) {
                    ++output_step_;
                    return true;
                }
                return false;
            }

            const TimeSchemeConfig& config() const override { return cfg_; }
        };

    } // namespace

    std::unique_ptr<TimeScheme> create_scheme(const TimeSchemeConfig& cfg)
    {
        switch (cfg.kind) {
        case TimeSchemeKind::Bdf1:
            return std::make_unique<Bdf1Scheme>(cfg);
        case TimeSchemeKind::Bdf2:
            return std::make_unique<Bdf2Scheme>(cfg);
        case TimeSchemeKind::AdaptiveBdf:
            return std::make_unique<AdaptiveBdfScheme>(cfg);
        }
        MHS_LOG_ERROR("Unknown TimeSchemeKind");
    }

} // namespace mhs::sim::time_scheme