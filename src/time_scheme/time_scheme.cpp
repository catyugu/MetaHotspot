#include "common/logger.hpp"
#include "detail/build_ops.hpp"
#include "time_scheme.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>

namespace mhs::sim::time_scheme {

    class Bdf1Scheme : public TimeScheme {
        TimeSchemeConfig cfg_;

    public:
        explicit Bdf1Scheme(TimeSchemeConfig cfg) : cfg_(std::move(cfg)) { }

        StepDecision select_step(const mhs::core::SolutionHistory&, double, double) const override
        {
            return {cfg_.initial_dt, 1};
        }

        LinearSystem build_system(
            const AssemblyResult& ops, const mhs::core::SolutionHistory& h, std::size_t, double dt) const override
        {
            return detail::build_bdf1_ls(ops, h, dt);
        }

        StepResult evaluate_step(const mhs::core::SolutionHistory&, const std::vector<double>&, double) const override
        {
            return {true};
        }

        const TimeSchemeConfig& config() const override { return cfg_; }
    };

    class Bdf2Scheme : public TimeScheme {
        TimeSchemeConfig cfg_;

    public:
        explicit Bdf2Scheme(TimeSchemeConfig cfg) : cfg_(std::move(cfg)) { }

        StepDecision select_step(const mhs::core::SolutionHistory& h, double, double) const override
        {
            return {cfg_.initial_dt, h.size() >= 3 ? 2 : 1ull};
        }

        LinearSystem build_system(
            const AssemblyResult& ops, const mhs::core::SolutionHistory& h, std::size_t order, double dt) const override
        {
            return (order == 1 || h.size() <= order) ? detail::build_bdf1_ls(ops, h, dt)
                                                     : detail::build_bdf2_ls(ops, h, dt);
        }

        StepResult evaluate_step(const mhs::core::SolutionHistory&, const std::vector<double>&, double) const override
        {
            return {true};
        }

        const TimeSchemeConfig& config() const override { return cfg_; }
    };

    class AdaptiveBdfScheme : public TimeScheme {
        TimeSchemeConfig cfg_;
        mutable double last_dt_, next_dt_, optimal_dt_;
        mutable int output_step_ = 0;

    public:
        explicit AdaptiveBdfScheme(TimeSchemeConfig cfg)
            : cfg_(std::move(cfg)), last_dt_(cfg_.initial_dt), next_dt_(cfg_.initial_dt), optimal_dt_(cfg_.initial_dt)
        {
        }

        void initialize(mhs::core::SolutionHistory& accepted, mhs::core::GlobalState& state) const override
        {
            TimeScheme::initialize(accepted, state);
            optimal_dt_ = next_dt_ = last_dt_ = cfg_.initial_dt;
            output_step_ = 0;
        }

        StepDecision select_step(const mhs::core::SolutionHistory&, double current_t, double duration) const override
        {
            optimal_dt_ = std::clamp(next_dt_ > 0.0 ? next_dt_ : cfg_.initial_dt, cfg_.min_dt, cfg_.max_dt);
            double dt = optimal_dt_;

            if (cfg_.output_dt > 0.0) {
                double t_next = (output_step_ + 1) * cfg_.output_dt;
                if (t_next > current_t + 1e-12 && t_next < current_t + dt) {
                    dt = t_next - current_t;
                }
            }

            last_dt_ = dt = std::min(dt, duration - current_t);

            // NOTE: original code clamped to order 1 via std::min(..., 1ull).
            // AdaptiveBdf with BDF2 is not yet validated — the different
            // linear system produces LTE estimates that far exceed abs_tol,
            // causing dt to spiral down to min_dt and the simulation to take
            // billions of steps.  Keep order=1 (BDF1 / Backward Euler) until
            // a proper BDF2 calibration is done.
            return {dt, 1};
        }

        LinearSystem build_system(
            const AssemblyResult& ops, const mhs::core::SolutionHistory& h, std::size_t order, double dt) const override
        {
            // 自适应算法现在会基于 order 在 BDF1 和 BDF2 之间切换
            return (order == 1 || h.size() <= order) ? detail::build_bdf1_ls(ops, h, dt)
                                                     : detail::build_bdf2_ls(ops, h, dt);
        }

        StepResult evaluate_step(const mhs::core::SolutionHistory& accepted, const std::vector<double>& trial_T,
            double trial_dt) const override
        {

            const int N = static_cast<int>(trial_T.size());
            if (N == 0)
                return {true};

            // 全局避免使用 for 循环遍历原生数组，通过 Eigen 进行快速的 SIMD 向量化计算
            Eigen::Map<const Eigen::VectorXd> T_curr(trial_T.data(), N);
            Eigen::Map<const Eigen::VectorXd> T_prev(accepted.current().data(), N);

            Eigen::VectorXd err_vec;
            if (accepted.size() >= 2) {
                Eigen::Map<const Eigen::VectorXd> T_prev2(accepted.at(1).data(), N);
                double dt_prev = accepted.previous_dt();
                double ratio = (dt_prev > 1e-12) ? (trial_dt / dt_prev) : 1.0;

                // 真正的局部截断误差(LTE)
                err_vec = ((T_curr - T_prev) - ratio * (T_prev - T_prev2)).cwiseAbs();
            }
            else {
                err_vec = (T_curr - T_prev).cwiseAbs();
            }

            Eigen::VectorXd max_T = T_curr.cwiseAbs().cwiseMax(1.0);
            double err = err_vec.cwiseQuotient(max_T).maxCoeff();

            double fac = cfg_.safety * std::pow(cfg_.abs_tol / std::max(err, 1e-30), 0.5);
            double calculated_dt = last_dt_ * std::clamp(fac, 0.5, 2.0);

            // 补偿逻辑：如果强制被输出边界切断步长，则在条件安全下恢复最优步长势能
            if (last_dt_ < optimal_dt_ && fac >= 1.0) {
                calculated_dt = std::max(calculated_dt, optimal_dt_);
            }

            next_dt_ = std::clamp(calculated_dt, cfg_.min_dt, cfg_.max_dt);

            return {true};
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
        MHS_FATAL("Unknown TimeSchemeKind");
    }

} // namespace mhs::sim::time_scheme
