#include "logging/logger.hpp"
#include "runtime/constants.hpp"
#include "solver/assembler.hpp"
#include "solver/nonlinear_solver.hpp"
#include "solver/probe_recorder.hpp"
#include "solver/scheduler.hpp"
#include "solver/solution_history.hpp"
#include "solver/time_integration.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace mhs::sim {

    namespace {

        /// Linear interpolation of the state between two snapshots.
        inline std::vector<double> interpolate_state(
            double t0, std::span<const double> x0, double t1, std::span<const double> x1, double t)
        {
            const double dt = t1 - t0;
            if (dt <= 0.0)
                return std::vector<double>(x0.begin(), x0.end());
            const double s = (t - t0) / dt; // ∈ [0, 1]
            std::vector<double> out(x0.size());
            for (std::size_t i = 0; i < x0.size(); ++i)
                out[i] = x0[i] + s * (x1[i] - x0[i]);
            return out;
        }

        struct SolveRun {
            mhs::core::SolveResult result;
            std::vector<mhs::core::ProbeTrace> probe_traces;
        };

        void append_block(std::vector<Eigen::Triplet<double>>& entries,
            const Eigen::SparseMatrix<double>& block, Eigen::Index row_offset, Eigen::Index column_offset)
        {
            for (Eigen::Index outer = 0; outer < block.outerSize(); ++outer) {
                for (Eigen::SparseMatrix<double>::InnerIterator entry(block, outer); entry; ++entry) {
                    entries.emplace_back(
                        entry.row() + row_offset, entry.col() + column_offset, entry.value());
                }
            }
        }

        Eigen::SparseMatrix<double> assemble_diagonal_blocks(const Eigen::SparseMatrix<double>& model,
            const Eigen::SparseMatrix<double>& port, Eigen::Index model_count, Eigen::Index state_count)
        {
            std::vector<Eigen::Triplet<double>> entries;
            entries.reserve(static_cast<std::size_t>(model.nonZeros() + port.nonZeros()));
            append_block(entries, model, 0, 0);
            append_block(entries, port, model_count, model_count);

            Eigen::SparseMatrix<double> result(state_count, state_count);
            result.setFromTriplets(entries.begin(), entries.end());
            return result;
        }

        void validate_operator_dimensions(
            const Operators& operators, std::size_t dof_count, const char* contribution)
        {
            if (static_cast<std::size_t>(operators.K.rows()) != dof_count
                || static_cast<std::size_t>(operators.K.cols()) != dof_count
                || static_cast<std::size_t>(operators.C.rows()) != dof_count
                || static_cast<std::size_t>(operators.C.cols()) != dof_count
                || static_cast<std::size_t>(operators.f.size()) != dof_count) {
                throw std::invalid_argument(
                    "solve_coupled: " + std::string(contribution) + " K/C/f dimensions do not match");
            }
        }

        void validate_matrix_blocks(
            const CouplingMatrixBlocks& blocks, std::size_t model_count, std::size_t port_count, const char* name)
        {
            const bool valid = static_cast<std::size_t>(blocks.model.rows()) == model_count
                && static_cast<std::size_t>(blocks.model.cols()) == model_count
                && static_cast<std::size_t>(blocks.model_to_port.rows()) == model_count
                && static_cast<std::size_t>(blocks.model_to_port.cols()) == port_count
                && static_cast<std::size_t>(blocks.port_to_model.rows()) == port_count
                && static_cast<std::size_t>(blocks.port_to_model.cols()) == model_count
                && static_cast<std::size_t>(blocks.port.rows()) == port_count
                && static_cast<std::size_t>(blocks.port.cols()) == port_count;
            if (!valid) {
                throw std::invalid_argument(
                    "solve_coupled: " + std::string(name) + " coupling block dimensions do not match");
            }
        }

        void validate_coupling(
            const CouplingOperators& coupling, std::size_t model_count, std::size_t port_count, const char* name)
        {
            validate_matrix_blocks(coupling.K, model_count, port_count, name);
            validate_matrix_blocks(coupling.C, model_count, port_count, name);
            if (static_cast<std::size_t>(coupling.f_model.size()) != model_count
                || static_cast<std::size_t>(coupling.f_port.size()) != port_count) {
                throw std::invalid_argument(
                    "solve_coupled: " + std::string(name) + " coupling RHS dimensions do not match");
            }
        }

        void validate_coupled_input(const Operators& macro_port, const InterfaceCoupling& interface,
            std::size_t model_count, std::span<const double> initial_state)
        {
            const auto port_count = static_cast<std::size_t>(macro_port.f.size());
            validate_operator_dimensions(macro_port, port_count, "macro port");
            if (initial_state.size() != model_count + port_count) {
                throw std::invalid_argument(
                    "solve_coupled: initial_state must contain Model FVM DoFs followed by macro port DoFs");
            }
            if (interface.fixed) {
                validate_coupling(*interface.fixed, model_count, port_count, "fixed");
            }
        }

        Eigen::SparseMatrix<double> assemble_coupling_matrix(const CouplingMatrixBlocks& blocks,
            Eigen::Index model_count, Eigen::Index state_count)
        {
            std::vector<Eigen::Triplet<double>> entries;
            entries.reserve(static_cast<std::size_t>(blocks.model.nonZeros() + blocks.model_to_port.nonZeros()
                + blocks.port_to_model.nonZeros() + blocks.port.nonZeros()));
            append_block(entries, blocks.model, 0, 0);
            append_block(entries, blocks.model_to_port, 0, model_count);
            append_block(entries, blocks.port_to_model, model_count, 0);
            append_block(entries, blocks.port, model_count, model_count);

            Eigen::SparseMatrix<double> result(state_count, state_count);
            result.setFromTriplets(entries.begin(), entries.end());
            return result;
        }

        void add_coupling(
            Operators& target, const CouplingOperators& coupling, std::size_t model_count, std::size_t port_count)
        {
            const auto eigen_model_count = static_cast<Eigen::Index>(model_count);
            const auto eigen_state_count = static_cast<Eigen::Index>(model_count + port_count);
            target.K += assemble_coupling_matrix(coupling.K, eigen_model_count, eigen_state_count);
            target.C += assemble_coupling_matrix(coupling.C, eigen_model_count, eigen_state_count);
            target.f.head(eigen_model_count) += coupling.f_model;
            target.f.tail(static_cast<Eigen::Index>(port_count)) += coupling.f_port;
        }

        Operators assemble_system(const mhs::core::Model& model, const Operators* macro_port,
            const InterfaceCoupling* interface, std::span<const double> state, std::size_t model_count, double time)
        {
            auto model_operators = assemble_thermal(model, state.first(model_count), time);
            if (macro_port == nullptr)
                return model_operators;

            const auto port_count = static_cast<std::size_t>(macro_port->f.size());
            const auto eigen_model_count = static_cast<Eigen::Index>(model_count);
            const auto eigen_state_count = static_cast<Eigen::Index>(model_count + port_count);

            Operators combined;
            combined.K = assemble_diagonal_blocks(
                model_operators.K, macro_port->K, eigen_model_count, eigen_state_count);
            combined.C = assemble_diagonal_blocks(
                model_operators.C, macro_port->C, eigen_model_count, eigen_state_count);
            combined.f.resize(eigen_state_count);
            combined.f.head(eigen_model_count) = model_operators.f;
            combined.f.tail(static_cast<Eigen::Index>(port_count)) = macro_port->f;

            if (interface->fixed) {
                add_coupling(combined, *interface->fixed, model_count, port_count);
            }
            if (interface->nonlinear) {
                auto nonlinear = interface->nonlinear(
                    state.first(model_count), state.subspan(model_count), time);
                validate_coupling(nonlinear, model_count, port_count, "nonlinear");
                add_coupling(combined, nonlinear, model_count, port_count);
            }
            return combined;
        }

        void record_fvm_state(
            ProbeRecorder& recorder, double time, std::span<const double> state, std::size_t fvm_count)
        {
            recorder.record(time, state.first(fvm_count));
        }

    } // anonymous namespace

    static SolveRun solve_model(const mhs::core::Model& model, const Operators* macro_port,
        const InterfaceCoupling* interface, std::span<const double> initial_state, const SolverOpts& opts)
    {
        const auto fvm_count = static_cast<std::size_t>(model.cells.cell_to_grid.size());
        std::size_t state_count = fvm_count;
        if (macro_port != nullptr) {
            validate_coupled_input(*macro_port, *interface, fvm_count, initial_state);
            state_count += static_cast<std::size_t>(macro_port->f.size());
        }
        else if (!initial_state.empty() && initial_state.size() != fvm_count) {
            throw std::invalid_argument("solve_thermal: initial_state.size() = " + std::to_string(initial_state.size())
                + " != cell_count = " + std::to_string(fvm_count));
        }

        auto solver = LinearSolver::create(opts.solver);

        std::vector<double> state;
        if (initial_state.empty())
            state.assign(fvm_count, model.initial_temperature);
        else
            state.assign(initial_state.begin(), initial_state.end());

        double current_time = 0.0;

        mhs::core::SolutionHistory accepted {state_count, 2};
        ProbeRecorder probe_recorder;
        probe_recorder.initialize(model);

        // Steady: single non-linear solve, then output.
        if (model.study_type == mhs::core::StudyType::Steady) {
            LinearSystemProvider build_ls = [&](std::span<const double> s) -> LinearSystem {
                auto ops = assemble_system(model, macro_port, interface, s, fvm_count, 0.0);
                return {std::move(ops.K), std::move(ops.f)};
            };
            auto nl_result = nonlinear_solve(build_ls, state, *solver, opts.nonlinear);
            record_fvm_state(probe_recorder, 0.0, state, fvm_count);
            return {{std::move(state), current_time, nl_result.converged}, probe_recorder.traces()};
        }

        // Transient.
        const double duration = model.transient_duration;
        const double output_dt = model.transient_time_step;

        const double min_dt = opts.min_dt;
        const double max_dt = opts.max_dt;

        time_scheme::StepController step_ctrl {opts.step_strategy, min_dt, max_dt, duration, output_dt, opts.fixed_dt};

        accepted.initialize(state, current_time);
        record_fvm_state(probe_recorder, current_time, state, fvm_count);

        double dt_sug = std::clamp(output_dt, min_dt, max_dt);

        while (current_time < duration - mhs::core::zero_guard) {
            double dt = step_ctrl.prepare(dt_sug, current_time, duration);
            if (dt <= 0.0)
                break;

            // Build the linearised system at (state, time + dt)
            LinearSystemProvider ls_provider = [&](std::span<const double> iter_state) -> LinearSystem {
                auto ops = assemble_system(model, macro_port, interface, iter_state, fvm_count, current_time + dt);
                return time_scheme::build_system(opts.integrator, ops, accepted, dt);
            };

            // Save state before trial so we can restore on rejection
            auto saved_state = state;

            // Non-linear solve (Picard/Anderson)
            auto nl = nonlinear_solve(ls_provider, state, *solver, opts.nonlinear);

            if (!nl.converged) {
                // Restore clean state before retrying
                state = std::move(saved_state);
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={} (nonlinear), retry dt={}", current_time, dt_sug);

                // Fatal: nonlinear divergence at minimum dt
                if (dt <= min_dt * 1.0001) {
                    MHS_LOG_WARN("Nonlinear solver diverged at minimum dt t={}", current_time);
                    auto final_out = step_ctrl.flush_outputs(current_time);
                    for (double t_out : final_out) {
                        record_fvm_state(probe_recorder, t_out, state, fvm_count);
                    }
                    return {{std::move(state), current_time, false}, probe_recorder.traces()};
                }
                continue;
            }

            bool accepted_step = true;
            double suggested_dt_factor = 1.0;

            // Fixed strategy: skip LTE-based rejection entirely
            if (opts.step_strategy == time_scheme::StepStrategy::Fixed) {
                accepted.accept(state, current_time + dt);
            }
            else {
                // Error estimation (LTE check for adaptive stepping)
                auto est = time_scheme::estimate_error(accepted, state, dt, {opts.error_abs_tol, opts.error_safety});
                suggested_dt_factor = est.suggested_factor;
                accepted_step = (est.error_ratio <= 1.0) || (dt <= min_dt * 1.0001);

                if (accepted_step) {
                    accepted.accept(state, current_time + dt);
                }
            }

            if (accepted_step) {
                current_time += dt;
                MHS_LOG_DEBUG("Time: {} solved (dt={})", current_time, dt);

                // AdaptiveFree mode: step end may overshoot output time → interpolate.
                auto out = step_ctrl.flush_outputs(current_time);
                for (double t_out : out) {
                    auto state_at_output = interpolate_state(
                        /* t0 = */ accepted.time_at(1),
                        /* x0 = */ accepted.at(1),
                        /* t1 = */ current_time,
                        /* x1 = */ state, t_out);
                    record_fvm_state(probe_recorder, t_out, state_at_output, fvm_count);
                }

                // Fixed mode → keep fixed dt; otherwise adapt from error estimate.
                dt_sug = (opts.step_strategy == time_scheme::StepStrategy::Fixed)
                    ? opts.fixed_dt
                    : std::clamp(dt * suggested_dt_factor, min_dt, max_dt);
            }
            else {
                // Restore clean state before retrying
                state = std::move(saved_state);
                dt_sug = dt * 0.5;
                MHS_LOG_DEBUG("Step rejected at t={} (LTE), retry dt={}", current_time, dt_sug);
            }
        }

        // Final flush — ensure last output times are recorded
        auto final_out = step_ctrl.flush_outputs(current_time);
        for (double t_out : final_out) {
            record_fvm_state(probe_recorder, t_out, state, fvm_count);
        }

        return {{std::move(state), current_time, true}, probe_recorder.traces()};
    }

    mhs::core::SolveResult solve_coupled(const mhs::core::Model& model, const Operators& macro_port,
        const InterfaceCoupling& interface, std::span<const double> initial_state, const SolverOpts& opts)
    {
        return solve_model(model, &macro_port, &interface, initial_state, opts).result;
    }

    mhs::core::ThermalSolution solve_thermal(
        const mhs::core::Model& model, const SolverOpts& opts, std::span<const double> initial_state)
    {
        auto run = solve_model(model, nullptr, nullptr, initial_state, opts);

        mhs::core::ThermalSolution result;
        result.temperature = std::move(run.result.state);
        result.time = run.result.time;
        result.converged = run.result.converged;
        result.probe_traces = std::move(run.probe_traces);
        return result;
    }

} // namespace mhs::sim
