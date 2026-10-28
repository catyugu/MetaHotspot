#include "reference.hpp"

#include "numerics/linear/linear_solver.hpp"

#include <Eigen/Sparse>

#include <chrono>
#include <cmath>
#include <cstddef>

namespace mhs::hdim {

    std::vector<double> solve_reference(const Model& model, const std::vector<double>& previous, double dt,
        double tolerance, int max_sweeps, RunInfo& info)
    {
        const auto start = std::chrono::steady_clock::now();
        const auto n = static_cast<Eigen::Index>(model.mesh.n);
        info = RunInfo {};

        std::vector<double> temperature = previous;
        if (temperature.empty())
            temperature.assign(static_cast<std::size_t>(n), kAmbient);
        // The memory term of eq. (3) is the *incoming* field: it must not follow
        // the sweeps, otherwise the sweeps advance the solution in time instead of
        // converging on one step.
        const std::vector<double> incoming = temperature;
        Eigen::VectorXd solution = Eigen::Map<const Eigen::VectorXd>(temperature.data(), n);

        const mhs::sim::SolverSpec spec;
        for (int sweep = 1; sweep <= max_sweeps; ++sweep) {
            const RegionSystem whole = assemble_whole(model, temperature, dt);
            Eigen::VectorXd rhs = whole.source;
            if (dt > 0.0) {
                const Eigen::Map<const Eigen::VectorXd> old(incoming.data(), n);
                rhs += whole.capacity.cwiseProduct(old) / dt;
            }

            const auto solver = mhs::sim::create_solver(spec);
            solver->compute(whole.K);
            const Eigen::VectorXd next = solver->solve(rhs, solution);
            ++info.solves;

            const double change = (next - solution).cwiseAbs().maxCoeff();
            solution = next;
            Eigen::Map<Eigen::VectorXd>(temperature.data(), n) = solution;
            info.sweeps = sweep;
            if (!(change >= tolerance))
                break;
        }

        info.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return temperature;
    }

} // namespace mhs::hdim
