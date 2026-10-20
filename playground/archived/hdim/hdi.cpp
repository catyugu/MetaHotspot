#include "hdi.hpp"

#include "numerics/linear/linear_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>

namespace mhs::hdim {

    namespace {

        mhs::sim::SolverPtr make_solver(const Eigen::SparseMatrix<double>& matrix, double tolerance)
        {
            mhs::sim::SolverSpec spec;
            spec.config.tolerance = tolerance;
            auto solver = mhs::sim::create_solver(spec);
            solver->compute(matrix);
            return solver;
        }

    } // namespace

    Partition Partition::build(const std::vector<InterfaceFace>& faces, double step)
    {
        Partition partition;
        partition.block.resize(faces.size());
        if (step <= 0.0) {
            partition.count = static_cast<int>(faces.size());
            for (std::size_t f = 0; f < faces.size(); ++f)
                partition.block[f] = static_cast<int>(f);
            return partition;
        }
        partition.nx = std::max(1, static_cast<int>(std::lround(kChipX / step)));
        partition.ny = std::max(1, static_cast<int>(std::lround(kChipY / step)));
        partition.count = partition.nx * partition.ny;
        for (std::size_t f = 0; f < faces.size(); ++f) {
            const double u = (faces[f].cx + 0.5 * kChipX) / kChipX;
            const double v = (faces[f].cy + 0.5 * kChipY) / kChipY;
            const int i = std::clamp(static_cast<int>(std::floor(u * partition.nx)), 0, partition.nx - 1);
            const int j = std::clamp(static_cast<int>(std::floor(v * partition.ny)), 0, partition.ny - 1);
            partition.block[f] = j * partition.nx + i;
        }
        return partition;
    }

    void LinearHalf::build(const Model& model, const Assembled& assembled, int region_index, double interface_step,
        double tolerance)
    {
        region = region_index;
        system = assembled.region[region];
        global_cell = model.region_cells[region];

        std::vector<InterfaceFace> own; // this interface, in mesh order
        for (std::size_t f = 0; f < assembled.interfaces.size(); ++f)
            if (assembled.interfaces[f].linear_region == region) {
                own.push_back(assembled.interfaces[f]);
                face.push_back(static_cast<int>(f));
            }
        partition = Partition::build(own, interface_step);

        cell.resize(own.size());
        area.resize(own.size());
        for (std::size_t f = 0; f < own.size(); ++f) {
            cell[f] = model.local[static_cast<std::size_t>(own[f].linear_cell)];
            area[f] = own[f].area;
        }

        solver = make_solver(system.K, tolerance);

        // Eq. (8): one unit-power excitation per block, spread over the block by
        // area, read back as the area-averaged interface temperature of a block.
        const auto n = system.capacity.size();
        const Eigen::VectorXd zero = Eigen::VectorXd::Zero(n);
        std::vector<std::vector<int>> members(static_cast<std::size_t>(partition.count));
        Eigen::VectorXd block_area = Eigen::VectorXd::Zero(partition.count);
        for (std::size_t f = 0; f < own.size(); ++f) {
            members[static_cast<std::size_t>(partition.block[f])].push_back(static_cast<int>(f));
            block_area(partition.block[f]) += area[f];
        }

        R = Eigen::MatrixXd::Zero(partition.count, partition.count);
        for (int excited = 0; excited < partition.count; ++excited) {
            const auto& sources = members[static_cast<std::size_t>(excited)];
            if (sources.empty())
                continue;
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
            for (const int f : sources)
                rhs(cell[static_cast<std::size_t>(f)]) += area[static_cast<std::size_t>(f)] / block_area(excited);
            const Eigen::VectorXd response = solver->solve(rhs, zero);
            for (int observed = 0; observed < partition.count; ++observed) {
                if (members[static_cast<std::size_t>(observed)].empty())
                    continue;
                double total = 0.0;
                for (const int f : members[static_cast<std::size_t>(observed)])
                    total += area[static_cast<std::size_t>(f)] * response(cell[static_cast<std::size_t>(f)]);
                R(observed, excited) = total / block_area(observed);
            }
        }
        // The exact map is symmetric; the linear solves that build it are not.
        R = 0.5 * (R + R.transpose());
    }

    void LinearHalf::solve_state(const std::vector<double>& previous, double dt)
    {
        Eigen::VectorXd rhs = system.source;
        if (dt > 0.0)
            for (std::size_t idx = 0; idx < global_cell.size(); ++idx) {
                const auto local = static_cast<Eigen::Index>(idx);
                rhs(local) += system.capacity(local) * previous[static_cast<std::size_t>(global_cell[idx])] / dt;
            }
        if (state.size() != rhs.size())
            state = Eigen::VectorXd::Zero(rhs.size());
        state = solver->solve(rhs, state);

        // Eq. (7)'s T0: with no flux through the interface the interface
        // temperature is the one of the first cell behind the block.
        load.setZero(partition.count);
        std::vector<double> block_area(static_cast<std::size_t>(partition.count), 0.0);
        for (std::size_t f = 0; f < face.size(); ++f) {
            load(partition.block[f]) += area[f] * state(cell[f]);
            block_area[static_cast<std::size_t>(partition.block[f])] += area[f];
        }
        for (int b = 0; b < partition.count; ++b)
            if (block_area[static_cast<std::size_t>(b)] > 0.0)
                load(b) /= block_area[static_cast<std::size_t>(b)];
    }

    Eigen::VectorXd LinearHalf::solve_flux(const std::vector<double>& flux) const
    {
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(system.capacity.size());
        for (std::size_t f = 0; f < face.size(); ++f)
            rhs(cell[f]) += flux[static_cast<std::size_t>(face[f])];
        return solver->solve(rhs, Eigen::VectorXd::Zero(rhs.size()));
    }

    Hdi::Hdi(const Model& model, double dt, double interface_step, double linear_tolerance)
        : model_(model)
        , dt_(dt)
        , linear_tolerance_(linear_tolerance)
    {
        const auto start = std::chrono::steady_clock::now();
        const Assembled assembled = assemble(model, {}, dt); // the linear regions do not depend on T
        for (int i = 0; i < kInterfaceCount; ++i) {
            half_[i].build(model, assembled, kInterfaceLinear[static_cast<std::size_t>(i)], interface_step,
                linear_tolerance);
            ti_[i] = Eigen::VectorXd::Zero(half_[i].partition.count);
        }
        presimulation_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    int Hdi::interface_blocks() const
    {
        return half_[0].partition.count + half_[1].partition.count;
    }

    void Hdi::interface_flux(const Assembled& assembled, const std::vector<double>& field)
    {
        flux_.assign(assembled.interfaces.size(), 0.0);
        for (int i = 0; i < kInterfaceCount; ++i) {
            const LinearHalf& half = half_[i];
            Eigen::VectorXd block_flux = Eigen::VectorXd::Zero(half.partition.count);
            gain_[i].assign(half.face.size(), 0.0);
            for (std::size_t f = 0; f < half.face.size(); ++f) {
                const InterfaceFace& face = assembled.interfaces[static_cast<std::size_t>(half.face[f])];
                const int block = half.partition.block[f];
                gain_[i][f] = interface_conductance(face);
                flux_[static_cast<std::size_t>(half.face[f])]
                    = gain_[i][f] * (field[static_cast<std::size_t>(face.nonlinear_cell)] - ti_[i](block));
                block_flux(block) += flux_[static_cast<std::size_t>(half.face[f])];
            }
            // Eq. (7) with the flux that actually crossed the interface, which is
            // eq. (10) once the nonlinear region is converged.
            ti_[i] = half.R * block_flux + half.load;
        }
    }

    double Hdi::sweep(std::vector<double>& field, RunInfo& info)
    {
        const Assembled assembled = assemble(model_, field, dt_, kNonlinear);
        interface_flux(assembled, field);

        // Eq. (10) as the Robin condition of the nonlinear region: the series
        // conductance of the two half cells stays in the operator, so the region
        // is well posed and the interface loop contracts; the interface
        // temperature of the previous flux is the load.
        const RegionSystem& nonlinear = assembled.region[kNonlinear];
        const auto n = nonlinear.capacity.size();
        Eigen::VectorXd rhs = nonlinear.source;
        if (dt_ > 0.0)
            for (std::size_t idx = 0; idx < model_.region_cells[kNonlinear].size(); ++idx) {
                const auto local = static_cast<Eigen::Index>(idx);
                rhs(local) += nonlinear.capacity(local) * incoming_[static_cast<std::size_t>(model_.region_cells[kNonlinear][idx])]
                    / dt_;
            }
        Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < kInterfaceCount; ++i) {
            const LinearHalf& half = half_[i];
            for (std::size_t f = 0; f < half.face.size(); ++f) {
                const InterfaceFace& face = assembled.interfaces[static_cast<std::size_t>(half.face[f])];
                const auto local = static_cast<Eigen::Index>(model_.local[static_cast<std::size_t>(face.nonlinear_cell)]);
                diagonal(local) += gain_[i][f];
                rhs(local) += gain_[i][f] * ti_[i](half.partition.block[f]);
            }
        }

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<std::size_t>(nonlinear.K.nonZeros()) + static_cast<std::size_t>(n));
        for (int k = 0; k < nonlinear.K.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(nonlinear.K, k); it; ++it)
                triplets.emplace_back(it.row(), it.col(), it.value());
        for (Eigen::Index idx = 0; idx < n; ++idx)
            if (diagonal(idx) > 0.0)
                triplets.emplace_back(idx, idx, diagonal(idx));
        Eigen::SparseMatrix<double> matrix(n, n);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();

        if (nonlinear_state_.size() != rhs.size())
            nonlinear_state_ = Eigen::VectorXd::Constant(rhs.size(), kAmbient);
        const auto solver = make_solver(matrix, linear_tolerance_);
        const Eigen::VectorXd next = solver->solve(rhs, nonlinear_state_);
        ++info.solves;

        double change = 0.0;
        for (std::size_t idx = 0; idx < model_.region_cells[kNonlinear].size(); ++idx) {
            const auto local = static_cast<Eigen::Index>(idx);
            change = std::max(change, std::abs(next(local) - nonlinear_state_(local)));
            field[static_cast<std::size_t>(model_.region_cells[kNonlinear][idx])] = next(local);
        }
        nonlinear_state_ = next;
        return change;
    }

    std::vector<double> Hdi::advance(const std::vector<double>& previous, double tolerance, int max_sweeps,
        RunInfo& info)
    {
        const auto start = std::chrono::steady_clock::now();
        info = RunInfo {};

        std::vector<double> field = previous;
        if (field.empty())
            field.assign(static_cast<std::size_t>(model_.mesh.n), kAmbient);
        incoming_ = field; // the memory term of eq. (3) must not follow the sweeps
        nonlinear_state_.resize(static_cast<Eigen::Index>(model_.region_cells[kNonlinear].size()));
        for (std::size_t idx = 0; idx < model_.region_cells[kNonlinear].size(); ++idx)
            nonlinear_state_(static_cast<Eigen::Index>(idx))
                = field[static_cast<std::size_t>(model_.region_cells[kNonlinear][idx])];

        // Eq. (4)-(5) with no interface flux yet: the zero-input response, which
        // carries the previous step, plus the fixed boundary load.
        for (LinearHalf& half : half_)
            half.solve_state(field, dt_);

        for (int sweep_index = 1; sweep_index <= max_sweeps; ++sweep_index) {
            const double change = sweep(field, info);
            info.sweeps = sweep_index;
            if (!(change >= tolerance))
                break;
        }

        // The flux of the converged interface completes eq. (5) for the two
        // linear regions.
        interface_flux(assemble(model_, field, dt_, kNonlinear), field);
        for (LinearHalf& half : half_) {
            const Eigen::VectorXd driven = half.solve_flux(flux_);
            ++info.solves;
            for (std::size_t idx = 0; idx < half.global_cell.size(); ++idx) {
                const auto local = static_cast<Eigen::Index>(idx);
                field[static_cast<std::size_t>(half.global_cell[idx])] = half.state(local) + driven(local);
            }
        }
        info.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return field;
    }

    std::vector<double> Hdi::steady(double tolerance, int max_sweeps, RunInfo& info)
    {
        return advance({}, tolerance, max_sweeps, info);
    }

    std::vector<double> Hdi::step(const std::vector<double>& previous, double tolerance, int max_sweeps,
        RunInfo& info)
    {
        return advance(previous, tolerance, max_sweeps, info);
    }

} // namespace mhs::hdim
