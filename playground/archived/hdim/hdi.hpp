// HDI: the three-level decomposition of Sec. II and the interconnection of
// eq. (7)-(10).
//
// Level 1 splits the stack into a nonlinear region (the two dies and the BEOL,
// where kappa depends on T) and two linear ones. Level 2 splits the linear
// region's response into a zero-input part, which carries the memory of the
// previous time step, and a zero-state part, which carries the interface flux.
// Level 3 replaces the zero-state part by the thermal resistance matrix R of a
// unit excitation on each interface block, so the linear region enters the
// nonlinear solve as one Robin boundary condition and its own unknowns never
// take part in the iteration.
#pragma once

#include "model.hpp"
#include "numerics/linear/linear_solver.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <memory>
#include <vector>

namespace mhs::hdim {

    /// Coarse partition of one interface plane: `nx` x `ny` blocks over the chip
    /// outline. Eq. (8) is evaluated once per block, which keeps R small enough to
    /// be dense; one block per mesh face reproduces the standard FVM exactly.
    struct Partition {
        int nx = 0;
        int ny = 0;
        int count = 0;
        std::vector<int> block; // per interface face

        /// `step` in mm; step <= 0 gives one block per face.
        static Partition build(const std::vector<InterfaceFace>& faces, double step);
    };

    /// One interface of Sec. II-C together with the linear region behind it: the
    /// operator of eq. (2), the macromodel of eq. (7)-(8), and the zero-input
    /// response of eq. (4).
    struct LinearHalf {
        int region = kLinearTop;
        RegionSystem system;
        mhs::sim::SolverPtr solver;
        Eigen::VectorXd state; // eq. (4)-(5) with no interface flux
        std::vector<int> global_cell; // cells of the region, in system order

        std::vector<int> face; // interface faces, indices into Assembled::interfaces
        std::vector<int> cell; // per face, position inside `system`
        std::vector<double> area; // per face, m^2
        Partition partition;
        Eigen::MatrixXd R; // eq. (8), blocks x blocks, K/W
        Eigen::VectorXd load; // eq. (7)'s T0, per block, K

        /// Pre-simulation: factorises the region and extracts R, one linear solve
        /// per interface block.
        void build(const Model& model, const Assembled& assembled, int region, double interface_step, double tolerance);
        /// Eq. (4)-(5) for the given previous field, refreshing `state` and `load`.
        void solve_state(const std::vector<double>& previous, double dt);
        /// Eq. (5) with the interface flux, i.e. what has to be added to `state`.
        Eigen::VectorXd solve_flux(const std::vector<double>& flux) const;
    };

    /// The HDI method of Sec. II for one time step: the pre-simulation that
    /// extracts the macromodels, then a steady solve or a march of eq. (3)-(10).
    class Hdi {
    public:
        /// `dt` <= 0 extracts the steady macromodels. `interface_step` (mm) is the
        /// block size of the interface partition.
        Hdi(const Model& model, double dt, double interface_step, double linear_tolerance);

        double presimulation_seconds() const { return presimulation_seconds_; }
        int interface_blocks() const;

        /// Steady state of eq. (2) from the ambient.
        std::vector<double> steady(double tolerance, int max_sweeps, RunInfo& info);
        /// One step of eq. (3) from `previous` (empty starts at the ambient).
        std::vector<double> step(const std::vector<double>& previous, double tolerance, int max_sweeps, RunInfo& info);
        /// The last step's interface temperatures, per block, for both interfaces.
        const std::array<Eigen::VectorXd, kInterfaceCount>& interface_temperature() const { return ti_; }

    private:
        /// One pass of Sec. II-D: eq. (7)-(10) for both interfaces, then one
        /// fixed-point update of the nonlinear region with that Robin condition.
        double sweep(std::vector<double>& field, RunInfo& info);
        /// Eq. (7)-(10): the interface temperature of every block and the flux
        /// through every interface face, from the current nonlinear temperatures.
        void interface_flux(const Assembled& assembled, const std::vector<double>& field);
        /// The shared body of `steady` and `step`.
        std::vector<double> advance(
            const std::vector<double>& previous, double tolerance, int max_sweeps, RunInfo& info);

        const Model& model_;
        double dt_ = 0.0;
        double linear_tolerance_ = 1e-10;
        double presimulation_seconds_ = 0.0;
        std::array<LinearHalf, kInterfaceCount> half_;
        std::array<Eigen::VectorXd, kInterfaceCount> ti_;
        std::array<std::vector<double>, kInterfaceCount> gain_; // per interface face, W/K
        std::vector<double> flux_; // per interface face, W
        std::vector<double> incoming_; // the field eq. (3) steps from, K
        Eigen::VectorXd nonlinear_state_; // warm start of the nonlinear solve
    };

} // namespace mhs::hdim
