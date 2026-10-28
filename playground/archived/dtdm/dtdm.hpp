// DTDM: Laguerre-domain macromodels and marching-on-order, paper Sec. II-III.
//
// The global operators come from the repository's TPFA assembler. A module's
// operator is the corresponding principal submatrix, which drops the couplings
// between modules but keeps the series conductance of every interface face on
// the diagonal. That is exactly the wrong boundary: the interface is not a 0 K
// node, it is a patch temperature that both neighbouring modules drive. The
// diagonal is therefore corrected to each module's own half conductance and the
// dropped couplings come back as the patch map W, so the interface is modelled
// once, at the patch, instead of twice, at the two cell rows.
#pragma once

#include "mesh.hpp"
#include "numerics/linear/linear_solver.hpp"

#include <Eigen/Sparse>
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace mhs::dtdm {

    /// Interface k joins modules k and k+1: CM I | KM, then KM | CM II.
    inline constexpr int kInterfaceCount = 2;
    inline constexpr std::array<std::pair<int, int>, kInterfaceCount> kInterfaceModules { {
        { 0, 1 },
        { 1, 2 },
    } };

    /// One face element of an interface: the module cell behind it and that
    /// side's half of the face resistance.
    struct PatchFace {
        int cell = -1;
        /// This module's own half of the face resistance: area * k / (half cell).
        double conductance = 0.0;
        /// The series conductance the principal submatrix already carries on the
        /// cell diagonal: area / (d_a / k_a + d_b / k_b).
        double series = 0.0;
    };

    /// One unknown interface temperature ("NI" of the paper): the face elements
    /// that share it. One patch per face is the exact limit of the model.
    struct Patch {
        std::vector<PatchFace> faces;
    };

    /// One composable module: its cells, its operators and its interface patches.
    struct Module {
        std::vector<int> cells; // global cell ids, ascending
        std::vector<int> to_local; // global id -> local index, -1 elsewhere
        Eigen::SparseMatrix<double> K;
        Eigen::SparseMatrix<double> C;
        std::vector<Patch> patches[kInterfaceCount];
    };

    struct PatchLayout {
        std::vector<int> global_id[kModuleCount]; // per module, its patches' global ids
        int interface_patch_count[kInterfaceCount] { 0, 0 };
        int total = 0;
    };

    /// Temperature field of one method plus what it cost.
    struct MarchResult {
        std::vector<double> sample_times;
        std::vector<std::vector<double>> samples;
        /// Orders marched by the transform, steps taken by the reference.
        int count = 0;
        /// Largest Laguerre coefficient of the last order.
        double last_coefficient = 0.0;
        /// What is left of the reference's own error: a third of the correction
        /// its extrapolation applied.
        double estimated_error = 0.0;
        /// What one march at --steps cost, which is what a time-stepping
        /// simulation of the paper's configuration costs.
        double single_march_seconds = 0.0;
        double setup_seconds = 0.0;
        double march_seconds = 0.0;
        /// AMG-CG solves spent on the macromodel of eq. (11).
        long long setup_solves = 0;
        /// Solves of eq. (6) and (7) spent on the marching.
        long long march_solves = 0;
        long long march_iterations = 0;
    };

    /// The whole stack at one mesh level, shared by the three methods.
    class System {
    public:
        System(StackMesh stack, int patches_x, int patches_y, mhs::sim::SolverType solver,
            double linear_tolerance = 1e-8);

        const StackMesh& stack() const { return stack_; }

        /// Thermal-WLP: one system for the whole stack, one solve per order.
        MarchResult monolithic_wlp(double scale, int orders, const std::vector<double>& sample_times) const;
        /// DTDM: the macromodels of eq. (11) plus one interface system.
        MarchResult dtdm(double scale, int orders, const std::vector<double>& sample_times) const;
        /// The reference: backward Euler on the same mesh and operators, with the
        /// step refined until two successive Richardson extrapolants agree. It
        /// throws rather than return a field it could not refine.
        MarchResult reference(int steps, double duration, const std::vector<double>& sample_times) const;

    private:
        void build_operators();
        void build_modules();
        void build_patches(int nx, int ny);
        /// The part of the response that does not decay: K theta = f. Its Laguerre
        /// coefficients are 2(-1)^q and never fall, so it is solved once and added
        /// back to every reconstruction.
        Eigen::VectorXd steady_state() const;

        StackMesh stack_;
        mhs::sim::SolverType solver_ = mhs::sim::SolverType::AmgCg;
        /// Relative residual of the iterative solves. It is a *residual* bound,
        /// and these operators span ten decades, so the field error it allows is
        /// orders of magnitude larger than the tolerance itself.
        double linear_tolerance_ = 1e-8;
        Eigen::SparseMatrix<double> K_, C_;
        Eigen::VectorXd f_;
        std::array<Module, kModuleCount> modules_;
        PatchLayout layout_;
    };

} // namespace mhs::dtdm
