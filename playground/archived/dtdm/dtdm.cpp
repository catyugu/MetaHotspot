#include "dtdm.hpp"

#include "solver/assembler.hpp"
#include "wlp.hpp"

#include <Eigen/Cholesky>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <tuple>

namespace mhs::dtdm {
    namespace {
        using Clock = std::chrono::steady_clock;

        /// Solver copies used to extract the macromodel columns in parallel. The
        /// modules are extracted at the same time too, so the module count times
        /// this is what the setup phase can keep busy.
        inline constexpr std::size_t kExtractionWorkers = 4;

        double seconds_since(const Clock::time_point& start)
        {
            return std::chrono::duration<double>(Clock::now() - start).count();
        }

        /// The modules meet only through the interface system, so everything one
        /// module does inside an order is independent of the others.
        template <class Task>
        void for_each_module(Task task)
        {
            tbb::parallel_for(0, kModuleCount, [&task](int m) { task(m); });
        }

        std::vector<double> to_vector(const Eigen::VectorXd& field)
        {
            return std::vector<double>(field.data(), field.data() + field.size());
        }

        /// The paper excites every order with f_q = 2 P (-1)^q and lets the series
        /// carry the constant. This implementation carries it exactly and the series
        /// carries theta - theta_ss, so the two reconstructions differ by
        /// theta_ss (1 - 2 sum_{q<N} (-1)^q phi_q(s t)), which vanishes as N grows.
        /// The factor is printed so the difference is not left implicit.
        void report_dc_remainder(int orders, double scale, const std::vector<double>& times)
        {
            std::fprintf(stderr, "[dc] paper form minus this form, in units of theta_ss:");
            std::vector<double> weights;
            for (const double t : times) {
                phi(orders, scale * t, weights);
                double alternating = 0.0;
                for (int q = 0; q < orders; ++q)
                    alternating += (q % 2 == 0 ? 1.0 : -1.0) * weights[static_cast<std::size_t>(q)];
                std::fprintf(stderr, " %.2e", 1.0 - 2.0 * alternating);
            }
            std::fprintf(stderr, "\n");
        }

        template <class Storage>
        Eigen::SparseMatrix<double> principal_submatrix(const Storage& matrix,
            const std::vector<int>& cells, const std::vector<int>& global_to_local)
        {
            std::vector<Eigen::Triplet<double>> triplets;
            for (std::size_t b = 0; b < cells.size(); ++b) {
                const auto column = static_cast<Eigen::Index>(cells[b]);
                for (typename Storage::InnerIterator it(matrix, column); it; ++it) {
                    const int row = static_cast<int>(it.row());
                    if (global_to_local[static_cast<std::size_t>(row)] < 0)
                        continue;
                    triplets.emplace_back(global_to_local[static_cast<std::size_t>(row)],
                        static_cast<int>(b), it.value());
                }
            }
            Eigen::SparseMatrix<double> out(static_cast<Eigen::Index>(cells.size()),
                static_cast<Eigen::Index>(cells.size()));
            out.setFromTriplets(triplets.begin(), triplets.end());
            out.makeCompressed();
            return out;
        }

        /// One module under the Laguerre shift, with the patch temperatures as its
        /// only coupling to the rest of the stack.
        ///
        /// The principal submatrix keeps the series conductance of every interface
        /// face on its diagonal, whereas the patch model measures each cell from
        /// its own half resistance to the face plane. The diagonal is therefore
        /// corrected by (half - series) and the other module supplies the second
        /// half, so 1/g_a + 1/g_b reproduces the assembled face exactly.
        class ModuleOperator {
        public:
            ModuleOperator() = default;
            ModuleOperator(const Module& module, double shift, mhs::sim::SolverType type, double tolerance)
            {
                const auto n = static_cast<Eigen::Index>(module.cells.size());
                for (int k = 0; k < kInterfaceCount; ++k)
                    patches_ += static_cast<int>(module.patches[k].size());
                half_conductance_ = Eigen::VectorXd::Zero(patches_);

                Eigen::VectorXd correction = Eigen::VectorXd::Zero(n);
                std::vector<Eigen::Triplet<double>> triplets;
                int offset = 0;
                for (int k = 0; k < kInterfaceCount; ++k)
                    for (int p = 0; p < static_cast<int>(module.patches[k].size()); ++p, ++offset)
                        for (const auto& face : module.patches[k][static_cast<std::size_t>(p)].faces) {
                            const auto row = module.to_local[static_cast<std::size_t>(face.cell)];
                            triplets.emplace_back(row, offset, face.conductance);
                            half_conductance_(offset) += face.conductance;
                            correction(row) += face.conductance - face.series;
                        }
                w_.resize(n, patches_);
                w_.setFromTriplets(triplets.begin(), triplets.end());
                w_.makeCompressed();
                wt_ = Eigen::SparseMatrix<double>(w_.transpose());

                std::vector<Eigen::Triplet<double>> diagonal;
                for (Eigen::Index i = 0; i < n; ++i)
                    if (correction(i) != 0.0)
                        diagonal.emplace_back(i, i, correction(i));
                Eigen::SparseMatrix<double> boundary(n, n);
                boundary.setFromTriplets(diagonal.begin(), diagonal.end());
                matrix_ = module.K + shift * module.C + boundary;
                matrix_.makeCompressed();

                solver_ = mhs::sim::create_solver({ type, { tolerance, 2000 } });
                solver_->compute(matrix_);
            }

            const Eigen::SparseMatrix<double>& W() const { return w_; }
            const Eigen::SparseMatrix<double>& Wt() const { return wt_; }
            const Eigen::SparseMatrix<double>& matrix() const { return matrix_; }
            const Eigen::VectorXd& half_conductance() const { return half_conductance_; }
            int patches() const { return patches_; }
            long long solves() const { return solves_; }
            long long iterations() const { return iterations_; }

            /// Eq. (6): the source-driven solve, with the interface at zero.
            Eigen::VectorXd solve_free(const Eigen::VectorXd& rhs) const { return solve(rhs, warm_free_); }
            /// Eq. (7): the same operator driven by the interface temperature.
            Eigen::VectorXd solve_with_interface(const Eigen::VectorXd& rhs, const Eigen::VectorXd& imposed) const
            {
                return solve(rhs + w_ * imposed, warm_driven_);
            }

        private:
            /// Warm-started AMG-CG solve. Consecutive Laguerre orders are close, so
            /// the previous field of the same family is the natural initial guess.
            Eigen::VectorXd solve(const Eigen::VectorXd& rhs, Eigen::VectorXd& warm) const
            {
                if (warm.size() != rhs.size())
                    warm = Eigen::VectorXd::Zero(rhs.size());
                warm = solver_->solve(rhs, warm);
                if (!solver_->success() || !warm.allFinite())
                    throw std::runtime_error("module solve failed");
                ++solves_;
                iterations_ += solver_->iterations();
                return warm;
            }

            mhs::sim::SolverPtr solver_;
            Eigen::SparseMatrix<double> matrix_;
            Eigen::SparseMatrix<double> w_;
            Eigen::SparseMatrix<double> wt_;
            Eigen::VectorXd half_conductance_;
            int patches_ = 0;
            mutable Eigen::VectorXd warm_free_;
            mutable Eigen::VectorXd warm_driven_;
            mutable long long solves_ = 0;
            mutable long long iterations_ = 0;
        };

        /// Eq. (11) of one module, plus what extracting it cost.
        struct Macromodel {
            Eigen::MatrixXd G;
            long long iterations = 0;
        };

        /// Eq. (11) column by column: G e_j = W^T A^-1 W e_j - g e_j. Each column is
        /// one AMG-CG solve, and only the P x P interface flux is kept, so the
        /// macromodel costs O(P^2) instead of O(n P) and the interface can be as
        /// fine as the volume mesh. The price is paid per order, where eq. (7) is
        /// one more solve on the same operator.
        ///
        /// The columns are independent and make up the bulk of the setup, so they
        /// run on the thread pool. An AMG hierarchy owns its work vectors, so every
        /// worker needs its own copy.
        Macromodel extract_macromodel(const ModuleOperator& op, mhs::sim::SolverType type, double tolerance)
        {
            const auto n = op.matrix().rows();
            const std::size_t columns = static_cast<std::size_t>(op.patches());
            const std::size_t workers = std::min(kExtractionWorkers, columns);
            std::vector<mhs::sim::SolverPtr> pool;
            pool.reserve(workers);
            for (std::size_t w = 0; w < workers; ++w) {
                pool.push_back(mhs::sim::create_solver({ type, { tolerance, 2000 } }));
                pool.back()->compute(op.matrix());
            }

            Macromodel out;
            out.G = Eigen::MatrixXd::Zero(op.patches(), op.patches());
            std::vector<long long> iterations(workers, 0);
            tbb::parallel_for(std::size_t(0), workers, [&](std::size_t w) {
                auto& solver = *pool[w];
                const auto begin = columns * w / workers;
                const auto end = columns * (w + 1) / workers;
                Eigen::VectorXd column = Eigen::VectorXd::Zero(n);
                Eigen::VectorXd field = Eigen::VectorXd::Zero(n);
                for (std::size_t j = begin; j < end; ++j) {
                    column.setZero();
                    for (Eigen::SparseMatrix<double>::InnerIterator it(op.W(), static_cast<Eigen::Index>(j)); it; ++it)
                        column(it.row()) += it.value();
                    field = solver.solve(column, field);
                    if (!solver.success() || !field.allFinite())
                        throw std::runtime_error("macromodel extraction failed");
                    iterations[w] += solver.iterations();
                    out.G.col(static_cast<Eigen::Index>(j)) = op.Wt() * field;
                }
            });
            for (const long long count : iterations)
                out.iterations += count;
            out.G.diagonal() -= op.half_conductance();
            return out;
        }

        /// Eqs. (12)-(13): the patch temperatures are the only unknowns the modules
        /// share, and each module contributes the flux its own macromodel predicts.
        /// The operator does not depend on the order, so it is factored once for
        /// the whole march.
        class InterfaceSystem {
        public:
            InterfaceSystem(const std::array<Macromodel, kModuleCount>& macromodels, const PatchLayout& layout)
                : layout_(layout)
            {
                Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(layout.total, layout.total);
                for (int m = 0; m < kModuleCount; ++m) {
                    const auto& ids = layout.global_id[m];
                    const Eigen::MatrixXd& G = macromodels[static_cast<std::size_t>(m)].G;
                    for (int i = 0; i < static_cast<int>(ids.size()); ++i)
                        for (int j = 0; j < static_cast<int>(ids.size()); ++j)
                            matrix(ids[static_cast<std::size_t>(i)], ids[static_cast<std::size_t>(j)]) += G(i, j);
                }
                factor_.compute(-matrix);
                if (factor_.info() != Eigen::Success)
                    throw std::runtime_error("interface system is not positive definite");
            }

            Eigen::VectorXd solve(const std::array<Eigen::VectorXd, kModuleCount>& flux) const
            {
                Eigen::VectorXd rhs = Eigen::VectorXd::Zero(layout_.total);
                for (int m = 0; m < kModuleCount; ++m) {
                    const auto& ids = layout_.global_id[m];
                    for (int i = 0; i < static_cast<int>(ids.size()); ++i)
                        rhs(ids[static_cast<std::size_t>(i)]) += flux[static_cast<std::size_t>(m)](i);
                }
                return factor_.solve(rhs);
            }

        private:
            PatchLayout layout_;
            Eigen::LLT<Eigen::MatrixXd> factor_;
        };

        /// The truncated Laguerre series sum_q T_q phi_q(s t), accumulated one
        /// order at a time at a fixed set of probe times. A coefficient field is
        /// either full length (the monolithic solve) or one module's block of the
        /// decomposition, scattered to its global cell ids.
        class Series {
        public:
            Series(int orders, double scale, const std::vector<double>& times)
                : weights_(times.size())
            {
                for (std::size_t k = 0; k < times.size(); ++k)
                    phi(orders, scale * times[k], weights_[k]);
            }

            void start_from(const std::vector<double>& field) { fields_.assign(weights_.size(), field); }

            void add(int q, const Eigen::VectorXd& state)
            {
                for (std::size_t k = 0; k < weights_.size(); ++k) {
                    const double weight = weights_[k][static_cast<std::size_t>(q)];
                    auto& field = fields_[k];
                    for (std::size_t i = 0; i < field.size(); ++i)
                        field[i] += weight * state(static_cast<Eigen::Index>(i));
                }
            }

            void add(int q, const Eigen::VectorXd& state, const std::vector<int>& cells)
            {
                for (std::size_t k = 0; k < weights_.size(); ++k) {
                    const double weight = weights_[k][static_cast<std::size_t>(q)];
                    auto& field = fields_[k];
                    for (std::size_t i = 0; i < cells.size(); ++i)
                        field[static_cast<std::size_t>(cells[i])] += weight * state(static_cast<Eigen::Index>(i));
                }
            }

            std::vector<std::vector<double>> take() { return std::move(fields_); }

        private:
            std::vector<std::vector<double>> weights_; // [probe][order]
            std::vector<std::vector<double>> fields_; // [probe][dof]
        };

        void require_samples(const std::vector<double>& sample_times)
        {
            if (sample_times.empty())
                throw std::invalid_argument("at least one probe time is required");
        }

        /// One fixed-step backward-Euler march of eq. (1) from a cold start, with
        /// the field recorded at the probe times. The step never changes, so the
        /// operator is factored once for the whole march.
        std::vector<std::vector<double>> march_backward_euler(const mhs::sim::Operators& operators, int steps,
            double duration, const std::vector<double>& times, mhs::sim::SolverType type, double tolerance)
        {
            const auto n = static_cast<Eigen::Index>(operators.f.size());
            const double dt = duration / static_cast<double>(steps);
            for (const double t : times)
                if (std::abs(std::round(t / dt) * dt - t) > 1e-9 * duration)
                    throw std::invalid_argument("the reference step does not land on every probe time");
            Eigen::SparseMatrix<double> matrix = operators.K;
            matrix += (1.0 / dt) * operators.C;
            matrix.makeCompressed();
            auto solver = mhs::sim::create_solver({ type, { tolerance, 2000 } });
            solver->compute(matrix);

            std::vector<std::vector<double>> samples(times.size());
            Eigen::VectorXd state = Eigen::VectorXd::Zero(n);
            std::size_t next = 0;
            // A probe at t = 0 is the initial condition itself.
            while (next < times.size() && times[next] <= 1e-9 * duration) {
                samples[next].assign(state.data(), state.data() + n);
                ++next;
            }
            for (int k = 1; k <= steps; ++k) {
                state = solver->solve(operators.f + (operators.C * state) / dt, state);
                if (!solver->success())
                    throw std::runtime_error("the backward-Euler reference did not converge");
                const double now = dt * static_cast<double>(k);
                while (next < times.size() && now >= times[next] - 1e-9 * duration) {
                    samples[next].assign(state.data(), state.data() + n);
                    ++next;
                }
            }
            if (next != times.size())
                throw std::runtime_error("the reference does not reach every probe time");
            return samples;
        }

        /// 2 fine - coarse: removes the first-order term of two fields that differ
        /// only in their step.
        std::vector<std::vector<double>> richardson(const std::vector<std::vector<double>>& coarse,
            const std::vector<std::vector<double>>& fine)
        {
            auto out = fine;
            for (std::size_t k = 0; k < out.size(); ++k)
                for (std::size_t i = 0; i < out[k].size(); ++i)
                    out[k][i] = 2.0 * fine[k][i] - coarse[k][i];
            return out;
        }

        double max_field_difference(
            const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b)
        {
            double worst = 0.0;
            for (std::size_t k = 0; k < a.size(); ++k)
                for (std::size_t i = 0; i < a[k].size(); ++i)
                    worst = std::max(worst, std::abs(a[k][i] - b[k][i]));
            return worst;
        }
    } // namespace

    System::System(StackMesh stack, int patches_x, int patches_y, mhs::sim::SolverType solver,
        double linear_tolerance)
        : stack_(std::move(stack))
        , solver_(solver)
        , linear_tolerance_(linear_tolerance)
    {
        build_operators();
        build_modules();
        build_patches(patches_x, patches_y);
    }

    void System::build_operators()
    {
        const std::vector<double> state(static_cast<std::size_t>(stack_.dof), 0.0);
        auto operators = mhs::sim::assemble_thermal(stack_.model, state, 0.0);
        K_ = std::move(operators.K);
        C_ = std::move(operators.C);
        f_ = std::move(operators.f);
    }

    void System::build_modules()
    {
        for (int m = 0; m < kModuleCount; ++m) {
            auto& module = modules_[static_cast<std::size_t>(m)];
            module.cells.assign(stack_.cells_of_module[m].begin(), stack_.cells_of_module[m].end());
            std::sort(module.cells.begin(), module.cells.end());
            module.to_local.assign(static_cast<std::size_t>(stack_.dof), -1);
            for (std::size_t i = 0; i < module.cells.size(); ++i)
                module.to_local[static_cast<std::size_t>(module.cells[i])] = static_cast<int>(i);
            module.K = principal_submatrix(K_, module.cells, module.to_local);
            module.C = principal_submatrix(C_, module.cells, module.to_local);
        }
        std::fprintf(stderr, "[modules] dof");
        for (int m = 0; m < kModuleCount; ++m)
            std::fprintf(stderr, " %s %zu", kModuleNames[static_cast<std::size_t>(m)],
                modules_[static_cast<std::size_t>(m)].cells.size());
        std::fprintf(stderr, "  (total %zu)\n", static_cast<std::size_t>(stack_.dof));
    }

    void System::build_patches(int nx, int ny)
    {
        layout_ = PatchLayout {};
        // A negative size selects one patch per interface face, the exact limit of
        // the model; nx <= 1 is the coarsest port, one patch per interface.
        const bool per_face = nx < 0 || ny < 0;
        nx = std::max(nx, 1);
        ny = std::max(ny, 1);

        // The inter-module couplings of K are exactly the interface faces, and the
        // face geometry gives each side its own half conductance.
        std::map<std::pair<int, int>, const mhs::core::FacePatch*> face_of_pair;
        for (const auto& face : stack_.model.discrete.faces) {
            if (!face.neighbor.valid())
                continue;
            face_of_pair[{ static_cast<int>(face.owner.value), static_cast<int>(face.neighbor.value) }] = &face;
            face_of_pair[{ static_cast<int>(face.neighbor.value), static_cast<int>(face.owner.value) }] = &face;
        }

        int next = 0;
        for (int k = 0; k < kInterfaceCount; ++k) {
            const auto [m0, m1] = kInterfaceModules[static_cast<std::size_t>(k)];
            const auto& a = modules_[static_cast<std::size_t>(m0)];
            const auto& b = modules_[static_cast<std::size_t>(m1)];

            std::vector<double> xs, ys;
            std::vector<std::tuple<int, int>> couplings;
            for (const int ca : a.cells)
                for (Eigen::SparseMatrix<double>::InnerIterator it(K_, ca); it; ++it) {
                    const int cb = static_cast<int>(it.row());
                    if (b.to_local[static_cast<std::size_t>(cb)] < 0)
                        continue;
                    const auto centre = stack_.model.discrete.cells[static_cast<std::size_t>(ca)].center();
                    xs.push_back(centre[0]);
                    ys.push_back(centre[1]);
                    couplings.emplace_back(ca, cb);
                }
            if (couplings.empty())
                throw std::runtime_error(std::string("interface ") + std::to_string(k) + " has no shared faces");

            const double xmin = *std::min_element(xs.begin(), xs.end());
            const double ymin = *std::min_element(ys.begin(), ys.end());
            const double spanx = std::max(*std::max_element(xs.begin(), xs.end()) - xmin, 1e-30);
            const double spany = std::max(*std::max_element(ys.begin(), ys.end()) - ymin, 1e-30);

            const auto slots = static_cast<std::size_t>(per_face ? couplings.size() : nx * ny);
            std::vector<Patch> side_a(slots), side_b(slots);
            for (std::size_t c = 0; c < couplings.size(); ++c) {
                const auto [ca, cb] = couplings[c];
                const auto found = face_of_pair.find({ ca, cb });
                if (found == face_of_pair.end())
                    throw std::runtime_error("interface face not found in the mesh");
                const auto& face = *found->second;
                const auto& cell_a = stack_.model.discrete.cells[static_cast<std::size_t>(ca)];
                const auto& cell_b = stack_.model.discrete.cells[static_cast<std::size_t>(cb)];
                const double d_a = 0.5 * cell_a.size()[static_cast<std::size_t>(face.axis)];
                const double d_b = 0.5 * cell_b.size()[static_cast<std::size_t>(face.axis)];
                const double g_a = face.area * stack_.cell_k[static_cast<std::size_t>(ca)] / d_a;
                const double g_b = face.area * stack_.cell_k[static_cast<std::size_t>(cb)] / d_b;
                const double series = -K_.coeff(ca, cb);

                std::size_t patch = c;
                if (!per_face) {
                    const auto centre = cell_a.center();
                    const int ix = std::clamp(static_cast<int>((centre[0] - xmin) / spanx * nx), 0, nx - 1);
                    const int iy = std::clamp(static_cast<int>((centre[1] - ymin) / spany * ny), 0, ny - 1);
                    patch = static_cast<std::size_t>(ix * ny + iy);
                }
                side_a[patch].faces.push_back({ ca, g_a, series });
                side_b[patch].faces.push_back({ cb, g_b, series });
            }

            // A coarse grid leaves some patch ids unused; number the used ones.
            std::vector<int> used;
            for (std::size_t p = 0; p < slots; ++p)
                if (!side_a[p].faces.empty())
                    used.push_back(static_cast<int>(p));

            auto& ma = modules_[static_cast<std::size_t>(m0)];
            auto& mb = modules_[static_cast<std::size_t>(m1)];
            ma.patches[k].assign(used.size(), {});
            mb.patches[k].assign(used.size(), {});
            for (std::size_t p = 0; p < used.size(); ++p) {
                ma.patches[k][p] = std::move(side_a[static_cast<std::size_t>(used[p])]);
                mb.patches[k][p] = std::move(side_b[static_cast<std::size_t>(used[p])]);
            }

            layout_.interface_patch_count[k] = static_cast<int>(used.size());
            for (int p = 0; p < layout_.interface_patch_count[k]; ++p) {
                layout_.global_id[m0].push_back(next + p);
                layout_.global_id[m1].push_back(next + p);
            }
            next += layout_.interface_patch_count[k];
        }
        layout_.total = next;
        std::fprintf(stderr, "[patches] total %d  iface0 %d iface1 %d\n", layout_.total,
            layout_.interface_patch_count[0], layout_.interface_patch_count[1]);
    }

    Eigen::VectorXd System::steady_state() const
    {
        auto solver = mhs::sim::create_solver({ solver_, { 0.1 * linear_tolerance_, 2000 } });
        solver->compute(K_);
        Eigen::VectorXd steady = solver->solve(f_, Eigen::VectorXd::Zero(f_.size()));
        if (!solver->success())
            throw std::runtime_error("steady-state solve failed");
        return steady;
    }

    MarchResult System::monolithic_wlp(double scale, int orders, const std::vector<double>& sample_times) const
    {
        require_samples(sample_times);
        const auto start = Clock::now();
        const auto n = static_cast<Eigen::Index>(stack_.dof);
        const Eigen::VectorXd steady = steady_state();

        // Eq. (4): one operator for the whole stack, independent of the order.
        Eigen::SparseMatrix<double> matrix = K_ + 0.5 * scale * C_;
        matrix.makeCompressed();
        std::fprintf(stderr, "[wlp] nnz K %lld  nnz C %lld  nnz(K + s/2 C) %lld\n",
            static_cast<long long>(K_.nonZeros()), static_cast<long long>(C_.nonZeros()),
            static_cast<long long>(matrix.nonZeros()));
        auto solver = mhs::sim::create_solver({ solver_, { linear_tolerance_, 2000 } });
        solver->compute(matrix);
        report_dc_remainder(orders, scale, sample_times);

        // The DC part is carried every order, so the drive keeps the steady state.
        const Eigen::VectorXd drive = -scale * (C_ * steady);
        Eigen::VectorXd history = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd previous = history;
        Series series(orders, scale, sample_times);
        series.start_from(to_vector(steady));

        MarchResult out;
        const auto midpoint = Clock::now();
        for (int q = 0; q < orders; ++q) {
            const Eigen::VectorXd state = solver->solve(drive - scale * (C_ * history), previous);
            if (!solver->success())
                throw std::runtime_error("monolithic WLP solve failed");
            previous = state;
            out.march_iterations += solver->iterations();
            out.last_coefficient = state.cwiseAbs().maxCoeff();
            history += state;
            series.add(q, state);
            if ((q + 1) % 50 == 0)
                std::fprintf(stderr, "[orders] %d / %d  %.1f s\n", q + 1, orders, seconds_since(midpoint));
        }
        out.march_seconds = seconds_since(midpoint);
        out.setup_seconds = seconds_since(start) - out.march_seconds;
        out.count = orders;
        out.march_solves = orders;
        out.samples = series.take();
        out.sample_times = sample_times;
        return out;
    }

    MarchResult System::dtdm(double scale, int orders, const std::vector<double>& sample_times) const
    {
        require_samples(sample_times);
        const auto start = Clock::now();
        const auto n = static_cast<Eigen::Index>(stack_.dof);
        const double shift = 0.5 * scale;

        // Line 2 of Algorithm 1: the macromodel G of eq. (11), one extraction per
        // module, all of them independent.
        std::array<ModuleOperator, kModuleCount> operators;
        std::array<Macromodel, kModuleCount> macromodels;
        for_each_module([&](int m) {
            const auto mi = static_cast<std::size_t>(m);
            operators[mi] = ModuleOperator(modules_[mi], shift, solver_, linear_tolerance_);
            macromodels[mi] = extract_macromodel(operators[mi], solver_, linear_tolerance_);
        });
        const InterfaceSystem interface(macromodels, layout_);

        // The steady state is solved on the whole stack and then split, so the
        // decomposition never accumulates a different DC level.
        const Eigen::VectorXd global_steady = steady_state();
        std::array<Eigen::VectorXd, kModuleCount> steady, drive, history;
        long long macromodel_bytes = 0;
        int macromodel_columns = 0;
        for (int m = 0; m < kModuleCount; ++m) {
            const auto mi = static_cast<std::size_t>(m);
            const auto& module = modules_[mi];
            steady[mi].resize(static_cast<Eigen::Index>(module.cells.size()));
            for (std::size_t i = 0; i < module.cells.size(); ++i)
                steady[mi](static_cast<Eigen::Index>(i)) = global_steady(module.cells[i]);
            drive[mi] = -scale * (module.C * steady[mi]);
            history[mi] = Eigen::VectorXd::Zero(steady[mi].size());
            macromodel_bytes += 8LL * macromodels[mi].G.size();
            macromodel_columns += operators[mi].patches();
        }
        std::fprintf(stderr, "[macromodel] %d columns, %lld iterations, %.1f MB\n",
            macromodel_columns, macromodels[0].iterations + macromodels[1].iterations + macromodels[2].iterations,
            static_cast<double>(macromodel_bytes) / (1024.0 * 1024.0));

        Series series(orders, scale, sample_times);
        series.start_from(to_vector(global_steady));

        MarchResult out;
        const auto midpoint = Clock::now();
        std::array<Eigen::VectorXd, kModuleCount> flux;
        for (int q = 0; q < orders; ++q) {
            std::array<Eigen::VectorXd, kModuleCount> rhs;
            for (int m = 0; m < kModuleCount; ++m)
                rhs[static_cast<std::size_t>(m)] = drive[static_cast<std::size_t>(m)]
                    - scale * (modules_[static_cast<std::size_t>(m)].C * history[static_cast<std::size_t>(m)]);

            // Line 5: the free solve of eq. (6) is independent per module.
            for_each_module([&](int m) {
                const auto mi = static_cast<std::size_t>(m);
                flux[mi] = operators[mi].Wt() * operators[mi].solve_free(rhs[mi]);
            });

            // Line 6-8: the only coupling, one small dense system per order.
            const Eigen::VectorXd patch = interface.solve(flux);

            // Line 9-10: the Dirichlet solve of eq. (7), again independent per module.
            std::array<double, kModuleCount> peak {};
            for_each_module([&](int m) {
                const auto mi = static_cast<std::size_t>(m);
                const auto& ids = layout_.global_id[m];
                Eigen::VectorXd imposed(operators[mi].patches());
                for (int i = 0; i < operators[mi].patches(); ++i)
                    imposed(i) = patch(ids[static_cast<std::size_t>(i)]);
                const Eigen::VectorXd state = operators[mi].solve_with_interface(rhs[mi], imposed);
                history[mi] += state;
                series.add(q, state, modules_[mi].cells);
                peak[mi] = state.cwiseAbs().maxCoeff();
            });
            out.last_coefficient = *std::max_element(peak.begin(), peak.end());
            if ((q + 1) % 50 == 0)
                std::fprintf(stderr, "[orders] %d / %d  %.1f s\n", q + 1, orders, seconds_since(midpoint));
        }
        out.march_seconds = seconds_since(midpoint);
        out.setup_seconds = seconds_since(start) - out.march_seconds;
        out.count = orders;
        out.samples = series.take();
        out.sample_times = sample_times;
        for (int m = 0; m < kModuleCount; ++m) {
            const auto mi = static_cast<std::size_t>(m);
            out.setup_solves += operators[mi].patches();
            out.march_solves += operators[mi].solves();
            out.march_iterations += operators[mi].iterations();
            std::fprintf(stderr, "[module %s] dof %zu patches %d solves %lld iterations %lld\n",
                kModuleNames[mi], modules_[mi].cells.size(), operators[mi].patches(), operators[mi].solves(),
                operators[mi].iterations());
        }
        return out;
    }

    MarchResult System::reference(int steps, double duration, const std::vector<double>& sample_times) const
    {
        require_samples(sample_times);
        const auto start = Clock::now();
        const mhs::sim::Operators operators { K_, C_, f_ };
        auto march = [&](int count) {
            return march_backward_euler(operators, count, duration, sample_times, solver_, linear_tolerance_);
        };

        // `steps` is the time-marching configuration this is compared against, the
        // paper's 5000 steps of 0.01 s. That march is run anyway, because its cost
        // is what a time-stepping simulation costs.
        //
        // It cannot be the reference. Backward Euler's global error is first order
        // in the step and a cold start is where most of it is made: the slow modes
        // are allowed to answer within one step, which at 0.01 s is 0.18 K at
        // t = 0.05 s. So the reference refines internally, and
        // 2 theta(dt/2) - theta(dt) over the two finest marches removes that term.
        // A third of the correction it applied is what is left of it.
        MarchResult out;
        const auto baseline = Clock::now();
        (void)march(steps); // timed only
        out.single_march_seconds = seconds_since(baseline);

        const auto coarse = march(2 * steps);
        const auto fine = march(4 * steps);
        out.samples = richardson(coarse, fine);
        out.estimated_error = max_field_difference(out.samples, fine) / 3.0;
        out.count = 4 * steps;
        out.sample_times = sample_times;
        out.march_seconds = seconds_since(start);
        return out;
    }

} // namespace mhs::dtdm
