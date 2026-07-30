#include "compiler/model_compiler.hpp"
#include "macromodel/modal_port.hpp"
#include "solver/solve.hpp"
#include <Eigen/LU>
#include <array>
#include <gtest/gtest.h>
#include <string>

using namespace mhs::macro;
using mhs::sim::build_model;

namespace {

    mhs::core::Model make_single_cell_model(const std::string& conductivity = "1")
    {
        mhs::model::ModelDefinition io;
        io.settings.study_type = mhs::model::StudyType::Steady;
        io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
        io.settings.initial_temperature = 0.0;

        io.mesh.x_vertices = {0.0, 1.0};
        io.mesh.y_vertices = {0.0, 1.0};
        io.mesh.z_vertices = {0.0, 1.0};

        mhs::model::LayerSpec layer;
        layer.thickness = "1";

        mhs::model::BlockSpec block;
        block.material = "solid";
        block.volumetric_heat_source = "0";

        mhs::model::RectOperation rect;
        rect.operation = mhs::model::GeometryOperation::Add;
        rect.rect.x = "0";
        rect.rect.y = "0";
        rect.rect.width = "1";
        rect.rect.height = "1";
        block.geometry.push_back(rect);

        layer.blocks.push_back(block);
        io.layers.push_back(layer);

        mhs::model::MaterialSpec material;
        material.conductivity_x = material.conductivity_y = material.conductivity_z = conductivity;
        material.density = "2";
        material.specific_heat = "3";
        io.materials.push_back({"solid", material});
        io.default_boundary = mhs::model::NeumannBoundary {};
        return build_model(io);
    }

    PortModel make_single_mode_port()
    {
        PortModel port;
        port.operators.K.resize(1, 1);
        port.operators.K.insert(0, 0) = 1.0;
        port.operators.C.resize(1, 1);
        port.operators.f = Eigen::VectorXd::Ones(1);
        port.basis = Eigen::MatrixXd::Ones(1, 1);
        port.physical_port_count = 1;
        return port;
    }

    /// Create coupling with model_cells = {0}, model_face = XP.
    PortCoupling make_single_port_coupling()
    {
        PortCoupling coupling;
        coupling.model_cells = {0};
        coupling.model_face = mhs::core::FaceDir::XP;
        return coupling;
    }

} // namespace

TEST(MacroModelTest, PortAssemblerProjectsPhysicalInterface)
{
    auto model = make_single_cell_model();
    auto port = make_single_mode_port();

    // Both FVM and macro are the same model; macro coupling just validates topology.
    auto coupling = make_single_port_coupling();
    const std::array state {0.0, 0.0};
    const auto operators = assemble(model, port, coupling, state, 0.0);

    // The 1 mm cube has FVM-side half conductance
    // k*A/(dx/2) = 1*1e-6/(0.5e-3) = 0.002 W/K.
    // The macro side is on the face — no series combination.
    const Eigen::Matrix2d dense = Eigen::MatrixXd(operators.K);
    EXPECT_NEAR(dense(0, 0), 0.002, 1e-12);
    EXPECT_NEAR(dense(0, 1), -0.002, 1e-12);
    EXPECT_NEAR(dense(1, 0), -0.002, 1e-12);
    EXPECT_NEAR(dense(1, 1), 1.002, 1e-12);
}

TEST(MacroModelTest, PortAssemblerReevaluatesInterfaceConductanceFromFvmState)
{
    auto model = make_single_cell_model("1 + T");
    auto port = make_single_mode_port();
    auto coupling = make_single_port_coupling();

    const std::array cold_state {0.0, 0.0};
    const std::array hot_state {1.0, 0.0};
    const auto cold = assemble(model, port, coupling, cold_state, 0.0);
    const auto hot = assemble(model, port, coupling, hot_state, 0.0);

    // T=0: k=1, conductance = 1*1e-6/0.5e-3 = 0.002
    EXPECT_NEAR(cold.K.coeff(0, 0), 0.002, 1e-12);
    // T=1: k=2, conductance = 2*1e-6/0.5e-3 = 0.004
    EXPECT_NEAR(hot.K.coeff(0, 0), 0.004, 1e-12);
    EXPECT_GT(hot.K.coeff(0, 0), cold.K.coeff(0, 0));
}

TEST(MacroModelTest, NoBasisEqualsExplicitUnitBasis)
{
    auto model = make_single_cell_model();

    // With explicit unit basis
    PortModel explicit_port;
    explicit_port.operators.K.resize(1, 1);
    explicit_port.operators.K.insert(0, 0) = 1.0;
    explicit_port.operators.C.resize(1, 1);
    explicit_port.operators.f = Eigen::VectorXd::Ones(1);
    explicit_port.basis = Eigen::MatrixXd::Identity(1, 1);
    explicit_port.physical_port_count = 1;

    // Without basis (empty = unit basis)
    PortModel no_basis_port;
    no_basis_port.operators.K.resize(1, 1);
    no_basis_port.operators.K.insert(0, 0) = 1.0;
    no_basis_port.operators.C.resize(1, 1);
    no_basis_port.operators.f = Eigen::VectorXd::Ones(1);
    // basis stays empty
    no_basis_port.physical_port_count = 1;

    auto coupling = make_single_port_coupling();
    const std::array state {0.0, 0.0};

    const auto explicit_ops = assemble(model, explicit_port, coupling, state, 0.0);
    const auto no_basis_ops = assemble(model, no_basis_port, coupling, state, 0.0);

    EXPECT_EQ(explicit_ops.K.nonZeros(), no_basis_ops.K.nonZeros());
    EXPECT_EQ(explicit_ops.C.nonZeros(), no_basis_ops.C.nonZeros());
    for (Eigen::Index k = 0; k < explicit_ops.K.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(explicit_ops.K, k); it; ++it) {
            EXPECT_NEAR(it.value(), no_basis_ops.K.coeff(it.row(), it.col()), 1e-15);
        }
    }
    EXPECT_NEAR((explicit_ops.f - no_basis_ops.f).norm(), 0.0, 1e-15);
}

TEST(MacroModelTest, NoBasisAndExplicitBasisSteadySolveAgree)
{
    auto model = make_single_cell_model();

    // Explicit unit-basis port
    PortModel explicit_port;
    explicit_port.operators.K.resize(1, 1);
    explicit_port.operators.K.insert(0, 0) = 1.0;
    explicit_port.operators.C.resize(1, 1);
    explicit_port.operators.f = Eigen::VectorXd::Ones(1);
    explicit_port.basis = Eigen::MatrixXd::Identity(1, 1);
    explicit_port.physical_port_count = 1;

    // No-basis port
    PortModel no_basis_port;
    no_basis_port.operators.K.resize(1, 1);
    no_basis_port.operators.K.insert(0, 0) = 1.0;
    no_basis_port.operators.C.resize(1, 1);
    no_basis_port.operators.f = Eigen::VectorXd::Ones(1);
    no_basis_port.physical_port_count = 1;

    auto coupling = make_single_port_coupling();
    const std::array initial_state {0.0, 0.0};

    mhs::sim::SolveOptions opts;
    opts.step_strategy = mhs::sim::SolveOptions::StepStrategy::Fixed;

    auto explicit_result = solve(model, explicit_port, coupling, initial_state, opts);
    auto no_basis_result = solve(model, no_basis_port, coupling, initial_state, opts);

    ASSERT_TRUE(explicit_result.converged);
    ASSERT_TRUE(no_basis_result.converged);
    ASSERT_EQ(explicit_result.state.size(), no_basis_result.state.size());
    for (std::size_t i = 0; i < explicit_result.state.size(); ++i) {
        EXPECT_NEAR(explicit_result.state[i], no_basis_result.state[i], 1e-12);
    }
}

TEST(MacroModelTest, PortSystemAdvancesTheFullCoupledTransientState)
{
    auto model = make_single_cell_model();

    PortModel port;
    port.operators.K.resize(1, 1);
    port.operators.K.insert(0, 0) = 1.0;
    port.operators.C.resize(1, 1);
    port.operators.C.insert(0, 0) = 0.5;
    port.operators.f = Eigen::VectorXd::Ones(1);
    port.basis = Eigen::MatrixXd::Ones(1, 1);
    port.physical_port_count = 1;

    PortCoupling coupling;
    coupling.model_cells = {0};
    coupling.model_face = mhs::core::FaceDir::XP;

    const std::array initial_state {300.0, 300.0};
    const double dt = 0.25;
    const auto initial_operators = assemble(model, port, coupling, initial_state, dt);
    const Eigen::Matrix2d expected_matrix
        = Eigen::MatrixXd(initial_operators.K) + Eigen::MatrixXd(initial_operators.C) / dt;
    const Eigen::Vector2d initial = Eigen::Map<const Eigen::Vector2d>(initial_state.data());
    const Eigen::Vector2d expected_rhs = initial_operators.f + initial_operators.C * initial / dt;
    const Eigen::Vector2d expected = expected_matrix.fullPivLu().solve(expected_rhs);

    mhs::sim::Study study {mhs::core::StudyType::Transient, dt, dt};
    mhs::sim::SolveOptions options;
    options.step_strategy = mhs::sim::SolveOptions::StepStrategy::Fixed;
    options.fixed_dt = dt;
    std::vector<double> observed_times;

    mhs::sim::SystemAssembler asm_fn
        = [&](std::span<const double> state, double time) { return assemble(model, port, coupling, state, time); };
    auto result = mhs::sim::solve_system(study, asm_fn, initial_state, options,
        [&](double time, std::span<const double>) { observed_times.push_back(time); });

    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.state.size(), 2u);
    EXPECT_NEAR(result.state[0], expected[0], 1e-10);
    EXPECT_NEAR(result.state[1], expected[1], 1e-10);
    EXPECT_EQ(observed_times, (std::vector<double> {0.0, dt}));
}
