#include "compiler/model_compiler.hpp"
#include "model_test_utils.hpp"
#include "macromodel/modal_port.hpp"
#include "solver/scheduler.hpp"
#include <Eigen/LU>
#include <array>
#include <gtest/gtest.h>
#include <string>

using namespace mhs::sim;

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

} // namespace

TEST(MacroModelTest, ModalPortAssemblerProjectsPhysicalInterface)
{
    auto model = make_single_cell_model();

    ModalPort macro;
    macro.operators.K.resize(1, 1);
    macro.operators.K.insert(0, 0) = 1.0;
    macro.operators.C.resize(1, 1);
    macro.operators.f = Eigen::VectorXd::Ones(1);
    macro.basis = Eigen::MatrixXd::Ones(1, 1);

    ThermalPortInterface interface;
    interface.model_cells = {0};
    interface.model_face = mhs::core::FaceDir::XP;
    interface.exterior_half_conductance = Eigen::VectorXd::Constant(1, 0.002);

    const std::array state {0.0, 0.0};
    const auto operators = assemble_modal_port_system(model, macro, interface, state, 0.0);

    // The 1 mm cube has model-side half conductance
    // k*A/(dx/2) = 1*1e-6/(0.5e-3) = 0.002 W/K. Two halves in series
    // yield 0.001 W/K at the physical interface.
    const Eigen::Matrix2d dense = Eigen::MatrixXd(operators.K);
    EXPECT_NEAR(dense(0, 0), 0.001, 1e-12);
    EXPECT_NEAR(dense(0, 1), -0.001, 1e-12);
    EXPECT_NEAR(dense(1, 0), -0.001, 1e-12);
    EXPECT_NEAR(dense(1, 1), 1.001, 1e-12);
}

TEST(MacroModelTest, ModalPortAssemblerReevaluatesInterfaceConductanceFromFvmState)
{
    auto model = make_single_cell_model("1 + T");

    ModalPort macro;
    macro.operators.K.resize(1, 1);
    macro.operators.C.resize(1, 1);
    macro.operators.f = Eigen::VectorXd::Zero(1);
    macro.basis = Eigen::MatrixXd::Ones(1, 1);

    ThermalPortInterface interface;
    interface.model_cells = {0};
    interface.model_face = mhs::core::FaceDir::XP;
    interface.exterior_half_conductance = Eigen::VectorXd::Constant(1, 0.002);

    const std::array cold_state {0.0, 0.0};
    const std::array hot_state {1.0, 0.0};
    const auto cold = assemble_modal_port_system(model, macro, interface, cold_state, 0.0);
    const auto hot = assemble_modal_port_system(model, macro, interface, hot_state, 0.0);

    EXPECT_NEAR(cold.K.coeff(0, 0), 0.001, 1e-12);
    EXPECT_NEAR(hot.K.coeff(0, 0), 4.0 / 3000.0, 1e-12);
    EXPECT_GT(hot.K.coeff(0, 0), cold.K.coeff(0, 0));
}

TEST(MacroModelTest, ModalPortSystemAdvancesTheFullCoupledTransientState)
{
    auto model = make_single_cell_model();

    ModalPort macro;
    macro.operators.K.resize(1, 1);
    macro.operators.K.insert(0, 0) = 1.0;
    macro.operators.C.resize(1, 1);
    macro.operators.C.insert(0, 0) = 0.5;
    macro.operators.f = Eigen::VectorXd::Ones(1);
    macro.basis = Eigen::MatrixXd::Ones(1, 1);

    ThermalPortInterface interface;
    interface.model_cells = {0};
    interface.model_face = mhs::core::FaceDir::XP;
    interface.exterior_half_conductance = Eigen::VectorXd::Constant(1, 0.002);

    const std::array initial_state {300.0, 300.0};
    const double dt = 0.25;
    const auto initial_operators = assemble_modal_port_system(model, macro, interface, initial_state, dt);
    const Eigen::Matrix2d expected_matrix
        = Eigen::MatrixXd(initial_operators.K) + Eigen::MatrixXd(initial_operators.C) / dt;
    const Eigen::Vector2d initial = Eigen::Map<const Eigen::Vector2d>(initial_state.data());
    const Eigen::Vector2d expected_rhs = initial_operators.f + initial_operators.C * initial / dt;
    const Eigen::Vector2d expected = expected_matrix.fullPivLu().solve(expected_rhs);

    Study study {mhs::core::StudyType::Transient, dt, dt};
    SolverOpts options;
    options.step_strategy = time_scheme::StepStrategy::Fixed;
    options.fixed_dt = dt;
    std::vector<double> observed_times;
    auto result = solve_system(study,
        [&](std::span<const double> state, double time) {
            return assemble_modal_port_system(model, macro, interface, state, time);
        },
        initial_state, options,
        [&](double time, std::span<const double>) {
            observed_times.push_back(time);
        });

    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.state.size(), 2u);
    EXPECT_NEAR(result.state[0], expected[0], 1e-10);
    EXPECT_NEAR(result.state[1], expected[1], 1e-10);
    EXPECT_EQ(observed_times, (std::vector<double> {0.0, dt}));
}
