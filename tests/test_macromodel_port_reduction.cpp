#include "compiler/model_compiler.hpp"
#include "macromodel/modal_port.hpp"
#include "solver/assembler.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

    mhs::core::Model make_line_model(std::size_t cell_count, mhs::model::StudyType study)
    {
        mhs::model::ModelDefinition definition;
        definition.settings.study_type = study;
        definition.settings.length_unit = mhs::model::LengthUnit::Millimeter;
        definition.settings.initial_temperature = 300.0;
        definition.settings.transient_duration = 0.1;
        definition.settings.transient_output_interval = 0.1;

        definition.mesh.x_vertices.reserve(cell_count + 1);
        for (std::size_t i = 0; i <= cell_count; ++i) {
            definition.mesh.x_vertices.push_back(static_cast<double>(i));
        }
        definition.mesh.y_vertices = {0.0, 1.0};
        definition.mesh.z_vertices = {0.0, 1.0};

        mhs::model::LayerSpec layer;
        layer.thickness = "1";
        mhs::model::BlockSpec block;
        block.material = "solid";
        block.volumetric_heat_source = "0";
        mhs::model::RectOperation rectangle;
        rectangle.operation = mhs::model::GeometryOperation::Add;
        rectangle.rect.x = "0";
        rectangle.rect.y = "0";
        rectangle.rect.width = std::to_string(cell_count);
        rectangle.rect.height = "1";
        block.geometry.push_back(rectangle);
        layer.blocks.push_back(block);
        definition.layers.push_back(layer);

        mhs::model::MaterialSpec material;
        material.conductivity_x = material.conductivity_y = material.conductivity_z = "1";
        material.density = "2";
        material.specific_heat = "3";
        definition.materials.push_back({"solid", material});
        definition.default_boundary = mhs::model::NeumannBoundary {};
        return mhs::sim::build_model(definition);
    }

    void expect_dense_near(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected, double tolerance)
    {
        ASSERT_EQ(actual.rows(), expected.rows());
        ASSERT_EQ(actual.cols(), expected.cols());
        for (Eigen::Index row = 0; row < actual.rows(); ++row) {
            for (Eigen::Index column = 0; column < actual.cols(); ++column) {
                EXPECT_NEAR(actual(row, column), expected(row, column), tolerance)
                    << "at (" << row << ", " << column << ")";
            }
        }
    }

} // namespace

TEST(MacroModelPortReductionTest, ReducedPortProjectionMatchesDenseBtGBFormula)
{
    auto model = make_line_model(3, mhs::model::StudyType::Steady);
    const std::array temperature {300.0, 300.0, 300.0};
    const auto base = mhs::sim::assemble_thermal(model, temperature, 0.0);

    mhs::macro::PortModel port;
    port.operators.K.resize(2, 2);
    port.operators.K.insert(0, 0) = 0.3;
    port.operators.K.insert(1, 1) = 0.4;
    port.operators.C.resize(2, 2);
    port.operators.f = Eigen::Vector2d::Zero();
    port.basis.resize(3, 2);
    port.basis << 1.0, 0.0, 0.5, 0.5, 0.0, 1.0;
    port.physical_port_count = 3;

    mhs::macro::PortCoupling coupling;
    coupling.model_cells = {0, 1, 2};
    coupling.model_face = mhs::core::FaceDir::YP;

    const std::array state {300.0, 300.0, 300.0, 0.0, 0.0};
    const auto assembled = mhs::macro::assemble(model, port, coupling, state, 0.0);

    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(5, 5);
    expected.topLeftCorner(3, 3) = Eigen::MatrixXd(base.K);
    expected.bottomRightCorner(2, 2) = Eigen::MatrixXd(port.operators.K);

    // Each 1 mm cube has Y-face half conductance
    // k*A/(dy/2) = 1 * 1e-6 / 0.5e-3 = 0.002 W/K.
    constexpr double conductance = 0.002;
    expected.topLeftCorner(3, 3).diagonal().array() += conductance;
    expected.topRightCorner(3, 2) -= conductance * port.basis;
    expected.bottomLeftCorner(2, 3) -= conductance * port.basis.transpose();
    expected.bottomRightCorner(2, 2) += conductance * port.basis.transpose() * port.basis;

    expect_dense_near(Eigen::MatrixXd(assembled.K), expected, 1.0e-13);
}

TEST(MacroModelPortReductionTest, TransientCapacityRegularizesSingularConductanceOperator)
{
    auto model = make_line_model(1, mhs::model::StudyType::Transient);

    mhs::macro::PortModel port;
    port.operators.K.resize(1, 1); // Deliberately singular isolated macro K.
    port.operators.C.resize(1, 1);
    port.operators.C.insert(0, 0) = 0.5;
    port.operators.f = Eigen::VectorXd::Zero(1);
    port.basis = Eigen::MatrixXd::Ones(1, 1);
    port.physical_port_count = 1;

    mhs::macro::PortCoupling coupling;
    coupling.model_cells = {0};
    coupling.model_face = mhs::core::FaceDir::XP;

    const std::array initial_state {300.0, 310.0};
    mhs::sim::SolveOptions options;
    options.linear_solver = mhs::sim::SolveOptions::LinearSolverType::EigenSparseLU;
    options.step_strategy = mhs::sim::SolveOptions::StepStrategy::Fixed;
    options.integrator = mhs::sim::SolveOptions::Integrator::Bdf1;
    options.fixed_dt = 0.1;
    options.min_dt = 0.1;
    options.max_dt = 0.1;

    const auto result = mhs::macro::solve(model, port, coupling, initial_state, options);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.state.size(), 2u);
    EXPECT_TRUE(std::isfinite(result.state[0]));
    EXPECT_TRUE(std::isfinite(result.state[1]));
    EXPECT_LT(std::abs(result.state[1] - result.state[0]), 10.0);
}
