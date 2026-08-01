#include "compiler/model_compiler.hpp"
#include "macromodel/modal_port.hpp"
#include "solver/assembler.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
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
        for (std::size_t i = 0; i <= cell_count; ++i)
            definition.mesh.x_vertices.push_back(static_cast<double>(i));
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

    std::vector<mhs::macro::PortPatch> top_patches(std::size_t count)
    {
        std::vector<mhs::macro::PortPatch> patches;
        for (std::size_t i = 0; i < count; ++i) {
            patches.push_back({mhs::core::FaceDir::YP, 1.0e-3, i * 1.0e-3, (i + 1) * 1.0e-3, 0.0, 1.0e-3});
        }
        return patches;
    }

} // namespace

TEST(MacroModelPortReductionTest, CompiledPatchesAssembleAnIsolatedDtNOperator)
{
    auto model = make_line_model(2, mhs::model::StudyType::Steady);
    const auto ports = mhs::macro::compile_port_map(model, top_patches(2));
    const std::array state {300.0, 300.0};
    const auto dtn = mhs::macro::assemble_dtn(model, ports, state, 0.0);

    ASSERT_EQ(ports.port_count, 2u);
    ASSERT_EQ(ports.faces.size(), 2u);
    ASSERT_EQ(dtn.K.rows(), 4);
    constexpr double g = 0.002;
    Eigen::MatrixXd dense = Eigen::MatrixXd(dtn.K);
    EXPECT_NEAR(dense(0, 0), g, 1.0e-13);
    EXPECT_NEAR(dense(0, 2), -g, 1.0e-13);
    EXPECT_NEAR(dense(2, 0), -g, 1.0e-13);
}

TEST(MacroModelPortReductionTest, ExactLeadingPortsPreserveSparseCoupling)
{
    auto model = make_line_model(3, mhs::model::StudyType::Steady);
    const auto ports = mhs::macro::compile_port_map(model, top_patches(3));
    const std::array temperature {300.0, 300.0, 300.0};
    const auto base = mhs::sim::assemble_thermal(model, temperature, 0.0);

    mhs::macro::DtNModel dtn;
    dtn.operators.K.resize(5, 5);
    dtn.operators.K.insert(0, 0) = 0.3;
    dtn.operators.K.insert(1, 1) = 0.4;
    dtn.operators.K.insert(2, 2) = 0.5;
    dtn.operators.K.insert(3, 3) = 0.6;
    dtn.operators.K.insert(4, 4) = 0.7;
    dtn.operators.K.insert(2, 3) = -0.1;
    dtn.operators.K.insert(3, 2) = -0.1;
    dtn.operators.C.resize(5, 5);
    dtn.operators.f = Eigen::VectorXd::Zero(5);

    const std::array state {300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 0.0, 0.0};
    const auto assembled = mhs::macro::assemble_coupled(model, dtn, ports, state, 0.0);
    const Eigen::MatrixXd dense = Eigen::MatrixXd(assembled.K);
    const Eigen::Index fvm_count = 3;
    constexpr double conductance = 0.002;

    for (Eigen::Index port = 0; port < 3; ++port) {
        EXPECT_NEAR(dense(port, port), Eigen::MatrixXd(base.K)(port, port) + conductance, 1.0e-13);
        EXPECT_NEAR(dense(port, fvm_count + port), -conductance, 1.0e-13);
        EXPECT_NEAR(dense(fvm_count + port, port), -conductance, 1.0e-13);
        EXPECT_NEAR(dense(fvm_count + port, fvm_count + port),
            Eigen::MatrixXd(dtn.operators.K)(port, port) + conductance, 1.0e-13);
    }
    for (Eigen::Index cell = 0; cell < fvm_count; ++cell) {
        EXPECT_DOUBLE_EQ(dense(cell, fvm_count + 3), 0.0);
        EXPECT_DOUBLE_EQ(dense(cell, fvm_count + 4), 0.0);
    }
    EXPECT_LE(assembled.K.nonZeros(), base.K.nonZeros() + dtn.operators.K.nonZeros() + 4 * ports.faces.size());
}

TEST(MacroModelPortReductionTest, RejectsFewerStatesThanPhysicalPorts)
{
    auto model = make_line_model(2, mhs::model::StudyType::Steady);
    const auto ports = mhs::macro::compile_port_map(model, top_patches(2));
    mhs::macro::DtNModel dtn;
    dtn.operators.K.resize(1, 1);
    dtn.operators.C.resize(1, 1);
    dtn.operators.f = Eigen::VectorXd::Zero(1);
    const std::array state {300.0, 300.0, 300.0};
    EXPECT_THROW(mhs::macro::assemble_coupled(model, dtn, ports, state, 0.0), std::invalid_argument);
}

TEST(MacroModelPortReductionTest, TransientCapacityRegularizesSingularDtNOperator)
{
    auto model = make_line_model(1, mhs::model::StudyType::Transient);
    const auto ports = mhs::macro::compile_port_map(model, top_patches(1));
    mhs::macro::DtNModel dtn;
    dtn.operators.K.resize(1, 1);
    dtn.operators.C.resize(1, 1);
    dtn.operators.C.insert(0, 0) = 0.5;
    dtn.operators.f = Eigen::VectorXd::Zero(1);

    const std::array initial_state {300.0, 310.0};
    mhs::sim::SolveOptions options;
    options.linear_solver = mhs::sim::SolveOptions::LinearSolverType::EigenSparseLU;
    options.step_strategy = mhs::sim::SolveOptions::StepStrategy::Fixed;
    options.integrator = mhs::sim::SolveOptions::Integrator::Bdf1;
    options.fixed_dt = options.min_dt = options.max_dt = 0.1;

    const auto result = mhs::macro::solve(model, dtn, ports, initial_state, options);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.state.size(), 2u);
    EXPECT_TRUE(std::isfinite(result.state[0]));
    EXPECT_TRUE(std::isfinite(result.state[1]));
    EXPECT_LT(std::abs(result.state[1] - result.state[0]), 10.0);
}
