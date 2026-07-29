#include "compiler/model_compiler.hpp"
#include "model_test_utils.hpp"
#include "solver/port_coupling.hpp"
#include "solver/scheduler.hpp"
#include <Eigen/LU>
#include <array>
#include <algorithm>
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

TEST(SchedulerTest, SolveSystemReassemblesWholeLinearizationDuringNonlinearIteration)
{
    std::vector<double> evaluated_conductances;
    SystemAssembler assemble = [&](std::span<const double> state, double time) {
        EXPECT_EQ(state.size(), 2u);
        EXPECT_DOUBLE_EQ(time, 0.0);

        const double conductance = 1.0 + state[0];
        evaluated_conductances.push_back(conductance);

        Operators operators;
        operators.K.resize(2, 2);
        operators.K.insert(0, 0) = conductance;
        operators.K.insert(0, 1) = -conductance;
        operators.K.insert(1, 0) = -conductance;
        operators.K.insert(1, 1) = conductance + 1.0;
        operators.C.resize(2, 2);
        operators.f = Eigen::Vector2d(0.0, 1.0);
        return operators;
    };

    const std::array initial_state {0.0, 0.0};
    Study study;
    auto result = solve_system(study, assemble, initial_state);

    ASSERT_TRUE(result.converged);
    ASSERT_GE(evaluated_conductances.size(), 2u);
    EXPECT_NEAR(evaluated_conductances.front(), 1.0, 1e-12);
    EXPECT_NEAR(evaluated_conductances.back(), 2.0, 1e-12);
    EXPECT_NEAR(result.state[0], 1.0, 1e-12);
    EXPECT_NEAR(result.state[1], 1.0, 1e-12);
}

TEST(SchedulerTest, ModalPortAssemblerProjectsPhysicalInterface)
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

TEST(SchedulerTest, ModalPortAssemblerReevaluatesInterfaceConductanceFromFvmState)
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

TEST(SchedulerTest, ModalPortSystemAdvancesTheFullCoupledTransientState)
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

TEST(SchedulerTest, SteadyHeatSourceProducesTemperatureGradient)
{
    // Cube with heat source + Dirichlet on bottom + convective BC on top
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0.0, 5.0, 10.0};
    io.mesh.y_vertices = {0.0, 5.0, 10.0};
    io.mesh.z_vertices = {0.0, 5.0, 10.0};

    mhs::model::LayerSpec layer;
    layer.thickness = "10";

    mhs::model::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "1e6"; // heat source

    mhs::model::RectOperation rect;
    rect.operation = mhs::model::GeometryOperation::Add;
    rect.rect.x = "0";
    rect.rect.y = "0";
    rect.rect.width = "10";
    rect.rect.height = "10";
    block.geometry.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    io.materials.push_back({"copper", mat});

    // Dirichlet 300K on bottom face (Z=0)
    mhs::model::BoundaryPatch boundary1;
    boundary1.condition = mhs::model::DirichletBoundary {"300"};
    boundary1.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    io.boundaries.push_back(boundary1);

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    auto result = solve_thermal(model);

    EXPECT_EQ(result.temperature.size(), model.cells.cell_to_grid.size());
    EXPECT_TRUE(std::equal(result.temperature.begin(), result.temperature.end(), result.temperature.begin()));

    // With heat source and Dirichlet 300K at bottom, temperatures should be > 300K
    double max_T = 0.0;
    for (const auto& t : result.temperature) {
        max_T = std::max(max_T, t);
    }
    EXPECT_GT(max_T, 300.0) << "Heat source should raise temperature above 300K";
}

TEST(SchedulerTest, ProbeRecorderCapturesPerStep)
{
    // 瞬态 5 步，2 个观察点。ProbeRecorder 应在 t=0 起点 + 5 个步末各记录 1 次。
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Transient;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0.0, 5.0, 10.0};
    io.mesh.y_vertices = {0.0, 5.0, 10.0};
    io.mesh.z_vertices = {0.0, 5.0, 10.0};

    mhs::model::LayerSpec layer;
    layer.thickness = "10";
    mhs::model::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "1e8";

    mhs::model::RectOperation rect;
    rect.operation = mhs::model::GeometryOperation::Add;
    rect.rect.x = "0";
    rect.rect.y = "0";
    rect.rect.width = "10";
    rect.rect.height = "10";
    block.geometry.push_back(rect);
    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    mat.density = "8920";
    mat.specific_heat = "385";
    io.materials.push_back({"copper", mat});

    io.settings.transient_duration = 5.0;
    io.settings.transient_output_interval = 1.0;

    io.default_boundary = mhs::model::NeumannBoundary {};

    // 两个观察点：中心 (5,5,5) mm + Dirichlet 面 z=0 上的 (5,5,0)
    mhs::model::ObservationPointSpec op1;
    op1.name = "center";
    op1.x = "5";
    op1.y = "5";
    op1.z = "5";
    io.observation_points.push_back(op1);
    mhs::model::ObservationPointSpec op2;
    op2.name = "z0";
    op2.x = "5";
    op2.y = "5";
    op2.z = "0";
    io.observation_points.push_back(op2);

    // z=0 设为 Dirichlet 500K，确保 op2 走 Dirichlet 早返回路径
    mhs::model::BoundaryPatch boundary;
    boundary.condition = mhs::model::DirichletBoundary {"500"};
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    io.boundaries.push_back(boundary);

    auto model = build_model(io);

    auto result = solve_thermal(model);

    const auto& traces = result.probe_traces;
    ASSERT_EQ(traces.size(), 2u);
    EXPECT_EQ(traces[0].name, "center");
    EXPECT_EQ(traces[1].name, "z0");

    // Sub-stepping 允许内部步比输出网格更细，因此采样点 ≥ duration/dt。
    for (const auto& tr : traces) {
        EXPECT_GE(tr.times.size(), 6u);
        EXPECT_GE(tr.values.size(), 6u);
        EXPECT_NEAR(tr.times.front(), 0.0, 1e-9);
        EXPECT_NEAR(tr.times.back(), 5.0, 1e-9);
        for (size_t i = 1; i < tr.times.size(); ++i) {
            EXPECT_GT(tr.times[i], tr.times[i - 1]) << "Times must be monotonically increasing";
        }
    }

    // op1 "center" 在体心；强热源下温度应随时间上升
    EXPECT_NEAR(traces[0].values.front(), 300.0, 1e-3) << "t=0 must be initial temperature";
    EXPECT_GT(traces[0].values.back(), traces[0].values.front())
        << "Center probe must rise over time with strong heat source";

    // op2 "z0" 落在 Dirichlet 面上 → 始终 500K（Dirichlet 是强约束，不随内部场变化）
    for (double v : traces[1].values) {
        EXPECT_NEAR(v, 500.0, 1e-6) << "z0 probe on Dirichlet face must stay at 500K";
    }
}

TEST(SchedulerTest, ProbeRecorderUsesCurrentTimeForTimeDependentBC)
{
    // Regression: ProbeRecorder::sample_one 之前把 FieldContext.t 硬编码成 0.0，
    // 导致时间依赖的 BC 表达式在 t>0 时被错误求值。本测试在 z=0 面上用
    // 时间依赖 Dirichlet "500 + 100*t"，跑 5 步瞬态 (dt=1)，验证每步末的
    // 探针温度严格等于 500 + 100*t，而非恒为 500。
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Transient;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0.0, 5.0, 10.0};
    io.mesh.y_vertices = {0.0, 5.0, 10.0};
    io.mesh.z_vertices = {0.0, 5.0, 10.0};

    mhs::model::LayerSpec layer;
    layer.thickness = "10";
    mhs::model::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "0";

    mhs::model::RectOperation rect;
    rect.operation = mhs::model::GeometryOperation::Add;
    rect.rect.x = "0";
    rect.rect.y = "0";
    rect.rect.width = "10";
    rect.rect.height = "10";
    block.geometry.push_back(rect);
    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    io.materials.push_back({"copper", mat});

    io.settings.transient_duration = 5.0;
    io.settings.transient_output_interval = 1.0;

    io.default_boundary = mhs::model::NeumannBoundary {};

    // Dirichlet 探针位于 z=0 面中心；BC 表达式随时间线性增长
    mhs::model::ObservationPointSpec op;
    op.name = "z0_dirichlet";
    op.x = "5";
    op.y = "5";
    op.z = "0";
    io.observation_points.push_back(op);

    mhs::model::BoundaryPatch boundary;
    boundary.condition = mhs::model::DirichletBoundary {"500 + 100*t"};
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    io.boundaries.push_back(boundary);

    auto model = build_model(io);

    auto result = solve_thermal(model);

    const auto& traces = result.probe_traces;
    ASSERT_EQ(traces.size(), 1u);
    const auto& tr = traces[0];
    ASSERT_GE(tr.times.size(), 6u); // t=0 + 5 步末（内部可 sub-stepping）
    ASSERT_GE(tr.values.size(), 6u);

    // 每步的时间值必须满足 T(t) = 500 + 100*t。旧实现把 t 写死 0，
    // 会让 tr.values 恒为 500.0；修复后必须随时间线性增长。
    for (size_t i = 0; i < tr.times.size(); ++i) {
        double expected = 500.0 + 100.0 * tr.times[i];
        EXPECT_NEAR(tr.values[i], expected, 1e-6) << "Time-dependent Dirichlet eval failed at t=" << tr.times[i]
                                                  << " (expected " << expected << ", got " << tr.values[i] << ")";
    }
}
