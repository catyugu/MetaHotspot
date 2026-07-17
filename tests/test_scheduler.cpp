#include "preprocessor/preprocessor.hpp"
#include "scheduler/scheduler.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace mhs::sim;

TEST(SchedulerTest, SteadyHeatSourceProducesTemperatureGradient)
{
    // Cube with heat source + Dirichlet on bottom + convective BC on top
    mhs::core::ModelDefinition io;
    io.study_type = mhs::core::StudyType::Steady;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    mhs::core::Layer layer;
    layer.thickness_expr = "10";

    mhs::core::Block block;
    block.material_name = "copper";
    block.ti_reyuan_expr = "1e6"; // heat source

    mhs::core::Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::core::Material mat;
    mat.kx = mat.ky = mat.kz = "400";
    io.materials["copper"] = mat;

    // Dirichlet 300K on bottom face (Z=0)
    mhs::core::Boundary boundary1;
    boundary1.bc = mhs::core::FirstTypeThermalBC {"300"};
    boundary1.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary1);

    io.other_bc = mhs::core::SecondTypeThermalBC {};

    auto model = build_model(io);

    auto result = solve(model);

    const auto& solution = result.temperature;
    EXPECT_EQ(solution.size(), model.cells.material_id.size());

    // With heat source and Dirichlet 300K at bottom, temperatures should be > 300K
    double max_T = 0.0;
    for (const auto& t : solution) {
        max_T = std::max(max_T, t);
    }
    EXPECT_GT(max_T, 300.0) << "Heat source should raise temperature above 300K";
}

TEST(SchedulerTest, ProbeRecorderCapturesPerStep)
{
    // 瞬态 5 步，2 个观察点。ProbeRecorder 应在 t=0 起点 + 5 个步末各记录 1 次。
    mhs::core::ModelDefinition io;
    io.study_type = mhs::core::StudyType::Transient;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    mhs::core::Layer layer;
    layer.thickness_expr = "10";
    mhs::core::Block block;
    block.material_name = "copper";
    block.ti_reyuan_expr = "1e8";

    mhs::core::Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);
    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::core::Material mat;
    mat.kx = mat.ky = mat.kz = "400";
    mat.midu = "8920";
    mat.bi_rerong = "385";
    io.materials["copper"] = mat;

    io.transient_duration = 5.0;
    io.transient_time_step = 1.0;

    io.other_bc = mhs::core::SecondTypeThermalBC {};

    // 两个观察点：中心 (5,5,5) mm + Dirichlet 面 z=0 上的 (5,5,0)
    mhs::core::ObservationPoint3D op1;
    op1.name = "center";
    op1.x = "5";
    op1.y = "5";
    op1.z = "5";
    io.observation_points.push_back(op1);
    mhs::core::ObservationPoint3D op2;
    op2.name = "z0";
    op2.x = "5";
    op2.y = "5";
    op2.z = "0";
    io.observation_points.push_back(op2);

    // z=0 设为 Dirichlet 500K，确保 op2 走 Dirichlet 早返回路径
    mhs::core::Boundary boundary;
    boundary.bc = mhs::core::FirstTypeThermalBC {"500"};
    boundary.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary);

    auto model = build_model(io);

    auto result = solve(model);

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
    mhs::core::ModelDefinition io;
    io.study_type = mhs::core::StudyType::Transient;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    mhs::core::Layer layer;
    layer.thickness_expr = "10";
    mhs::core::Block block;
    block.material_name = "copper";
    block.ti_reyuan_expr = "0";

    mhs::core::Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);
    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::core::Material mat;
    mat.kx = mat.ky = mat.kz = "400";
    io.materials["copper"] = mat;

    io.transient_duration = 5.0;
    io.transient_time_step = 1.0;

    io.other_bc = mhs::core::SecondTypeThermalBC {};

    // Dirichlet 探针位于 z=0 面中心；BC 表达式随时间线性增长
    mhs::core::ObservationPoint3D op;
    op.name = "z0_dirichlet";
    op.x = "5";
    op.y = "5";
    op.z = "0";
    io.observation_points.push_back(op);

    mhs::core::Boundary boundary;
    boundary.bc = mhs::core::FirstTypeThermalBC {"500 + 100*t"};
    boundary.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary);

    auto model = build_model(io);

    auto result = solve(model);

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
