#include "compiler/fom_compiler.hpp"
#include "compiler/project_compiler.hpp"
#include "logging/logger.hpp"
#include "solver/assembler.hpp"
#include "solver/solver.hpp"
#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>
#include <sstream>

using namespace mhs;
using namespace mhs::model;

namespace {
    std::shared_ptr<const Fom> cube(bool split = false, bool interface_probe = false)
    {
        FomDefinition definition;
        MaterialSpec material;
        material.conductivity_x = material.conductivity_y = material.conductivity_z = "1";
        material.density = material.specific_heat = "1";
        definition.materials.push_back({"solid", material});
        definition.volumes = {{0, core::invalid_id, {{GeometryOperation::Add, {}, {1, 1, 1}}}}};
        definition.thermal_boundaries = {{{{Axis::X, 0, {0, 1, 0, 1}}}, DirichletBoundary {"400"}},
            {{{Axis::X, 1, {0, 1, 0, 1}}}, DirichletBoundary {"300"}}};
        if (interface_probe)
            definition.point_probes = {{"contact", 1, 0.75, 0.5}, {"exposed", 1, 0.25, 0.5}};
        return std::make_shared<const Fom>(compiler::compile_fom(
            definition, {{0, 1}, split ? core::Vector<double> {0, 0.5, 1} : core::Vector<double> {0, 1}, {0, 1}}));
    }

    std::shared_ptr<Erom> modal(double reference = 280, bool left_open = true)
    {
        auto e = std::make_shared<Erom>();
        e->K = core::Matrix<double>(1, 1);
        e->C = core::Matrix<double>(1, 1);
        e->C(0, 0) = 2;
        e->B = core::Matrix<double>(1, 2);
        e->B(0, 0) = 1;
        e->B(0, 1) = 2;
        e->L = core::Matrix<double>(2, 1);
        e->L(0, 0) = 1;
        e->L(1, 0) = 0.5;
        e->Vb = core::Matrix<double>(left_open ? 2 : 1, 1);
        e->Vb(0, 0) = 1;
        const Facet left {Axis::X, -1, 0, {0, 1, 0, 1}};
        const Facet right {Axis::X, 1, 1, {0, 1, 0, 1}};
        if (left_open) {
            e->Vb(1, 0) = 1;
            e->interfaces.push_back({left, 0.25, 0});
            e->boundary_groups.push_back({"left", 1, 10, 2, 310});
        }
        else
            e->outline.push_back({"left", {left}});
        e->interfaces.push_back({right, 0.25, static_cast<BoundaryGroupId>(e->boundary_groups.size())});
        e->boundary_groups.push_back({"right", 1, 10, 1, 330});
        e->reference_temperature = reference;
        e->uniform_temperature_mode = {1};
        e->sources = {{"first", 0}, {"second", 1}};
        e->outputs = {{"sensor", Vec3 {0.5, 0.25, 0.5}}, {"source mean", std::nullopt}};
        return e;
    }

    ComponentInstance instance(ComponentId id, std::shared_ptr<const Fom> model, Vec3 at = {})
    {
        return {id, at, std::move(model), {}, {}};
    }

    ComponentInstance instance(ComponentId id, std::shared_ptr<const Erom> model, Vec3 at = {})
    {
        return {id, at, std::move(model), {}, {}};
    }
} // namespace

TEST(Project, ComponentCountsAndSharedAssets)
{
    const auto empty = mhs::compiler::compile_project({});
    EXPECT_EQ(empty.dof_count(), 0u);
    EXPECT_THROW(solver::solve(empty), std::invalid_argument);
    auto fom = cube();
    auto erom = modal();
    for (const auto& [foms, eroms] : {std::pair {2, 0}, {0, 1}, {0, 2}, {1, 1}, {2, 2}}) {
        ProjectDefinition definition;
        for (int i = 0; i < foms; ++i)
            definition.components.push_back(instance(static_cast<ComponentId>(i), fom, {2.0 * i, 0, 0}));
        for (int i = foms; i < foms + eroms; ++i)
            definition.components.push_back(instance(static_cast<ComponentId>(i), erom, {2.0 * i, 0, 0}));
        const auto project = mhs::compiler::compile_project(std::move(definition));
        EXPECT_EQ(project.dof_count(), static_cast<std::size_t>(foms + eroms));
        EXPECT_TRUE(project.patches().empty());
        const auto result = solver::solve(project);
        EXPECT_TRUE(result.converged);
        ASSERT_EQ(result.output_traces.size(), 2u * eroms);
        for (int i = 0; i < eroms; ++i) {
            EXPECT_EQ(result.output_traces[2 * i].name, "sensor");
            EXPECT_EQ(result.output_traces[2 * i].component, static_cast<ComponentId>(i + foms));
            EXPECT_EQ(result.output_traces[2 * i].output, 0u);
        }
    }
}

TEST(Project, MatchingAndNonmatchingFomContactConserveHeat)
{
    for (bool split : {false, true}) {
        const auto project = mhs::compiler::compile_project(
            {{instance(1, cube()), instance(2, cube(split), {1, 0, 0})}, {{{1, 2}, 0.25}}});
        const auto count = split ? 2u : 1u;
        ASSERT_EQ(project.patches().size(), count);
        solver::ProjectAssembler assembler(project);
        const core::Vector<double> t(project.dof_count(), 300.0);
        const auto& ops = assembler.assemble(t, 0);
        double area = 0;
        for (const auto& patch : project.patches()) {
            area += patch.area();
            EXPECT_NEAR(patch.area(), 1.0 / count, 1e-14);
        }
        EXPECT_NEAR(area, 1.0, 1e-14);
        EXPECT_NEAR(ops.K.coeff(0, 0), 2.8, 1e-14);
        for (std::size_t i = 1; i <= count; ++i)
            EXPECT_NEAR(ops.K.coeff(0, i), -0.8 / count, 1e-14);
        const auto result = solver::solve(project);
        const double heat = 100 / 2.25;
        ASSERT_TRUE(result.converged);
        EXPECT_NEAR(result.state[0], 400 - heat / 2, 1e-6);
        for (std::size_t i = 1; i <= count; ++i)
            EXPECT_NEAR(result.state[i], 300 + heat / 2, 1e-6);
        EXPECT_NEAR(0.8 * (result.state[0] - result.state[1]), heat, 1e-6);
    }
}

TEST(Project, PartialMixedContactUsesExposedBoundaryAndReferenceOffsets)
{
    const auto erom = modal();
    const auto project = mhs::compiler::compile_project(
        {{instance(1, cube(false, true), {7, 13, 19}), instance(2, erom, {8, 13.5, 19})}, {{{1, 2}, 0.25}}});
    ASSERT_EQ(project.patches().size(), 3u);
    EXPECT_TRUE(project.patches()[0].contact);
    EXPECT_DOUBLE_EQ(project.patches()[0].area(), 0.5);
    EXPECT_EQ(project.output_position(2, 0), (Vec3 {8.5, 13.75, 19.5}));
    EXPECT_FALSE(project.output_position(2, 1));
    EXPECT_EQ(*erom->outputs[0].point, (Vec3 {0.5, 0.25, 0.5}));

    solver::ProjectAssembler assembler(project);
    const core::Vector<double> state {320, 40};
    const auto& ops = assembler.assemble(state, 0);
    EXPECT_NEAR(ops.K.coeff(0, 0), 2 + 1 + 0.5, 1e-13);
    EXPECT_NEAR(ops.K.coeff(0, 1), -0.5, 1e-13);
    EXPECT_NEAR(ops.K.coeff(1, 1), 0.5 + 2.0 / 3 + 0.8, 1e-13);
    EXPECT_NEAR(ops.f[0], 800 + 300 + 140, 1e-12);
    EXPECT_NEAR(ops.f[1], 2 - 140 + 20 + 40, 1e-12);
    EXPECT_NEAR(ops.C.coeff(1, 1), 2, 1e-13);

    const auto result = solver::solve(project);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.output_traces.size(), 4u);
    const auto& x = result.state;
    EXPECT_NEAR(result.output_traces[0].values[0], x[0] + 0.5 * (280 + x[1] - x[0]), 1e-8);
    EXPECT_DOUBLE_EQ(result.output_traces[1].values[0], 300);
    EXPECT_NEAR(result.output_traces[2].values[0], 280 + x[1], 1e-12);
    EXPECT_NEAR(result.output_traces[3].values[0], 280 + 0.5 * x[1], 1e-12);
}

TEST(Project, EromContactUsesModalOuterProductsAndInstanceSchedules)
{
    auto left = modal(280);
    auto right = modal(310);
    // Two contact patches share the same pair of modal blocks.
    auto split = left->interfaces[1];
    left->interfaces[1].facet.bounds[1] = split.facet.bounds[0] = 0.5;
    left->interfaces.push_back(split);
    left->Vb = core::Matrix<double>(3, 1);
    left->Vb(0, 0) = 1;
    left->Vb(1, 0) = left->Vb(2, 0) = 2;
    right->Vb(0, 0) = -3;
    auto l = instance(1, left), r = instance(2, right, {1, 0, 0});
    l.powers[0] = [](double t) { return 7 * t; };
    const auto project = mhs::compiler::compile_project({{l, r}, {}});
    ASSERT_EQ(project.patches().size(), 2u);
    EXPECT_EQ(project.dof_count(), 2u);
    const core::Vector<double> state {0, 0};
    solver::ProjectAssembler assembler(project);
    const auto& ops = assembler.assemble(state, 2);
    // A/Rsum=2 and the difference vector is (2, 3).
    EXPECT_DOUBLE_EQ(ops.K.coeff(0, 1), 12);
    EXPECT_DOUBLE_EQ(ops.K.coeff(1, 0), 12);
    EXPECT_NEAR(ops.K.coeff(0, 0), 8 + 4.0 / 3, 1e-13);
    EXPECT_NEAR(ops.K.coeff(1, 1), 18 + 0.8, 1e-13);
    EXPECT_NEAR(ops.f[0], 16 + 30 * (4.0 / 3 + 4), 1e-12);
    EXPECT_NEAR(ops.f[1], 2 + 20 * 0.8 + 30 * 6, 1e-12);
    const auto& next = assembler.assemble(state, 3);
    EXPECT_EQ(next.K.nnz(), 4u);
    EXPECT_DOUBLE_EQ(next.K.coeff(0, 1), 12);
    EXPECT_NEAR(next.f[0], 23 + 30 * (4.0 / 3 + 4), 1e-12);
    EXPECT_NEAR(next.f[1], 2 + 20 * 0.8 + 30 * 6, 1e-12);
}

TEST(Project, UntrainedContactWarnsAndKeepsTheEnvironment)
{
    struct Capture {
        std::ostringstream log;
        std::shared_ptr<spdlog::sinks::sink> sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log);
        std::vector<spdlog::sink_ptr>& sinks = logger::detail::instance().sinks();
        Capture() { sinks.push_back(sink); }
        ~Capture() { sinks.pop_back(); }
    } capture;
    const auto project
        = mhs::compiler::compile_project({{instance(21, cube()), instance(42, modal(280, false), {1, 0, 0})}, {}});
    EXPECT_TRUE(project.patches().empty());
#if MHS_LOG_COMPILED_LEVEL <= 3
    const auto message = capture.log.str();
    const auto warning = message.find("thermal contact is ignored");
    ASSERT_NE(warning, std::string::npos);
    EXPECT_EQ(message.find("thermal contact is ignored", warning + 1), std::string::npos);
    EXPECT_NE(message.find("21 (boundary 1)"), std::string::npos);
    EXPECT_NE(message.find("42 (left)"), std::string::npos);
#endif
    const auto result = solver::solve(project);
    EXPECT_NEAR(result.state[0], 350, 1e-7);
}

TEST(Project, RejectsDuplicateIdentityAndOverlappingContacts)
{
    const auto fom = cube();
    EXPECT_THROW(
        mhs::compiler::compile_project({{instance(1, fom), instance(1, fom, {1, 0, 0})}, {}}), std::invalid_argument);
    EXPECT_THROW(mhs::compiler::compile_project(
                     {{instance(1, fom), instance(2, fom, {1, 0, 0}), instance(3, fom, {1, 0.2, 0})}, {}}),
        std::invalid_argument);
    auto e = modal();
    e->interfaces[1].boundary_group = 0;
    EXPECT_THROW(mhs::compiler::compile_project({{instance(1, e)}, {}}), std::invalid_argument);
}
