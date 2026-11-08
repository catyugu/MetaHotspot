#include "compiler/model_compiler.hpp"
#include "solver/assembler.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

namespace {
    mhs::model::ModelDefinition refined_bar()
    {
        using namespace mhs::model;
        ModelDefinition d;
        auto& region = d.domains[0];
        region.mesh.x_vertices = {0, 1, 2};
        region.mesh.y_vertices = region.mesh.z_vertices = {0, 1};
        region.refinements.push_back({{0, 0, 0}, {1, 1, 1}, 1});
        MaterialSpec material;
        material.conductivity_x = material.conductivity_y = material.conductivity_z = "1";
        material.density = material.specific_heat = "1";
        d.materials.push_back({"solid", material});
        region.volumes.push_back({0, mhs::core::invalid_id,
            {{GeometryOperation::Add, {0, 0, 0}, {2, 1, 1}}}});
        region.default_thermal_boundary = NeumannBoundary {};
        region.thermal_boundaries.push_back(
            {{{Axis::X, 0, {0, 1, 0, 1}}}, DirichletBoundary {"0"}});
        region.thermal_boundaries.push_back(
            {{{Axis::X, 2, {0, 1, 0, 1}}}, DirichletBoundary {"2"}});
        return d;
    }
}

TEST(NonconformingFom, OctreeChildrenUseConservativeInternalTpfa)
{
    const auto compiled = mhs::compiler::build_model(refined_bar());
    ASSERT_EQ(compiled.domains.size(), 1u);
    ASSERT_TRUE(compiled.connections.empty());
    const auto& mesh = compiled.domains[0].discrete;
    ASSERT_EQ(mesh.cells.size(), 9u);
    EXPECT_EQ(compiled.layout.state_count, 9u);
    double plane_area = 0.0;
    std::size_t subfaces = 0;
    for (const auto& face : mesh.faces) {
        EXPECT_GT(face.area(), 0.0);
        if (!face.exterior() && face.axis == mhs::model::Axis::X) {
            const auto right = mesh.cells[face.minus].max[0];
            if (right == 1.0) {
                plane_area += face.area();
                ++subfaces;
                EXPECT_NEAR(face.area(), 0.25, 1e-14);
            }
        }
    }
    EXPECT_EQ(subfaces, 4u);
    EXPECT_NEAR(plane_area, 1.0, 1e-14);

    for (std::size_t c = 0; c < mesh.cells.size(); ++c) {
        mhs::model::Vec3 closed {};
        for (auto face_id : mesh.incident_faces(static_cast<mhs::model::CellId>(c))) {
            const auto& f = mesh.faces[face_id];
            const auto sign = f.minus == c ? 1.0 : -1.0;
            closed[static_cast<std::size_t>(f.axis)] += sign * f.area();
        }
        for (const auto v : closed)
            EXPECT_NEAR(v, 0.0, 1e-14);
    }
    mhs::core::Vector<double> exact(mesh.cells.size());
    for (std::size_t c = 0; c < mesh.cells.size(); ++c)
        exact[c] = mesh.cells[c].center()[0];

    const auto ops = mhs::solver::assemble_thermal(compiled, exact, 0);
    for (std::size_t row = 0; row < exact.size(); ++row) {
        double residual = -ops.f[row];
        for (std::size_t col = 0; col < exact.size(); ++col)
            residual += ops.K.coeff(row, col) * exact[col];
        EXPECT_NEAR(residual, 0.0, 1e-12) << "cell " << row;
    }
    double total_heat_capacity = 0;
    for (std::size_t c = 0; c < exact.size(); ++c)
        total_heat_capacity += ops.C.coeff(c, c);
    EXPECT_NEAR(total_heat_capacity, 2.0, 1e-14);
}

TEST(NonconformingFom, RefinementBoxesMustBeAlignedAndHaveValidLevels)
{
    auto definition = refined_bar();
    definition.domains[0].refinements[0].max[0] = 1.5;
    EXPECT_THROW(mhs::compiler::build_model(definition), std::invalid_argument);
    definition.domains[0].refinements[0].max[0] = 1.0;
    definition.domains[0].refinements[0].levels = 0;
    EXPECT_THROW(mhs::compiler::build_model(definition), std::invalid_argument);
}
