#include "compiler/model_compiler.hpp"
#include "core/vector.hpp"
#include "model/model.hpp"
#include "model/model_definition.hpp"
#include "solver/assembler.hpp"
#include <gtest/gtest.h>

using namespace mhs::solver;
using namespace mhs::compiler;

// Helper: build a minimal SI ModelDefinition for a simple uniform cube
static mhs::model::ModelDefinition make_simple_cube_io()
{
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0.0, 0.005, 0.01};
    io.mesh.y_vertices = {0.0, 0.005, 0.01};
    io.mesh.z_vertices = {0.0, 0.005, 0.01};

    io.sources = {"0"};
    io.volumes = {{0, 0, {{mhs::model::GeometryOperation::Add, {0, 0, 0}, {0.01, 0.01, 0.01}}}}};

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    mat.density = "8920";
    mat.specific_heat = "385";
    io.materials.push_back({"copper", mat});

    io.default_thermal_boundary = mhs::model::NeumannBoundary {};

    return io;
}

TEST(AssemblerTest, AssembleCapacityMatrixMatchesExpected)
{
    auto io = make_simple_cube_io();
    auto model = mhs::compiler::build_model(io);

    int N = static_cast<int>(model.cell_material.size());
    mhs::core::Vector<double> T(static_cast<std::size_t>(N), 300.0);

    auto ops = assemble_thermal(model, T, 0.0);

    EXPECT_EQ(ops.C.rows(), N);
    EXPECT_EQ(ops.C.cols(), N);
    EXPECT_EQ(ops.C.nnz(), N);
    // Each cell: rho=8920, c=385, vol = (5e-3)^3 = 1.25e-7
    // C(i,i) = 8920 * 385 * 1.25e-7 = 0.4293
    double expected = 8920.0 * 385.0 * 1.25e-7;
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(ops.C.coeff(i, i), expected, 1e-6) << "Cell " << i;
    }
}

TEST(AssemblerTest, AssembleReadsCellTemperatureForMaterialProperties)
{
    // Material k is T-dependent in this case ("100 + T"). Different T should
    // produce different K.
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0.0, 0.005, 0.01};
    io.mesh.y_vertices = {0.0, 0.005, 0.01};
    io.mesh.z_vertices = {0.0, 0.005, 0.01};

    io.sources = {"0"};
    io.volumes = {{0, 0, {{mhs::model::GeometryOperation::Add, {0, 0, 0}, {0.01, 0.01, 0.01}}}}};

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "100 + T";
    io.materials.push_back({"mat", mat});

    io.default_thermal_boundary = mhs::model::NeumannBoundary {};

    auto model = mhs::compiler::build_model(io);

    int N = static_cast<int>(model.cell_material.size());

    mhs::core::Vector<double> T300(static_cast<std::size_t>(N), 300.0);
    mhs::core::Vector<double> T500(static_cast<std::size_t>(N), 500.0);
    auto k1 = assemble_thermal(model, T300, 0.0).K;
    auto k2 = assemble_thermal(model, T500, 0.0).K;

    const bool differs = !std::equal(k1.values().begin(), k1.values().end(), k2.values().begin(),
        [](double a, double b) { return std::abs(a - b) <= 1e-9; });
    EXPECT_TRUE(differs) << "T-dependent k should produce different K for different T";
}

TEST(AssemblerTest, AssembleProducesZeroRhsForAdiabaticNoSource)
{
    auto io = make_simple_cube_io();
    auto model = mhs::compiler::build_model(io);

    int N = static_cast<int>(model.cell_material.size());
    mhs::core::Vector<double> T(static_cast<std::size_t>(N), 300.0);

    auto ops = assemble_thermal(model, T, 0.0);

    // Adiabatic Neumann(0) on all faces with no source => b = 0 everywhere.
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(ops.f[i], 0.0, 1e-12);
    }
    // And K should be non-empty.
    EXPECT_GT(ops.K.nnz(), 0);
}
