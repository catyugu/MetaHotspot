#include "compiler/model_compiler.hpp"
#include "model/model_definition.hpp"
#include "runtime/model.hpp"
#include "solver/assembler.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace mhs::sim;

// Helper: build a default-state vector from the model's initial_temperature.
static std::vector<double> default_state(const mhs::core::Model& model)
{
    return std::vector<double>(static_cast<std::size_t>(model.cells.cell_to_grid.size()), model.initial_temperature);
}

// Helper: build a minimal mhs::model::ModelDefinition for a simple uniform cube
static mhs::model::ModelDefinition make_simple_cube_io()
{
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
    mat.density = "8920";
    mat.specific_heat = "385";
    io.materials.push_back({"copper", mat});

    io.default_boundary = mhs::model::NeumannBoundary {};

    return io;
}

TEST(AssemblerTest, CompileBuildsCellStateLayout)
{
    auto model = build_model(make_simple_cube_io());
    const auto cell_count = model.cells.material_id.size();

    EXPECT_EQ(model.cells.cell_to_grid.size(), cell_count);
    auto state = default_state(model);
    ASSERT_EQ(state.size(), cell_count);
    EXPECT_TRUE(
        std::all_of(state.begin(), state.end(), [&](double value) { return value == model.initial_temperature; }));
}

TEST(AssemblerTest, AssembleCapacityMatrixMatchesExpected)
{
    auto io = make_simple_cube_io();
    auto model = build_model(io);

    int N = static_cast<int>(model.cells.material_id.size());
    std::vector<double> T(static_cast<std::size_t>(N), 300.0);

    auto ops = assemble_thermal(model, T, 0.0);

    EXPECT_EQ(ops.C.rows(), N);
    EXPECT_EQ(ops.C.cols(), N);
    EXPECT_EQ(ops.C.nonZeros(), N);
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
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0.0, 5.0, 10.0};
    io.mesh.y_vertices = {0.0, 5.0, 10.0};
    io.mesh.z_vertices = {0.0, 5.0, 10.0};

    mhs::model::LayerSpec layer;
    layer.thickness = "10";
    mhs::model::BlockSpec block;
    block.material = "mat";
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
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "100 + T";
    io.materials.push_back({"mat", mat});

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    int N = static_cast<int>(model.cells.material_id.size());

    std::vector<double> T300(static_cast<std::size_t>(N), 300.0);
    std::vector<double> T500(static_cast<std::size_t>(N), 500.0);
    auto k1 = assemble_thermal(model, T300, 0.0).K;
    auto k2 = assemble_thermal(model, T500, 0.0).K;

    bool differs = false;
    for (int k = 0; k < k1.outerSize() && !differs; ++k) {
        for (typename Eigen::SparseMatrix<double>::InnerIterator it(k1, k); it && !differs; ++it) {
            if (std::abs(it.value() - k2.coeff(it.row(), it.col())) > 1e-9) {
                differs = true;
            }
        }
    }
    EXPECT_TRUE(differs) << "T-dependent k should produce different K for different T";
}

TEST(AssemblerTest, AssembleProducesZeroRhsForAdiabaticNoSource)
{
    // Sanity: for steady-state (dt=0), assemble returns A = K.
    // The b vector should equal what the old `assemble` would have computed for
    // steady (no mass term). The diagonal of K should be the same (negative).
    auto io = make_simple_cube_io();
    auto model = build_model(io);

    int N = static_cast<int>(model.cells.material_id.size());
    std::vector<double> T(static_cast<std::size_t>(N), 300.0);

    auto ops = assemble_thermal(model, T, 0.0);

    // Adiabatic Neumann(0) on all faces with no source => b = 0 everywhere.
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(ops.f(i), 0.0, 1e-12);
    }
    // And K should be non-empty.
    EXPECT_GT(ops.K.nonZeros(), 0);
}
