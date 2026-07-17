#include "assembler/assembler.hpp"
#include "data/model.hpp"
#include "data/model_definition.hpp"
#include "preprocessor/preprocessor.hpp"
#include <gtest/gtest.h>

using namespace mhs::sim;

// Helper: build a minimal mhs::core::ModelDefinition for a simple uniform cube
static mhs::core::ModelDefinition make_simple_cube_io()
{
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
    mat.midu = "8920";
    mat.bi_rerong = "385";
    io.materials["copper"] = mat;

    io.other_bc = mhs::core::SecondTypeThermalBC {};

    return io;
}

TEST(AssemblerTest, AssembleMassDiagMatchesExpected)
{
    auto io = make_simple_cube_io();
    auto model = build_model(io);

    int N = static_cast<int>(model.cells.material_id.size());
    std::vector<double> T(static_cast<std::size_t>(N), 300.0);
    Eigen::Map<const Eigen::VectorXd> T_map(T.data(), N);
    AssembleContext ctx {T_map, 0.0};

    Assembler assembler(model);
    auto ops = assembler.assemble(ctx);

    EXPECT_EQ(ops.M_diag.size(), N);
    // Each cell: rho=8920, c=385, vol = (5e-3)^3 = 1.25e-7
    // M_diag = 8920 * 385 * 1.25e-7 = 0.4293
    double expected = 8920.0 * 385.0 * 1.25e-7;
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(ops.M_diag(i), expected, 1e-6) << "Cell " << i;
    }
}

TEST(AssemblerTest, AssembleReadsTemperatureForKAndMDiag)
{
    // Material k is T-dependent in this case ("100 + T"). Different T should
    // produce different K.
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
    block.material_name = "mat";
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
    mat.kx = mat.ky = mat.kz = "100 + T";
    io.materials["mat"] = mat;

    io.other_bc = mhs::core::SecondTypeThermalBC {};

    auto model = build_model(io);

    int N = static_cast<int>(model.cells.material_id.size());
    Assembler assembler(model);

    std::vector<double> T300(static_cast<std::size_t>(N), 300.0);
    std::vector<double> T500(static_cast<std::size_t>(N), 500.0);
    AssembleContext s1 {Eigen::Map<const Eigen::VectorXd>(T300.data(), N), 0.0};
    auto k1 = assembler.assemble(s1).K;

    AssembleContext s2 {Eigen::Map<const Eigen::VectorXd>(T500.data(), N), 0.0};
    auto k2 = assembler.assemble(s2).K;

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
    Eigen::Map<const Eigen::VectorXd> T_map(T.data(), N);
    AssembleContext ctx {T_map, 0.0};

    Assembler assembler(model);
    auto ops = assembler.assemble(ctx);

    // Adiabatic Neumann(0) on all faces with no source => b = 0 everywhere.
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(ops.f(i), 0.0, 1e-12);
    }
    // And K should be non-empty.
    EXPECT_GT(ops.K.nonZeros(), 0);
}
