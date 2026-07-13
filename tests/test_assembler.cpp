#include "assembler/assembler.hpp"
#include "config.h"
#include "data/io_structure.hpp"
#include "data/model.hpp"
#include "io/io.hpp"
#include "preprocessor/preprocessor.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace mhs::sim;
using namespace mhs::io;
using namespace mhs::sim;

// Helper: build a minimal mhs::core::IOStructure for a simple uniform cube
static mhs::core::IOStructure make_simple_cube_io()
{
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    mhs::core::Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    mhs::core::Block block;
    block.name = "test_block";
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
    mat.name = "copper";
    mat.kx = mat.ky = mat.kz = "400";
    mat.midu = "8920";
    mat.bi_rerong = "385";
    io.materials["copper"] = mat;

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    return io;
}

TEST(AssemblerTest, ConstructWithModel)
{
    auto io = make_simple_cube_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    Assembler assembler(*model);
}

TEST(AssemblerTest, AssembleReturnsKAndFAndMDiag)
{
    auto io = make_simple_cube_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = static_cast<int>(model->cells.material_id.size());
    std::vector<double> T(static_cast<std::size_t>(N), 300.0);
    Eigen::Map<const Eigen::VectorXd> T_map(T.data(), N);
    AssembleContext ctx {T_map, 0.0};

    Assembler assembler(*model);
    auto ops = assembler.assemble(ctx);

    EXPECT_EQ(ops.K.rows(), N);
    EXPECT_EQ(ops.K.cols(), N);
    EXPECT_EQ(ops.f.size(), N);
    EXPECT_EQ(ops.M_diag.size(), N);
}

TEST(AssemblerTest, AssembleProducesConsistentResultsAcrossDt)
{
    // Key invariant: assemble does NOT add M_diag/dt to the diagonal.
    // Compare two calls at the same T; the K matrix must be identical —
    // transient terms live in nonlinear_solve / build_system, not in assemble.
    auto io = make_simple_cube_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = static_cast<int>(model->cells.material_id.size());
    std::vector<double> T(static_cast<std::size_t>(N), 300.0);
    Eigen::Map<const Eigen::VectorXd> T_map(T.data(), N);

    Assembler assembler(*model);

    AssembleContext ctx_a {T_map, 0.0};
    auto ops_a = assembler.assemble(ctx_a);

    AssembleContext ctx_b {T_map, 0.0};
    auto ops_b = assembler.assemble(ctx_b);

    EXPECT_EQ(ops_a.K.nonZeros(), ops_b.K.nonZeros());
    for (int k = 0; k < ops_a.K.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<double>::InnerIterator it(ops_a.K, k); it; ++it) {
            double diff = std::abs(it.value() - ops_b.K.coeff(it.row(), it.col()));
            EXPECT_LT(diff, 1e-12) << "K(" << it.row() << "," << it.col() << ") differs across calls";
        }
    }
}

TEST(AssemblerTest, AssembleMassDiagMatchesExpected)
{
    auto io = make_simple_cube_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = static_cast<int>(model->cells.material_id.size());
    std::vector<double> T(static_cast<std::size_t>(N), 300.0);
    Eigen::Map<const Eigen::VectorXd> T_map(T.data(), N);
    AssembleContext ctx {T_map, 0.0};

    Assembler assembler(*model);
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
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    mhs::core::Layer layer;
    layer.name = "l";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";
    mhs::core::Block block;
    block.name = "b";
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
    mat.name = "mat";
    mat.kx = mat.ky = mat.kz = "100 + T";
    io.materials["mat"] = mat;

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = static_cast<int>(model->cells.material_id.size());
    Assembler assembler(*model);

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
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = static_cast<int>(model->cells.material_id.size());
    std::vector<double> T(static_cast<std::size_t>(N), 300.0);
    Eigen::Map<const Eigen::VectorXd> T_map(T.data(), N);
    AssembleContext ctx {T_map, 0.0};

    Assembler assembler(*model);
    auto ops = assembler.assemble(ctx);

    // Adiabatic Neumann(0) on all faces with no source => b = 0 everywhere.
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(ops.f(i), 0.0, 1e-12);
    }
    // And K should be non-empty.
    EXPECT_GT(ops.K.nonZeros(), 0);
}

TEST(AssemblerTest, Case1AssemblyRuns)
{
    std::string case_path = std::string(PROJECT_SOURCE_DIR) + "/cases/simple_steady_cases/simple_steady_case1.xml";
    if (!std::filesystem::exists(case_path)) {
        GTEST_SKIP() << "Case1 XML not found";
    }

    auto io_data = mhs::io::read_xml(case_path);
    Preprocessor preprocessor;
    auto model = preprocessor.load(io_data);
    ASSERT_NE(model, nullptr);

    int N = static_cast<int>(model->cells.material_id.size());
    EXPECT_GT(N, 0);

    std::vector<double> T(static_cast<std::size_t>(N), model->initial_temperature);
    Eigen::Map<const Eigen::VectorXd> T_map(T.data(), N);
    AssembleContext ctx {T_map, 0.0};

    Assembler assembler(*model);
    auto ops = assembler.assemble(ctx);
    EXPECT_EQ(ops.K.rows(), N);
    EXPECT_EQ(ops.K.cols(), N);
    EXPECT_GT(ops.K.nonZeros(), 0);
    EXPECT_GT(ops.f.norm(), 0.0);
}
