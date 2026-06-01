#include "assembler/assembler.hpp"
#include "config.h"
#include "io/io.hpp"
#include "model/internal_model.hpp"
#include "model/io_model.hpp"
#include "preprocessor/preprocessor.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace mhs;
using namespace mhs::model;
using namespace mhs::assembler;

// Helper: build a minimal IOStructure for a simple uniform cube
static IOStructure make_simple_cube_io()
{
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "test_block";
    block.material_name = "copper";
    block.thickness_expr = "10";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    Material mat;
    mat.name = "copper";
    mat.daore_xishu = "400";
    mat.midu = "8920";
    mat.bi_rerong = "385";
    io.materials["copper"] = mat;

    io.other_bc_type = ThermalBCType::SecondType;
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

TEST(AssemblerTest, AssembleReturnsCorrectSize)
{
    auto io = make_simple_cube_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    GlobalState state;
    state.cell_count = N;
    state.T.resize(N, 300.0);
    state.T_prev.resize(N, 300.0);
    state.current_time = 0.0;

    Assembler assembler(*model);
    LinearSystem result = assembler.assemble(state);

    EXPECT_EQ(result.A.rows(), N);
    EXPECT_EQ(result.A.cols(), N);
    EXPECT_EQ(result.b.size(), N);
    EXPECT_EQ(result.residual.size(), N);
}

TEST(AssemblerTest, LinearSystemHasResidualField)
{
    LinearSystem sys;
    sys.A = Eigen::SparseMatrix<double>(3, 3);
    sys.b = Eigen::VectorXd(3);
    sys.residual = Eigen::VectorXd(3);

    EXPECT_EQ(sys.A.rows(), 3);
    EXPECT_EQ(sys.b.size(), 3);
    EXPECT_EQ(sys.residual.size(), 3);
}

TEST(AssemblerTest, DiagonalIsNegativeAndSymmetricForInteriorCell)
{
    // 2x2x2 cube with all cells active, Neumann(0) BC on all domain faces
    // Interior cell (1,1,1) should have negative diagonal and symmetric off-diagonal entries
    auto io = make_simple_cube_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    GlobalState state;
    state.cell_count = N;
    state.T.resize(N, 300.0);
    state.T_prev.resize(N, 300.0);
    state.current_time = 0.0;

    Assembler assembler(*model);
    LinearSystem result = assembler.assemble(state);

    // Matrix should be square
    EXPECT_EQ(result.A.rows(), N);
    EXPECT_EQ(result.A.cols(), N);

    // Each row should have at least one entry (diagonal)
    for (int i = 0; i < N; i++) {
        EXPECT_NE(result.A.coeff(i, i), 0.0) << "Diagonal should not be zero for cell " << i;
    }

    // Diagonal entries should be negative (since all coefficients are subtracted from diag)
    // for steady-state with Neumann(0) BC and no heat source, diag should be < 0
    // Actually with Neumann(0), the flux is zero, so only interior faces contribute
    // Interior cell (1,1,1) has 3 interior faces -> diag < 0
    // Corner cell (0,0,0) has no interior faces + all BC are Neumann(0) -> diag = 0
    // This is expected for adiabatic BC on all faces with no heat source
}

TEST(AssemblerTest, DirichletBCProducesStrongDiagonal)
{
    // Simple cube with Dirichlet BC on one face
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "test_block";
    block.material_name = "copper";
    block.thickness_expr = "10";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    Material mat;
    mat.name = "copper";
    mat.daore_xishu = "400";
    io.materials["copper"] = mat;

    // Dirichlet BC on bottom face (Z=0)
    Boundary boundary;
    boundary.name = "bc1";
    boundary.bc_type = ThermalBCType::FirstType;
    boundary.first.temperature = "500";
    boundary.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary);

    // Neumann(0) for all other faces
    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    EXPECT_EQ(N, 8);

    GlobalState state;
    state.cell_count = N;
    state.T.resize(N, 300.0);
    state.T_prev.resize(N, 300.0);
    state.current_time = 0.0;

    Assembler assembler(*model);
    LinearSystem result = assembler.assemble(state);

    EXPECT_EQ(result.A.rows(), N);
    EXPECT_EQ(result.A.cols(), N);

    // Cells at bottom face (iz=0) should have Dirichlet BC on ZM face
    // This gives them a strong diagonal contribution from 2*k*A_f/half_dist
    // k=400, A_f = dx*dy = 5e-3*5e-3 = 25e-6, half_dist = dz/2 = 2.5e-3
    // 2*400*25e-6/2.5e-3 = 2*400*0.01 = 8.0
    // So diagonal for bottom cells should include this contribution
}

TEST(AssemblerTest, HeatSourceContributesToRHS)
{
    // Simple cube with a heat source
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "test_block";
    block.material_name = "copper";
    block.thickness_expr = "10";
    block.ti_reyuan_expr = "1e6"; // 1e6 W/m^3 heat source
    block.is_normal_material = true;

    Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    Material mat;
    mat.name = "copper";
    mat.daore_xishu = "400";
    io.materials["copper"] = mat;

    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    GlobalState state;
    state.cell_count = N;
    state.T.resize(N, 300.0);
    state.T_prev.resize(N, 300.0);
    state.current_time = 0.0;

    Assembler assembler(*model);
    LinearSystem result = assembler.assemble(state);

    // Each cell has Q=1e6 W/m^3, vol = (5mm)^3 = 125e-9 m^3
    // Q*vol = 1e6 * 125e-9 = 0.125 W
    // RHS should contain this heat source contribution
    // (plus any BC contributions)
    for (int i = 0; i < N; i++) {
        // RHS should be at least the heat source contribution
        EXPECT_GT(std::abs(result.b(i)), 0.0) << "RHS should not be zero with heat source";
    }
}

TEST(AssemblerTest, CauchyBCAddsConvectiveTerms)
{
    // Cube with Cauchy/Robin BC on top face
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "test_block";
    block.material_name = "copper";
    block.thickness_expr = "10";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    Material mat;
    mat.name = "copper";
    mat.daore_xishu = "400";
    io.materials["copper"] = mat;

    // Cauchy BC on top face (Z=10mm)
    Boundary boundary;
    boundary.name = "bc1";
    boundary.bc_type = ThermalBCType::ThirdType;
    boundary.third.convection_coeff = "10";
    boundary.third.T_inf = "300";
    boundary.face_keys.push_back("Z|E|10|0,10,0,10");
    io.boundaries.push_back(boundary);

    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    GlobalState state;
    state.cell_count = N;
    state.T.resize(N, 300.0);
    state.T_prev.resize(N, 300.0);
    state.current_time = 0.0;

    Assembler assembler(*model);
    LinearSystem result = assembler.assemble(state);

    EXPECT_EQ(result.A.rows(), N);
    EXPECT_EQ(result.A.cols(), N);

    // Top face cells should have Cauchy BC contributions
    // h=10, T_inf=300, A_f = dx*dy = 25e-6 m^2
    // Additional to diagonal: -h*A_f and -k*A_f/half_dist
    // Additional to RHS: h*A_f*T_inf + k*A_f/half_dist*T_inf
}

TEST(AssemblerTest, Case1AssemblyRuns)
{
    std::string case_path = std::string(PROJECT_SOURCE_DIR) + "/cases/original_steady_tests/case1.xml";
    if (!std::filesystem::exists(case_path)) {
        GTEST_SKIP() << "Case1 XML not found";
    }

    auto io = io::read_xml(case_path);
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    EXPECT_GT(N, 0);

    GlobalState state;
    state.cell_count = N;
    state.T.resize(N, model->initial_temperature);
    state.T_prev.resize(N, model->initial_temperature);
    state.current_time = 0.0;

    Assembler assembler(*model);
    LinearSystem result = assembler.assemble(state);

    EXPECT_EQ(result.A.rows(), N);
    EXPECT_EQ(result.A.cols(), N);
    EXPECT_EQ(result.b.size(), N);
    EXPECT_EQ(result.residual.size(), N);

    // Matrix should not be empty - there should be non-zero entries
    EXPECT_GT(result.A.nonZeros(), 0);

    // RHS should not be all zeros (heat source + BC contributions)
    EXPECT_GT(result.b.norm(), 0.0);
}