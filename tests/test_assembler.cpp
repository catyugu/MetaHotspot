#include "assembler/assembler.hpp"
#include "model/internal_model.hpp"
#include <gtest/gtest.h>

using namespace mhs;
using namespace mhs::model;
using namespace mhs::assembler;

TEST(AssemblerTest, ConstructWithModel)
{
    InternalModel model;
    model.mesh.nx = 2;
    model.mesh.ny = 2;
    model.mesh.nz = 2;
    model.mesh.total_cell_count = 8;
    model.cells.cell_count = 8;

    Assembler assembler(model);
    // Construction succeeded — model_ is bound
}

TEST(AssemblerTest, AssembleReturnsLinearSystem)
{
    InternalModel model;
    model.mesh.nx = 2;
    model.mesh.ny = 2;
    model.mesh.nz = 2;
    model.mesh.total_cell_count = 8;
    model.cells.cell_count = 8;

    GlobalState state;
    state.cell_count = 8;
    state.T.resize(8, 300.0);
    state.T_prev.resize(8, 300.0);

    Assembler assembler(model);
    LinearSystem result = assembler.assemble(state);

    // Stub returns empty LinearSystem
    EXPECT_EQ(result.A.rows(), 0);
    EXPECT_EQ(result.A.cols(), 0);
    EXPECT_EQ(result.b.size(), 0);
    EXPECT_EQ(result.residual.size(), 0);
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