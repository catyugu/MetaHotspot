#include "compiler/model_compiler.hpp"
#include "model/model_builder.hpp"

#include <gtest/gtest.h>

namespace {

    mhs::model::FaceRegion bottom_face() { return {mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}}; }

} // namespace

TEST(ModelBuilderTest, PreservesOrderedModelingOperations)
{
    mhs::model::ModelBuilder builder;

    mhs::model::ModelSettings settings;
    settings.length_unit = mhs::model::LengthUnit::Millimeter;
    builder.set_settings(settings);
    builder.set_mesh({{0.0, 5.0, 10.0}, {0.0, 10.0}, {0.0, 10.0}});

    builder.add_material({"substrate", {"400", "400", "400", "8920", "385", std::nullopt}});
    builder.add_material({"chip", {"130", "130", "130", "2330", "700", std::nullopt}});

    const auto layer = builder.add_layer(mhs::model::LayerParams {"10", "0", "0"});
    const auto substrate = builder.add_block(layer, mhs::model::BlockParams {"substrate", "0", "0", "0", std::nullopt});
    builder.add_rect(substrate, {mhs::model::GeometryOperation::Add, {"0", "0", "10", "10"}});
    builder.add_rect(substrate, {mhs::model::GeometryOperation::Subtract, {"5", "0", "5", "10"}});

    const auto chip = builder.add_block(layer, mhs::model::BlockParams {"chip", "1e7", "0", "0", std::nullopt});
    builder.add_rect(chip, {mhs::model::GeometryOperation::Add, {"0", "0", "5", "10"}});

    builder.add_boundary({{bottom_face()}, mhs::model::DirichletBoundary {"310"}});
    builder.add_boundary({{bottom_face()}, mhs::model::ConvectionBoundary {"42", "280"}});
    builder.set_default_boundary(mhs::model::NeumannBoundary {});

    auto definition = std::move(builder).finish();

    ASSERT_EQ(definition.layers.size(), 1u);
    ASSERT_EQ(definition.layers[0].blocks.size(), 2u);
    ASSERT_EQ(definition.layers[0].blocks[0].geometry.size(), 2u);
    EXPECT_EQ(definition.layers[0].blocks[0].geometry[0].operation, mhs::model::GeometryOperation::Add);
    EXPECT_EQ(definition.layers[0].blocks[0].geometry[1].operation, mhs::model::GeometryOperation::Subtract);
    EXPECT_EQ(definition.layers[0].blocks[1].material, "chip");
    EXPECT_EQ(definition.layers[0].blocks[1].volumetric_heat_source, "1e7");
    ASSERT_EQ(definition.boundaries.size(), 2u);
    EXPECT_TRUE(std::holds_alternative<mhs::model::ConvectionBoundary>(definition.boundaries.back().condition));
}

TEST(ModelBuilderTest, BuildsWithExistingCompilerAndKeepsLastWinsSemantics)
{
    mhs::model::ModelBuilder builder;
    mhs::model::ModelSettings settings;
    settings.length_unit = mhs::model::LengthUnit::Millimeter;
    builder.set_settings(settings);
    builder.set_mesh({{0.0, 5.0, 10.0}, {0.0, 10.0}, {0.0, 10.0}});

    builder.add_material({"background", {"400", "400", "400", "1", "1", std::nullopt}});
    builder.add_material({"foreground", {"130", "130", "130", "1", "1", std::nullopt}});

    const auto layer = builder.add_layer(mhs::model::LayerParams {"10", "0", "0"});
    const auto background
        = builder.add_block(layer, mhs::model::BlockParams {"background", "0", "0", "0", std::nullopt});
    builder.add_rect(background, {mhs::model::GeometryOperation::Add, {"0", "0", "10", "10"}});
    const auto foreground
        = builder.add_block(layer, mhs::model::BlockParams {"foreground", "1e7", "0", "0", std::nullopt});
    builder.add_rect(foreground, {mhs::model::GeometryOperation::Add, {"0", "0", "5", "10"}});
    builder.set_default_boundary(mhs::model::NeumannBoundary {});

    const auto model = mhs::sim::build_model(std::move(builder).finish());
    const auto first_cell = model.cells.grid_to_cell[0];
    ASSERT_NE(first_cell, mhs::core::invalidIndex);
    EXPECT_DOUBLE_EQ(model.material_table[model.cells.material_id[first_cell]].kx.eval({}), 130.0);
    EXPECT_DOUBLE_EQ(model.heat_source_table[model.cells.heat_source_idx[first_cell]].eval({}), 1e7);
}

TEST(ModelBuilderTest, KeepsHandlesStableAcrossInterleavedAssembly)
{
    mhs::model::ModelBuilder builder;

    const auto lower = builder.add_layer(mhs::model::LayerParams {"1", "0", "0"});
    const auto lower_block = builder.add_block(lower, mhs::model::BlockParams {"lower", "1", "0", "0", std::nullopt});
    const auto upper = builder.add_layer(mhs::model::LayerParams {"2", "0", "0"});
    const auto upper_block = builder.add_block(upper, mhs::model::BlockParams {"upper", "2", "0", "0", std::nullopt});

    builder.add_rect(lower_block, {mhs::model::GeometryOperation::Add, {"0", "0", "1", "1"}});
    builder.add_rect(upper_block, {mhs::model::GeometryOperation::Subtract, {"1", "1", "2", "2"}});

    auto definition = std::move(builder).finish();
    ASSERT_EQ(definition.layers.size(), 2u);
    ASSERT_EQ(definition.layers[0].blocks.size(), 1u);
    ASSERT_EQ(definition.layers[1].blocks.size(), 1u);
    EXPECT_EQ(definition.layers[0].blocks[0].material, "lower");
    EXPECT_EQ(definition.layers[1].blocks[0].material, "upper");
    EXPECT_EQ(definition.layers[0].blocks[0].geometry[0].operation, mhs::model::GeometryOperation::Add);
    EXPECT_EQ(definition.layers[1].blocks[0].geometry[0].operation, mhs::model::GeometryOperation::Subtract);
}
