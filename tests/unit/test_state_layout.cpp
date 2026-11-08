#include "model/state_layout.hpp"
#include <gtest/gtest.h>

TEST(StateLayout, UsesCanonicalContiguousFieldRanges)
{
    using namespace mhs::model;
    StateLayout layout;
    layout.state_count = 8;
    layout.fields = {
        {0, FieldQuantity::Temperature, 0, 2},
        {0, FieldQuantity::Pressure, 2, 2},
        {0, FieldQuantity::Velocity, 4, 3},
        {1, FieldQuantity::ReducedCoordinate, 7, 1},
    };
    EXPECT_EQ(layout.find(0, FieldQuantity::Pressure), 1u);
    EXPECT_EQ(layout.find(0, FieldQuantity::Velocity), 2u);
    EXPECT_EQ(layout.field(0).dof(1), 1u);
    EXPECT_EQ(layout.field(2).dof(2), 6u);
    EXPECT_EQ(layout.field(2).support(), EntitySupport::Face);
    EXPECT_EQ(layout.field(3).support(), EntitySupport::Reduced);
    EXPECT_THROW(layout.find(2, FieldQuantity::Temperature), std::out_of_range);
    EXPECT_THROW(layout.field(99), std::out_of_range);
}
