#include "model/state_layout.hpp"
#include <gtest/gtest.h>

TEST(StateLayout, SeparatesSupportEntitiesFromGlobalDofs)
{
    using namespace mhs::model;
    StateLayout layout {6,
        {{0, FieldQuantity::Temperature, EntitySupport::Cell, 0, {7, 2}, {4, 0}},
            {0, FieldQuantity::Pressure, EntitySupport::Cell, 0, {2}, {3}},
            {0, FieldQuantity::Velocity, EntitySupport::Face, 1, {11}, {1}},
            {1, FieldQuantity::ReducedCoordinate, EntitySupport::Reduced, 0, {0, 1}, {5, 2}}}};
    EXPECT_EQ(layout.find(0, FieldQuantity::Pressure), 1u);
    EXPECT_EQ(layout.find(0, FieldQuantity::Velocity, 1), 2u);
    EXPECT_EQ(layout.field(0).dofs[0], 4u);
    EXPECT_EQ(layout.field(0).entities[0], 7u);
    EXPECT_EQ(layout.field(3).domain, 1u);
    EXPECT_THROW(layout.find(2, FieldQuantity::Temperature), std::out_of_range);
    EXPECT_THROW(layout.field(99), std::out_of_range);
}
