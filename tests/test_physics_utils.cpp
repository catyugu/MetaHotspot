#include "common/physics_utils.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mhs::utils;

TEST(PhysicsUtilsTest, NusseltSquareDuct)
{
    // Square duct: AR = 1.0 → Nu ≈ 3.61 (Shah & London)
    double Nu = nusselt_rectangular(1.0, 1.0);
    EXPECT_NEAR(Nu, 3.610224, 1e-4);
}

TEST(PhysicsUtilsTest, NusseltVeryNarrow)
{
    // Very narrow channel: AR → 0 → Nu → 8.235 (infinite parallel-plate limit)
    double Nu = nusselt_rectangular(1e-6, 1.0);
    EXPECT_NEAR(Nu, 8.235, 1e-2);
}

TEST(PhysicsUtilsTest, NusseltTypicalAspect)
{
    // Typical AR = 0.5 → Nu ≈ 4.126 (from the polynomial)
    double Nu = nusselt_rectangular(0.5, 1.0);
    EXPECT_NEAR(Nu, 4.125812203125, 1e-4);
    // Sanity: Nu must be positive and between narrow and square limits
    EXPECT_GT(Nu, 0.0);
    EXPECT_LT(Nu, 8.235);
}

TEST(PhysicsUtilsTest, NusseltSymmetric)
{
    // nusselt_rectangular(w, h) == nusselt_rectangular(h, w)
    double Nu1 = nusselt_rectangular(0.3, 0.6);
    double Nu2 = nusselt_rectangular(0.6, 0.3);
    EXPECT_NEAR(Nu1, Nu2, 1e-12);
}
