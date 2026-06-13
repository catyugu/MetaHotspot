#include "data/time_step_buffer.hpp"
#include <gtest/gtest.h>
#include <vector>

using mhs::core::TimeStepBuffer;

namespace {
    std::vector<double> make_T(int n, double fill = 0.0)
    {
        std::vector<double> v(static_cast<std::size_t>(n), fill);
        for (int i = 0; i < n; ++i)
            v[static_cast<std::size_t>(i)] = fill + i;
        return v;
    }
}

TEST(TimeStepBufferTest, EmptyBufferHasSizeZero)
{
    TimeStepBuffer buf(3, 3);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.capacity(), 3u);
}

TEST(TimeStepBufferTest, PushThenLatest)
{
    TimeStepBuffer buf(3, 3);
    auto T = make_T(3, 100.0);
    buf.push(T, 0.1);
    EXPECT_EQ(buf.size(), 1u);
    EXPECT_EQ(buf.latest().size(), 3u);
    EXPECT_DOUBLE_EQ(buf.latest()[0], 100.0);
    EXPECT_DOUBLE_EQ(buf.latest()[1], 101.0);
    EXPECT_DOUBLE_EQ(buf.latest()[2], 102.0);
}

TEST(TimeStepBufferTest, AtRelative)
{
    TimeStepBuffer buf(3, 3);
    buf.push(make_T(3, 100.0), 0.0); // T_1
    buf.push(make_T(3, 200.0), 0.1); // T_2
    EXPECT_EQ(buf.size(), 2u);
    EXPECT_DOUBLE_EQ(buf.at(0)[0], 200.0);
    EXPECT_DOUBLE_EQ(buf.at(0)[1], 201.0);
    EXPECT_DOUBLE_EQ(buf.at(1)[0], 100.0);
    EXPECT_DOUBLE_EQ(buf.at(1)[1], 101.0);
}

TEST(TimeStepBufferTest, WrapAround)
{
    TimeStepBuffer buf(2, 3);
    for (int i = 0; i < 5; ++i) {
        buf.push(make_T(2, static_cast<double>(i * 10)), static_cast<double>(i));
    }
    EXPECT_EQ(buf.size(), 3u);
    // last 3 pushes: i=2 (T=20,30, t=2), i=3 (30,40, t=3), i=4 (40,50, t=4)
    EXPECT_DOUBLE_EQ(buf.at(0)[0], 40.0); // latest = i=4
    EXPECT_DOUBLE_EQ(buf.at(1)[0], 30.0); // i=3
    EXPECT_DOUBLE_EQ(buf.at(2)[0], 20.0); // i=2
    EXPECT_DOUBLE_EQ(buf.time_at(0), 4.0);
    EXPECT_DOUBLE_EQ(buf.time_at(1), 3.0);
    EXPECT_DOUBLE_EQ(buf.time_at(2), 2.0);
}

TEST(TimeStepBufferTest, TimeAtAndDtTo)
{
    TimeStepBuffer buf(1, 3);
    buf.push({1.0}, 0.5);
    buf.push({2.0}, 1.0);
    buf.push({3.0}, 2.5);
    EXPECT_DOUBLE_EQ(buf.time_at(0), 2.5);
    EXPECT_DOUBLE_EQ(buf.time_at(1), 1.0);
    EXPECT_DOUBLE_EQ(buf.time_at(2), 0.5);
    EXPECT_DOUBLE_EQ(buf.dt_to(1), 1.5);
    EXPECT_DOUBLE_EQ(buf.dt_to(2), 2.0);
}

TEST(TimeStepBufferTest, Reset)
{
    TimeStepBuffer buf(3, 3);
    buf.push(make_T(3, 100.0), 0.1);
    buf.push(make_T(3, 200.0), 0.2);
    EXPECT_EQ(buf.size(), 2u);

    auto T0 = make_T(3, 50.0);
    buf.reset(T0);
    EXPECT_EQ(buf.size(), 1u);
    EXPECT_DOUBLE_EQ(buf.latest()[0], 50.0);
    EXPECT_DOUBLE_EQ(buf.latest()[1], 51.0);
    EXPECT_DOUBLE_EQ(buf.latest()[2], 52.0);
    EXPECT_DOUBLE_EQ(buf.time_at(0), 0.0);
}

TEST(TimeStepBufferTest, LatestAndAtReturnRefForMultipleSlots)
{
    TimeStepBuffer buf(2, 3);
    buf.push({1.0, 2.0}, 0.0);
    buf.push({3.0, 4.0}, 0.1);
    // Verify the reference points to the right vector.
    const auto& a = buf.at(1);
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    EXPECT_DOUBLE_EQ(a[1], 2.0);
}
