#include "common/logger.hpp"
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace {

    TEST(LoggerInit, ConsoleOutputEnabled)
    {
        EXPECT_NO_THROW(mhs::logger::init("", true));
        EXPECT_NO_THROW(MHS_LOG_INFO("Test console init"));
    }

    TEST(LoggerInit, FileOutputCreated)
    {
        const std::string test_log = "test_output.log";
        std::remove(test_log.c_str());

        mhs::logger::init(test_log, true);
        MHS_LOG_INFO("Log message to file");

        // 强制刷新以确保立即可读
        mhs::logger::flush();

        std::ifstream file(test_log);
        ASSERT_TRUE(file.is_open()) << "Log file should exist";

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("Log message to file") != std::string::npos);

        file.close();
        std::remove(test_log.c_str());
    }

    TEST(LoggerInit, ConsoleOnlyNoCrash)
    {
        EXPECT_NO_THROW(mhs::logger::init("", false));
        EXPECT_NO_THROW(MHS_LOG_INFO("Console only mode"));
    }

    TEST(LoggerAPI, CanLogAllLevels)
    {
        mhs::logger::init("", true);
        // 使用宏直接测试行为，确保没有编译期错误且不抛异常
        EXPECT_NO_THROW(MHS_LOG_DEBUG("Debug level test"));
        EXPECT_NO_THROW(MHS_LOG_INFO("Info level test"));
        EXPECT_NO_THROW(MHS_LOG_WARN("Warn level test"));
    }

    TEST(LoggerAPI, FormattedMessages)
    {
        mhs::logger::init("", true);
        EXPECT_NO_THROW(MHS_LOG_INFO("Value: {}, String: {}", 42, "test"));
        EXPECT_NO_THROW(MHS_LOG_INFO("Multi: {} {} {}", 1, 2, 3));
    }

    TEST(LoggerPanic, PanicExits)
    {
        mhs::logger::init("", true);
        EXPECT_DEATH(mhs::logger::panic(), "");
    }

    TEST(LoggerPanic, ErrorThenPanic)
    {
        mhs::logger::init("", true);
        EXPECT_DEATH(MHS_FATAL("Fatal error test"), "");
    }

} // namespace