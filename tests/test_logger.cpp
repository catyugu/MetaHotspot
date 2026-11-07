#include "logging/logger.hpp"
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace {

    TEST(LoggerInit, FileOutputCreated)
    {
#if MHS_LOG_COMPILED_LEVEL >= 6
        GTEST_SKIP() << "All log messages are compiled out";
#else
        const std::string test_log = "test_output.log";
        std::remove(test_log.c_str());

        mhs::logger::init(test_log, false);
        mhs::logger::set_log_level(mhs::logger::Level::critical);
        MHS_LOG_CRITICAL("Log message to file");
        mhs::logger::flush();

        std::ifstream file(test_log);
        ASSERT_TRUE(file.is_open()) << "Log file should exist";

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("Log message to file") != std::string::npos);

        file.close();
        std::remove(test_log.c_str());
#endif
    }

} // namespace
