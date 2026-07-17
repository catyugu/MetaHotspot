#include "logger/logger.hpp"
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace {

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

} // namespace
