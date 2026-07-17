#pragma once

#include <spdlog/spdlog.h> // 提供格式化推导

#include <string_view>
#include <utility>

namespace mhs::logger {

    // 初始化日志系统（仅需在程序入口调用一次）
    void init(std::string_view log_file = {}, bool console_output = true);

    // 手动刷新日志缓冲
    void flush();

    // ---------------------------------------------------------
    // 内部转发接口 (用户通常应通过下方的宏来调用以实现条件编译)
    // ---------------------------------------------------------

    template <typename... Args> void debug(spdlog::format_string_t<Args...> fmt, Args&&... args)
    {
        spdlog::debug(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args> void info(spdlog::format_string_t<Args...> fmt, Args&&... args)
    {
        spdlog::info(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args> void warn(spdlog::format_string_t<Args...> fmt, Args&&... args)
    {
        spdlog::warn(fmt, std::forward<Args>(args)...);
    }

} // namespace mhs::logger

// ---------------------------------------------------------
// 极简易用的日志宏
// ---------------------------------------------------------

#define MHS_LOG_DEBUG(...) ::mhs::logger::debug(__VA_ARGS__)
#define MHS_LOG_INFO(...) ::mhs::logger::info(__VA_ARGS__)
#define MHS_LOG_WARN(...) ::mhs::logger::warn(__VA_ARGS__)