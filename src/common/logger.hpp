#pragma once

#include <spdlog/spdlog.h> // 提供格式化推导

#include <string_view>
#include <utility>

namespace mhs::logger {

    // 初始化日志系统（仅需在程序入口调用一次）
    void init(std::string_view log_file = {}, bool console_output = true);

    // 手动刷新日志缓冲
    void flush();

    // 记录错误、刷新缓冲并退出进程 (由 MHS_LOG_ERROR 宏调用)
    [[noreturn]] void panic();

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

    template <typename... Args> void error(spdlog::format_string_t<Args...> fmt, Args&&... args)
    {
        spdlog::error(fmt, std::forward<Args>(args)...);
    }

} // namespace mhs::logger

// ---------------------------------------------------------
// 极简易用的日志宏
// ---------------------------------------------------------

#ifdef VERBOSE
#define MHS_LOG_DEBUG(...) ::mhs::logger::debug(__VA_ARGS__)
#else
#define MHS_LOG_DEBUG(...) (void)0
#endif

#define MHS_LOG_INFO(...) ::mhs::logger::info(__VA_ARGS__)
#define MHS_LOG_WARN(...) ::mhs::logger::warn(__VA_ARGS__)
#define MHS_LOG_ERROR(...) (::mhs::logger::error(__VA_ARGS__), ::mhs::logger::panic())