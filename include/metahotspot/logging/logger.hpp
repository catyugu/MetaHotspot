#pragma once

#include <spdlog/spdlog.h>

#include <functional>
#include <string_view>
#include <utility>

namespace mhs::logger {

    enum class LogLevel : int { trace, debug, info, warn, error, critical, off };

    // The callback receives the formatted message (without the sink's timestamp
    // or prefix). The message view is valid only during the callback.
    using LogCallback = std::function<void(LogLevel, std::string_view)>;

    // Configure output sinks. SDK defaults to WARN; the CLI explicitly chooses INFO.
    // MHS_LOG_LEVEL overrides the default, and set_level() overrides both.
    void init(std::string_view log_file = {}, bool console_output = true, LogLevel default_level = LogLevel::warn);
    void set_level(LogLevel level);
    [[nodiscard]] LogLevel get_level();
    [[nodiscard]] bool enabled(LogLevel level);
    void set_callback(LogCallback callback);
    void flush();

    template <typename... Args> void trace(spdlog::format_string_t<Args...> fmt, Args&&... args)
    { spdlog::trace(fmt, std::forward<Args>(args)...); }

    template <typename... Args> void debug(spdlog::format_string_t<Args...> fmt, Args&&... args)
    { spdlog::debug(fmt, std::forward<Args>(args)...); }

    template <typename... Args> void info(spdlog::format_string_t<Args...> fmt, Args&&... args)
    { spdlog::info(fmt, std::forward<Args>(args)...); }

    template <typename... Args> void warn(spdlog::format_string_t<Args...> fmt, Args&&... args)
    { spdlog::warn(fmt, std::forward<Args>(args)...); }

    template <typename... Args> void error(spdlog::format_string_t<Args...> fmt, Args&&... args)
    { spdlog::error(fmt, std::forward<Args>(args)...); }

    template <typename... Args> void critical(spdlog::format_string_t<Args...> fmt, Args&&... args)
    { spdlog::critical(fmt, std::forward<Args>(args)...); }

} // namespace mhs::logger

// Numeric levels match spdlog's ordering. CMake publishes the compile-time
// threshold to SDK consumers via mhs_logging's PUBLIC compile definitions.
#define MHS_LOG_LEVEL_TRACE 0
#define MHS_LOG_LEVEL_DEBUG 1
#define MHS_LOG_LEVEL_INFO 2
#define MHS_LOG_LEVEL_WARN 3
#define MHS_LOG_LEVEL_ERROR 4
#define MHS_LOG_LEVEL_CRITICAL 5
#define MHS_LOG_LEVEL_OFF 6

#ifndef MHS_LOG_COMPILED_LEVEL
#define MHS_LOG_COMPILED_LEVEL MHS_LOG_LEVEL_DEBUG
#endif

#if MHS_LOG_COMPILED_LEVEL < MHS_LOG_LEVEL_TRACE || MHS_LOG_COMPILED_LEVEL > MHS_LOG_LEVEL_OFF
#error "MHS_LOG_COMPILED_LEVEL must be between TRACE (0) and OFF (6)"
#endif

// The runtime check precedes argument evaluation; disabled messages incur no
// formatting or argument side effects. Compile-time exclusions remove the
// entire call, including its argument expressions.
#define MHS_LOG_IMPL(level, ...)                                                                                      \
    do {                                                                                                              \
        if (::mhs::logger::enabled(::mhs::logger::LogLevel::level))                                                    \
            ::mhs::logger::level(__VA_ARGS__);                                                                         \
    } while (false)

#if MHS_LOG_COMPILED_LEVEL <= MHS_LOG_LEVEL_TRACE
#define MHS_LOG_TRACE(...) MHS_LOG_IMPL(trace, __VA_ARGS__)
#else
#define MHS_LOG_TRACE(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= MHS_LOG_LEVEL_DEBUG
#define MHS_LOG_DEBUG(...) MHS_LOG_IMPL(debug, __VA_ARGS__)
#else
#define MHS_LOG_DEBUG(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= MHS_LOG_LEVEL_INFO
#define MHS_LOG_INFO(...) MHS_LOG_IMPL(info, __VA_ARGS__)
#else
#define MHS_LOG_INFO(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= MHS_LOG_LEVEL_WARN
#define MHS_LOG_WARN(...) MHS_LOG_IMPL(warn, __VA_ARGS__)
#else
#define MHS_LOG_WARN(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= MHS_LOG_LEVEL_ERROR
#define MHS_LOG_ERROR(...) MHS_LOG_IMPL(error, __VA_ARGS__)
#else
#define MHS_LOG_ERROR(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= MHS_LOG_LEVEL_CRITICAL
#define MHS_LOG_CRITICAL(...) MHS_LOG_IMPL(critical, __VA_ARGS__)
#else
#define MHS_LOG_CRITICAL(...) ((void)0)
#endif
