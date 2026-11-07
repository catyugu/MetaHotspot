#pragma once

#include <spdlog/spdlog.h>

#include <functional>
#include <string_view>
#include <utility>

namespace mhs::logger {

    using Level = spdlog::level::level_enum;
    using LogCallback = std::function<void(Level, std::string_view)>;

    // Reads MHS_LOG_LEVEL once, on first use. SDK default is WARN.
    // The CLI passes INFO as its default; the environment overrides either.
    void init(std::string_view log_file = {}, bool console_output = true, Level default_level = Level::warn);
    void set_log_level(Level level);
    [[nodiscard]] Level log_level();
    [[nodiscard]] bool enabled(Level level);
    // An empty callback removes the current callback. The callback must not throw.
    void set_log_callback(LogCallback callback);
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

// Propagated by the mhs::logging CMake target to downstream translation units.
#ifndef MHS_LOG_COMPILED_LEVEL
#define MHS_LOG_COMPILED_LEVEL 1 // DEBUG for direct header users without CMake
#endif

// The level check precedes the evaluation of formatting arguments.
// clang-format off
#if MHS_LOG_COMPILED_LEVEL <= 0
#define MHS_LOG_TRACE(...) do { if (::mhs::logger::enabled(::mhs::logger::Level::trace)) ::mhs::logger::trace(__VA_ARGS__); } while (false)
#else
#define MHS_LOG_TRACE(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 1
#define MHS_LOG_DEBUG(...) do { if (::mhs::logger::enabled(::mhs::logger::Level::debug)) ::mhs::logger::debug(__VA_ARGS__); } while (false)
#else
#define MHS_LOG_DEBUG(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 2
#define MHS_LOG_INFO(...) do { if (::mhs::logger::enabled(::mhs::logger::Level::info)) ::mhs::logger::info(__VA_ARGS__); } while (false)
#else
#define MHS_LOG_INFO(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 3
#define MHS_LOG_WARN(...) do { if (::mhs::logger::enabled(::mhs::logger::Level::warn)) ::mhs::logger::warn(__VA_ARGS__); } while (false)
#else
#define MHS_LOG_WARN(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 4
#define MHS_LOG_ERROR(...) do { if (::mhs::logger::enabled(::mhs::logger::Level::err)) ::mhs::logger::error(__VA_ARGS__); } while (false)
#else
#define MHS_LOG_ERROR(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 5
#define MHS_LOG_CRITICAL(...) do { if (::mhs::logger::enabled(::mhs::logger::Level::critical)) ::mhs::logger::critical(__VA_ARGS__); } while (false)
#else
#define MHS_LOG_CRITICAL(...) ((void)0)
#endif
// clang-format on
