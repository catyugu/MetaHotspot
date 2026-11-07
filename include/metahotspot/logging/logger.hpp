#pragma once

#include <spdlog/spdlog.h>

#include <string_view>

namespace mhs::logger {
    using Level = spdlog::level::level_enum;

    // Call init only at startup, before concurrent logging.
    // Reads MHS_LOG_LEVEL once; SDK defaults to WARN and CLI passes INFO.
    void init(std::string_view log_file = {}, bool console_output = true, Level default_level = Level::warn);
    void set_log_level(Level level);
    [[nodiscard]] Level log_level();
    [[nodiscard]] bool enabled(Level level);
    void flush();

    namespace detail {
        // A MetaHotspot-owned logger: never change the host's spdlog registry.
        spdlog::logger& instance(Level default_level = Level::warn);
    }
} // namespace mhs::logger

#ifndef MHS_LOG_COMPILED_LEVEL
#define MHS_LOG_COMPILED_LEVEL 1 // DEBUG for consumers without the CMake target
#endif

// The runtime check prevents evaluation of disabled log arguments.
// clang-format off
#define MHS_LOG_IMPL(level, ...) \
    do { \
        auto& mhs_log_instance = ::mhs::logger::detail::instance(); \
        if (mhs_log_instance.should_log(level)) \
            mhs_log_instance.log(level, __VA_ARGS__); \
    } while (false)

#if MHS_LOG_COMPILED_LEVEL <= 0
#define MHS_LOG_TRACE(...) MHS_LOG_IMPL(::mhs::logger::Level::trace, __VA_ARGS__)
#else
#define MHS_LOG_TRACE(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 1
#define MHS_LOG_DEBUG(...) MHS_LOG_IMPL(::mhs::logger::Level::debug, __VA_ARGS__)
#else
#define MHS_LOG_DEBUG(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 2
#define MHS_LOG_INFO(...) MHS_LOG_IMPL(::mhs::logger::Level::info, __VA_ARGS__)
#else
#define MHS_LOG_INFO(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 3
#define MHS_LOG_WARN(...) MHS_LOG_IMPL(::mhs::logger::Level::warn, __VA_ARGS__)
#else
#define MHS_LOG_WARN(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 4
#define MHS_LOG_ERROR(...) MHS_LOG_IMPL(::mhs::logger::Level::err, __VA_ARGS__)
#else
#define MHS_LOG_ERROR(...) ((void)0)
#endif

#if MHS_LOG_COMPILED_LEVEL <= 5
#define MHS_LOG_CRITICAL(...) MHS_LOG_IMPL(::mhs::logger::Level::critical, __VA_ARGS__)
#else
#define MHS_LOG_CRITICAL(...) ((void)0)
#endif
// clang-format on
