#include "logging/logger.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mhs::logger {
    namespace {
        std::once_flag level_once;

        Level configured_level(Level fallback)
        {
            std::string value;
#ifdef _WIN32
            char* raw = nullptr;
            size_t length = 0;
            if (_dupenv_s(&raw, &length, "MHS_LOG_LEVEL") == 0 && raw) {
                value = raw;
                std::free(raw);
            }
#else
            if (const char* raw = std::getenv("MHS_LOG_LEVEL"))
                value = raw;
#endif
            std::transform(value.begin(), value.end(), value.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            if (value == "trace")
                return Level::trace;
            if (value == "debug")
                return Level::debug;
            if (value == "info")
                return Level::info;
            if (value == "warn" || value == "warning")
                return Level::warn;
            if (value == "error")
                return Level::err;
            if (value == "critical")
                return Level::critical;
            if (value == "off")
                return Level::off;
            return fallback;
        }
    } // namespace

    namespace detail {
        spdlog::logger& instance(Level default_level)
        {
            static spdlog::logger log("mhs", std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
            std::call_once(level_once, [&] {
                log.set_level(configured_level(default_level));
                log.flush_on(Level::warn);
            });
            return log;
        }
    } // namespace detail

    void init(std::string_view log_file, bool console_output, Level default_level)
    {
        // Configure sinks at startup, before other threads emit logs.
        auto& log = detail::instance(default_level);
        std::vector<spdlog::sink_ptr> sinks;
        if (console_output) {
            auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console->set_pattern("%H:%M:%S:%e [%^%l%$] %v");
            sinks.push_back(console);
        }
        if (!log_file.empty()) {
            auto file = std::make_shared<spdlog::sinks::basic_file_sink_mt>(std::string(log_file), true);
            file->set_pattern("%H:%M:%S:%e [%^%l%$] %v");
            sinks.push_back(file);
        }
        log.sinks() = std::move(sinks);
    }

    void set_log_level(Level level) { detail::instance().set_level(level); }
    Level log_level() { return detail::instance().level(); }
    bool enabled(Level level) { return detail::instance().should_log(level); }
    void flush() { detail::instance().flush(); }
} // namespace mhs::logger
