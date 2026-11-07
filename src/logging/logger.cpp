#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "core/vector.hpp"
#include "logging/logger.hpp"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace mhs::logger {
    namespace {
        std::atomic<int> explicit_level {-1};

        spdlog::level::level_enum to_spdlog(LogLevel level)
        {
            return static_cast<spdlog::level::level_enum>(level);
        }

        std::optional<LogLevel> parse_level(const char* text)
        {
            if (!text)
                return std::nullopt;
            std::string name(text);
            for (char& c : name)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (name == "trace")
                return LogLevel::trace;
            if (name == "debug")
                return LogLevel::debug;
            if (name == "info")
                return LogLevel::info;
            if (name == "warn" || name == "warning")
                return LogLevel::warn;
            if (name == "error" || name == "err")
                return LogLevel::error;
            if (name == "critical")
                return LogLevel::critical;
            if (name == "off")
                return LogLevel::off;
            return std::nullopt;
        }

        // Cached on first initialization; getenv is never used on a log path.
        std::optional<LogLevel> environment_level()
        {
            static const auto level = parse_level(std::getenv("MHS_LOG_LEVEL"));
            return level;
        }

        class CallbackSink final : public spdlog::sinks::base_sink<std::mutex> {
        public:
            void set(LogCallback callback)
            {
                std::lock_guard lock(mutex_);
                callback_ = std::move(callback);
            }

        protected:
            void sink_it_(const spdlog::details::log_msg& message) override
            {
                if (callback_)
                    callback_(static_cast<LogLevel>(message.level),
                        std::string_view(message.payload.data(), message.payload.size()));
            }

            void flush_() override {}

        private:
            LogCallback callback_;
        };

        std::shared_ptr<CallbackSink> callback_sink()
        {
            static auto sink = std::make_shared<CallbackSink>();
            return sink;
        }
    } // namespace

    void init(std::string_view log_file, bool console_output, LogLevel default_level)
    {
        mhs::core::Vector<spdlog::sink_ptr> sinks;
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
        // Without any output selection, retain the historical console fallback.
        if (sinks.empty()) {
            auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console->set_pattern("%H:%M:%S:%e [%^%l%$] %v");
            sinks.push_back(console);
        }
        sinks.push_back(callback_sink());

        auto global_logger = std::make_shared<spdlog::logger>("mhs", sinks.begin(), sinks.end());
        spdlog::set_default_logger(global_logger);
        spdlog::flush_on(spdlog::level::warn);

        LogLevel level = environment_level().value_or(default_level);
        const int override_level = explicit_level.load(std::memory_order_relaxed);
        if (override_level >= 0)
            level = static_cast<LogLevel>(override_level);
        spdlog::set_level(to_spdlog(level));
    }

    void set_level(LogLevel level)
    {
        explicit_level.store(static_cast<int>(level), std::memory_order_relaxed);
        spdlog::set_level(to_spdlog(level));
    }

    LogLevel get_level()
    {
        return static_cast<LogLevel>(spdlog::get_level());
    }

    bool enabled(LogLevel level)
    {
        return spdlog::get_level() <= to_spdlog(level);
    }

    void set_callback(LogCallback callback)
    {
        callback_sink()->set(std::move(callback));
    }

    void flush()
    {
        if (auto logger = spdlog::default_logger())
            logger->flush();
    }

} // namespace mhs::logger
