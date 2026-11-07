#include "logging/logger.hpp"

#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

namespace mhs::logger {
    namespace {
        class CallbackSink final : public spdlog::sinks::base_sink<std::mutex> {
        public:
            void set_callback(LogCallback callback)
            {
                std::atomic_store(&callback_, callback ? std::make_shared<const LogCallback>(std::move(callback))
                                                       : std::shared_ptr<const LogCallback> {});
            }

        protected:
            void sink_it_(const spdlog::details::log_msg& msg) override
            {
                auto callback = std::atomic_load(&callback_);
                if (callback) {
                    const std::string message(msg.payload.data(), msg.payload.size());
                    try {
                        (*callback)(msg.level, message);
                    }
                    catch (...) {
                        // Host callbacks must not interrupt a numerical solve.
                    }
                }
            }
            void flush_() override {}

        private:
            std::shared_ptr<const LogCallback> callback_;
        };

        std::once_flag config_once;

        std::shared_ptr<CallbackSink> callback_sink()
        {
            static auto sink = std::make_shared<CallbackSink>();
            return sink;
        }

        void configure_level(Level fallback)
        {
            std::call_once(config_once, [fallback] {
                auto level = fallback;
                if (const char* raw = std::getenv("MHS_LOG_LEVEL")) {
                    std::string value(raw);
                    std::transform(value.begin(), value.end(), value.begin(),
                        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    if (value == "trace")
                        level = Level::trace;
                    else if (value == "debug")
                        level = Level::debug;
                    else if (value == "info")
                        level = Level::info;
                    else if (value == "warn" || value == "warning")
                        level = Level::warn;
                    else if (value == "error")
                        level = Level::err;
                    else if (value == "critical")
                        level = Level::critical;
                    else if (value == "off")
                        level = Level::off;
                }
                spdlog::set_level(level);
            });
        }
    } // namespace

    void init(std::string_view log_file, bool console_output, Level default_level)
    {
        configure_level(default_level);
        const auto active_level = spdlog::get_level();
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
        sinks.push_back(callback_sink());

        auto global_logger = std::make_shared<spdlog::logger>("mhs", sinks.begin(), sinks.end());
        spdlog::set_default_logger(global_logger);
        spdlog::set_level(active_level);
        spdlog::flush_on(Level::warn);
    }

    void set_log_level(Level level)
    {
        configure_level(Level::warn);
        spdlog::set_level(level);
    }

    Level log_level()
    {
        configure_level(Level::warn);
        return spdlog::get_level();
    }

    bool enabled(Level level)
    {
        configure_level(Level::warn);
        return spdlog::default_logger_raw()->should_log(level);
    }

    void set_log_callback(LogCallback callback)
    {
        configure_level(Level::warn);
        // Attach the callback sink even when the host never calls init().
        // Keep any existing sink configuration otherwise.
        auto logger = spdlog::default_logger();
        const auto sink = callback_sink();
        if (std::find(logger->sinks().begin(), logger->sinks().end(), sink) == logger->sinks().end())
            logger->sinks().push_back(sink);
        sink->set_callback(std::move(callback));
    }

    void flush()
    {
        if (auto logger = spdlog::default_logger())
            logger->flush();
    }
} // namespace mhs::logger
