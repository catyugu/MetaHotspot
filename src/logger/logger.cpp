#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <vector>

#include "logger.hpp"

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

namespace mhs::logger {

    void init(std::string_view log_file, bool console_output)
    {
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

        // 默认兜底：如果没有提供任何输出渠道，强制使用控制台
        if (sinks.empty()) {
            auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console->set_pattern("%H:%M:%S:%e [%^%l%$] %v");
            sinks.push_back(console);
        }

        // 配置并设置默认的全局 Logger
        auto global_logger = std::make_shared<spdlog::logger>("mhs", sinks.begin(), sinks.end());
        spdlog::set_default_logger(global_logger);

        // 遇到 Warning 及以上级别自动将缓冲刷入文件
        spdlog::flush_on(spdlog::level::warn);

#ifdef VERBOSE
        spdlog::set_level(spdlog::level::debug);
#else
        spdlog::set_level(spdlog::level::info);
#endif
    }

    void flush()
    {
        if (auto logger = spdlog::default_logger()) {
            logger->flush();
        }
    }

} // namespace mhs::logger
