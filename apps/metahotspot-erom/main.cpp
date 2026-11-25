#include "input.hpp"
#include "logging/logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc, char* argv[])
{
    constexpr std::string_view usage = "Usage: metahotspot-erom --input project.xml --output package.xml\n";
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << usage;
        return 0;
    }
    try {
        std::filesystem::path input, output;
        for (int i = 1; i < argc; ++i) {
            const std::string_view option = argv[i];
            auto* destination = option == "--input" ? &input : option == "--output" ? &output : nullptr;
            if (!destination || !destination->empty() || i + 1 >= argc
                || std::string_view(argv[i + 1]).starts_with("--"))
                throw std::invalid_argument("invalid or duplicate command-line option: " + std::string(option));
            *destination = argv[++i];
        }
        if (input.empty() || output.empty())
            throw std::invalid_argument("--input and --output are required");
        if (std::filesystem::exists(output) && std::filesystem::equivalent(input, output))
            throw std::invalid_argument("input and output must be different files");

        mhs::logger::init({}, false, mhs::logger::Level::warn);
        mhs::logger::detail::instance().sinks().push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());

        const auto started = std::chrono::steady_clock::now();
        const auto definition = mhs::app::erom::read_input(input);
        const auto extraction = mhs::app::erom::extract(
            definition.model, definition.mesh, definition.ranges, definition.tolerance);
        mhs::app::erom::write_erom(output, extraction.asset);
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
        std::cout << "EROM extracted: order=" << extraction.asset.K.rows()
                  << ", output_error=" << extraction.statistics.output_error << ", elapsed=" << elapsed.count()
                  << " s\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "EROM extraction failed: " << error.what() << '\n';
        return 1;
    }
}
