#include "compiler/fom_compiler.hpp"
#include "extraction.hpp"
#include "pta/io.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace {
    using namespace mhs::app::erom;

    template <typename T> T number(std::string_view text)
    {
        if constexpr (std::is_same_v<T, double>) {
            std::size_t consumed = 0;
            const double value = std::stod(std::string(text), &consumed);
            if (consumed != text.size() || !std::isfinite(value))
                throw std::invalid_argument("invalid finite number: " + std::string(text));
            return value;
        }
        else {
            T value;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (error != std::errc {} || end != text.data() + text.size())
                throw std::invalid_argument("invalid integer: " + std::string(text));
            return value;
        }
    }

    struct Arguments {
        std::filesystem::path input, output;
        Selection selection;
        Options options;
        bool list = false, help = false;
    };

    Arguments parse(int argc, char** argv)
    {
        Arguments result;
        bool no_probes = false;
        for (int i = 1; i < argc; ++i) {
            const std::string_view option = argv[i];
            auto next = [&]() -> std::string_view {
                if (++i >= argc)
                    throw std::invalid_argument("missing value for " + std::string(option));
                return argv[i];
            };
            if (option == "--help" || option == "-h")
                result.help = true;
            else if (option == "--list-surfaces")
                result.list = true;
            else if (option == "--htc") {
                const auto id = number<mhs::model::SurfaceId>(next());
                const auto lo = number<double>(next()), hi = number<double>(next());
                if (lo <= 0 || hi < lo)
                    throw std::invalid_argument("HTC range requires 0 < minimum <= maximum");
                if (!result.selection.htc.emplace(id, mhs::model::EromSurfaceTraining {lo, hi}).second)
                    throw std::invalid_argument("duplicate HTC surface");
            }
            else if (option == "--probe") {
                result.selection.all_probes = false;
                result.selection.probes.emplace(next());
            }
            else if (option == "--no-point-probes") {
                result.selection.all_probes = false;
                no_probes = true;
            }
            else if (option == "--no-source-temperatures")
                result.selection.source_temperatures = false;
            else if (option == "--tolerance")
                result.options.tolerance = number<double>(next());
            else if (option == "--max-order")
                result.options.max_order = number<int>(next());
            else if (option == "--output-error")
                result.options.output_error = number<double>(next());
            else if (option.starts_with('-'))
                throw std::invalid_argument("unknown extraction option: " + std::string(option));
            else if (result.input.empty())
                result.input = option;
            else if (result.output.empty())
                result.output = option;
            else
                throw std::invalid_argument("unexpected argument: " + std::string(option));
        }
        if (result.help)
            return result;
        if (result.input.empty())
            throw std::invalid_argument("provide a PTA input path; use --help for usage");
        if (no_probes && !result.selection.probes.empty())
            throw std::invalid_argument("--probe and --no-point-probes are mutually exclusive");
        if (!result.list) {
            if (result.output.extension() != ".mherom" || result.selection.htc.empty())
                throw std::invalid_argument("provide a .mherom output path and at least one --htc group");
            if (std::filesystem::weakly_canonical(result.output) == std::filesystem::weakly_canonical(result.input)
                || (std::filesystem::exists(result.output) && std::filesystem::equivalent(result.input, result.output)))
                throw std::invalid_argument("the output path must differ from the PTA input path");
        }
        return result;
    }
} // namespace

int main(int argc, char** argv)
{
    try {
        auto args = parse(argc, argv);
        if (args.help) {
            std::cout << "usage: metahotspot-erom PTA.xml [OUTPUT.mherom] [options]\n"
                         "  --list-surfaces              list compiled surface IDs\n"
                         "  --htc ID MIN MAX             train this surface (repeatable, W/(m^2 K))\n"
                         "  --probe NAME                 retain a named point probe (repeatable)\n"
                         "  --no-point-probes            omit point temperature outputs\n"
                         "  --no-source-temperatures     omit source mean temperature outputs\n"
                         "  --tolerance VALUE            response/SVD tolerance (default 0.001)\n"
                         "  --max-order COUNT            maximum response solves (default 512)\n"
                         "  --output-error VALUE         sampled output error limit (default 0.02)\n";
            return 0;
        }
        const auto started = std::chrono::steady_clock::now();
        const auto input = mhs::app::pta::read_xml(args.input.string());
        const auto fom = mhs::compiler::compile_fom(input.model, input.mesh);
        if (args.list) {
            for (const auto& surface : fom.surfaces)
                if (!surface.facets.empty())
                    std::cout << surface.id << '\t' << surface.name << '\t' << surface.facets.size() << " facets\n";
            return 0;
        }
        const auto prepared = prepare(fom, input.study.initial_temperature, std::move(args.selection));
        const auto result = extract(prepared, args.options);
        write_erom(args.output, result.asset);
        const auto& stats = result.statistics;
        std::cout << args.output.string() << "  cells=" << prepared.K.rows() << "  order=" << result.asset.K.rows()
                  << "  output error=" << stats.output_error << "\nspectrum=[" << stats.lambda_min << ", "
                  << stats.lambda_max << "]  shifts=" << stats.shifts << "  response solves=" << stats.response_solves
                  << "  response AMG setups=" << stats.response_amg_setups << "  CG iterations=" << stats.cg_iterations
                  << "  seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                  << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "metahotspot-erom: " << error.what() << '\n';
        return 1;
    }
}
