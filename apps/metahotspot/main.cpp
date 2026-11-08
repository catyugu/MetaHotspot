#include "cli.hpp"
#include "compiler/model_compiler.hpp"
#include "logging/logger.hpp"
#include "output/vtu.hpp"
#include "postprocess.hpp"
#include "pta/io.hpp"
#include "solver/solver.hpp"
#include <array>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    auto cli = mhs::app::cli::parse(argc, argv);

    if (cli.status == mhs::app::cli::ParseStatus::HelpRequested) {
        std::cout << cli.message;
        return 0;
    }
    if (cli.status == mhs::app::cli::ParseStatus::Error || !cli.options.has_value()) {
        std::cerr << cli.message << std::endl;
        std::cerr << "\n" << mhs::app::cli::usage_text(cli.program_name);
        return 1;
    }

    const auto& opts = *cli.options;
    const std::string& input_path = opts.input;
    const std::string& output_vtu = opts.output_vtu;
    const std::string& output_xml = opts.output_xml;

    // Initialize logger
    mhs::logger::init(opts.log_file, opts.console_log, mhs::logger::Level::info);

    MHS_LOG_INFO("Starting MetaHotspot simulation");
    MHS_LOG_INFO("Input: {}", input_path);

    try {
        // Read input XML
        auto definition = mhs::app::pta::read_xml(input_path);

        MHS_LOG_INFO("Loaded {} volumes, {} materials, {} boundaries", definition.model.fom(0).volumes.size(),
            definition.model.component(0).library->materials.size(),
            definition.model.component(0).thermal_boundaries.size());

        auto model = mhs::compiler::build_model(definition.model);

        MHS_LOG_INFO("Compiled {} cells and {} face patches", model.fom(0).discrete.cells.size(),
            model.fom(0).discrete.faces.size());

        // Run simulation via the standard solve entry point
        MHS_LOG_INFO("Running simulation...");
        auto result = mhs::solver::solve(model, definition.study);

        MHS_LOG_INFO("Simulation complete.");
        const auto& solution = result.state;

        const auto& grid = definition.model.fom(0).mesh;
        const std::array<std::size_t, 3> shape {grid.x_vertices.size(), grid.y_vertices.size(), grid.z_vertices.size()};
        // Write outputs
        // VTU: writes cell-centered body temperature directly (no node interpolation)
        mhs::app::output::write_vtu(output_vtu, model.fom(0).discrete, solution);
        MHS_LOG_INFO("VTU written to: {}", output_vtu);

        // XML output uses node-centered data (interpolated from cell centers)
        auto node_temperature
            = mhs::app::postprocess::interpolate_cell_to_node(model.fom(0), grid, solution, result.time);
        mhs::app::pta::write_xml(
            input_path, output_xml, shape, node_temperature, result.probe_traces, result.snapshot_times);
        MHS_LOG_INFO("XML written to: {}", output_xml);

        // Print statistics
        double max_T = mhs::app::postprocess::max_temperature(solution);
        double min_T = mhs::app::postprocess::min_temperature(solution);
        MHS_LOG_INFO("Temperature range: {:.2f}K to {:.2f}K", min_T, max_T);
    }
    catch (const std::exception& e) {
        std::cerr << "Simulation failed: " << e.what() << '\n';
        return 1;
    }

    MHS_LOG_INFO("Done.");
    return 0;
}
