#include "cli.hpp"
#include "compiler/model_compiler.hpp"
#include "core/solver.hpp"
#include "io/model_io.hpp"
#include "io/result_io.hpp"
#include "logging/logger.hpp"
#include "solver/postprocessor.hpp"
#include <iostream>
#include <optional>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    auto cli = mhs::cli::parse(argc, argv);

    if (cli.status == mhs::cli::ParseStatus::HelpRequested) {
        std::cout << cli.message;
        return 0;
    }
    if (cli.status == mhs::cli::ParseStatus::Error || !cli.options.has_value()) {
        std::cerr << cli.message << std::endl;
        std::cerr << "\n" << mhs::cli::usage_text(cli.program_name);
        return 1;
    }

    const auto& opts = *cli.options;
    const std::string& input_path = opts.input;
    const std::string& output_vtu = opts.output_vtu;
    const std::string& output_xml = opts.output_xml;

    std::optional<std::string> fluidOverlayPath;
    fluidOverlayPath = opts.fluid_overlay;

    // Initialize logger
    mhs::logger::init(opts.log_file, opts.console_log);

    MHS_LOG_INFO("Starting MetaHotspot simulation");
    MHS_LOG_INFO("Input: {}", input_path);

    try {
        // Read input XML
        auto definition = mhs::io::read_xml(input_path);

        MHS_LOG_INFO("Loaded {} layers, {} materials, {} boundaries", definition.layers.size(),
            definition.materials.size(), definition.boundaries.size());

        if (fluidOverlayPath.has_value()) {
            mhs::io::merge_fluid_xml(*fluidOverlayPath, definition);
            MHS_LOG_INFO("Merged fluid data with {} boundaries", definition.fluid_boundaries.size());
        }

        auto model = mhs::sim::build_model(definition);

        MHS_LOG_INFO("Created mesh with {} cells ({} x {} x {})", model.mesh.nx * model.mesh.ny * model.mesh.nz,
            model.mesh.nx, model.mesh.ny, model.mesh.nz);

        // Count fluid cells for diagnostics
        if (!model.fluid.fluid_to_global.empty()) {
            const auto fluidCount = model.fluid.fluid_to_global.size();
            if (fluidCount > 0) {
                MHS_LOG_INFO("Fluid cells: {}", fluidCount);
            }
        }

        // Run simulation via the standard solve entry point
        MHS_LOG_INFO("Running simulation...");
        auto result = mhs::sim::solve(model);

        MHS_LOG_INFO("Simulation complete.");

        const auto& solution = result.state;

        // Write outputs
        // VTU: writes cell-centered body temperature directly (no node interpolation)
        mhs::io::write_vtu(output_vtu, model, solution);
        MHS_LOG_INFO("VTU written to: {}", output_vtu);

        // XML output uses node-centered data (interpolated from cell centers)
        auto node_temperature = mhs::post::interpolate_cell_to_node(model, solution, result.time);
        mhs::io::write_xml(input_path, output_xml, model, node_temperature, result.probe_traces);
        MHS_LOG_INFO("XML written to: {}", output_xml);

        // Print statistics
        double max_T = mhs::post::max_temperature(solution);
        double min_T = mhs::post::min_temperature(solution);
        MHS_LOG_INFO("Temperature range: {:.2f}K to {:.2f}K", min_T, max_T);
    }
    catch (const std::exception& e) {
        printf("Simulation failed: %s", e.what());
        return 1;
    }

    MHS_LOG_INFO("Done.");
    return 0;
}
