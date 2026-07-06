#include "common/logger.hpp"
#include "io/io.hpp"
#include "linear_solver/linear_solver.hpp"
#include "postprocessor/postprocessor.hpp"
#include "preprocessor/preprocessor.hpp"
#include "scheduler/scheduler.hpp"
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.xml> [output.vtu] [output.xml]" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_vtu = argc > 2 ? argv[2] : "./output.vtu";
    std::string output_xml = argc > 3 ? argv[3] : "./output.xml";

    // Auto-infer fluid-overlay path from input filename: insert "_additional" before extension.
    // E.g. "steady_case1.xml" → "steady_case1_additional.xml"
    std::optional<std::string> fluidOverlayPath;
    {
        auto dot = input_path.rfind('.');
        if (dot != std::string::npos) {
            fluidOverlayPath = input_path.substr(0, dot) + "_additional" + input_path.substr(dot);
        }
        else {
            fluidOverlayPath = input_path + "_additional";
        }
    }

    // Initialize logger
    mhs::logger::init("metahotspot.log", true);

    MHS_LOG_INFO("Starting MetaHotspot simulation");
    MHS_LOG_INFO("Input: {}", input_path);

    try {
        // Read input XML
        auto io_structure = mhs::io::read_xml(input_path);

        MHS_LOG_INFO("Loaded {} layers, {} materials, {} boundaries", io_structure.layers.size(),
            io_structure.materials.size(), io_structure.boundaries.size());

        // Auto-inferred fluid overlay: e.g. steady_case1.xml → steady_case1_additional.xml
        std::optional<mhs::core::FluidOverlay> fluidOverlay;
        if (fluidOverlayPath.has_value()) {
            std::error_code ec;
            if (std::filesystem::exists(*fluidOverlayPath, ec)) {
                fluidOverlay = mhs::io::read_fluid_overlay_xml(*fluidOverlayPath);
                if (fluidOverlay.has_value()) {
                    MHS_LOG_INFO("Loaded fluid overlay with {} materials, {} boundaries",
                        fluidOverlay->fluid_materials.size(), fluidOverlay->boundaries.size());
                }
                else {
                    MHS_LOG_WARN(
                        "Fluid overlay file '{}' contained no FluidOverlay element; skipping", *fluidOverlayPath);
                }
            }
        }

        // Preprocess (apply fluid overlay inside, if any)
        mhs::sim::Preprocessor preprocessor;
        auto model = preprocessor.load(io_structure, fluidOverlay);

        MHS_LOG_INFO("Created mesh with {} cells ({} x {} x {})", model->mesh.nx * model->mesh.ny * model->mesh.nz,
            model->mesh.nx, model->mesh.ny, model->mesh.nz);

        // Count fluid cells for diagnostics
        if (!model->is_fluid.empty()) {
            int fluidCount = 0;
            for (uint8_t v : model->is_fluid) {
                if (v)
                    ++fluidCount;
            }
            if (fluidCount > 0) {
                MHS_LOG_INFO("Fluid cells: {}", fluidCount);
            }
        }

        // Create solver
        auto solver = mhs::sim::LinearSolver::create(mhs::sim::SolverType::Pardiso);

        // Create scheduler. setModel 时 ProbeRecorder 同步 initialize；瞬态 + 存在观察点时
        // 才会写入 trace，稳态路径或空观察点下 scheduler.probeTraces() 为空。
        mhs::sim::Scheduler scheduler;
        scheduler.setModel(model.get());
        scheduler.setSolver(std::move(solver));

        // Run simulation
        MHS_LOG_INFO("Running simulation...");
        scheduler.run();

        const auto& solution = scheduler.solution();

        MHS_LOG_INFO("Simulation complete.");

        // Postprocess
        auto node_temperature = mhs::post::interpolate_cell_to_node(*model, solution, scheduler.currentTime());

        // Write outputs
        mhs::io::write_vtu(output_vtu, *model, node_temperature);
        MHS_LOG_INFO("VTU written to: {}", output_vtu);

        mhs::io::write_xml(input_path, output_xml, *model, node_temperature, scheduler.probeTraces());
        MHS_LOG_INFO("XML written to: {}", output_xml);

        // Print statistics
        double max_T = mhs::post::max_temperature(solution);
        double min_T = mhs::post::min_temperature(solution);
        MHS_LOG_INFO("Temperature range: {:.2f}K to {:.2f}K", min_T, max_T);
    }
    catch (const std::exception& e) {
        MHS_FATAL("Simulation failed: {}", e.what());
        return 1;
    }

    MHS_LOG_INFO("Done.");
    return 0;
}
