#include "cli.hpp"
#include "logger/logger.hpp"
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

    // Fluid overlay 路径策略：
    //   - --fluid-overlay <file>     -> 显式覆盖；只有显式传入时才执行流体相关逻辑
    //   - (默认)                    -> 不加载 fluid overlay，所有流体逻辑跳过
    std::optional<std::string> fluidOverlayPath;
    fluidOverlayPath = opts.fluid_overlay;

    // Initialize logger
    mhs::logger::init(opts.log_file, opts.console_log);

    MHS_LOG_INFO("Starting MetaHotspot simulation");
    MHS_LOG_INFO("Input: {}", input_path);

    try {
        // Read input XML
        auto io_structure = mhs::io::read_xml(input_path);

        MHS_LOG_INFO("Loaded {} layers, {} materials, {} boundaries", io_structure.layers.size(),
            io_structure.materials.size(), io_structure.boundaries.size());

        // Fluid overlay: 加载与否由 CLI 决定；只有显式传入 --fluid-overlay 时才执行流体相关逻辑。
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
        auto case_dir = std::filesystem::path(input_path).parent_path().string();
        auto trained_models = mhs::io::load_smart_macro_models(io_structure, case_dir);
        auto model = preprocessor.load(io_structure, fluidOverlay, trained_models);

        MHS_LOG_INFO("Created mesh with {} cells ({} x {} x {})", model->mesh.nx * model->mesh.ny * model->mesh.nz,
            model->mesh.nx, model->mesh.ny, model->mesh.nz);

        // Count fluid cells for diagnostics
        if (!model->fluid.is_fluid.empty()) {
            int fluidCount = 0;
            for (uint8_t v : model->fluid.is_fluid) {
                if (v)
                    ++fluidCount;
            }
            if (fluidCount > 0) {
                MHS_LOG_INFO("Fluid cells: {}", fluidCount);
            }
        }

        // Create solver
        auto solver = mhs::sim::LinearSolver::create();

        // Create scheduler. setModel 时 ProbeRecorder 同步 initialize；瞬态 + 存在观察点时
        // 才会写入 trace，稳态路径或空观察点下 scheduler.probeTraces() 为空。
        mhs::sim::Scheduler scheduler;
        scheduler.setModel(model.get());
        scheduler.setSolver(std::move(solver));

        // Run simulation
        MHS_LOG_INFO("Running simulation...");
        scheduler.run();

        MHS_LOG_INFO("Simulation complete.");

        // scheduler.solution() returns physical cell-center temperatures only
        // (modal DOFs are stripped internally).
        const auto& solution = scheduler.solution();

        // Postprocess
        auto node_temperature = mhs::post::interpolate_cell_to_node(*model, solution, scheduler.currentTime());

        // Write outputs
        mhs::io::write_vtu(output_vtu, *model, node_temperature);
        MHS_LOG_INFO("VTU written to: {}", output_vtu);

        mhs::io::write_xml(input_path, output_xml, *model, node_temperature, scheduler.probeTraces());
        MHS_LOG_INFO("XML written to: {}", output_xml);

        // Print statistics (physical DOFs only — modal amplitudes are not temperatures)
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
