#include "io/io.hpp"
#include "preprocessor/preprocessor.hpp"
#include "assembler/assembler.hpp"
#include "solver/solver.hpp"
#include "scheduler/scheduler.hpp"
#include "postprocessor/postprocessor.hpp"
#include "logger/logger.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.xml> [output.vtu] [output.xml]" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_vtu = argc > 2 ? argv[2] : "output.vtu";
    std::string output_xml = argc > 3 ? argv[3] : "output.xml";

    // Initialize logger
    mhs::logger::init("metahotspot.log", true);

    MHS_LOG_INFO("Starting MetaHotspot simulation");
    MHS_LOG_INFO("Input: {}", input_path);

    try {
        // Read input XML
        mhs::io::Reader reader(input_path);
        auto io_structure = reader.read_structure();

        MHS_LOG_INFO("Loaded {} layers, {} materials, {} boundaries",
                     io_structure.layers.size(),
                     io_structure.materials.size(),
                     io_structure.boundaries.size());

        // Preprocess
        mhs::Preprocessor preprocessor;
        auto model = preprocessor.load(io_structure);

        MHS_LOG_INFO("Created mesh with {} cells ({} x {} x {})",
                     model->mesh.cell_count,
                     model->mesh.nx,
                     model->mesh.ny,
                     model->mesh.nz);

        // Create solver
        auto solver = mhs::Solver::create(mhs::SolverType::SparseLU);

        // Create scheduler
        mhs::Scheduler scheduler;
        scheduler.setModel(std::move(model));
        scheduler.setSolver(std::move(solver));

        // Run simulation
        MHS_LOG_INFO("Running simulation...");
        scheduler.run();

        const auto& solution = scheduler.solution();

        MHS_LOG_INFO("Simulation complete. {} cells computed.", solution.size());

        // Postprocess
        mhs::Postprocessor postprocessor;

        // Write outputs
        postprocessor.writeVTU(output_vtu, *scheduler.getModel(), solution);
        MHS_LOG_INFO("VTU written to: {}", output_vtu);

        postprocessor.writeXML(output_xml, *scheduler.getModel(), solution);
        MHS_LOG_INFO("XML written to: {}", output_xml);

        // Print statistics
        double max_T = postprocessor.max_temperature(solution);
        double min_T = postprocessor.min_temperature(solution);
        MHS_LOG_INFO("Temperature range: {:.2f}K to {:.2f}K", min_T, max_T);

    } catch (const std::exception& e) {
        MHS_LOG_ERROR("Simulation failed: {}", e.what());
        return 1;
    }

    MHS_LOG_INFO("Done.");
    return 0;
}