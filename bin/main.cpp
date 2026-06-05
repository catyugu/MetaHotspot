#include "common/logger.hpp"
#include "io/io.hpp"
#include "postprocessor/postprocessor.hpp"
#include "preprocessor/preprocessor.hpp"
#include "scheduler/scheduler.hpp"
#include "solver/solver.hpp"
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
        auto io_structure = mhs::io::read_xml(input_path);

        MHS_LOG_INFO("Loaded {} layers, {} materials, {} boundaries", io_structure.layers.size(),
            io_structure.materials.size(), io_structure.boundaries.size());

        // Preprocess
        mhs::Preprocessor preprocessor;
        auto model = preprocessor.load(io_structure);

        MHS_LOG_INFO("Created mesh with {} cells ({} x {} x {})", model->mesh.total_cell_count, model->mesh.nx,
            model->mesh.ny, model->mesh.nz);

        // 准备探针 traces：仅瞬态 + 存在观察点时启用
        std::vector<mhs::ProbeTrace> traces;
        for (const auto& p : model->observation_points) {
            mhs::ProbeTrace t;
            t.name = p.name;
            traces.push_back(std::move(t));
        }
        const bool probe_enabled
            = (model->study_type == mhs::StudyType::Transient) && !model->observation_points.empty();

        // Create solver
        auto solver = mhs::Solver::create(mhs::SolverType::Pardiso);

        // Create scheduler
        mhs::Scheduler scheduler;
        scheduler.setModel(model.get());
        scheduler.setSolver(std::move(solver));

        // 装配时间步回调：每步求解后做节点插值 + 探针采样。
        // 复用 main 中的 Postprocessor（lambda 持有引用；main 生命周期覆盖 scheduler.run()）
        mhs::Postprocessor postprocessor;
        if (probe_enabled) {
            mhs::StepCallback cb;
            cb.on_step_done = [&postprocessor, &model = *model, &traces, &io_structure](
                                  double time, int step, const std::vector<double>& cell_T) {
                (void)step;
                auto node_T = postprocessor.interpolate_cell_to_node(model, cell_T);
                // 检查 traces 容量
                for (size_t i = 0; i < traces.size() && i < model.observation_points.size(); ++i) {
                    double v = postprocessor.sample_point(node_T, model, model.observation_points[i]);
                    traces[i].times.push_back(time);
                    traces[i].values.push_back(v);
                }
                (void)io_structure;
            };
            scheduler.setCallback(std::move(cb));
        }

        // Run simulation
        MHS_LOG_INFO("Running simulation...");
        scheduler.run();

        const auto& solution = scheduler.solution();

        MHS_LOG_INFO("Simulation complete. {} cells computed.", solution.size());

        // Postprocess
        auto node_temperature = postprocessor.interpolate_cell_to_node(*model, solution);

        // Write outputs
        mhs::io::write_vtu(output_vtu, *model, node_temperature);
        MHS_LOG_INFO("VTU written to: {}", output_vtu);

        mhs::io::write_xml(input_path, output_xml, *model, node_temperature, traces);
        MHS_LOG_INFO("XML written to: {}", output_xml);

        // Print statistics
        double max_T = postprocessor.max_temperature(solution);
        double min_T = postprocessor.min_temperature(solution);
        MHS_LOG_INFO("Temperature range: {:.2f}K to {:.2f}K", min_T, max_T);
    }
    catch (const std::exception& e) {
        MHS_LOG_ERROR("Simulation failed: {}", e.what());
        return 1;
    }

    MHS_LOG_INFO("Done.");
    return 0;
}