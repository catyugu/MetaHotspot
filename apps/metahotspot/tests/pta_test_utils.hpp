#pragma once
// Existing PTA fixtures exercise translation, compilation and presentation together.
#include "compiler/model_compiler.hpp"
#include "solver/frozen_flow.hpp"

#include "postprocess.hpp"
#include "pta/definition.hpp"
#include "pta/io.hpp"
namespace test_pta {
    inline mhs::model::SurfaceRegion face_region(
        mhs::model::Axis axis, double coordinate, mhs::model::RegionRect rectangle)
    { return {axis, coordinate, rectangle}; }

    struct Model : mhs::model::FomDomain {
        mhs::solver::FrozenFlow flow;
        mhs::model::CompiledLibrary library;
        mhs::model::MeshSpec sampling_grid;
        struct {
            mhs::core::Vector<mhs::model::Vec3> points;
            std::array<std::size_t, 3> cartesian_shape {};
        } output;
    };
    inline Model build_model(const mhs::model::ModelDefinition& d)
    {
        Model m;
        auto compiled = mhs::compiler::build_model(d);
        static_cast<mhs::model::FomDomain&>(m) = std::move(compiled.fom(0));
        m.library = std::move(compiled.library);
        m.flow = mhs::solver::initialize_frozen_flow(m, m.library->material_table, m.library->fluid_conditions,
            mhs::core::Vector<double>(m.discrete.cells.size(), 300));
        const auto& mesh = d.fom(0).mesh;
        m.sampling_grid = mesh;
        m.output.cartesian_shape = {mesh.x_vertices.size(), mesh.y_vertices.size(), mesh.z_vertices.size()};
        for (double x : mesh.x_vertices)
            for (double y : mesh.y_vertices)
                for (double z : mesh.z_vertices)
                    m.output.points.push_back({x, y, z});
        return m;
    }
    inline Model build_model(const mhs::app::pta::Definition& d) { return build_model(mhs::app::pta::normalize(d)); }
    inline Model build_model(const mhs::app::pta::Input& input) { return build_model(input.model); }
    inline auto interpolate_cell_to_node(const Model& m, std::span<const double> t, double time)
    { return mhs::app::postprocess::interpolate_cell_to_node(m, m.library, m.sampling_grid, t, time); }
}
