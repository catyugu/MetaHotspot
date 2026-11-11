#pragma once
// Existing PTA fixtures exercise translation, compilation and presentation together.
#include "compiler/model_compiler.hpp"

#include "postprocess.hpp"
#include "pta/definition.hpp"
#include "pta/io.hpp"
namespace test_pta {
    inline mhs::model::SurfaceRegion face_region(
        mhs::model::Axis axis, double coordinate, mhs::model::RegionRect rectangle)
    { return {axis, coordinate, rectangle}; }

    struct Model : mhs::model::Model {
        mhs::model::ConformalMeshSpec sampling_grid;
        struct {
            mhs::core::Vector<mhs::model::Vec3> points;
            std::array<std::size_t, 3> cartesian_shape {};
        } output;
    };
    inline Model build_model(const mhs::model::ModelDefinition& d, const mhs::model::ConformalMeshSpec& mesh)
    {
        Model m;
        static_cast<mhs::model::Model&>(m) = mhs::compiler::build_model(d, mesh);
        m.sampling_grid = mesh;
        m.output.cartesian_shape = {mesh.x_vertices.size(), mesh.y_vertices.size(), mesh.z_vertices.size()};
        for (double x : mesh.x_vertices)
            for (double y : mesh.y_vertices)
                for (double z : mesh.z_vertices)
                    m.output.points.push_back({x, y, z});
        return m;
    }
    inline Model build_model(mhs::app::pta::Definition d)
    {
        mhs::app::pta::normalize(d);
        return build_model(d.model, d.mesh);
    }
    inline Model build_model(const mhs::app::pta::Input& input) { return build_model(input.model, input.mesh); }
    inline auto interpolate_cell_to_node(const Model& m, std::span<const double> t, double time)
    { return mhs::app::postprocess::interpolate_cell_to_node(m, m.sampling_grid, t, time); }
}
