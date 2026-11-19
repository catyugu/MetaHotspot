#pragma once
// Existing PTA fixtures exercise translation, compilation and presentation together.
#include "compiler/fom_compiler.hpp"

#include "postprocess.hpp"
#include "pta/definition.hpp"
#include "pta/io.hpp"
namespace test_pta {
    inline mhs::model::SurfaceRegion face_region(mhs::model::Axis axis, double coordinate, mhs::model::Rect rectangle)
    {
        return {axis, coordinate, rectangle};
    }

    struct Model : mhs::model::Fom {
        mhs::model::MeshSpec sampling_grid;
        struct {
            mhs::core::Vector<mhs::model::Vec3> points;
            std::array<std::size_t, 3> cartesian_shape {};
        } output;
    };
    inline Model compile_fom(const mhs::model::FomDefinition& d, const mhs::model::MeshSpec& mesh)
    {
        Model m;
        static_cast<mhs::model::Fom&>(m) = mhs::compiler::compile_fom(d, mesh);
        m.sampling_grid = mesh;
        m.output.cartesian_shape = {mesh.x_vertices.size(), mesh.y_vertices.size(), mesh.z_vertices.size()};
        for (double x : mesh.x_vertices)
            for (double y : mesh.y_vertices)
                for (double z : mesh.z_vertices)
                    m.output.points.push_back({x, y, z});
        return m;
    }
    inline Model compile_fom(mhs::app::pta::Definition d)
    {
        mhs::app::pta::normalize(d);
        return compile_fom(d.model, d.mesh);
    }
    inline Model compile_fom(const mhs::app::pta::Input& input) { return compile_fom(input.model, input.mesh); }
    inline auto interpolate_cell_to_node(const Model& m, std::span<const double> t, double time)
    {
        return mhs::app::postprocess::interpolate_cell_to_node(m, m.sampling_grid, t, time);
    }
}
