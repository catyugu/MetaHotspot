#pragma once
// Existing PTA fixtures exercise translation, compilation and presentation together.
#include "compiler/model_compiler.hpp"
#include "pta/xml.hpp"
#include "postprocessor.hpp"
#include "pta/definition.hpp"
namespace test_pta {
    struct Model : mhs::model::Model {
        mhs::model::MeshSpec sampling_grid;
        struct {
            mhs::core::Vector<mhs::model::Vec3> points;
            std::array<std::size_t, 3> cartesian_shape {};
        } output;
    };
    inline Model build_model(const mhs::model::ModelDefinition& d, const mhs::model::DiscreteMesh* mesh = nullptr)
    {
        Model m;
        static_cast<mhs::model::Model&>(m)
            = mesh ? mhs::compiler::build_model(d, *mesh) : mhs::compiler::build_model(d);
        m.sampling_grid = d.mesh;
        m.output.cartesian_shape = {d.mesh.x_vertices.size(), d.mesh.y_vertices.size(), d.mesh.z_vertices.size()};
        for (double x : d.mesh.x_vertices)
            for (double y : d.mesh.y_vertices)
                for (double z : d.mesh.z_vertices)
                    m.output.points.push_back({x, y, z});
        return m;
    }
    inline Model build_model(const mhs::model::ModelDefinition& d, const mhs::model::DiscreteMesh& mesh)
    { return build_model(d, &mesh); }
    inline Model build_model(const mhs::app::pta::Definition& d) { return build_model(mhs::app::pta::normalize(d)); }
    inline Model build_model(const mhs::app::pta::Definition& d, const mhs::model::DiscreteMesh& mesh)
    { return build_model(mhs::app::pta::normalize(d), mesh); }
    inline auto interpolate_cell_to_node(const Model& m, std::span<const double> t, double time)
    { return mhs::app::interpolate_cell_to_node(m, m.sampling_grid, t, time); }
}
