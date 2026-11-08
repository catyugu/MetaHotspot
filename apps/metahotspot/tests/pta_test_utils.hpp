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
        mhs::model::MeshSpec sampling_grid;
        struct {
            mhs::core::Vector<mhs::model::Vec3> points;
            std::array<std::size_t, 3> cartesian_shape {};
        } output;
    };
    inline Model build_model(const mhs::model::ModelDefinition& d)
    {
        Model m;
        static_cast<mhs::model::FomDomain&>(m) = mhs::compiler::build_model(d).domains.front();
        m.flow = mhs::solver::initialize_frozen_flow(m, mhs::core::Vector<double>(m.discrete.cells.size(), 300));
        m.sampling_grid = d.domains[0].mesh;
        m.output.cartesian_shape = {d.domains[0].mesh.x_vertices.size(), d.domains[0].mesh.y_vertices.size(),
            d.domains[0].mesh.z_vertices.size()};
        for (double x : d.domains[0].mesh.x_vertices)
            for (double y : d.domains[0].mesh.y_vertices)
                for (double z : d.domains[0].mesh.z_vertices)
                    m.output.points.push_back({x, y, z});
        return m;
    }
    inline Model build_model(const mhs::app::pta::Definition& d) { return build_model(mhs::app::pta::normalize(d)); }
    inline Model build_model(const mhs::app::pta::Input& input) { return build_model(input.model); }
    inline auto interpolate_cell_to_node(const Model& m, std::span<const double> t, double time)
    { return mhs::app::postprocess::interpolate_cell_to_node(m, m.sampling_grid, t, time); }
}
