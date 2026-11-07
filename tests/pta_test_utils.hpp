#pragma once
// Existing PTA fixtures exercise translation, compilation and presentation together.
#include "compiler/model_compiler.hpp"
#include "io/result_io.hpp"
#include "postprocessor.hpp"
#include "pta_definition.hpp"
namespace test_pta {
    struct Model : mhs::model::Model {
        mhs::app::OutputMesh output;
    };
    inline Model build_model(const mhs::model::ModelDefinition& d, const mhs::model::DiscreteMesh* mesh = nullptr)
    {
        Model m;
        static_cast<mhs::model::Model&>(m)
            = mesh ? mhs::compiler::build_model(d, *mesh) : mhs::compiler::build_model(d);
        m.output = mhs::app::build_output_mesh(m, mesh ? nullptr : &d.mesh);
        return m;
    }
    inline Model build_model(const mhs::model::ModelDefinition& d, const mhs::model::DiscreteMesh& mesh)
    { return build_model(d, &mesh); }
    inline Model build_model(const mhs::app::pta::Definition& d) { return build_model(mhs::app::pta::normalize(d)); }
    inline Model build_model(const mhs::app::pta::Definition& d, const mhs::model::DiscreteMesh& mesh)
    { return build_model(mhs::app::pta::normalize(d), mesh); }
    inline auto interpolate_cell_to_node(const Model& m, std::span<const double> t, double time)
    { return mhs::app::interpolate_cell_to_node(m, m.output, t, time); }
}
