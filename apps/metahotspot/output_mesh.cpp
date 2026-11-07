#include "output_mesh.hpp"
#include "model/model.hpp"
#include <algorithm>
#include <map>
namespace mhs::app {
    OutputMesh build_output_mesh(const mhs::model::Model& model, const mhs::model::MeshSpec* mesh)
    {
        OutputMesh output;
        std::map<Vec3, PointId> ids;
        if (mesh) {
            output.cartesian_shape = {mesh->x_vertices.size(), mesh->y_vertices.size(), mesh->z_vertices.size()};
            for (auto x : mesh->x_vertices)
                for (auto y : mesh->y_vertices)
                    for (auto z : mesh->z_vertices) {
                        ids[{x, y, z}] = static_cast<PointId>(output.points.size());
                        output.points.push_back({x, y, z});
                    }
        }
        constexpr int corners[8][3]
            = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        for (const auto& c : model.discrete.cells) {
            std::array<PointId, 8> vertices;
            for (int i = 0; i < 8; ++i) {
                Vec3 p;
                for (int a = 0; a < 3; ++a)
                    p[a] = corners[i][a] ? c.max[a] : c.min[a];
                auto [it, inserted] = ids.try_emplace(p, static_cast<PointId>(output.points.size()));
                if (inserted)
                    output.points.push_back(p);
                vertices[i] = it->second;
            }
            output.cell_vertices.push_back(vertices);
        }
        return output;
    }
}
