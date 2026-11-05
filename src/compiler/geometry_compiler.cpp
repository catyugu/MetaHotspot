#include "compiler/geometry_compiler.hpp"
#include "core/vector.hpp"
#include "model/constants.hpp"

#include "expression/expression.hpp"
#include <algorithm>

namespace mhs::compiler {

    constexpr double EPS = mhs::model::geometry_eps;

    namespace {

        bool point_in_region(const CompiledBoundaryRegion& region, double a, double b)
        {
            return std::any_of(region.rectangles.begin(), region.rectangles.end(), [&](const auto& rectangle) {
                return a >= rectangle.a_min - EPS && a <= rectangle.a_max + EPS && b >= rectangle.b_min - EPS
                    && b <= rectangle.b_max + EPS;
            });
        }

        std::optional<std::size_t> find_block_for_cell(
            const ResolvedLayerGeometry& resolved_layer, double cx, double cy, double cz)
        {
            for (std::size_t b = resolved_layer.blocks.size(); b-- > 0;) {
                const auto& block = resolved_layer.blocks[b];
                if (cz < block.z_start - EPS || cz > block.z_end + EPS)
                    continue;
                bool is_inside = false;
                for (const auto& rect : block.rects) {
                    if (cx >= rect.x - EPS && cx <= rect.x + rect.width + EPS && cy >= rect.y - EPS
                        && cy <= rect.y + rect.height + EPS) {
                        is_inside = rect.operation == mhs::model::GeometryOperation::Add;
                    }
                }
                if (is_inside)
                    return b;
            }
            return std::nullopt;
        }

    } // anonymous namespace

    mhs::core::Vector<ResolvedLayerGeometry> resolve_geometry(const mhs::core::Vector<mhs::model::LayerSpec>& layers,
        double si_scale, const mhs::expression::SymbolTable& symbols)
    {
        const auto num_layers = layers.size();
        mhs::core::Vector<ResolvedLayerGeometry> resolved(num_layers);

        mhs::core::Vector<double> thickness(num_layers);
        double z_cursor = 0.0;
        for (std::size_t l = 0; l < num_layers; l++) {
            if (l == 0) {
                double max_t = 0.0;
                for (const auto& b : layers[l].blocks) {
                    if (b.thickness.has_value()) {
                        double t = mhs::expression::eval_geometry(*b.thickness, symbols) * si_scale;
                        if (t > max_t)
                            max_t = t;
                    }
                }
                double layer_t = layers[l].thickness.empty()
                    ? 0.0
                    : mhs::expression::eval_geometry(layers[l].thickness, symbols) * si_scale;
                thickness[l] = std::max(max_t, layer_t);
            }
            else {
                thickness[l] = mhs::expression::eval_geometry(layers[l].thickness, symbols) * si_scale;
            }
            z_cursor += thickness[l];
        }
        for (std::size_t l = 0; l < num_layers; l++) {
            resolved[l].z_start = z_cursor - thickness[l];
            resolved[l].z_end = z_cursor;
            z_cursor -= thickness[l];
        }

        for (std::size_t l = 0; l < num_layers; l++) {
            const auto& layer = layers[l];
            double layer_x_off_si = mhs::expression::eval_geometry(layer.x_offset, symbols) * si_scale;
            double layer_y_off_si = mhs::expression::eval_geometry(layer.y_offset, symbols) * si_scale;

            for (const auto& block : layer.blocks) {
                ResolvedBlock rb;
                double block_x_off_si = mhs::expression::eval_geometry(block.x_offset, symbols) * si_scale;
                double block_y_off_si = mhs::expression::eval_geometry(block.y_offset, symbols) * si_scale;

                if (l == 0 && block.thickness.has_value()) {
                    double b_thick = mhs::expression::eval_geometry(*block.thickness, symbols) * si_scale;
                    rb.z_start = resolved[l].z_start;
                    rb.z_end = resolved[l].z_start + b_thick;
                }
                else {
                    rb.z_start = resolved[l].z_start;
                    rb.z_end = resolved[l].z_end;
                }

                for (const auto& operation : block.geometry) {
                    const auto& rect = operation.rect;
                    ResolvedRect rr;
                    rr.operation = operation.operation;
                    double x_val = mhs::expression::eval_geometry(rect.x, symbols);
                    double y_val = mhs::expression::eval_geometry(rect.y, symbols);
                    double w_val = mhs::expression::eval_geometry(rect.width, symbols);
                    double h_val = mhs::expression::eval_geometry(rect.height, symbols);
                    if (w_val < 0) {
                        x_val += w_val;
                        w_val = -w_val;
                    }
                    if (h_val < 0) {
                        y_val += h_val;
                        h_val = -h_val;
                    }
                    rr.x = x_val * si_scale + block_x_off_si + layer_x_off_si;
                    rr.y = y_val * si_scale + block_y_off_si + layer_y_off_si;
                    rr.width = w_val * si_scale;
                    rr.height = h_val * si_scale;
                    rb.rects.push_back(rr);
                }
                resolved[l].blocks.push_back(rb);
            }
        }
        return resolved;
    }

    std::optional<std::pair<std::size_t, std::size_t>> classify_point(
        const mhs::core::Vector<ResolvedLayerGeometry>& layers, const mhs::model::Vec3& p)
    {
        for (std::size_t l = 0; l < layers.size(); ++l) {
            if (p[2] < layers[l].z_start - EPS || p[2] > layers[l].z_end + EPS)
                continue;
            auto b = find_block_for_cell(layers[l], p[0], p[1], p[2]);
            if (b)
                return std::pair {l, *b};
        }
        return std::nullopt;
    }

    void project_cell_regions(mhs::model::Model& model, const mhs::core::Vector<ResolvedLayerGeometry>& layers)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        model.cell_material.clear();
        model.source_terms.clear();
        for (std::size_t c = 0; c < model.discrete.cells.size(); ++c) {
            const auto& cell = model.discrete.cells[c];
            std::array<mhs::core::Vector<double>, 3> cuts;
            for (int a = 0; a < 3; ++a)
                cuts[a] = {cell.min[a], cell.max[a]};
            auto cut = [&](int a, double v) {
                if (v > cell.min[a] + EPS && v < cell.max[a] - EPS)
                    cuts[a].push_back(v);
            };
            for (const auto& layer : layers)
                for (const auto& block : layer.blocks) {
                    if (block.z_end <= cell.min[2] || block.z_start >= cell.max[2])
                        continue;
                    for (const auto& r : block.rects) {
                        if (r.x + r.width <= cell.min[0] || r.x >= cell.max[0] || r.y + r.height <= cell.min[1]
                            || r.y >= cell.max[1])
                            continue;
                        cut(0, r.x);
                        cut(0, r.x + r.width);
                        cut(1, r.y);
                        cut(1, r.y + r.height);
                        cut(2, block.z_start);
                        cut(2, block.z_end);
                    }
                }
            for (auto& v : cuts) {
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
            }
            MaterialId material = mhs::core::invalid_id;
            for (std::size_t x = 1; x < cuts[0].size(); ++x)
                for (std::size_t y = 1; y < cuts[1].size(); ++y)
                    for (std::size_t z = 1; z < cuts[2].size(); ++z) {
                        Vec3 p {(cuts[0][x - 1] + cuts[0][x]) / 2, (cuts[1][y - 1] + cuts[1][y]) / 2,
                            (cuts[2][z - 1] + cuts[2][z]) / 2};
                        const auto location = classify_point(layers, p);
                        // A block edge can cut a control volume and leave a piece
                        // outside every block. Such a piece carries no source.
                        if (!location)
                            continue;
                        const auto [l, b] = *location;
                        const auto& block = layers[l].blocks[b];
                        material = block.material_id;
                        const double volume = (cuts[0][x] - cuts[0][x - 1]) * (cuts[1][y] - cuts[1][y - 1])
                            * (cuts[2][z] - cuts[2][z - 1]);
                        model.source_terms.push_back({static_cast<CellId>(c), block.heat_source_idx, volume, p});
                    }
            model.cell_material.push_back(material);
        }
    }

    mhs::core::Vector<mhs::model::BoundaryFace> project_surface_regions(const mhs::model::DiscreteMesh& mesh,
        const mhs::core::Vector<CompiledBoundaryRegion>& boundaries, mhs::model::BoundaryConditionId fallback)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        mhs::core::Vector<BoundaryFace> faces;
        for (std::size_t id = 0; id < mesh.faces.size(); ++id) {
            const auto& face = mesh.faces[id];
            if (!face.exterior())
                continue;
            const int a = face.axis == Axis::X ? 1 : 0, b = face.axis == Axis::Z ? 1 : 2;
            mhs::core::Vector<double> ca {face.bounds[0], face.bounds[1]}, cb {face.bounds[2], face.bounds[3]};
            for (const auto& region : boundaries) {
                if (region.axis != face.axis || std::abs(region.coordinate - face.coordinate(mesh.cells)) >= EPS)
                    continue;
                for (const auto& r : region.rectangles) {
                    if (r.a_max <= face.bounds[0] || r.a_min >= face.bounds[1] || r.b_max <= face.bounds[2]
                        || r.b_min >= face.bounds[3])
                        continue;
                    ca.push_back(std::clamp(r.a_min, face.bounds[0], face.bounds[1]));
                    ca.push_back(std::clamp(r.a_max, face.bounds[0], face.bounds[1]));
                    cb.push_back(std::clamp(r.b_min, face.bounds[2], face.bounds[3]));
                    cb.push_back(std::clamp(r.b_max, face.bounds[2], face.bounds[3]));
                }
            }
            auto unique = [](auto& cuts) {
                std::sort(cuts.begin(), cuts.end());
                cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
            };
            unique(ca);
            unique(cb);
            for (std::size_t i = 1; i < ca.size(); ++i)
                for (std::size_t j = 1; j < cb.size(); ++j) {
                    const std::array<double, 4> bounds {ca[i - 1], ca[i], cb[j - 1], cb[j]};
                    const auto p = face.center(mesh.cells, bounds);
                    BoundaryConditionId condition = fallback;
                    for (auto it = boundaries.rbegin(); it != boundaries.rend(); ++it) {
                        if (it->axis == face.axis && std::abs(it->coordinate - face.coordinate(mesh.cells)) < EPS
                            && point_in_region(*it, p[a], p[b])) {
                            condition = it->condition_index;
                            break;
                        }
                    }
                    if (condition != mhs::core::invalid_id)
                        faces.push_back({static_cast<FaceId>(id), condition, bounds});
                }
        }
        return faces;
    }

} // namespace mhs::compiler
