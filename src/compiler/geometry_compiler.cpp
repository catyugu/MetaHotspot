#include "compiler/geometry_compiler.hpp"
#include "core/constants.hpp"

#include "numerics/expression/expr.hpp"
#include <algorithm>
#include <cstddef>

namespace mhs::sim {

    constexpr double EPS = mhs::core::geometry_eps;

    namespace {

        bool point_in_region(const CompiledBoundaryRegion &region, double a, double b) {
            return std::any_of(region.rectangles.begin(), region.rectangles.end(), [&](const auto &rectangle) {
                return a >= rectangle.a_min - EPS && a <= rectangle.a_max + EPS && b >= rectangle.b_min - EPS &&
                       b <= rectangle.b_max + EPS;
            });
        }

        mhs::core::Index find_block_for_cell(const ResolvedLayerGeometry &resolved_layer, double cx, double cy,
                                             double cz) {
            for (mhs::core::Index b = static_cast<mhs::core::Index>(resolved_layer.blocks.size()) - 1;
                 b != mhs::core::invalidIndex; b--) {
                const auto &block = resolved_layer.blocks[b];
                if (cz < block.z_start - EPS || cz > block.z_end + EPS)
                    continue;
                bool is_inside = false;
                for (const auto &rect : block.rects) {
                    if (cx >= rect.x - EPS && cx <= rect.x + rect.width + EPS && cy >= rect.y - EPS &&
                        cy <= rect.y + rect.height + EPS) {
                        is_inside = rect.operation == mhs::model::GeometryOperation::Add;
                    }
                }
                if (is_inside)
                    return b;
            }
            return mhs::core::invalidIndex;
        }

    } // anonymous namespace

    std::vector<ResolvedLayerGeometry> resolve_geometry(const std::vector<mhs::model::LayerSpec> &layers,
                                                        double si_scale, const mhs::core::SymbolTable &symbols) {
        mhs::core::Index num_layers = static_cast<mhs::core::Index>(layers.size());
        std::vector<ResolvedLayerGeometry> resolved(num_layers);

        std::vector<double> thickness(num_layers);
        double z_cursor = 0.0;
        for (mhs::core::Index l = 0; l < num_layers; l++) {
            if (l == 0) {
                double max_t = 0.0;
                for (const auto &b : layers[l].blocks) {
                    if (b.thickness.has_value()) {
                        double t = mhs::core::eval_geometry(*b.thickness, symbols) * si_scale;
                        if (t > max_t)
                            max_t = t;
                    }
                }
                double layer_t = layers[l].thickness.empty()
                                     ? 0.0
                                     : mhs::core::eval_geometry(layers[l].thickness, symbols) * si_scale;
                thickness[l] = std::max(max_t, layer_t);
            } else {
                thickness[l] = mhs::core::eval_geometry(layers[l].thickness, symbols) * si_scale;
            }
            z_cursor += thickness[l];
        }
        for (mhs::core::Index l = 0; l < num_layers; l++) {
            resolved[l].z_start = z_cursor - thickness[l];
            resolved[l].z_end = z_cursor;
            z_cursor -= thickness[l];
        }

        for (mhs::core::Index l = 0; l < num_layers; l++) {
            const auto &layer = layers[l];
            double layer_x_off_si = mhs::core::eval_geometry(layer.x_offset, symbols) * si_scale;
            double layer_y_off_si = mhs::core::eval_geometry(layer.y_offset, symbols) * si_scale;

            for (const auto &block : layer.blocks) {
                ResolvedBlock rb;
                double block_x_off_si = mhs::core::eval_geometry(block.x_offset, symbols) * si_scale;
                double block_y_off_si = mhs::core::eval_geometry(block.y_offset, symbols) * si_scale;

                if (l == 0 && block.thickness.has_value()) {
                    double b_thick = mhs::core::eval_geometry(*block.thickness, symbols) * si_scale;
                    rb.z_start = resolved[l].z_start;
                    rb.z_end = resolved[l].z_start + b_thick;
                } else {
                    rb.z_start = resolved[l].z_start;
                    rb.z_end = resolved[l].z_end;
                }

                for (const auto &operation : block.geometry) {
                    const auto &rect = operation.rect;
                    ResolvedRect rr;
                    rr.operation = operation.operation;
                    double x_val = mhs::core::eval_geometry(rect.x, symbols);
                    double y_val = mhs::core::eval_geometry(rect.y, symbols);
                    double w_val = mhs::core::eval_geometry(rect.width, symbols);
                    double h_val = mhs::core::eval_geometry(rect.height, symbols);
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

    std::pair<mhs::core::Index, mhs::core::Index> classify_point(const std::vector<ResolvedLayerGeometry> &layers,
                                                                 const mhs::core::Vec3 &p) {
        for (mhs::core::Index l = 0; l < layers.size(); ++l) {
            if (p[2] < layers[l].z_start - EPS || p[2] > layers[l].z_end + EPS)
                continue;
            auto b = find_block_for_cell(layers[l], p[0], p[1], p[2]);
            if (b != mhs::core::invalidIndex)
                return {l, b};
        }
        return {mhs::core::invalidIndex, mhs::core::invalidIndex};
    }

    void project_cell_regions(mhs::core::Model &model, const std::vector<ResolvedLayerGeometry> &layers) {
        using namespace mhs::core;
        model.cells = {};
        for (const auto &cell : model.discrete.cells) {
            const auto [l, b] = classify_point(layers, cell.center());
            if (l == invalidIndex)
                throw std::invalid_argument("mesh cell is outside the physical domain");
            const auto &block = layers[l].blocks[b];
            Index region = b;
            for (Index k = 0; k < l; ++k)
                region += layers[k].blocks.size();
            model.cells.region_id.push_back({region});
            model.cells.layer_id.push_back(static_cast<TableIndex>(l));
            model.cells.block_id.push_back(static_cast<TableIndex>(b));
            model.cells.material_id.push_back(block.material_id);
            model.cells.heat_source_idx.push_back(block.heat_source_idx);
        }
    }

    void project_boundary_patches(mhs::core::Model &model, const std::vector<CompiledBoundaryRegion> &boundaries,
                                  const DefaultBoundary &fallback) {
        using namespace mhs::core;
        std::vector<FacePatch> faces;
        std::vector<FaceBC> assignments;
        for (const auto &face : model.discrete.faces) {
            if (face.neighbor.valid()) {
                faces.push_back(face);
                assignments.push_back({});
                continue;
            }
            const int a = face.axis == 0 ? 1 : 0, b = face.axis == 2 ? 1 : 2;
            std::vector<double> ca{face.min[a], face.max[a]}, cb{face.min[b], face.max[b]};
            for (const auto &region : boundaries) {
                if (static_cast<int>(region.axis) != face.axis ||
                    std::abs(region.coordinate - face.min[face.axis]) >= EPS)
                    continue;
                for (const auto &r : region.rectangles) {
                    if (r.a_max <= face.min[a] || r.a_min >= face.max[a] || r.b_max <= face.min[b] ||
                        r.b_min >= face.max[b])
                        continue;
                    ca.push_back(std::clamp(r.a_min, face.min[a], face.max[a]));
                    ca.push_back(std::clamp(r.a_max, face.min[a], face.max[a]));
                    cb.push_back(std::clamp(r.b_min, face.min[b], face.max[b]));
                    cb.push_back(std::clamp(r.b_max, face.min[b], face.max[b]));
                }
            }
            auto unique = [](auto &cuts) {
                std::sort(cuts.begin(), cuts.end());
                cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
            };
            unique(ca);
            unique(cb);
            for (Index i = 1; i < ca.size(); ++i)
                for (Index j = 1; j < cb.size(); ++j) {
                    auto patch = face;
                    patch.min[a] = ca[i - 1];
                    patch.max[a] = ca[i];
                    patch.min[b] = cb[j - 1];
                    patch.max[b] = cb[j];
                    patch.area = (ca[i] - ca[i - 1]) * (cb[j] - cb[j - 1]);
                    const auto p = patch.center();
                    FaceBC bc{fallback.type, fallback.parameter_index};
                    for (auto it = boundaries.rbegin(); it != boundaries.rend(); ++it) {
                        if (static_cast<int>(it->axis) == face.axis && std::abs(it->coordinate - p[face.axis]) < EPS &&
                            point_in_region(*it, p[a], p[b])) {
                            bc = {it->type, it->parameter_index};
                            break;
                        }
                    }
                    faces.push_back(patch);
                    assignments.push_back(bc);
                }
        }
        model.discrete.faces = std::move(faces);
        model.patch_bcs = std::move(assignments);
        model.discrete.build_incidence();
        model.discrete.validate();
    }

} // namespace mhs::sim
