#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mhs::dtdm {
    namespace {
        constexpr double kGrowth = 1.30;

        // In-plane footprint of each stack level: the sink spans the whole grid,
        // the spreader and the interposer-side stack their own squares, and the
        // chip-side stack plus the TIM the chiplet union.
        bool layer_inside(int layer, double interposer, double x, double y)
        {
            const double half_sink = kSink / 2 + 1e-12;
            const double half_spread = kSpreader / 2 + 1e-12;
            const double half_interp = interposer / 2 + 1e-12;
            switch (kStack[layer].footprint) {
            case kFootprintSink:
                return std::abs(x) <= half_sink && std::abs(y) <= half_sink;
            case kFootprintSpreader:
                return std::abs(x) <= half_spread && std::abs(y) <= half_spread;
            case kFootprintChiplets:
                return false; // decided per chiplet by the caller
            default:
                return std::abs(x) <= half_interp && std::abs(y) <= half_interp;
            }
        }

        bool inside(const Chiplet& c, double x, double y, double eps = 1e-12)
        {
            return x >= c.x - eps && x <= c.x + c.width + eps && y >= c.y - eps && y <= c.y + c.height + eps;
        }

        std::vector<int> split_by_thickness(const int total, const std::vector<double>& weight)
        {
            const std::size_t n = weight.size();
            std::vector<int> base(n, 1);
            int used = static_cast<int>(n);
            if (total < used)
                throw std::runtime_error("group z count smaller than its layer count");
            const double wsum = std::accumulate(weight.begin(), weight.end(), 0.0);
            std::vector<double> share(n);
            for (std::size_t i = 0; i < n; ++i)
                share[i] = weight[i] / wsum * total;
            for (std::size_t i = 0; i < n; ++i)
                base[i] = std::max(1, static_cast<int>(std::floor(share[i])));
            used = std::accumulate(base.begin(), base.end(), 0);
            while (used < total) {
                std::size_t best = 0;
                double best_gap = -1e300;
                for (std::size_t i = 0; i < n; ++i) {
                    const double gap = share[i] - base[i];
                    if (gap > best_gap) {
                        best_gap = gap;
                        best = i;
                    }
                }
                ++base[best];
                ++used;
            }
            while (used > total) {
                std::size_t best = n;
                double best_gap = 1e300;
                for (std::size_t i = 0; i < n; ++i) {
                    if (base[i] <= 1)
                        continue;
                    const double gap = share[i] - base[i];
                    if (gap < best_gap) {
                        best_gap = gap;
                        best = i;
                    }
                }
                if (best == n)
                    break;
                --base[best];
                --used;
            }
            return base;
        }
    } // namespace

    long covered_cells(const std::vector<double>& x, const std::vector<double>& y,
        const std::vector<Chiplet>& chiplets)
    {
        long count = 0;
        for (std::size_t i = 0; i + 1 < x.size(); ++i) {
            const double cx = 0.5 * (x[i] + x[i + 1]);
            for (std::size_t j = 0; j + 1 < y.size(); ++j) {
                const double cy = 0.5 * (y[j] + y[j + 1]);
                for (const auto& c : chiplets)
                    if (inside(c, cx, cy)) {
                        ++count;
                        break;
                    }
            }
        }
        return count;
    }

    std::vector<double> build_axis(const Layout& layout, const std::vector<Chiplet>& chiplets, int axis)
    {
        auto lo_of = [axis](const Chiplet& c) { return axis == 0 ? c.x : c.y; };
        auto size_of = [axis](const Chiplet& c) { return axis == 0 ? c.width : c.height; };

        double widest = 0.0;
        std::vector<double> edges;
        for (const auto& c : chiplets) {
            widest = std::max(widest, std::max(c.width, c.height));
            edges.push_back(lo_of(c));
            edges.push_back(lo_of(c) + size_of(c));
        }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

        // One cell size for the whole fine region, so a chiplet and the gap next
        // to it are resolved the same way.
        const double fine = widest / static_cast<double>(layout.n_fine);

        std::vector<double> inner { edges.front() };
        for (std::size_t i = 0; i + 1 < edges.size(); ++i) {
            const double a = edges[i];
            const double b = edges[i + 1];
            const int count = std::max(1, static_cast<int>(std::llround((b - a) / fine)));
            for (int k = 1; k <= count; ++k)
                inner.push_back(a + (b - a) * k / count);
        }

        // Outwards the cell size grows geometrically from the chiplet resolution,
        // so the heat sink and the substrate cost a handful of cells instead of
        // carrying unknowns the physics does not need.
        auto outward = [&](double from, double to) {
            std::vector<double> weights;
            double cell = fine;
            double covered = 0.0;
            while (covered < std::abs(to - from)) {
                weights.push_back(cell);
                covered += cell;
                cell *= kGrowth;
            }
            const double scale = (to - from) / covered;
            std::vector<double> out { from };
            double acc = from;
            for (double weight : weights) {
                acc += weight * scale;
                out.push_back(acc);
            }
            out.back() = to;
            return out;
        };

        std::vector<double> out;
        const auto left = outward(inner.front(), -kSink / 2);
        out.insert(out.end(), left.rbegin(), left.rend() - 1);
        out.insert(out.end(), inner.begin(), inner.end());
        const auto right = outward(inner.back(), kSink / 2);
        out.insert(out.end(), right.begin() + 1, right.end());
        return out;
    }

    std::vector<int> solve_layer_nz(const StackMesh& probe, const Layout& layout,
        const std::vector<Chiplet>& chiplets, std::array<int, 4>& group_nz)
    {
        const auto& x = probe.axes[0];
        const auto& y = probe.axes[1];
        const int nx = static_cast<int>(x.size()) - 1;
        const int ny = static_cast<int>(y.size()) - 1;
        // Footprints counted with the same predicate the builder uses, so the
        // integer solve and the assembled mesh cannot disagree.
        auto count = [&](int layer) {
            long total = 0;
            for (int i = 0; i < nx; ++i) {
                const double cx = 0.5 * (x[static_cast<std::size_t>(i)] + x[static_cast<std::size_t>(i) + 1]);
                for (int j = 0; j < ny; ++j) {
                    const double cy = 0.5 * (y[static_cast<std::size_t>(j)] + y[static_cast<std::size_t>(j) + 1]);
                    bool hit = layer_inside(layer, kInterposer, cx, cy);
                    if (kStack[layer].footprint == kFootprintChiplets)
                        for (const auto& c : chiplets)
                            if (inside(c, cx, cy)) { hit = true; break; }
                    if (hit) ++total;
                }
            }
            return total;
        };
        const long sink = count(0);
        const long spread = count(1);
        const long chip = count(2);
        const long interp = count(6);

        // Which layers sit in each z group, and how many layers that group holds.
        std::vector<double> chip_w, interp_w;
        std::vector<int> chip_idx, interp_idx;
        for (int l = 0; l < static_cast<int>(kStack.size()); ++l) {
            if (kStack[l].footprint == kFootprintChiplets) {
                chip_w.push_back(kStack[l].thickness);
                chip_idx.push_back(l);
            }
            else if (kStack[l].footprint == kFootprintInterposer) {
                interp_w.push_back(kStack[l].thickness);
                interp_idx.push_back(l);
            }
        }

        // The z direction only has to resolve the stack. In-plane carries the
        // resolution, so the search stays inside what a thin layer can justify:
        // at least one cell per layer, at most a few more.
        const int chip_layers = static_cast<int>(chip_idx.size());
        const int interp_layers = static_cast<int>(interp_idx.size());
        long best_miss = -1;
        std::vector<int> best;
        for (int a = 1; a <= 4; ++a)
            for (int b = 1; b <= 4; ++b)
                for (int c = chip_layers; c <= chip_layers + 3; ++c)
                    for (int d = interp_layers; d <= interp_layers + 3; ++d) {
                        const long total = a * sink + b * spread + c * chip + d * interp;
                        const long miss = std::abs(total - layout.target);
                        if (best_miss < 0 || miss < best_miss) {
                            best_miss = miss;
                            const auto split_chip = split_by_thickness(c, chip_w);
                            const auto split_interp = split_by_thickness(d, interp_w);
                            std::vector<int> nz(kStack.size(), 1);
                            nz[0] = a;
                            nz[1] = b;
                            for (std::size_t i = 0; i < chip_idx.size(); ++i)
                                nz[chip_idx[i]] = split_chip[i];
                            for (std::size_t i = 0; i < interp_idx.size(); ++i)
                                nz[interp_idx[i]] = split_interp[i];
                            best = nz;
                            group_nz = { { a, b, c, d } };
                        }
                    }
        if (best.empty())
            throw std::runtime_error("mesh level has no admissible z distribution");
        return best;
    }

    StackMesh build_stack_mesh(const SystemSpec& spec, const Layout& layout)
    {
        StackMesh out;
        out.axes[0] = build_axis(layout, spec.chiplets, 0);
        out.axes[1] = build_axis(layout, spec.chiplets, 1);

        // probe pass: solve the z distribution so the DoF count is exact
        std::array<int, 4> group { { 0, 0, 0, 0 } };
        const auto nz = solve_layer_nz(out, layout, spec.chiplets, group);
        out.group_nz = group;

        const auto& x = out.axes[0];
        const auto& y = out.axes[1];
        const int nx = static_cast<int>(x.size()) - 1;
        const int ny = static_cast<int>(y.size()) - 1;

        // z vertices: layer 0 is the top of the stack
        const double total = [&] {
            double t = 0.0;
            for (const auto& layer : kStack)
                t += layer.thickness;
            return t;
        }();
        std::array<std::vector<double>, 11> z;
        double top = total;
        for (int l = 0; l < static_cast<int>(kStack.size()); ++l) {
            const double bottom = top - kStack[l].thickness;
            z[l].push_back(bottom);
            for (int k = 1; k <= nz[l]; ++k)
                z[l].push_back(bottom + kStack[l].thickness * k / nz[l]);
            top = bottom;
        }

        auto& mesh = out.model.discrete;
        std::vector<std::vector<long>> index(kStack.size());
        for (int l = 0; l < static_cast<int>(kStack.size()); ++l) {
            index[l].assign(static_cast<std::size_t>(nx) * ny * nz[l], -1);
            for (int k = 0; k < nz[l]; ++k)
                for (int i = 0; i < nx; ++i) {
                    const double cx = 0.5 * (x[i] + x[i + 1]);
                    for (int j = 0; j < ny; ++j) {
                        const double cy = 0.5 * (y[j] + y[j + 1]);
                        bool hit = layer_inside(l, spec.interposer, cx, cy);
                        if (kStack[l].footprint == kFootprintChiplets)
                            for (const auto& c : spec.chiplets)
                                if (inside(c, cx, cy)) {
                                    hit = true;
                                    break;
                                }
                        if (!hit)
                            continue;
                        const auto id = static_cast<long>(mesh.cells.size());
                        mesh.cells.push_back({ { x[i], y[j], z[l][k] }, { x[i + 1], y[j + 1], z[l][k + 1] } });
                        index[l][(static_cast<std::size_t>(k) * nx + i) * ny + j] = id;
                        out.cells_of_module[kStack[l].module].push_back(id);
                        out.cell_k.push_back(kStack[l].material.k);
                    }
                }
            out.cells_of_layer.push_back({});
            for (long id : index[l])
                if (id >= 0)
                    out.cells_of_layer.back().push_back(static_cast<mhs::core::Index>(id));
        }
        out.dof = static_cast<long>(mesh.cells.size());

        // material table: one constant entry per distinct material
        std::vector<Material> materials;
        std::vector<int> material_of_layer(kStack.size());
        for (int l = 0; l < static_cast<int>(kStack.size()); ++l) {
            const auto& m = kStack[l].material;
            auto it = std::find_if(materials.begin(), materials.end(),
                [&](const Material& e) { return std::abs(e.k - m.k) < 1e-12 && std::abs(e.rho_c - m.rho_c) < 1e-6; });
            if (it == materials.end()) {
                materials.push_back(m);
                it = materials.end() - 1;
            }
            material_of_layer[l] = static_cast<int>(it - materials.begin());
        }
        for (const auto& m : materials) {
            mhs::core::MaterialProps props;
            // rho * c is what the assembler multiplies by the volume
            props.kx = mhs::core::CompiledExpression::make_constant(m.k);
            props.ky = props.kx;
            props.kz = props.kx;
            props.rho = mhs::core::CompiledExpression::make_constant(m.rho_c);
            props.c = mhs::core::CompiledExpression::make_constant(1.0);
            out.model.material_table.push_back(std::move(props));
        }
        out.model.cell_material.assign(mesh.cells.size(), 0);
        for (int l = 0; l < static_cast<int>(kStack.size()); ++l)
            for (long id : index[l])
                if (id >= 0)
                    out.model.cell_material[static_cast<std::size_t>(id)] = static_cast<mhs::core::TableIndex>(material_of_layer[l]);

        // heat sources: one per chiplet, constant volumetric density
        out.model.heat_source_table.clear();
        for (const auto& c : spec.chiplets) {
            const double volume = c.width * c.height * kStack[3].thickness;
            out.model.heat_source_table.push_back(
                mhs::core::CompiledExpression::make_constant(c.power / volume));
        }
        out.total_power = 0.0;
        for (std::size_t ci = 0; ci < spec.chiplets.size(); ++ci) {
            const auto& c = spec.chiplets[ci];
            const double density = c.power / (c.width * c.height * kStack[3].thickness);
            for (long id : index[3])
                if (id >= 0) {
                    const auto centre = mesh.cells[static_cast<std::size_t>(id)].center();
                    if (!inside(c, centre[0], centre[1]))
                        continue;
                    const double volume = mesh.cells[static_cast<std::size_t>(id)].volume();
                    out.model.source_terms.push_back(
                        { { static_cast<mhs::core::Index>(id) }, static_cast<mhs::core::TableIndex>(ci), volume,
                            centre });
                    out.total_power += density * volume;
                }
        }

        // faces: one per interior pair (emitted from the lower cell index) plus
        // one per exterior side
        auto& faces = mesh.faces;
        auto emit = [&](long owner, long neighbor, int axis, int side) {
            mhs::core::FacePatch f;
            f.owner = { static_cast<mhs::core::Index>(owner) };
            f.neighbor = mhs::core::CellId {};
            if (neighbor >= 0)
                f.neighbor.value = static_cast<mhs::core::Index>(neighbor);
            f.axis = axis;
            const auto& co = mesh.cells[static_cast<std::size_t>(owner)];
            f.owner_distance = 0.5 * co.size()[static_cast<std::size_t>(axis)];
            f.min = co.min;
            f.max = co.max;
            if (neighbor >= 0) {
                const auto& cn = mesh.cells[static_cast<std::size_t>(neighbor)];
                f.neighbor_distance = 0.5 * cn.size()[static_cast<std::size_t>(axis)];
                const double plane = 0.5 * (co.center()[static_cast<std::size_t>(axis)]
                    + cn.center()[static_cast<std::size_t>(axis)]);
                f.min[static_cast<std::size_t>(axis)] = plane;
                f.max[static_cast<std::size_t>(axis)] = plane;
            } else {
                // exterior: the face sits on the cell side the missing neighbour
                // would be on, which is `side` along the axis
                const double plane = side > 0 ? co.max[static_cast<std::size_t>(axis)]
                                              : co.min[static_cast<std::size_t>(axis)];
                f.min[static_cast<std::size_t>(axis)] = plane;
                f.max[static_cast<std::size_t>(axis)] = plane;
            }
            f.area = (co.max[(axis + 1) % 3] - co.min[(axis + 1) % 3])
                * (co.max[(axis + 2) % 3] - co.min[(axis + 2) % 3]);
            f.sign = side;
            faces.push_back(f);
            return static_cast<long>(faces.size() - 1);
        };

        std::vector<long> sink_top_faces, pcb_bottom_faces;
        auto lookup = [&](int l, int k, int i, int j) -> long {
            if (k < 0 || k >= nz[l] || i < 0 || i >= nx || j < 0 || j >= ny)
                return -1;
            return index[l][(static_cast<std::size_t>(k) * nx + i) * ny + j];
        };
        for (int l = 0; l < static_cast<int>(kStack.size()); ++l) {
            for (int k = 0; k < nz[l]; ++k)
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j) {
                        const long self = lookup(l, k, i, j);
                        if (self < 0)
                            continue;
                        // z neighbours, possibly in an adjacent layer
                        for (int dir = -1; dir <= 1; dir += 2) {
                            long other = lookup(l, k + dir, i, j);
                            if (other < 0) {
                                const int nl = l - dir; // layer below (+1) or above (-1)
                                if (nl >= 0 && nl < static_cast<int>(kStack.size())) {
                                    const int nk = dir > 0 ? 0 : nz[nl] - 1;
                                    other = lookup(nl, nk, i, j);
                                }
                            }
                            if (other >= 0) {
                                if (other > self)
                                    emit(self, other, 2, dir);
                            } else {
                                const long id = emit(self, -1, 2, dir);
                                if (l == 0 && dir > 0)
                                    sink_top_faces.push_back(id);
                                if (l == static_cast<int>(kStack.size()) - 1 && dir < 0)
                                    pcb_bottom_faces.push_back(id);
                            }
                        }
                        for (int axis = 0; axis < 2; ++axis)
                            for (int dir = -1; dir <= 1; dir += 2) {
                                const long other = axis == 0 ? lookup(l, k, i + dir, j) : lookup(l, k, i, j + dir);
                                if (other >= 0) {
                                    if (other > self)
                                        emit(self, other, axis, dir);
                                } else {
                                    emit(self, -1, axis, dir);
                                }
                            }
                    }
        }
        mesh.build_incidence();

        // boundary conditions in RISE coordinates: the convection ambient equals
        // the initial temperature, so its contribution to the right-hand side is
        // zero and the WLP expansion sees a homogeneous initial value problem
        out.model.boundary_conditions.push_back(mhs::core::Convection {
            mhs::core::CompiledExpression::make_constant(kHeatSinkH),
            mhs::core::CompiledExpression::make_constant(0.0) });
        out.model.boundary_conditions.push_back(mhs::core::Convection {
            mhs::core::CompiledExpression::make_constant(kPcbH),
            mhs::core::CompiledExpression::make_constant(0.0) });

        auto add_term = [&](long face_id, std::size_t condition) {
            const auto& f = faces[static_cast<std::size_t>(face_id)];
            const int axis = f.axis;
            const int a = axis == 0 ? 1 : 0;
            const int b = axis == 2 ? 1 : 2;
            out.model.boundary_terms.push_back(
                { { static_cast<mhs::core::Index>(face_id) }, static_cast<mhs::core::TableIndex>(condition),
                    { f.min[a], f.max[a], f.min[b], f.max[b] } });
        };
        for (long id : sink_top_faces)
            add_term(id, 0);
        for (long id : pcb_bottom_faces)
            add_term(id, 1);
        std::sort(out.model.boundary_terms.begin(), out.model.boundary_terms.end(),
            [](const mhs::core::BoundaryTerm& l, const mhs::core::BoundaryTerm& r) {
                return l.face.value < r.face.value;
            });

        out.model.initial_temperature = kAmbient;
        out.model.study_type = mhs::core::StudyType::Transient;
        return out;
    }

} // namespace mhs::dtdm


