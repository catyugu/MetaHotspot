#include "preprocessor/boundary_processor.hpp"

#include "data/tolerance_config.hpp"

namespace mhs::sim {

    std::vector<CompiledBoundaryRegion> compile_boundary_patches(
        const std::vector<mhs::model::BoundaryPatch>& boundaries, mhs::core::BCParamTable& parameters,
        double si_scale, const std::function<std::string(const std::string&)>& rewriter,
        const mhs::core::SymbolTable& symbols)
    {
        std::vector<CompiledBoundaryRegion> compiled;

        for (const auto& boundary : boundaries) {
            DefaultBoundary value;
            std::visit(overloaded {
                           [&](const mhs::model::DirichletBoundary& condition) {
                               value.type = mhs::core::BcType::FirstType;
                               value.parameter_index = static_cast<uint16_t>(parameters.dirichlet_T.size());
                               parameters.dirichlet_T.push_back(
                                   mhs::core::parse(rewriter(condition.temperature), symbols));
                           },
                           [&](const mhs::model::NeumannBoundary& condition) {
                               value.type = mhs::core::BcType::SecondType;
                               value.parameter_index = static_cast<uint16_t>(parameters.neumann_q.size());
                               parameters.neumann_q.push_back(
                                   mhs::core::parse(rewriter(condition.heat_flux), symbols));
                           },
                           [&](const mhs::model::ConvectionBoundary& condition) {
                               value.type = mhs::core::BcType::ThirdType;
                               value.parameter_index = static_cast<uint16_t>(parameters.cauchy_h.size());
                               parameters.cauchy_h.push_back(
                                   mhs::core::parse(rewriter(condition.coefficient), symbols));
                               parameters.cauchy_T_inf.push_back(
                                   mhs::core::parse(rewriter(condition.ambient_temperature), symbols));
                           },
                       },
                boundary.condition);

            for (const auto& region : boundary.regions) {
                CompiledBoundaryRegion item;
                item.axis = region.axis;
                item.coordinate = region.coordinate * si_scale;
                item.type = value.type;
                item.parameter_index = value.parameter_index;
                item.rectangles.reserve(region.rectangles.size());
                for (const auto& rectangle : region.rectangles) {
                    item.rectangles.push_back({rectangle.a_min * si_scale, rectangle.a_max * si_scale,
                        rectangle.b_min * si_scale, rectangle.b_max * si_scale});
                }
                compiled.push_back(std::move(item));
            }
        }

        return compiled;
    }

    bool point_in_region(const CompiledBoundaryRegion& region, double a, double b)
    {
        for (const auto& rectangle : region.rectangles) {
            if (a >= rectangle.a_min - mhs::core::geometry_eps && a <= rectangle.a_max + mhs::core::geometry_eps
                && b >= rectangle.b_min - mhs::core::geometry_eps
                && b <= rectangle.b_max + mhs::core::geometry_eps) {
                return true;
            }
        }
        return false;
    }

} // namespace mhs::sim
