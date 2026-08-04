#include "api/metahotspot_macromodel.h"
#include "api/internal.h"

#include "macromodel/modal_port.hpp"
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

struct mhs_macro_port_map_t {
    std::shared_ptr<const mhs::core::Model> model;
    mhs::macro::PortMap map;
};

namespace {

    mhs::core::FaceDir to_face(mhs_face_t face)
    {
        switch (face) {
        case MHS_FACE_XM:
            return mhs::core::FaceDir::XM;
        case MHS_FACE_XP:
            return mhs::core::FaceDir::XP;
        case MHS_FACE_YM:
            return mhs::core::FaceDir::YM;
        case MHS_FACE_YP:
            return mhs::core::FaceDir::YP;
        case MHS_FACE_ZM:
            return mhs::core::FaceDir::ZM;
        case MHS_FACE_ZP:
            return mhs::core::FaceDir::ZP;
        default:
            throw std::invalid_argument("invalid face value: " + std::to_string(face));
        }
    }

} // namespace

MHS_API mhs_status_t mhs_macromodel_port_map_create(const mhs_compiled_t* compiled,
    const mhs_macro_port_patch_t* patches, size_t patch_count, mhs_macro_port_map_t** out)
{
    CHECK_NULL(compiled);
    CHECK_NULL(patches);
    CHECK_NULL(out);
    *out = nullptr;
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        if (patch_count == 0)
            throw std::invalid_argument("patch_count must be > 0");
        std::vector<mhs::macro::PortPatch> cpp_patches;
        cpp_patches.reserve(patch_count);
        for (size_t i = 0; i < patch_count; ++i) {
            const auto& patch = patches[i];
            cpp_patches.push_back({to_face(patch.face), patch.coordinate, patch.rectangle.a_min, patch.rectangle.a_max,
                patch.rectangle.b_min, patch.rectangle.b_max});
        }
        auto* result = new (std::nothrow) mhs_macro_port_map_t;
        if (!result)
            throw std::bad_alloc();
        result->model = compiled->model;
        result->map
            = mhs::macro::compile_port_map(*compiled->model, std::span<const mhs::macro::PortPatch>(cpp_patches));
        *out = result;
    });
}

MHS_API void mhs_macromodel_port_map_destroy(mhs_macro_port_map_t* map) { delete map; }

MHS_API mhs_status_t mhs_macromodel_assemble_dtn(
    const mhs_macro_port_map_t* ports, const double* state, size_t state_count, double time, mhs_operators_t** out)
{
    CHECK_NULL(ports);
    CHECK_NULL(state);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_ASSEMBLE, {
        const auto cell_count = ports->model->cells.cell_to_grid.size();
        if (state_count != cell_count)
            throw std::invalid_argument("state_count must equal compiled cell count");
        *out = nullptr;
        auto result = std::make_unique<mhs_operators_t>();
        result->operators
            = mhs::macro::assemble_dtn(*ports->model, ports->map, std::span<const double>(state, state_count), time);
        *out = result.release();
    });
}

MHS_API mhs_status_t mhs_macromodel_solve(const mhs_macro_port_map_t* ports, const mhs_operators_t* dtn,
    const double* state, size_t state_count, const mhs_solve_options_t* opts, mhs_solution_t** out)
{
    CHECK_NULL(ports);
    CHECK_NULL(dtn);
    CHECK_NULL(state);
    CHECK_NULL(out);
    *out = nullptr;
    MHS_TRY(MHS_ERR_SOLVE, {
        const auto fvm_count = ports->model->cells.cell_to_grid.size();
        const auto dtn_state_count = static_cast<size_t>(dtn->operators.f.size());
        if (dtn_state_count < ports->map.port_count)
            throw std::invalid_argument("DtN states must begin with one state per physical port");
        if (state_count != fvm_count + dtn_state_count)
            throw std::invalid_argument("state_count must equal cell_count + DtN state count");

        mhs::macro::DtNModel model;
        model.operators = dtn->operators;

        auto result = mhs::macro::solve(*ports->model, model, ports->map, std::span<const double>(state, state_count),
            to_solve_options(opts, ports->model->transient_duration));
        auto* solution = new (std::nothrow) mhs_solution_t;
        if (!solution)
            throw std::bad_alloc();
        solution->sol = std::move(result);
        solution->sol.fvm_count = fvm_count;
        solution->model = ports->model;
        *out = solution;
    });
}
