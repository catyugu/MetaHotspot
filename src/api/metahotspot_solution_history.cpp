#include "api/metahotspot_solution_history.h"
#include "api/internal.h"

MHS_API mhs_status_t mhs_solution_history_view(
    const mhs_solution_t* solution, mhs_solution_history_view_t* out)
{
    CHECK_NULL(solution);
    CHECK_NULL(out);

    const auto state_count = solution->sol.state.size();
    const auto record_count = solution->sol.snapshot_times.size();
    if (record_count > 0 && solution->sol.snapshot_states.size() != record_count * state_count) {
        SET_ERR("solution history storage is inconsistent");
        return MHS_ERR_RUNTIME;
    }

    out->times = record_count > 0 ? solution->sol.snapshot_times.data() : nullptr;
    out->states = record_count > 0 ? solution->sol.snapshot_states.data() : nullptr;
    out->record_count = record_count;
    out->state_count = state_count;
    mhs_detail_clear_last_error();
    return MHS_OK;
}
